"""
extracao_features_v8.py
═══════════════════════════════════════════════════════════════════════════════
OranDet v8.0 — Features focadas em CONTAGEM (MAPE como métrica alvo)

PROBLEMA v7: 878 features com R²=0.19 — 386 features de ruído puro
(cor/textura global) afogavam o sinal dos estimadores de contagem.

SOLUÇÃO v8: 3 princípios de curadoria
  1. Só entra feature que varia COM O NÚMERO de frutas na cena
  2. Features de textura/cor GLOBAL foram removidas (G1, G2, G6, G7, G9)
  3. Dos grupos parciais, só as razões fruta/fundo e contagens sobreviveram

GRUPOS REMOVIDOS (ruído puro, R² ~ 0 com contagem):
  ✗ G1  — HSV histogramas + first-order global (150 features)
  ✗ G2  — RGB/LAB/YCbCr first-order global    (166 features)
  ✗ G6  — LBP global                          ( 22 features)
  ✗ G7  — GLCM Haralick global                ( 32 features)
  ✗ G9  — HOG global                          ( 16 features)
  Total removido: 386 features

GRUPOS MANTIDOS INTEIROS:
  ✓ G5  — Gabor isotropy (4 orientações × 2 freqs) — textura DIRECIONAL de fruta
  ✓ G10 — Geometria + MSER circular — conta blobs diretamente
  ✓ G11 — Hough Circles (3 faixas) — conta círculos diretamente
  ✓ G12 — Grade espacial 4×4 — distribuição espacial e nº de células ativas
  ✓ G14 — Contagem direta (6 estimadores + ensemble) — núcleo do modelo
  ✓ mascara_prop_fruta — proporção da cena ocupada por fruta

GRUPOS PARCIAIS — só features que variam com contagem:
  ✓ G3  → v_razao_fruta_global, v_original_mean, v_eq_mean        (3)
  ✓ G4  → textura_razao_fundo_fruta, textura_lap_fruta_norm        (2)
  ✓ G8  → satd_razao_fundo_fruta, satd_prop_lisa_global, satd_fruta_norm (3)
  ✓ G13 → n_blobs, log_blobs, prop_fruta, gabor_isotropy nas 2 escalas (12)
  ✓ G15 → brilho_razao_vertical_* + brilho_ratio_mean/std          (5)

TARGET: log(1 + contagem)
  - MAPE penaliza erros proporcionalmente → log lineariza a escala
  - Reduz viés de "regressão à média" para contagens baixas
  - Na avaliação: exp(pred) - 1 para voltar à escala original
  - O CSV salva AMBOS: contagem_log (target de treino) e contagem (avaliação)

OUTPUTS:
  orandet_v8_train_raw.csv   → XGBoost  (~156 features + contagem + contagem_log)
  orandet_v8_test_raw.csv    → XGBoost
  orandet_v8_train_norm.csv  → MLP + SVR (mesmas features, normalizadas [0,1])
  orandet_v8_test_norm.csv   → MLP + SVR
  orandet_v8_scaler.joblib   → MinMaxScaler (fit só no treino)
  orandet_v8_info.json       → metadados e contagem de features por grupo

COMO USAR O TARGET NOS MODELOS:
  XGBoost / SVR / MLP: treinar em 'contagem_log', avaliar MAPE em 'contagem'
    y_train = df['contagem_log']
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)       # expm1 = exp(x) - 1
    mape = mean_absolute_percentage_error(df_test['contagem'], y_pred)

Referências mantidas:
  Kurtulmus et al. (2011) — Gabor circular para citrus verde
  Zhao & Lee (2016)       — SATD 83.4% citrus verde
  Maldonado & Barbosa (2016) — bas-relief + razão vertical
  Li et al. (2016)        — LAB b* para carotenoides em cítricos
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = "/Users/antonioreis/Downloads/dataverse_files"
OUTPUT_DIR = "./dataset_preparado_v8"
IMG_SIZE   = 416
N_JOBS     = -1

# contagem_log é o TARGET de treino; contagem é para avaliação de MAPE
COLUNAS_META = ["image_id", "file_name", "split", "contagem", "contagem_log", "augmentacao"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PRÉ-CÔMPUTO DE KERNELS GABOR — 1× por processo, não por imagem
# ═══════════════════════════════════════════════════════════════════════════════

def _build_gabor_kernels():
    kernels = {}
    orientacoes = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # G5: features Gabor principais (2 lambdas × 4 ângulos)
    for lam in [10, 20]:
        for theta in orientacoes:
            k = cv2.getGaborKernel(
                (21, 21), sigma=4.0, theta=theta,
                lambd=float(lam), gamma=1.0, psi=0,
            )
            kernels[(lam, int(np.degrees(theta)))] = k

    # Máscara: lambda=10 apenas
    for theta in orientacoes:
        k = cv2.getGaborKernel(
            (21, 21), sigma=4.0, theta=theta,
            lambd=10.0, gamma=1.0, psi=0,
        )
        kernels[("mask", int(np.degrees(theta)))] = k

    # G13 multi-escala: kernels menores (15×15)
    for lam in [8, 15]:
        for theta in orientacoes:
            k = cv2.getGaborKernel((15, 15), 3.0, theta, float(lam), 1.0, 0)
            kernels[("ms", lam, int(np.degrees(theta)))] = k

    return kernels

_GABOR_KERNELS = _build_gabor_kernels()


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES DA MÁSCARA
# ═══════════════════════════════════════════════════════════════════════════════

def _gabor_isotropy_map(gray):
    """
    Isotropia Gabor: alta = textura igual em todas direções = fruta esférica.
    Ref: Kurtulmus et al. (2011).
    """
    respostas = [
        np.abs(cv2.filter2D(gray, cv2.CV_64F, _GABOR_KERNELS[("mask", ang)]))
        for ang in [0, 45, 90, 135]
    ]
    stacked   = np.stack(respostas, axis=0)
    mean_resp = stacked.mean(axis=0) + 1e-9
    return (1.0 - np.clip(stacked.std(axis=0) / mean_resp, 0, 1)).astype(np.float32)


def _satd_map(gray):
    """Superfície lisa (fruta) → SATD baixo. Nervura → SATD alto. Ref: Zhao & Lee (2016)."""
    gray_f = gray.astype(np.float32)
    return (np.abs(gray_f - cv2.blur(gray_f, (5, 5))) +
            np.abs(gray_f - cv2.blur(gray_f, (11, 11)))) / 2.0


def _laplacian_smooth_map(gray):
    """Laplaciano suavizado: detecta nervuras (bordas internas)."""
    return cv2.GaussianBlur(
        np.abs(cv2.Laplacian(gray, cv2.CV_64F, ksize=3)).astype(np.float32),
        (31, 31), 0
    )


def _hough_seed_map(gray, img_size):
    """Campo de atração ao redor de círculos Hough detectados."""
    seed = np.zeros((img_size, img_size), dtype=np.float32)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    for rmin, rmax in [(15, 40), (40, 80), (80, 120)]:
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=30, param1=50, param2=40,
            minRadius=rmin, maxRadius=rmax,
        )
        if circles is not None:
            for cx, cy, r in circles[0]:
                cv2.circle(seed, (int(cx), int(cy)), int(r * 1.1), 1.0, -1)
    return cv2.GaussianBlur(seed, (21, 21), 0)


def _bas_relief_map(V_eq):
    """
    Bas-relief Maldonado: Sobel_X + Laplaciano + Blur 11×11.
    Fruta esférica sob luz superior → padrão "metade clara / metade escura".
    Ref: Maldonado & Barbosa (2016).
    """
    sx  = np.abs(cv2.Sobel(V_eq, cv2.CV_64F, 1, 0, ksize=3)).astype(np.float32)
    lap = np.abs(cv2.Laplacian(V_eq, cv2.CV_64F, ksize=3)).astype(np.float32)
    return cv2.normalize(
        cv2.GaussianBlur(cv2.addWeighted(sx, 0.6, lap, 0.4, 0), (11, 11), 0),
        None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )


def _vertical_brightness_ratio(V, mascara=None):
    """
    Razão upper_half / lower_half. Fruta esférica → 1.2–1.8. Folha → ~1.0.
    Ref: Maldonado & Barbosa (2016).
    """
    mid = V.shape[0] // 2
    if mascara is not None and mascara.sum() > 0:
        up = V[:mid, :][mascara[:mid, :] > 0]
        lo = V[mid:, :][mascara[mid:, :] > 0]
    else:
        up, lo = V[:mid, :].flatten(), V[mid:, :].flatten()
    if len(up) == 0 or len(lo) == 0:
        return 1.0
    return (float(np.mean(up)) + 1e-6) / (float(np.mean(lo)) + 1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# MÁSCARA v6.1 — 6 critérios, votação 4/6
# ═══════════════════════════════════════════════════════════════════════════════

def construir_mascara_fruta_verde(img_bgr, gray, hsv, lab):
    """
    Máscara baseada em forma/textura (não em cor).
    Recebe canais pré-computados para evitar reconversão redundante.
    """
    h, w = img_bgr.shape[:2]
    V    = hsv[:, :, 2]

    # A: Gabor isotropy — fruta esférica vs folha nervurada
    iso  = _gabor_isotropy_map(gray)
    crit_A = (iso >= float(np.percentile(iso, 65))).astype(np.float32)

    # B: SATD — superfície lisa vs nervuras
    satd = _satd_map(gray)
    crit_B = (satd <= float(np.percentile(satd, 40))).astype(np.float32)

    # C: Laplaciano suavizado — sem bordas internas
    lap  = _laplacian_smooth_map(gray)
    crit_C = (lap <= float(np.percentile(lap, 45))).astype(np.float32)

    # D: LAB b* — carotenoides elevam b* em cítricos (Li et al., 2016)
    b_ch = lab[:, :, 2].astype(np.float32)
    crit_D = (b_ch >= float(np.percentile(b_ch, 55))).astype(np.float32)

    # E: Hough seed — geometria circular independente de cor
    crit_E = (_hough_seed_map(gray, h) > 0.05).astype(np.float32)

    # F: Razão de brilho vertical — Maldonado (2016), threshold adaptativo
    ratio_map = np.zeros((h, w), dtype=np.float32)
    step = 32
    for i in range(0, h - step, step // 2):
        for j in range(0, w - step, step // 2):
            p   = V[i:i + step, j:j + step]
            mid = step // 2
            ratio_map[i:i + step, j:j + step] = np.clip(
                (np.mean(p[:mid, :]) + 1e-6) / (np.mean(p[mid:, :]) + 1e-6), 0, 3
            )
    valid     = ratio_map[ratio_map > 0]
    thr_ratio = float(np.percentile(valid, 60)) if len(valid) > 0 else 1.15
    crit_F    = (ratio_map >= thr_ratio).astype(np.float32)

    # Votação 4/6 + morfologia
    mask = ((crit_A + crit_B + crit_C + crit_D + crit_E + crit_F) >= 4).astype(np.uint8) * 255
    k3   = np.ones((3, 3), np.uint8)
    k9   = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k9, iterations=3)

    # Filtra blobs por circularidade mínima
    cnts, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final    = np.zeros_like(mask)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 150:
            continue
        perim = cv2.arcLength(cnt, True)
        if 4 * np.pi * area / (perim ** 2 + 1e-6) >= 0.35:
            cv2.drawContours(final, [cnt], -1, 255, -1)

    prop = float(final.sum()) / (255.0 * h * w)
    return final if 0.003 <= prop <= 0.40 else np.zeros_like(final)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_skew(arr):
    v = skew(arr.flatten())
    return float(v) if np.isfinite(v) else 0.0

def _safe_kurtosis(arr):
    v = kurtosis(arr.flatten())
    return float(v) if np.isfinite(v) else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# G5 — Gabor Circular: isotropy score (MANTIDO INTEIRO)
#
# Por que manter: prop_isotropy_high estima a PROPORÇÃO DA CENA com textura
# de fruta — cresce com o número de frutas visíveis.
# Ref: Kurtulmus et al. (2011) — 75.3% acurácia citrus verde.
# ═══════════════════════════════════════════════════════════════════════════════

def features_gabor(gray):
    feats = {}
    for lam in [10, 20]:
        respostas = []
        for ang in [0, 45, 90, 135]:
            resp     = cv2.filter2D(gray, cv2.CV_64F, _GABOR_KERNELS[(lam, ang)])
            resp_abs = np.abs(resp)
            resp_u8  = np.clip(resp_abs / (resp_abs.max() + 1e-9) * 255, 0, 255).astype(np.uint8)
            respostas.append(resp_abs)
            feats[f"gabor_lam{lam}_a{ang:03d}_mean"]   = float(resp_u8.mean() / 255.0)
            feats[f"gabor_lam{lam}_a{ang:03d}_std"]    = float(resp_u8.std()  / 255.0)
            feats[f"gabor_lam{lam}_a{ang:03d}_energy"] = float(np.mean(resp_abs ** 2) / 255.0 ** 2)

        stacked  = np.stack(respostas, axis=0)
        mean_r   = stacked.mean(axis=0) + 1e-9
        isotropy = 1.0 - np.clip(stacked.std(axis=0) / mean_r, 0, 1)

        feats[f"gabor_lam{lam}_isotropy_mean"]      = float(isotropy.mean())
        feats[f"gabor_lam{lam}_isotropy_std"]       = float(isotropy.std())
        feats[f"gabor_lam{lam}_isotropy_p75"]       = float(np.percentile(isotropy, 75))
        feats[f"gabor_lam{lam}_isotropy_p90"]       = float(np.percentile(isotropy, 90))
        feats[f"gabor_lam{lam}_prop_isotropy_high"] = float((isotropy > 0.7).mean())

    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# G10 — Geometria de contornos + MSER (MANTIDO INTEIRO)
#
# Por que manter: conta blobs circulares diretamente. Cada blob circular
# na máscara = 1 fruta candidata. MSER_circular é o melhor proxy de contagem
# entre todos os estimadores quando a máscara funciona bem.
# ═══════════════════════════════════════════════════════════════════════════════

def features_geometria(gray, mascara):
    feats = {}
    k3    = np.ones((3, 3), np.uint8)
    mask  = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k3)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas, solidities, aspect_ratios, circularities = [], [], [], []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        areas.append(area)
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidities.append(area / hull_area if hull_area > 0 else 0.0)
        _, _, cw, ch = cv2.boundingRect(cnt)
        aspect_ratios.append(float(cw) / ch if ch > 0 else 0.0)
        perim = cv2.arcLength(cnt, True)
        circularities.append(4 * np.pi * area / perim ** 2 if perim > 0 else 0.0)

    for nome, lista in [("area", areas), ("solidity", solidities),
                        ("aspect_ratio", aspect_ratios), ("circularity", circularities)]:
        arr = np.array(lista) if lista else np.array([0.0])
        feats[f"geom_{nome}_mean"]  = float(arr.mean())
        feats[f"geom_{nome}_std"]   = float(arr.std())
        feats[f"geom_{nome}_max"]   = float(arr.max())
        feats[f"geom_{nome}_count"] = float(len(lista))

    n_circ = sum(1 for c in circularities if c >= 0.5)
    feats["geom_n_blobs_circulares"]    = float(n_circ)
    feats["geom_prop_blobs_circulares"] = float(n_circ / (len(circularities) + 1e-6))

    try:
        mser = cv2.MSER_create(
            _delta=8, _min_area=400, _max_area=25000,
            _max_variation=0.12, _min_diversity=0.20,
        )
        regs, _ = mser.detectRegions(gray)
        n_total = n_circ_mser = 0
        for pts in regs:
            if len(pts) < 400:
                continue
            hull = cv2.convexHull(pts.reshape(-1, 1, 2))
            ha   = cv2.contourArea(hull)
            if ha > 0 and (len(pts) / ha) > 0.55:
                n_total += 1
                if len(pts) >= 5:
                    _, (ma, mi), _ = cv2.fitEllipse(pts.reshape(-1, 1, 2))
                    if mi > 0 and (ma / mi) < 1.5:
                        n_circ_mser += 1
    except Exception:
        n_total = n_circ_mser = 0

    feats["geom_mser_total"]    = float(n_total)
    feats["geom_mser_circular"] = float(n_circ_mser)
    feats["geom_log_mser"]      = float(np.log1p(n_circ_mser))
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# G11 — Hough Circles (3 faixas de raio) (MANTIDO INTEIRO)
#
# Por que manter: estimador mais direto de contagem disponível.
# param2=40 para menor taxa de falsos positivos.
# ═══════════════════════════════════════════════════════════════════════════════

def features_hough_circles(gray, mascara):
    feats = {}
    blur  = cv2.GaussianBlur(gray, (9, 9), 2)

    total_circ = 0
    for rmin, rmax, nome in [(15, 40, "pequeno"), (40, 80, "medio"), (80, 120, "grande")]:
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=30, param1=50, param2=40,
            minRadius=rmin, maxRadius=rmax,
        )
        n_v   = int(len(circles[0])) if circles is not None else 0
        radii = [float(r) for _, _, r in circles[0]] if circles is not None else []
        feats[f"hough_{nome}_count"]     = float(n_v)
        feats[f"hough_{nome}_raio_mean"] = float(np.mean(radii)) if radii else 0.0
        feats[f"hough_{nome}_raio_std"]  = float(np.std(radii))  if radii else 0.0
        total_circ += n_v

    feats["hough_total_estimado"] = float(total_circ)
    feats["hough_log_total"]      = float(np.log1p(total_circ))
    feats["hough_sqrt_total"]     = float(np.sqrt(total_circ))
    feats["hough_prop_pequenos"]  = float(feats["hough_pequeno_count"] / (total_circ + 1e-7))
    feats["hough_prop_medios"]    = float(feats["hough_medio_count"]   / (total_circ + 1e-7))
    feats["hough_prop_grandes"]   = float(feats["hough_grande_count"]  / (total_circ + 1e-7))

    if mascara.sum() > 0:
        gray_m           = gray.copy()
        gray_m[mascara == 0] = 0
        circles_m = cv2.HoughCircles(
            cv2.GaussianBlur(gray_m, (9, 9), 2), cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=30, param1=50, param2=40,
            minRadius=15, maxRadius=120
        )
        n_m = int(len(circles_m[0])) if circles_m is not None else 0
    else:
        n_m = 0
    feats["hough_mascara_count"] = float(n_m)
    feats["hough_log_mascara"]   = float(np.log1p(n_m))
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# G12 — Grade espacial 4×4 (MANTIDO INTEIRO)
#
# Por que manter: células ativas e entropia espacial capturam QUANTAS regiões
# da cena têm fruta — proxy robusto de contagem mesmo com máscara imperfeita.
# ═══════════════════════════════════════════════════════════════════════════════

def features_grade_espacial(hsv, mascara, grid=(4, 4)):
    feats  = {}
    V      = hsv[:, :, 2].astype(np.float64)
    H_img, W_img = mascara.shape
    gh, gw = H_img // grid[0], W_img // grid[1]

    densidades = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            y1, y2 = i * gh, (i + 1) * gh
            x1, x2 = j * gw, (j + 1) * gw
            dens   = float(mascara[y1:y2, x1:x2].mean()) / 255.0
            brilho = float(V[y1:y2, x1:x2].mean()) / 255.0
            feats[f"grade_{i}_{j}_densidade"] = dens
            feats[f"grade_{i}_{j}_brilho"]    = brilho
            densidades.append(dens)

    d = np.array(densidades)
    feats["grade_dens_mean"]        = float(d.mean())
    feats["grade_dens_std"]         = float(d.std())
    feats["grade_dens_max"]         = float(d.max())
    feats["grade_dens_min"]         = float(d.min())
    feats["grade_dens_range"]       = float(d.max() - d.min())
    feats["grade_n_celulas_ativas"] = float((d > 0.05).sum())
    feats["grade_frac_ativa"]       = float((d > 0.05).mean())

    d_norm = d / (d.sum() + 1e-9)
    d_nz   = d_norm[d_norm > 0]
    feats["grade_entropia_espacial"] = float(-np.sum(d_nz * np.log2(d_nz)))

    n        = len(d)
    d_sorted = np.sort(d)
    feats["grade_concentracao"] = float(
        (2 * np.sum(np.arange(1, n + 1) * d_sorted)) /
        (n * d_sorted.sum() + 1e-9) - (n + 1) / n
    )
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# G14 — Contagem direta: 6 estimadores + ensemble (MANTIDO INTEIRO)
#
# Por que manter: este é o grupo com maior correlação direta com o target.
# Transformações log/sqrt linearizam a distribuição de Poisson da contagem.
# ═══════════════════════════════════════════════════════════════════════════════

def features_contagem_direta(gray, hsv, mascara):
    feats = {}
    V     = hsv[:, :, 2]

    # Estimador 1: Hough
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    total_hough = 0
    for rmin, rmax in [(15, 40), (40, 80), (80, 120)]:
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=30, param1=50, param2=40,
            minRadius=rmin, maxRadius=rmax,
        )
        total_hough += int(len(circles[0])) if circles is not None else 0
    feats["cnt_hough_n"]    = float(total_hough)
    feats["cnt_hough_log"]  = float(np.log1p(total_hough))
    feats["cnt_hough_sqrt"] = float(np.sqrt(total_hough))

    # Estimador 2: MSER circular
    try:
        mser = cv2.MSER_create(
            _delta=8, _min_area=400, _max_area=25000,
            _max_variation=0.12, _min_diversity=0.20,
        )
        regs, _ = mser.detectRegions(gray)
        n_mser  = 0
        for pts in regs:
            if len(pts) < 400:
                continue
            hull = cv2.convexHull(pts.reshape(-1, 1, 2))
            ha   = cv2.contourArea(hull)
            if ha > 0 and (len(pts) / ha) > 0.55 and len(pts) >= 5:
                _, (ma, mi), _ = cv2.fitEllipse(pts.reshape(-1, 1, 2))
                if mi > 0 and (ma / mi) < 1.5:
                    n_mser += 1
    except Exception:
        n_mser = 0
    feats["cnt_mser_n"]    = float(n_mser)
    feats["cnt_mser_log"]  = float(np.log1p(n_mser))
    feats["cnt_mser_sqrt"] = float(np.sqrt(n_mser))

    # Estimador 3: Blobs circulares na máscara
    k3     = np.ones((3, 3), np.uint8)
    mask_c = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k3)
    cnts, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_blob  = sum(
        1 for cnt in cnts
        if cv2.contourArea(cnt) >= 100 and
        4 * np.pi * cv2.contourArea(cnt) / (cv2.arcLength(cnt, True) ** 2 + 1e-6) >= 0.35
    )
    feats["cnt_blob_n"]    = float(n_blob)
    feats["cnt_blob_log"]  = float(np.log1p(n_blob))
    feats["cnt_blob_sqrt"] = float(np.sqrt(n_blob))

    # Estimador 4: Área de isotropy como proxy de área de fruta
    iso_map        = _gabor_isotropy_map(gray)
    prop_iso       = float((iso_map > 0.7).mean())
    cnt_iso_est    = (prop_iso * IMG_SIZE * IMG_SIZE) / (np.pi * 30 ** 2 + 1e-6)
    feats["cnt_estimativa_area_iso"]      = float(cnt_iso_est)
    feats["cnt_estimativa_area_iso_log"]  = float(np.log1p(cnt_iso_est))
    feats["cnt_estimativa_area_iso_sqrt"] = float(np.sqrt(max(0, cnt_iso_est)))

    # Estimador 5: Células ativas na grade 4×4
    H_img, W_img = mascara.shape
    gh, gw = H_img // 4, W_img // 4
    celulas_ativas = sum(
        1 for i in range(4) for j in range(4)
        if (mascara[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw].mean() / 255.0) > 0.05
    )
    feats["cnt_celulas_ativas"]      = float(celulas_ativas)
    feats["cnt_celulas_ativas_log"]  = float(np.log1p(celulas_ativas))
    feats["cnt_celulas_ativas_sqrt"] = float(np.sqrt(celulas_ativas))

    # Estimador 6: Bas-relief + razão vertical (Maldonado 2016)
    bas_relief  = _bas_relief_map(V)
    ratio_map   = np.zeros_like(V, dtype=np.float32)
    step = 32
    for i in range(0, V.shape[0] - step, step // 2):
        for j in range(0, V.shape[1] - step, step // 2):
            p   = V[i:i + step, j:j + step]
            mid = step // 2
            ratio_map[i:i + step, j:j + step] = (
                (np.mean(p[:mid, :]) + 1e-6) / (np.mean(p[mid:, :]) + 1e-6)
            )
    candidatos  = (
        (bas_relief >= float(np.percentile(bas_relief, 55))) & (ratio_map >= 1.18)
    ).astype(np.uint8) * 255
    cnts_br, _  = cv2.findContours(candidatos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_basrelief = sum(1 for cnt in cnts_br if cv2.contourArea(cnt) >= 200)
    feats["cnt_basrelief_n"]    = float(n_basrelief)
    feats["cnt_basrelief_log"]  = float(np.log1p(n_basrelief))
    feats["cnt_basrelief_sqrt"] = float(np.sqrt(n_basrelief))

    # Ensemble ponderado
    ensemble = (
        0.30 * total_hough  +
        0.25 * n_mser       +
        0.15 * n_blob       +
        0.15 * cnt_iso_est  +
        0.15 * n_basrelief
    )
    feats["cnt_ensemble"]      = float(ensemble)
    feats["cnt_ensemble_log"]  = float(np.log1p(ensemble))
    feats["cnt_ensemble_sqrt"] = float(np.sqrt(max(0, ensemble)))

    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURES PARCIAIS — só as razões/métricas que variam com contagem
# ═══════════════════════════════════════════════════════════════════════════════

def features_parciais_uteis(gray, hsv, mascara, V_eq):
    """
    Extrai APENAS as features selecionadas dos grupos parciais G3, G4, G8, G15.
    Nenhuma estatística global de cor ou textura — só razões e métricas
    que crescem/diminuem com o número de frutas na cena.
    """
    feats = {}
    V = hsv[:, :, 2]

    # ── G3: razão de brilho fruta/global (a fruta tem V maior que o fundo) ──
    tem = mascara.sum() > 0
    v_fruta  = float(V[mascara > 0].mean()) if tem else 0.0
    v_global = float(V.mean()) + 1e-6
    feats["v_razao_fruta_global"] = v_fruta / v_global
    feats["v_original_mean"]      = float(V.mean() / 255.0)
    feats["v_eq_mean"]            = float(V_eq.mean() / 255.0)

    # ── G4: razão de textura fundo/fruta (fruta lisa → Laplaciano baixo) ──
    V_blur   = cv2.GaussianBlur(V_eq, (5, 5), 0)
    lap      = np.abs(cv2.Laplacian(V_blur, cv2.CV_64F, ksize=3)).astype(np.uint8)
    lap_g    = float(lap.mean()) + 1e-6
    fundo    = mascara == 0
    if tem:
        lap_fruta = float(lap[mascara > 0].mean()) + 1e-6
        lap_fundo = float(lap[fundo].mean())        + 1e-6 if fundo.sum() > 0 else lap_g
    else:
        lap_fruta = lap_fundo = lap_g
    feats["textura_razao_fundo_fruta"] = float(lap_fundo / lap_fruta)
    feats["textura_lap_fruta_norm"]    = float(lap_fruta / 255.0)

    # ── G8: razão SATD fundo/fruta (fruta lisa → SATD baixo) ──────────────
    satd_map  = _satd_map(gray)
    satd_g    = float(satd_map.mean()) + 1e-6
    if tem:
        satd_fruta = float(satd_map[mascara > 0].mean()) + 1e-6
        satd_fundo = float(satd_map[fundo].mean())       + 1e-6 if fundo.sum() > 0 else satd_g
    else:
        satd_fruta = satd_fundo = satd_g
    feats["satd_razao_fundo_fruta"] = float(satd_fundo / satd_fruta)
    feats["satd_fruta_norm"]        = float(satd_fruta / 255.0)
    feats["satd_prop_lisa_global"]  = float((satd_map <= float(np.percentile(satd_map, 35))).mean())

    # ── G15: razão de brilho vertical (Maldonado 2016) ───────────────────
    ratio_global = _vertical_brightness_ratio(V_eq)
    ratio_fruta  = _vertical_brightness_ratio(V_eq, mascara) if tem else ratio_global
    feats["brilho_razao_vertical_global"] = float(np.clip(ratio_global, 0, 5))
    feats["brilho_razao_vertical_fruta"]  = float(np.clip(ratio_fruta,  0, 5))
    feats["brilho_razao_vertical_diff"]   = float(ratio_fruta - ratio_global)

    # histograma local de razões verticais → distribuição de regiões esféricas
    h, w = V_eq.shape
    ratios_local = []
    step = 32
    for i in range(0, h - step, step // 2):
        for j in range(0, w - step, step // 2):
            patch = V_eq[i:i + step, j:j + step]
            if patch.size < step * step * 0.3:
                continue
            mid = step // 2
            ratios_local.append(np.clip(
                (np.mean(patch[:mid, :]) + 1e-6) / (np.mean(patch[mid:, :]) + 1e-6), 0, 3
            ))
    feats["brilho_ratio_mean"] = float(np.mean(ratios_local))  if ratios_local else 0.0
    feats["brilho_ratio_std"]  = float(np.std(ratios_local))   if ratios_local else 0.0

    return feats


def features_multiescala_reduzido(img_bgr):
    """
    G13 reduzido: só n_blobs, prop_fruta e Gabor isotropy por escala.
    Remove histogramas HSV (ruído) e mantém apenas métricas de contagem.
    """
    feats = {}
    for fator, nome in [(0.5, "escala_208"), (0.25, "escala_104")]:
        sz     = (int(IMG_SIZE * fator), int(IMG_SIZE * fator))
        img_r  = cv2.resize(img_bgr, sz, interpolation=cv2.INTER_AREA)
        hsv_r  = cv2.cvtColor(img_r, cv2.COLOR_BGR2HSV)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        # Máscara na escala reduzida
        lab_r  = cv2.cvtColor(img_r, cv2.COLOR_BGR2LAB)
        mask_r = construir_mascara_fruta_verde(img_r, gray_r, hsv_r, lab_r)
        feats[f"{nome}_prop_fruta"] = float(mask_r.mean()) / 255.0

        # Contagem de blobs circulares na escala reduzida
        k3     = np.ones((3, 3), np.uint8)
        mask_c = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, k3)
        cnts, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_blobs = sum(
            1 for cnt in cnts
            if cv2.contourArea(cnt) >= 10 and
            4 * np.pi * cv2.contourArea(cnt) / (cv2.arcLength(cnt, True) ** 2 + 1e-6) >= 0.35
        )
        feats[f"{nome}_n_blobs"]   = float(n_blobs)
        feats[f"{nome}_log_blobs"] = float(np.log1p(n_blobs))

        # Gabor isotropy na escala reduzida (multi-escala de textura)
        for lam in [8, 15]:
            resps = [
                np.abs(cv2.filter2D(gray_r, cv2.CV_64F, _GABOR_KERNELS[("ms", lam, ang)]))
                for ang in [0, 45, 90, 135]
            ]
            stk = np.stack(resps, axis=0)
            iso = 1.0 - np.clip(stk.std(axis=0) / (stk.mean(axis=0) + 1e-9), 0, 1)
            feats[f"{nome}_gabor_lam{lam}_isotropy_mean"] = float(iso.mean())
            feats[f"{nome}_gabor_lam{lam}_prop_high"]     = float((iso > 0.7).mean())

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline completo por imagem
# ─────────────────────────────────────────────────────────────────────────────

def _extrair_de_img(img_bgr):
    """
    Conversões de cor calculadas UMA VEZ e repassadas.
    Grupos de ruído puro (G1, G2, G6, G7, G9) REMOVIDOS.
    """
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    # Conversões únicas
    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    V_eq  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(hsv[:, :, 2])

    mascara = construir_mascara_fruta_verde(img_bgr, gray, hsv, lab)

    f = {}
    f.update(features_gabor(gray))                             # G5  — isotropy
    f.update(features_geometria(gray, mascara))                # G10 — blobs + MSER
    f.update(features_hough_circles(gray, mascara))            # G11 — círculos
    f.update(features_grade_espacial(hsv, mascara))            # G12 — grade 4×4
    f.update(features_contagem_direta(gray, hsv, mascara))     # G14 — estimadores
    f.update(features_parciais_uteis(gray, hsv, mascara, V_eq))# G3+G4+G8+G15 selecionados
    f.update(features_multiescala_reduzido(img_bgr))           # G13 reduzido

    f["mascara_prop_fruta"] = float(mascara.mean()) / 255.0
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Augmentação
# ─────────────────────────────────────────────────────────────────────────────

def _ajusta_brilho(img, fator):
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h[:, :, 2] = np.clip(h[:, :, 2] * fator, 0, 255)
    return cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2BGR)

def augmentar_imagem(img_bgr):
    return [
        (cv2.flip(img_bgr, 1), "flip_h"),
        (cv2.flip(img_bgr, 0), "flip_v"),
        (_ajusta_brilho(img_bgr, 0.75),  "bright_75"),
        (_ajusta_brilho(img_bgr, 1.25), "bright_125"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Leitura de anotações COCO
# ─────────────────────────────────────────────────────────────────────────────

def carregar_anotacoes(ann_file, img_dir):
    with open(ann_file, "r") as f:
        coco = json.load(f)
    id_para_img = {img["id"]: img for img in coco["images"]}
    contagem    = {img["id"]: 0   for img in coco["images"]}
    for ann in coco["annotations"]:
        contagem[ann["image_id"]] += 1
    return [
        {
            "image_id": img_id,
            "file_name": info["file_name"],
            "caminho":   os.path.join(img_dir, info["file_name"]),
            "contagem":  contagem[img_id],
        }
        for img_id, info in id_para_img.items()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Worker paralelo
# ─────────────────────────────────────────────────────────────────────────────

def _processar_registro(reg, aplicar_augmentacao):
    linhas = []
    try:
        img = cv2.imread(reg["caminho"])
        if img is None:
            raise FileNotFoundError(reg["caminho"])
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        feats = _extrair_de_img(img)
        linha = {
            "image_id":     reg["image_id"],
            "file_name":    reg["file_name"],
            "split":        reg.get("split", ""),
            "contagem":     reg["contagem"],
            "contagem_log": float(np.log1p(reg["contagem"])),  # TARGET de treino
            "augmentacao":  "original",
        }
        linha.update(feats)
        linhas.append(linha)

        if aplicar_augmentacao:
            for img_aug, aug_nome in augmentar_imagem(img):
                try:
                    fa = _extrair_de_img(img_aug)
                    la = {
                        "image_id":     f"{reg['image_id']}_{aug_nome}",
                        "file_name":    f"{aug_nome}_{reg['file_name']}",
                        "split":        reg.get("split", ""),
                        "contagem":     reg["contagem"],
                        "contagem_log": float(np.log1p(reg["contagem"])),
                        "augmentacao":  aug_nome,
                    }
                    la.update(fa)
                    linhas.append(la)
                except Exception:
                    pass
    except Exception:
        pass
    return linhas


# ─────────────────────────────────────────────────────────────────────────────
# Processamento paralelo
# ─────────────────────────────────────────────────────────────────────────────

def processar_split(registros, nome_split, aplicar_augmentacao=False):
    total = len(registros)
    mult  = 5 if aplicar_augmentacao else 1
    print(f"\n  {nome_split}: {total} imagens → ~{total * mult} amostras (n_jobs={N_JOBS})...")

    resultados = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(_processar_registro)({**reg, "split": nome_split}, aplicar_augmentacao)
        for reg in registros
    )
    linhas = [l for sub in resultados for l in sub]
    print(f"  {nome_split}: {len(linhas)} amostras geradas.")
    return pd.DataFrame(linhas)


# ─────────────────────────────────────────────────────────────────────────────
# Normalização — fit APENAS no treino, contagem e contagem_log NUNCA normalizadas
# ─────────────────────────────────────────────────────────────────────────────

def normalizar(df_train, df_test):
    cols = [c for c in df_train.columns if c not in COLUNAS_META]

    for df in [df_train, df_test]:
        df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)

    medianas        = df_train[cols].median()
    df_train[cols]  = df_train[cols].fillna(medianas)
    df_test[cols]   = df_test[cols].fillna(medianas)

    scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    scaler.fit(df_train[cols])

    df_tn = df_train.copy()
    df_tt = df_test.copy()
    df_tn[cols] = scaler.transform(df_train[cols])
    df_tt[cols] = scaler.transform(df_test[cols])

    # Diagnóstico de clipping
    test_raw = df_test[cols].values
    clipped  = np.any((test_raw < scaler.data_min_) | (test_raw > scaler.data_max_), axis=0)
    n_clip   = clipped.sum()
    if n_clip > 0:
        cols_clip = [c for c, cl in zip(cols, clipped) if cl]
        print(f"  [aviso] {n_clip} feature(s) clipadas no teste: {cols_clip[:10]}"
              + (" ..." if n_clip > 10 else ""))

    constantes = [c for c, mn, mx in zip(cols, scaler.data_min_, scaler.data_max_) if mn == mx]
    if constantes:
        print(f"  [aviso] {len(constantes)} feature(s) com variância zero: {constantes[:5]}")

    print(f"  Treino norm: [{df_tn[cols].min().min():.4f}, {df_tn[cols].max().max():.4f}]")
    print(f"  Teste  norm: [{df_tt[cols].min().min():.4f}, {df_tt[cols].max().max():.4f}]")

    return df_tn, df_tt, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Metadados
# ─────────────────────────────────────────────────────────────────────────────

def gerar_info(df_train, df_test):
    cols    = [c for c in df_train.columns if c not in COLUNAS_META]
    df_orig = df_train[df_train["augmentacao"] == "original"]

    grupos = {
        "G5_gabor":           [c for c in cols if c.startswith("gabor_")],
        "G10_geometria_mser": [c for c in cols if c.startswith("geom_")],
        "G11_hough":          [c for c in cols if c.startswith("hough_")],
        "G12_grade":          [c for c in cols if c.startswith("grade_")],
        "G14_contagem":       [c for c in cols if c.startswith("cnt_")],
        "G13_multiescala":    [c for c in cols if c.startswith("escala_")],
        "G3_v_ratio":         [c for c in cols if c.startswith("v_")],
        "G4_textura_ratio":   [c for c in cols if c.startswith("textura_")],
        "G8_satd_ratio":      [c for c in cols if c.startswith("satd_")],
        "G15_brilho":         [c for c in cols if c.startswith("brilho_")],
        "mascara":            [c for c in cols if c.startswith("mascara_")],
    }

    removidos = {
        "G1_hsv_histogramas":     "~150 features — cor global não varia com nº de frutas",
        "G2_rgb_lab_ycbcr":       "~166 features — estatísticas de cor global = ruído puro",
        "G6_lbp":                 " ~22 features — textura local global independente de contagem",
        "G7_glcm_haralick":       " ~32 features — Haralick global independente de contagem",
        "G9_hog":                 " ~16 features — gradiente global não correlaciona com contagem",
        "G3_parcial_removido":    "first_order v_original/v_eq (36 features) — só razão mantida",
        "G4_parcial_removido":    "first_order sobel/laplace/basrelief (90 features) — só razão mantida",
        "G8_parcial_removido":    "first_order satd_global/fruta (36 features) — só razão mantida",
        "G13_parcial_removido":   "HSV histogramas multi-escala — ruído, não conta frutas",
        "G15_parcial_removido":   "first_order basrelief_maldonado (36 features) — só razão mantida",
    }

    return {
        "gerado_em":        datetime.now().strftime("%d/%m/%Y %H:%M"),
        "versao":           "8.0 — features focadas em contagem, target log(1+y)",
        "img_size":         IMG_SIZE,
        "n_features_total": len(cols),
        "n_por_grupo":      {k: len(v) for k, v in grupos.items()},
        "features_removidas_v8": removidos,
        "target": {
            "coluna_treino":   "contagem_log = log(1 + contagem)",
            "coluna_avaliacao":"contagem (escala original)",
            "como_usar": (
                "y_train = df['contagem_log']; "
                "y_pred = np.expm1(model.predict(X_test)); "
                "mape = mean_absolute_percentage_error(df_test['contagem'], y_pred)"
            ),
            "motivo": (
                "MAPE penaliza erros proporcionalmente. log(1+y) reduz o viés de "
                "regressão à média para contagens baixas (principal causa do MAPE "
                "de 123% para contagem=1 na v7). expm1 garante pred >= 0."
            ),
        },
        "outputs": {
            "orandet_v8_train_raw.csv":  "XGBoost — ~{} features + contagem + contagem_log".format(len(cols)),
            "orandet_v8_test_raw.csv":   "XGBoost",
            "orandet_v8_train_norm.csv": "MLP + SVR — mesmas features normalizadas [0,1]",
            "orandet_v8_test_norm.csv":  "MLP + SVR",
            "orandet_v8_scaler.joblib":  "MinMaxScaler (fit só no treino)",
        },
        "treino": {
            "n_originais": int(len(df_orig)),
            "n_total":     int(len(df_train)),
            "aug":         ["flip_h", "flip_v", "bright_75", "bright_125"],
            "cnt_min":     int(df_orig["contagem"].min()),
            "cnt_max":     int(df_orig["contagem"].max()),
            "cnt_media":   round(float(df_orig["contagem"].mean()), 2),
            "n_zero":      int((df_orig["contagem"] == 0).sum()),
        },
        "teste": {
            "n_imagens": int(len(df_test)),
            "cnt_min":   int(df_test["contagem"].min()),
            "cnt_max":   int(df_test["contagem"].max()),
            "cnt_media": round(float(df_test["contagem"].mean()), 2),
            "n_zero":    int((df_test["contagem"] == 0).sum()),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    data_dir  = Path(DATA_DIR)
    img_dir   = str(data_dir / "images")
    ann_train = str(data_dir / "annotations_coco" / "instances_train.json")
    ann_test  = str(data_dir / "annotations_coco" / "instances_test.json")

    print("\n" + "═" * 65)
    print("  OranDet v8.0 — Features focadas em contagem")
    print("  Removidos: G1 HSV | G2 RGB/LAB | G6 LBP | G7 GLCM | G9 HOG")
    print("  Target: log(1 + contagem)  →  avaliação em escala original")
    print("  Grupos: G5 G10 G11 G12 G14 + parciais de G3 G4 G8 G13 G15")
    print("═" * 65)

    print("\n[1/5] Carregando anotações...")
    reg_train = carregar_anotacoes(ann_train, img_dir)
    reg_test  = carregar_anotacoes(ann_test,  img_dir)
    print(f"  Treino: {len(reg_train)} | Teste: {len(reg_test)}")

    print("\n[2/5] Extraindo features (paralelo)...")
    df_train = processar_split(reg_train, "train", aplicar_augmentacao=True)
    df_test  = processar_split(reg_test,  "test",  aplicar_augmentacao=False)

    print("\n[3/5] Salvando raw (XGBoost)...")
    df_train.to_csv(os.path.join(OUTPUT_DIR, "orandet_v8_train_raw.csv"), index=False)
    df_test.to_csv( os.path.join(OUTPUT_DIR, "orandet_v8_test_raw.csv"),  index=False)
    print(f"  Salvo: orandet_v8_train_raw.csv  ({len(df_train)} amostras)")
    print(f"  Salvo: orandet_v8_test_raw.csv   ({len(df_test)}  amostras)")

    print("\n[4/5] Normalizando [0,1] (MLP + SVR)...")
    df_train_norm, df_test_norm, scaler = normalizar(df_train, df_test)
    df_train_norm.to_csv(os.path.join(OUTPUT_DIR, "orandet_v8_train_norm.csv"), index=False)
    df_test_norm.to_csv( os.path.join(OUTPUT_DIR, "orandet_v8_test_norm.csv"),  index=False)
    joblib.dump(scaler,  os.path.join(OUTPUT_DIR, "orandet_v8_scaler.joblib"))
    print("  Salvo: orandet_v8_train_norm.csv / test_norm.csv / scaler.joblib")

    print("\n[5/5] Salvando metadados...")
    info = gerar_info(df_train, df_test)
    with open(os.path.join(OUTPUT_DIR, "orandet_v8_info.json"), "w",
              encoding="utf-8") as fj:
        json.dump(info, fj, indent=2, ensure_ascii=False)

    cols = [c for c in df_train.columns if c not in COLUNAS_META]
    print(f"\n{'═' * 65}")
    print(f"  Total de features: {len(cols)}  (era 878 na v7 — redução de "
          f"{100*(878-len(cols))/878:.0f}%)")
    for grupo, n in info["n_por_grupo"].items():
        print(f"    {grupo:<28} {n:>4} features")
    print(f"\n  Treino: {info['treino']['n_originais']} imgs → "
          f"{info['treino']['n_total']} amostras (aug × 5)")
    print(f"  Teste:  {info['teste']['n_imagens']} imgs | "
          f"média {info['teste']['cnt_media']:.1f} laranjas/img")
    print(f"\n  Target de treino : contagem_log = log(1 + contagem)")
    print(f"  Avaliação MAPE   : np.expm1(pred) vs contagem")
    print(f"\n  ┌─ XGBoost ─── orandet_v8_train_raw.csv  / test_raw.csv")
    print(f"  └─ MLP + SVR ── orandet_v8_train_norm.csv / test_norm.csv")
    print(f"\n  Arquivos em: {OUTPUT_DIR}/")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()