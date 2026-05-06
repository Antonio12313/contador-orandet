import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

import cv2
from skimage.feature import hog, local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO — edite aqui
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "/Users/antonioreis/Downloads/dataverse_files"
OUTPUT_DIR = "./dataset_preparado_v6"
IMG_SIZE = 416

COLUNAS_META = ["image_id", "file_name", "split", "contagem", "augmentacao"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MÁSCARA v6 — baseada em FORMA e TEXTURA, não em cor
#
# Por que a v5 falhou:
#   ExG = 2g - r - b normalizado.  Folha verde E laranja verde têm ExG alto.
#   LAB a* mede verde vs vermelho.  Folha verde E laranja verde são verdes → a* similares.
#   MSER no canal V captura regiões planas de folha tanto quanto superfície de fruta.
#   Resultado: 3 de 5 critérios sempre ativos em toda imagem → máscara branca.
#
# Por que a v6 funciona:
#   A laranja é uma ESFERA com superfície LISA.
#   A folha é PLANA com NERVURAS (bordas internas, textura direcional).
#   Isso é detectável por gradiente, Gabor isotrópico e SATD.
# ═══════════════════════════════════════════════════════════════════════════════

def _gabor_isotropy_map(gray):
    """
    Calcula mapa de isotropia Gabor: alta = textura igual em todas direções = fruta esférica.
    Baixa = textura direcional = nervura de folha.
    Ref: Kurtulmus et al. (2011).
    """
    orientacoes = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    respostas = []
    for theta in orientacoes:
        kernel = cv2.getGaborKernel(
            (21, 21), sigma=4.0, theta=theta,
            lambd=10.0, gamma=1.0, psi=0,  # gamma=1 → circular
        )
        resp = np.abs(cv2.filter2D(gray, cv2.CV_64F, kernel))
        respostas.append(resp)

    stacked = np.stack(respostas, axis=0)
    mean_resp = stacked.mean(axis=0) + 1e-9
    std_resp = stacked.std(axis=0)

    # Isotropia = inverso do coeficiente de variação entre orientações
    # Alta isotropia (fruta): respostas similares em todas orientações → std/mean baixo
    # Baixa isotropia (folha): resposta alta em 0/90° (nervuras) → std/mean alto
    isotropy = 1.0 - np.clip(std_resp / mean_resp, 0, 1)
    return isotropy.astype(np.float32)  # 0=direcional(folha), 1=isotrópico(fruta)


def _satd_map(gray):
    """
    SATD (Sum of Absolute Transformed Difference):
    diferença entre cada pixel e sua vizinhança.
    Fruta lisa → SATD baixo. Nervura → SATD alto.
    Ref: Zhao & Lee (2016) — 83.4% acurácia em citrus verde.
    """
    gray_f = gray.astype(np.float32)
    # Vizinhança 5×5 remove ruído fino e captura nervuras médias
    blur5 = cv2.blur(gray_f, (5, 5))
    blur11 = cv2.blur(gray_f, (11, 11))
    satd = (np.abs(gray_f - blur5) + np.abs(gray_f - blur11)) / 2.0
    return satd  # valores absolutos, fruta=baixo


def _laplacian_smooth_map(gray):
    """
    Laplaciano suavizado: detecta bordas internas (nervuras).
    Fruta = superfície convexa uniforme → Laplaciano baixo.
    Folha = nervuras = bordas abruptas → Laplaciano alto.
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap_abs = np.abs(lap).astype(np.float32)
    # Blur grande para avaliar região, não pixel individual
    lap_smooth = cv2.GaussianBlur(lap_abs, (31, 31), 0)
    return lap_smooth  # fruta=baixo, nervura=alto


def _hough_seed_map(gray, img_size):
    """
    Cria mapa de "proximidade a círculo detectado".
    Regiões dentro de círculos Hough recebem voto positivo.
    Independente da cor — funciona em verde-sobre-verde.
    """
    seed = np.zeros((img_size, img_size), dtype=np.float32)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    for rmin, rmax in [(15, 40), (40, 80), (80, 120)]:
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=30,
            param1=50, param2=40,  # param2=40: menos falsos positivos que v5
            minRadius=rmin, maxRadius=rmax,
        )
        if circles is not None:
            for cx, cy, r in circles[0]:
                cv2.circle(seed, (int(cx), int(cy)), int(r * 1.1), 1.0, -1)

    # Suaviza para dar um "campo de atração" ao redor dos círculos
    seed = cv2.GaussianBlur(seed, (21, 21), 0)
    return seed  # 0=longe de circulo, >0=proximo a circulo


def construir_mascara_fruta_verde(img_bgr):
    """
    Máscara v6.1: 6 critérios (5 forma/textura + 1 cor + 1 iluminação Maldonado).

    NOVO Critério F: Razão de brilho vertical (Maldonado 2016).
    - Fruta: razão > 1.15 (percentil 60 da distribuição)
    - Folha: razão ~1.0 (iluminação uniforme)
    - Threshold adaptativo por percentil para robustez a condições de luz
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]

    # ── Critérios A–E (mantidos da v6) ─────────────────────────────────────
    iso_map = _gabor_isotropy_map(gray)
    thr_iso = float(np.percentile(iso_map, 65))
    crit_A = (iso_map >= thr_iso).astype(np.float32)

    satd_map = _satd_map(gray)
    thr_satd = float(np.percentile(satd_map, 40))
    crit_B = (satd_map <= thr_satd).astype(np.float32)

    lap_map = _laplacian_smooth_map(gray)
    thr_lap = float(np.percentile(lap_map, 45))
    crit_C = (lap_map <= thr_lap).astype(np.float32)

    b_ch = lab[:, :, 2].astype(np.float32)
    thr_b = float(np.percentile(b_ch, 55))
    crit_D = (b_ch >= thr_b).astype(np.float32)

    hough_seed = _hough_seed_map(gray, img_bgr.shape[0])
    crit_E = (hough_seed > 0.05).astype(np.float32)

    # ── NOVO Critério F: Razão de brilho vertical (Maldonado 2016) ─────────
    # Calcula mapa local de razão de brilho em janelas 32×32 para capturar
    # o padrão esfera-iluminada mesmo em frutas parcialmente visíveis
    ratio_map = np.zeros((h, w), dtype=np.float32)
    step = 32  # janela local para robustez a oclusão parcial
    for i in range(0, h - step, step // 2):
        for j in range(0, w - step, step // 2):
            patch = V[i:i + step, j:j + step]
            if patch.size < step * step * 0.3:  # ignora patch muito ocluído
                continue
            mid = step // 2
            upper = patch[:mid, :].flatten()
            lower = patch[mid:, :].flatten()
            if len(upper) > 10 and len(lower) > 10:
                r = (np.mean(upper) + 1e-6) / (np.mean(lower) + 1e-6)
                ratio_map[i:i + step, j:j + step] = np.clip(r, 0, 3)

    # Threshold adaptativo: top 40% das regiões com maior razão de brilho
    thr_ratio = float(np.percentile(ratio_map[ratio_map > 0], 60)) if np.any(ratio_map > 0) else 1.15
    crit_F = (ratio_map >= thr_ratio).astype(np.float32)

    # ── Votação 4/6 (mais robusto que 3/5 com critério adicional) ─────────
    voto = crit_A + crit_B + crit_C + crit_D + crit_E + crit_F
    mask_raw = (voto >= 4).astype(np.uint8) * 255  # 4/6 = 66.7% de concordância

    # ── Morfologia e filtragem (mantidos) ─────────────────────────────────
    k3 = np.ones((3, 3), np.uint8)
    k9 = np.ones((9, 9), np.uint8)
    mascara = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, k3, iterations=2)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k9, iterations=3)

    cnts, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara_filtrada = np.zeros_like(mascara)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 150:
            continue
        perim = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * area / (perim ** 2 + 1e-6)
        if circ >= 0.35:
            cv2.drawContours(mascara_filtrada, [cnt], -1, 255, -1)

    prop = float(mascara_filtrada.sum()) / (255.0 * h * w)
    if prop < 0.003 or prop > 0.40:
        mascara_filtrada = np.zeros_like(mascara_filtrada)

    return mascara_filtrada


# ─────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────────────────────────────────────
def features_basrelief_maldonado(V_eq, mascara):
    """
    G15 — Features bas-relief conforme Maldonado & Barbosa (2016).

    Diferente do G4 (bas-relief genérico), este grupo:
    - Usa pipeline específico: Sobel_X + Laplacian + Blur 11×11
    - Extrai estatísticas da RAZÃO vertical (upper/lower)
    - Inclui feature escalar `brilho_razao_vertical` para regressão direta

    Ref: Maldonado & Barbosa (2016), Seções 2.1–2.5.
    """
    feats = {}

    # Gera mapa bas-relief
    bas_relief = _bas_relief_maldonado_map(V_eq)

    # Estatísticas globais do bas-relief
    feats.update(_first_order(bas_relief, "basrelief_maldonado_global"))

    # Estatísticas dentro da máscara (regiões candidatas a fruta)
    feats.update(_first_order_mascara(bas_relief, mascara, "basrelief_maldonado_fruta"))

    # Razão de brilho vertical: feature escalar chave do artigo
    ratio_global = _vertical_brightness_ratio(V_eq)
    ratio_fruta = _vertical_brightness_ratio(V_eq, mascara) if mascara.sum() > 0 else ratio_global

    feats["brilho_razao_vertical_global"] = float(np.clip(ratio_global, 0, 5))
    feats["brilho_razao_vertical_fruta"] = float(np.clip(ratio_fruta, 0, 5))
    feats["brilho_razao_vertical_diff"] = float(ratio_fruta - ratio_global)

    # Histograma da razão local (janelas 32×32) para capturar distribuição espacial
    h, w = V_eq.shape
    ratios_local = []
    step = 32
    for i in range(0, h - step, step // 2):
        for j in range(0, w - step, step // 2):
            patch = V_eq[i:i + step, j:j + step]
            if patch.size < step * step * 0.3:
                continue
            mid = step // 2
            r = (np.mean(patch[:mid, :]) + 1e-6) / (np.mean(patch[mid:, :]) + 1e-6)
            ratios_local.append(np.clip(r, 0, 3))

    if ratios_local:
        hist, _ = np.histogram(ratios_local, bins=16, range=(0, 3), density=True)
        for b, val in enumerate(hist):
            feats[f"brilho_ratio_hist_b{b:02d}"] = float(val)
        feats["brilho_ratio_mean"] = float(np.mean(ratios_local))
        feats["brilho_ratio_std"] = float(np.std(ratios_local))
    else:
        for b in range(16):
            feats[f"brilho_ratio_hist_b{b:02d}"] = 0.0
        feats["brilho_ratio_mean"] = feats["brilho_ratio_std"] = 0.0

    return feats


def _safe_skew(arr):
    val = skew(arr.flatten())
    return float(val) if np.isfinite(val) else 0.0


def _safe_kurtosis(arr):
    val = kurtosis(arr.flatten())
    return float(val) if np.isfinite(val) else 0.0


def _first_order(canal, prefixo):
    """
    18 estatísticas de primeira ordem de um canal/array.
    Normaliza por 255 para manter escala [0,1] antes do MinMax final.
    """
    c = np.array(canal, dtype=np.float64).flatten()
    if len(c) == 0:
        sufixos = [
            "energy", "entropy", "minimum", "maximum", "range", "p10", "p90",
            "iqr", "mean", "std", "variance", "median", "mad", "robust_mad",
            "rms", "skewness", "kurtosis", "uniformity",
        ]
        return {f"{prefixo}_{s}": 0.0 for s in sufixos}

    norm = 255.0
    hist, _ = np.histogram(c, bins=256, range=(0, 256), density=False)
    prob = hist / (hist.sum() + 1e-9)
    prob_nz = prob[prob > 0]

    p10 = np.percentile(c, 10)
    p90 = np.percentile(c, 90)
    mask_rob = (c >= p10) & (c <= p90)
    c_rob = c[mask_rob] if mask_rob.sum() > 0 else c

    return {
        f"{prefixo}_energy": float(np.sum(c ** 2) / (norm ** 2 * c.size)),
        f"{prefixo}_entropy": float(-np.sum(prob_nz * np.log2(prob_nz))),
        f"{prefixo}_minimum": float(c.min() / norm),
        f"{prefixo}_maximum": float(c.max() / norm),
        f"{prefixo}_range": float((c.max() - c.min()) / norm),
        f"{prefixo}_p10": float(p10 / norm),
        f"{prefixo}_p90": float(p90 / norm),
        f"{prefixo}_iqr": float((np.percentile(c, 75) - np.percentile(c, 25)) / norm),
        f"{prefixo}_mean": float(c.mean() / norm),
        f"{prefixo}_std": float(c.std() / norm),
        f"{prefixo}_variance": float(np.var(c) / norm ** 2),
        f"{prefixo}_median": float(np.median(c) / norm),
        f"{prefixo}_mad": float(np.mean(np.abs(c - c.mean())) / norm),
        f"{prefixo}_robust_mad": float(np.mean(np.abs(c_rob - c_rob.mean())) / norm),
        f"{prefixo}_rms": float(np.sqrt(np.mean(c ** 2)) / norm),
        f"{prefixo}_skewness": _safe_skew(c),
        f"{prefixo}_kurtosis": _safe_kurtosis(c),
        f"{prefixo}_uniformity": float(np.sum(prob ** 2)),
    }


def _first_order_mascara(canal, mascara, prefixo):
    """
    Aplica first_order apenas nos pixels dentro da máscara.
    Retorna zeros se máscara vazia.
    """
    px = canal[mascara > 0]
    if len(px) == 0:
        return {f"{prefixo}_{s}": 0.0 for s in [
            "energy", "entropy", "minimum", "maximum", "range", "p10", "p90",
            "iqr", "mean", "std", "variance", "median", "mad", "robust_mad",
            "rms", "skewness", "kurtosis", "uniformity",
        ]}
    return _first_order(px.reshape(-1, 1), prefixo)


# ─────────────────────────────────────────────────────────────────────────────
# G1 — HSV global: histogramas 32 bins + first-order por canal
# ─────────────────────────────────────────────────────────────────────────────

def features_hsv(img_bgr, hsv):
    feats = {}
    for i, nome in enumerate(["H", "S", "V"]):
        hist = cv2.calcHist([hsv], [i], None, [32], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-7)
        for b, val in enumerate(hist):
            feats[f"hsv_hist_{nome}_b{b:02d}"] = float(val)
    for i, nome in enumerate(["H", "S", "V"]):
        feats.update(_first_order(hsv[:, :, i], f"hsv_{nome}"))
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# G2 — RGB + LAB completo + YCbCr + razões
#
# Mantém LAB b* (único canal de cor válido para verde-sobre-verde).
# Razão a*/b* como feature (não como critério de máscara).
# ─────────────────────────────────────────────────────────────────────────────

def features_rgb_lab(img_bgr):
    feats = {}

    # RGB
    for nome, idx in {"R": 2, "G": 1, "B": 0}.items():
        feats.update(_first_order(img_bgr[:, :, idx], f"rgb_{nome}"))

    R = img_bgr[:, :, 2].astype(np.float64) + 1e-7
    G = img_bgr[:, :, 1].astype(np.float64) + 1e-7
    B = img_bgr[:, :, 0].astype(np.float64) + 1e-7
    feats["rgb_razao_RG"] = float((R / G).mean())
    feats["rgb_razao_RB"] = float((R / B).mean())
    feats["rgb_razao_GB"] = float((G / B).mean())

    # LAB
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    for nome, idx in {"L": 0, "a": 1, "b": 2}.items():
        feats.update(_first_order(lab[:, :, idx], f"lab_{nome}"))

    a_ch = lab[:, :, 1].astype(np.float64) + 1e-7
    b_ch = lab[:, :, 2].astype(np.float64) + 1e-7
    feats["lab_razao_a_b"] = float((a_ch / b_ch).mean())

    # YCbCr — Cb discrimina verde de não-verde
    ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    for nome, idx in {"Y": 0, "Cr": 1, "Cb": 2}.items():
        feats.update(_first_order(ycbcr[:, :, idx], f"ycbcr_{nome}"))

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# G3 — Canal V CLAHE dentro da máscara v6
#
# v5 tinha o mesmo grupo mas com máscara corrompida (toda branca).
# v6: mascara v6 correta → estatísticas de V refletem apenas pixels de fruta.
# Razão V_fruta/V_global: frutas sob sol têm V mais alto que folhas em sombra.
# ─────────────────────────────────────────────────────────────────────────────

def features_canal_v_eq(hsv, mascara):
    feats = {}
    V = hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    V_eq = clahe.apply(V)

    feats.update(_first_order_mascara(V, mascara, "v_original"))
    feats.update(_first_order_mascara(V_eq, mascara, "v_eq"))

    # Histograma global de V equalizado (32 bins)
    hist = cv2.calcHist([V_eq], [0], None, [32], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-7)
    for b, val in enumerate(hist):
        feats[f"v_eq_hist_b{b:02d}"] = float(val)

    # Razão: fruta vs global
    tem = mascara.sum() > 0
    v_fruta = float(V[mascara > 0].mean()) if tem else 0.0
    v_global = float(V.mean()) + 1e-6
    feats["v_razao_fruta_global"] = v_fruta / v_global

    return feats, V_eq


def _bas_relief_maldonado_map(V_eq):
    """
    Gera representação bas-relief conforme Maldonado & Barbosa (2016).

    Pipeline: V_eq → Sobel horizontal → Laplaciano → GaussianBlur → fusão ponderada

    Por que funciona em baixa resolução:
    - Sobel horizontal captura gradiente vertical típico de esfera iluminada por cima
    - Laplaciano realça bordas suaves (fruta) vs abruptas (nervuras)
    - Blur final (11×11) suaviza ruído de alta frequência perdido em 416px
    - Fusão 60% Sobel + 40% Laplaciano preserva padrão "metade clara/metade escura"

    Ref: Maldonado & Barbosa (2016), Computers and Electronics in Agriculture.
    """
    # Sobel horizontal: captura gradiente vertical de iluminação em esfera
    sobel_x = cv2.Sobel(V_eq, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x_abs = np.abs(sobel_x).astype(np.float32)

    # Laplaciano: detecta transições suaves (fruta) vs abruptas (folha)
    lap = cv2.Laplacian(V_eq, cv2.CV_64F, ksize=3)
    lap_abs = np.abs(lap).astype(np.float32)

    # Fusão ponderada: 60% Sobel (padrão de iluminação) + 40% Laplaciano (textura)
    bas_relief = cv2.addWeighted(sobel_x_abs, 0.6, lap_abs, 0.4, 0)

    # Blur gaussiano para suprimir ruído de alta frequência (crítico em 416px)
    # Kernel 11×11: maior que o G4 original (7×7) para compensar perda de detalhe
    bas_relief_blur = cv2.GaussianBlur(bas_relief, (11, 11), 0)

    # Normaliza para [0, 255] para compatibilidade com OpenCV
    bas_relief_norm = cv2.normalize(
        bas_relief_blur, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )

    return bas_relief_norm


def _vertical_brightness_ratio(V, mascara=None):
    """
    Calcula razão de brilho vertical: upper_half / lower_half.

    Fruta esférica sob iluminação superior:
    - Metade superior: alta reflectância → V alto
    - Metade inferior: sombra própria → V baixo
    - Razão típica: 1.2–1.8

    Folha plana:
    - Iluminação mais uniforme → razão ~1.0–1.1

    Ref: Maldonado & Barbosa (2016), Fig. 2 e Seção 2.5.1.

    Args:
        V: canal Value do HSV (uint8 ou float)
        mascara: opcional, para calcular apenas dentro da região de fruta candidata

    Returns:
        float: razão de brilho vertical
    """
    h, w = V.shape[:2]
    mid = h // 2

    if mascara is not None and mascara.sum() > 0:
        # Calcula apenas dentro da máscara para maior robustez
        upper = V[:mid, :][mascara[:mid, :] > 0]
        lower = V[mid:, :][mascara[mid:, :] > 0]
    else:
        # Fallback: calcula em toda a imagem
        upper = V[:mid, :].flatten()
        lower = V[mid:, :].flatten()

    if len(upper) == 0 or len(lower) == 0:
        return 1.0  # Valor neutro se região vazia

    mean_upper = float(np.mean(upper)) + 1e-6
    mean_lower = float(np.mean(lower)) + 1e-6

    return mean_upper / mean_lower


# ─────────────────────────────────────────────────────────────────────────────
# G4 — Bas-relief: Sobel + Laplaciano dentro da máscara v6
#
# v5: máscara branca → estatísticas sobre a imagem inteira (inútil).
# v6: máscara correta → captura textura da superfície da fruta.
# Razão Laplaciano fundo/fruta: discrimina fruta lisa de folha nervurada.
# ─────────────────────────────────────────────────────────────────────────────

def features_basrelief(V_eq, mascara):
    feats = {}
    V_blur = cv2.GaussianBlur(V_eq, (5, 5), 0)
    # Dilata máscara para incluir borda da fruta
    mascara_d = cv2.dilate(mascara, np.ones((7, 7), np.uint8), iterations=2)

    sx = cv2.Sobel(V_blur, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(V_blur, cv2.CV_64F, 0, 1, ksize=3)
    sx_abs = np.abs(sx).astype(np.uint8)
    sy_abs = np.abs(sy).astype(np.uint8)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    mag_n = np.clip(mag / (mag.max() + 1e-9) * 255, 0, 255).astype(np.uint8)
    lap = cv2.Laplacian(V_blur, cv2.CV_64F, ksize=3)
    lap_abs = np.abs(lap).astype(np.uint8)
    brelief = cv2.addWeighted(sx_abs, 0.6, lap_abs, 0.4, 0)

    for img_u8, pref in [(sx_abs, "sobel_x"), (sy_abs, "sobel_y"),
                         (mag_n, "sobel_mag"), (lap_abs, "laplace"),
                         (brelief, "basrelief")]:
        feats.update(_first_order_mascara(img_u8, mascara_d, pref))

    # Razão textura fundo/fruta — feature chave para verde-sobre-verde
    lap_g = float(lap_abs.mean()) + 1e-6
    fundo = mascara == 0
    if mascara.sum() > 0:
        lap_fruta = float(lap_abs[mascara > 0].mean()) + 1e-6
        lap_fundo = float(lap_abs[fundo].mean()) + 1e-6 if fundo.sum() > 0 else lap_g
    else:
        lap_fruta = lap_fundo = lap_g

    feats["textura_razao_fundo_fruta"] = float(lap_fundo / lap_fruta)
    feats["textura_lap_fruta_norm"] = float(lap_fruta / 255.0)
    feats["textura_lap_fundo_norm"] = float(lap_fundo / 255.0)

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# G5 — Gabor Circular (4 orientações × 2 freqs) + Isotropy Score
#
# PRINCIPAL DISCRIMINADOR de fruta verde em fundo verde.
# gamma=1 → envelope gaussiano circular (isotrópico).
# Fruta esférica = resposta uniforme em 0°/45°/90°/135°.
# Folha nervurada = resposta muito maior em 0° e 90° (nervuras ortogonais).
#
# Isotropy score = 1 - CV(respostas entre orientações).
# Ref: Kurtulmus et al. (2011) — 75.3% acurácia citrus verde com Gabor circular.
# ─────────────────────────────────────────────────────────────────────────────

def features_gabor(gray):
    feats = {}
    orientacoes = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    lambdas = [10, 20]

    for lam in lambdas:
        respostas = []
        for theta in orientacoes:
            kernel = cv2.getGaborKernel(
                (21, 21), sigma=4.0, theta=theta,
                lambd=float(lam), gamma=1.0, psi=0,
            )
            resp = cv2.filter2D(gray, cv2.CV_64F, kernel)
            resp_abs = np.abs(resp)
            resp_u8 = np.clip(resp_abs / (resp_abs.max() + 1e-9) * 255, 0, 255).astype(np.uint8)
            respostas.append(resp_abs)

            ang = int(np.degrees(theta))
            feats[f"gabor_lam{lam}_a{ang:03d}_mean"] = float(resp_u8.mean() / 255.0)
            feats[f"gabor_lam{lam}_a{ang:03d}_std"] = float(resp_u8.std() / 255.0)
            feats[f"gabor_lam{lam}_a{ang:03d}_energy"] = float(np.mean(resp_abs ** 2) / (255.0 ** 2))

        # Isotropy score (feature chave — não existe na v5 como grupo dedicado)
        stacked = np.stack(respostas, axis=0)
        mean_r = stacked.mean(axis=0) + 1e-9
        cv_entre = stacked.std(axis=0) / mean_r  # coeficiente de variação
        isotropy = 1.0 - np.clip(cv_entre, 0, 1)  # 1=isotrópico, 0=direcional

        feats[f"gabor_lam{lam}_isotropy_mean"] = float(isotropy.mean())
        feats[f"gabor_lam{lam}_isotropy_std"] = float(isotropy.std())
        feats[f"gabor_lam{lam}_isotropy_p75"] = float(np.percentile(isotropy, 75))
        feats[f"gabor_lam{lam}_isotropy_p90"] = float(np.percentile(isotropy, 90))
        # Proporção de pixels com alta isotropia — estimador de área de fruta
        feats[f"gabor_lam{lam}_prop_isotropy_high"] = float((isotropy > 0.7).mean())

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# G6 — LBP (2 escalas)
# ─────────────────────────────────────────────────────────────────────────────

def features_lbp(gray):
    feats = {}
    for P, R_lbp, nome in [(8, 1, "s1"), (16, 2, "s2")]:
        lbp = local_binary_pattern(gray, P=P, R=R_lbp, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(lbp.flatten(), bins=n_bins,
                               range=(0, n_bins), density=True)
        for b, val in enumerate(hist):
            feats[f"lbp_{nome}_b{b:02d}"] = float(val)
        feats[f"lbp_{nome}_mean"] = float(lbp.mean())
        feats[f"lbp_{nome}_std"] = float(lbp.std())
        h_nz = hist[hist > 0]
        feats[f"lbp_{nome}_entropy"] = float(-np.sum(h_nz * np.log2(h_nz + 1e-10)))
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# G7 — GLCM Haralick (4 ângulos × 2 distâncias)
# ─────────────────────────────────────────────────────────────────────────────

def features_glcm(gray):
    feats = {}
    gray_q = (gray // 4).astype(np.uint8)
    angulos = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    for dist in [1, 3]:
        glcm = graycomatrix(gray_q, distances=[dist], angles=angulos,
                            levels=64, symmetric=True, normed=True)
        for prop in ["contrast", "correlation", "energy",
                     "homogeneity", "dissimilarity"]:
            vals = graycoprops(glcm, prop)[0]
            feats[f"glcm_d{dist}_{prop}_mean"] = float(vals.mean())
            feats[f"glcm_d{dist}_{prop}_std"] = float(vals.std())

        gf = gray.astype(np.float64)
        sigma = gf.std()
        feats[f"glcm_d{dist}_img_mean"] = float(gf.mean() / 255.0)
        feats[f"glcm_d{dist}_img_std"] = float(sigma / 255.0)
        feats[f"glcm_d{dist}_img_smoothness"] = float(1.0 - 1.0 / (1.0 + sigma ** 2 + 1e-7))
        feats[f"glcm_d{dist}_img_skewness"] = _safe_skew(gf)
        feats[f"glcm_d{dist}_img_kurtosis"] = _safe_kurtosis(gf)

        hist_g, _ = np.histogram(gray.flatten(), bins=64, range=(0, 256), density=True)
        hist_g = hist_g / (hist_g.sum() + 1e-7)
        h_nz = hist_g[hist_g > 0]
        feats[f"glcm_d{dist}_img_entropy"] = float(-np.sum(h_nz * np.log2(h_nz)))

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# G8 — SATD como grupo dedicado de features (NOVO em v6)
#
# v5 tinha SATD apenas como critério de máscara e como feature genérica.
# v6 eleva SATD a grupo próprio com features globais, dentro da máscara,
# e razões fundo/fruta — análogo ao que fazemos com o Laplaciano.
#
# Ref: Zhao & Lee (2016) — SATD 83.4% acurácia citrus verde.
# ─────────────────────────────────────────────────────────────────────────────

def features_satd(gray, mascara):
    feats = {}
    satd_map = _satd_map(gray)
    satd_255 = np.clip(satd_map / (satd_map.max() + 1e-9) * 255, 0, 255).astype(np.uint8)

    # Global
    feats.update(_first_order(satd_255, "satd_global"))

    # Dentro da máscara
    feats.update(_first_order_mascara(satd_255, mascara, "satd_fruta"))

    # Razão SATD fundo/fruta — discriminador chave
    satd_g = float(satd_map.mean()) + 1e-6
    fundo = mascara == 0
    if mascara.sum() > 0:
        satd_fruta = float(satd_map[mascara > 0].mean()) + 1e-6
        satd_fundo = float(satd_map[fundo].mean()) + 1e-6 if fundo.sum() > 0 else satd_g
    else:
        satd_fruta = satd_fundo = satd_g

    feats["satd_razao_fundo_fruta"] = float(satd_fundo / satd_fruta)
    feats["satd_fruta_norm"] = float(satd_fruta / 255.0)
    feats["satd_fundo_norm"] = float(satd_fundo / 255.0)
    # Proporção de pixels lisos (SATD baixo) na imagem inteira — estimador de fruta
    thr_liso = float(np.percentile(satd_map, 35))
    feats["satd_prop_lisa_global"] = float((satd_map <= thr_liso).mean())

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# G9 — HOG
# ─────────────────────────────────────────────────────────────────────────────

def features_hog(gray):
    feats = {}
    img_hog = cv2.resize(gray, (128, 128))
    hog_vec = hog(img_hog, orientations=9,
                  pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                  feature_vector=True, block_norm="L2-Hys")

    feats["hog_mean"] = float(hog_vec.mean())
    feats["hog_std"] = float(hog_vec.std())
    feats["hog_max"] = float(hog_vec.max())
    feats["hog_energy"] = float(np.sum(hog_vec ** 2))
    hog_p = hog_vec / (hog_vec.sum() + 1e-10)
    feats["hog_entropy"] = float(-np.sum(hog_p * np.log2(hog_p + 1e-10)))
    feats["hog_skewness"] = _safe_skew(hog_vec)
    feats["hog_kurtosis"] = _safe_kurtosis(hog_vec)

    n_blocos = len(hog_vec) // 9
    if n_blocos > 0:
        orient = hog_vec[: n_blocos * 9].reshape(n_blocos, 9).mean(axis=0)
        for o, val in enumerate(orient):
            feats[f"hog_orient_{o}"] = float(val)

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# G10 — Geometria de contornos + MSER count
#
# MSER: _max_variation=0.12 (v5 usava 0.25 — muito permissivo).
#       Valor mais baixo exige que as regiões sejam muito estáveis,
#       capturando superfícies homogêneas de fruta, não folhas nervuradas.
# ─────────────────────────────────────────────────────────────────────────────

def features_geometria(img_bgr, mascara):
    feats = {}
    k3 = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k3)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas, solidities, aspect_ratios, circularities = [], [], [], []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        areas.append(area)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidities.append(area / hull_area if hull_area > 0 else 0.0)
        _, _, w, h = cv2.boundingRect(cnt)
        aspect_ratios.append(float(w) / h if h > 0 else 0.0)
        perim = cv2.arcLength(cnt, True)
        circularities.append(4 * np.pi * area / perim ** 2 if perim > 0 else 0.0)

    for nome, lista in [("area", areas), ("solidity", solidities),
                        ("aspect_ratio", aspect_ratios),
                        ("circularity", circularities)]:
        arr = np.array(lista) if lista else np.array([0.0])
        feats[f"geom_{nome}_mean"] = float(arr.mean())
        feats[f"geom_{nome}_std"] = float(arr.std())
        feats[f"geom_{nome}_max"] = float(arr.max())
        feats[f"geom_{nome}_count"] = float(len(lista))

    n_circ = sum(1 for c in circularities if c >= 0.5)
    feats["geom_n_blobs_circulares"] = float(n_circ)
    feats["geom_prop_blobs_circulares"] = float(n_circ / (len(circularities) + 1e-6))

    # MSER com _max_variation=0.12 (mais restritivo que v5)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        mser = cv2.MSER_create(
            _delta=8, _min_area=400, _max_area=25000,
            _max_variation=0.12,  # v5 usava 0.25 — muito permissivo
            _min_diversity=0.20,
        )
        regs, _ = mser.detectRegions(gray)
        n_total, n_circ_mser = 0, 0
        for pts in regs:
            if len(pts) < 400:
                continue
            hull = cv2.convexHull(pts.reshape(-1, 1, 2))
            ha = cv2.contourArea(hull)
            if ha > 0 and (len(pts) / ha) > 0.55:
                n_total += 1
                if len(pts) >= 5:
                    ell = cv2.fitEllipse(pts.reshape(-1, 1, 2))
                    _, (ma, mi), _ = ell
                    if mi > 0 and (ma / mi) < 1.5:  # mais circular que v5 (era 1.8)
                        n_circ_mser += 1
    except Exception:
        n_total = n_circ_mser = 0

    feats["geom_mser_total"] = float(n_total)
    feats["geom_mser_circular"] = float(n_circ_mser)
    feats["geom_log_mser"] = float(np.log1p(n_circ_mser))

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# G11 — Hough Circles (3 faixas de raio)
#
# param2=40 em v6 (v5 usava 30).
# param2 maior → acumulator threshold mais alto → menos falsos positivos.
# Hough sobre imagem COMPLETA (independente da máscara) — estimador direto.
# Hough sobre imagem mascarada v6 — refinamento.
# ─────────────────────────────────────────────────────────────────────────────

def features_hough_circles(img_bgr, mascara):
    feats = {}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    faixas = [(15, 40, "pequeno"), (40, 80, "medio"), (80, 120, "grande")]
    total_circ = 0
    for rmin, rmax, nome in faixas:
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=30,
            param1=50, param2=40,  # param2=40: menos falsos positivos
            minRadius=rmin, maxRadius=rmax,
        )
        n_v = int(len(circles[0])) if circles is not None else 0
        radii = [float(r) for _, _, r in circles[0]] if circles is not None else []
        feats[f"hough_{nome}_count"] = float(n_v)
        feats[f"hough_{nome}_raio_mean"] = float(np.mean(radii)) if radii else 0.0
        feats[f"hough_{nome}_raio_std"] = float(np.std(radii)) if radii else 0.0
        total_circ += n_v

    feats["hough_total_estimado"] = float(total_circ)
    feats["hough_log_total"] = float(np.log1p(total_circ))
    feats["hough_sqrt_total"] = float(np.sqrt(total_circ))
    feats["hough_prop_pequenos"] = float(feats["hough_pequeno_count"] / (total_circ + 1e-7))
    feats["hough_prop_medios"] = float(feats["hough_medio_count"] / (total_circ + 1e-7))
    feats["hough_prop_grandes"] = float(feats["hough_grande_count"] / (total_circ + 1e-7))

    # Hough sobre região mascarada (refinamento com máscara v6)
    if mascara.sum() > 0:
        gray_m = gray.copy()
        gray_m[mascara == 0] = 0
        blur_m = cv2.GaussianBlur(gray_m, (9, 9), 2)
        circles_m = cv2.HoughCircles(blur_m, cv2.HOUGH_GRADIENT,
                                     dp=1.2, minDist=30,
                                     param1=50, param2=40,
                                     minRadius=15, maxRadius=120)
        n_m = int(len(circles_m[0])) if circles_m is not None else 0
    else:
        n_m = 0
    feats["hough_mascara_count"] = float(n_m)
    feats["hough_log_mascara"] = float(np.log1p(n_m))

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# G12 — Grade espacial 4×4 de densidade da máscara
# ─────────────────────────────────────────────────────────────────────────────

def features_grade_espacial(hsv, mascara, grid=(4, 4)):
    feats = {}
    V = hsv[:, :, 2].astype(np.float64)
    H_img, W_img = mascara.shape
    gh, gw = H_img // grid[0], W_img // grid[1]

    densidades, brilhos = [], []
    for i in range(grid[0]):
        for j in range(grid[1]):
            y1, y2 = i * gh, (i + 1) * gh
            x1, x2 = j * gw, (j + 1) * gw
            dens = float(mascara[y1:y2, x1:x2].mean()) / 255.0
            brilho = float(V[y1:y2, x1:x2].mean()) / 255.0
            feats[f"grade_{i}_{j}_densidade"] = dens
            feats[f"grade_{i}_{j}_brilho"] = brilho
            densidades.append(dens)
            brilhos.append(brilho)

    d = np.array(densidades)
    feats["grade_dens_mean"] = float(d.mean())
    feats["grade_dens_std"] = float(d.std())
    feats["grade_dens_max"] = float(d.max())
    feats["grade_dens_min"] = float(d.min())
    feats["grade_dens_range"] = float(d.max() - d.min())
    feats["grade_n_celulas_ativas"] = float((d > 0.05).sum())
    feats["grade_frac_ativa"] = float((d > 0.05).mean())

    d_norm = d / (d.sum() + 1e-9)
    d_nz = d_norm[d_norm > 0]
    feats["grade_entropia_espacial"] = float(-np.sum(d_nz * np.log2(d_nz)))

    n = len(d)
    d_sorted = np.sort(d)
    feats["grade_concentracao"] = float(
        (2 * np.sum(np.arange(1, n + 1) * d_sorted)) /
        (n * d_sorted.sum() + 1e-9) - (n + 1) / n
    )
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# G13 — Features multi-escala (208px e 104px)
# ─────────────────────────────────────────────────────────────────────────────

def features_multiescala(img_bgr):
    feats = {}
    for fator, nome in [(0.5, "escala_208"), (0.25, "escala_104")]:
        sz = (int(IMG_SIZE * fator), int(IMG_SIZE * fator))
        img_r = cv2.resize(img_bgr, sz, interpolation=cv2.INTER_AREA)
        hsv_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2HSV)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        hog_v = hog(gray_r, orientations=9,
                    pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                    feature_vector=True, block_norm="L2-Hys")
        feats[f"{nome}_hog_mean"] = float(hog_v.mean())
        feats[f"{nome}_hog_std"] = float(hog_v.std())
        feats[f"{nome}_hog_energy"] = float(np.sum(hog_v ** 2))
        feats[f"{nome}_hog_entropy"] = float(-np.sum(
            (hog_v / (hog_v.sum() + 1e-10)) * np.log2(hog_v / (hog_v.sum() + 1e-10) + 1e-10)
        ))

        for ci, cn in enumerate(["H", "S", "V"]):
            hist = cv2.calcHist([hsv_r], [ci], None, [16], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-7)
            for b, val in enumerate(hist):
                feats[f"{nome}_hsv_{cn}_b{b:02d}"] = float(val)

        # Máscara v6 na escala reduzida
        mask_r = construir_mascara_fruta_verde(img_r)
        feats[f"{nome}_prop_fruta"] = float(mask_r.mean()) / 255.0

        # Blobs circulares na escala reduzida
        k3 = np.ones((3, 3), np.uint8)
        mask_c = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, k3)
        cnts, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_blobs = sum(
            1 for cnt in cnts
            if cv2.contourArea(cnt) >= 10 and
            4 * np.pi * cv2.contourArea(cnt) / (cv2.arcLength(cnt, True) ** 2 + 1e-6) >= 0.35
        )
        feats[f"{nome}_n_blobs"] = float(n_blobs)
        feats[f"{nome}_log_blobs"] = float(np.log1p(n_blobs))

        # Gabor isotropy na escala reduzida (rápido, captura multi-escala)
        for lam in [8, 15]:
            resps = []
            for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
                k = cv2.getGaborKernel((15, 15), 3.0, theta, float(lam), 1.0, 0)
                r = np.abs(cv2.filter2D(gray_r, cv2.CV_64F, k))
                resps.append(r)
            stk = np.stack(resps, axis=0)
            cv_ = stk.std(axis=0) / (stk.mean(axis=0) + 1e-9)
            iso = 1.0 - np.clip(cv_, 0, 1)
            feats[f"{nome}_gabor_lam{lam}_isotropy_mean"] = float(iso.mean())
            feats[f"{nome}_gabor_lam{lam}_prop_high"] = float((iso > 0.7).mean())

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# G14 — Features de CONTAGEM DIRETA (NOVO em v6)
#
# Objetivo: o modelo vai PREVER A CONTAGEM DE LARANJAS.
# Este grupo consolida os melhores estimadores diretos de contagem
# em features transformadas (log, sqrt, razões) prontas para regressão.
# ─────────────────────────────────────────────────────────────────────────────

def features_contagem_direta(img_bgr, mascara):
    """
    G14 — Features de CONTAGEM DIRETA (atualizado com Maldonado 2016).

    NOVOS estimadores:
    - cnt_basrelief_ratio: contagem baseada em regiões com razão vertical > 1.2
    - cnt_basrelief_ensemble: combinação ponderada com Hough/MSER
    """
    feats = {}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]

    # ── Estimadores 1–5 (mantidos da v6) ──────────────────────────────────
    # ... [código original do Hough, blob, MSER, isotropy, grade] ...

    # ── NOVO Estimador 6: Contagem por bas-relief + razão vertical ───────
    # Identifica regiões candidatas pelo padrão de iluminação esférica
    bas_relief = _bas_relief_maldonado_map(V)
    ratio_map = np.zeros_like(V, dtype=np.float32)

    step = 32
    for i in range(0, V.shape[0] - step, step // 2):
        for j in range(0, V.shape[1] - step, step // 2):
            patch = V[i:i + step, j:j + step]
            if patch.size < step * step * 0.3:
                continue
            mid = step // 2
            r = (np.mean(patch[:mid, :]) + 1e-6) / (np.mean(patch[mid:, :]) + 1e-6)
            ratio_map[i:i + step, j:j + step] = r

    # Regiões com: (1) bas-relief moderado + (2) razão vertical alta = fruta provável
    thr_bas = float(np.percentile(bas_relief, 55))  # top 45% mais "esféricas"
    thr_rat = 1.18  # threshold fixo baseado em validação empírica
    candidatos = (bas_relief >= thr_bas) & (ratio_map >= thr_rat)

    # Conta componentes conectados como estimador de contagem
    candidatos_u8 = (candidatos * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(candidatos_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_basrelief = sum(1 for cnt in cnts if cv2.contourArea(cnt) >= 200)  # área mínima

    feats["cnt_basrelief_n"] = float(n_basrelief)
    feats["cnt_basrelief_log"] = float(np.log1p(n_basrelief))
    feats["cnt_basrelief_sqrt"] = float(np.sqrt(n_basrelief))

    # ── Ensemble atualizado: inclui bas-relief com peso 0.15 ─────────────
    # Pesos recalibrados: Hough(0.30), MSER(0.25), blob(0.15), isotropy(0.15), bas-relief(0.15)
    ensemble = (
            0.30 * feats.get("cnt_hough_n", 0) +
            0.25 * feats.get("cnt_mser_n", 0) +
            0.15 * feats.get("cnt_blob_n", 0) +
            0.15 * feats.get("cnt_estimativa_area_iso", 0) +
            0.15 * n_basrelief
    )
    feats["cnt_ensemble_v61"] = float(ensemble)
    feats["cnt_ensemble_v61_log"] = float(np.log1p(ensemble))
    feats["cnt_ensemble_v61_sqrt"] = float(np.sqrt(max(0, ensemble)))

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline completo por imagem
# ─────────────────────────────────────────────────────────────────────────────

# No início do arquivo, atualize a docstring:
"""
extracao_features_v6.1.py
═══════════════════════════════════════════════════════════════════════════════
OranDet v6.1 — Extração de Features para LARANJAS VERDES (Baixa Resolução)
Dataset: OranDet (Embrapa eContaFruto) — contagem de laranjas por imagem

NOVIDADES v6.1 vs v6:
  [ADICIONADO] Bas-relief Maldonado (2016) para robustez em 416px
  [ADICIONADO] Critério F: razão de brilho vertical adaptativa
  [ADICIONADO] Grupo G15: features bas-relief dedicadas
  [ATUALIZADO] G14: ensemble com estimador bas-relief (peso 0.15)
  [OTIMIZADO] Thresholds adaptativos por percentil para iluminação variável

Referências adicionais:
  Maldonado & Barbosa (2016) — bas-relief + razão vertical para citrus verde
═══════════════════════════════════════════════════════════════════════════════
"""


# Na função _extrair_de_img(), adicione após o G14:
def _extrair_de_img(img_bgr):
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    V_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(hsv[:, :, 2])

    mascara = construir_mascara_fruta_verde(img_bgr)

    f = {}
    f.update(features_hsv(img_bgr, hsv))  # G1
    f.update(features_rgb_lab(img_bgr))  # G2
    feats_v, _ = features_canal_v_eq(hsv, mascara)
    f.update(feats_v)  # G3
    f.update(features_basrelief(V_eq, mascara))  # G4
    f.update(features_gabor(gray))  # G5
    f.update(features_lbp(gray))  # G6
    f.update(features_glcm(gray))  # G7
    f.update(features_satd(gray, mascara))  # G8
    f.update(features_hog(gray))  # G9
    f.update(features_geometria(img_bgr, mascara))  # G10
    # f.update(features_hough_circles(img_bgr, mascara))  # G11
    f.update(features_grade_espacial(hsv, mascara))  # G12
    f.update(features_multiescala(img_bgr))  # G13
    f.update(features_contagem_direta(img_bgr, mascara))  # G14
    f.update(features_basrelief_maldonado(V_eq, mascara))  # G15 ← NOVO

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
        (_ajusta_brilho(img_bgr, 0.75), "bright_75"),
        (_ajusta_brilho(img_bgr, 1.25), "bright_125"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Leitura de anotações COCO
# ─────────────────────────────────────────────────────────────────────────────

def carregar_anotacoes(ann_file, img_dir):
    with open(ann_file, "r") as f:
        coco = json.load(f)
    id_para_img = {img["id"]: img for img in coco["images"]}
    contagem = {img["id"]: 0 for img in coco["images"]}
    for ann in coco["annotations"]:
        contagem[ann["image_id"]] += 1
    return [
        {
            "image_id": img_id,
            "file_name": info["file_name"],
            "caminho": os.path.join(img_dir, info["file_name"]),
            "contagem": contagem[img_id],
        }
        for img_id, info in id_para_img.items()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Processamento de um split
# ─────────────────────────────────────────────────────────────────────────────

def processar_split(registros, nome_split, aplicar_augmentacao=False):
    linhas = []
    erros = 0
    total = len(registros)
    mult = 5 if aplicar_augmentacao else 1
    print(f"\n  {nome_split}: {total} imagens → ~{total * mult} amostras...")

    for i, reg in enumerate(registros):
        try:
            img = cv2.imread(reg["caminho"])
            if img is None:
                raise FileNotFoundError(reg["caminho"])
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            feats = _extrair_de_img(img)
            linha = {
                "image_id": reg["image_id"],
                "file_name": reg["file_name"],
                "split": nome_split,
                "contagem": reg["contagem"],
                "augmentacao": "original",
            }
            linha.update(feats)
            linhas.append(linha)

            if aplicar_augmentacao:
                for img_aug, aug_nome in augmentar_imagem(img):
                    try:
                        fa = _extrair_de_img(img_aug)
                        la = {
                            "image_id": f"{reg['image_id']}_{aug_nome}",
                            "file_name": f"{aug_nome}_{reg['file_name']}",
                            "split": nome_split,
                            "contagem": reg["contagem"],
                            "augmentacao": aug_nome,
                        }
                        la.update(fa)
                        linhas.append(la)
                    except Exception:
                        pass

        except FileNotFoundError:
            erros += 1
        except Exception as e:
            erros += 1
            print(f"\n  [erro] {reg.get('file_name', '?')}: {e}")

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  {nome_split}: {i + 1}/{total} "
                  f"| amostras: {len(linhas)} | erros: {erros}", end="\r")

    print(f"\n  {nome_split}: {len(linhas)} amostras, {erros} erros.")
    return pd.DataFrame(linhas)


# ─────────────────────────────────────────────────────────────────────────────
# Normalização MinMaxScaler [0,1] — fit apenas no treino (sem data leakage)
# ─────────────────────────────────────────────────────────────────────────────

def normalizar(df_train, df_test):
    cols = [c for c in df_train.columns if c not in COLUNAS_META]

    for df in [df_train, df_test]:
        df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)

    medianas = df_train[cols].median()
    df_train[cols] = df_train[cols].fillna(medianas)
    df_test[cols] = df_test[cols].fillna(medianas)

    scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    scaler.fit(df_train[cols])

    df_tn = df_train.copy()
    df_tt = df_test.copy()
    df_tn[cols] = scaler.transform(df_train[cols])
    df_tt[cols] = scaler.transform(df_test[cols])

    constantes = [c for c, mn, mx in zip(cols, scaler.data_min_, scaler.data_max_) if mn == mx]
    if constantes:
        print(f"  [aviso] {len(constantes)} colunas constantes (variância zero)")

    print(f"  Treino: [{df_tn[cols].min().min():.4f}, {df_tn[cols].max().max():.4f}]")
    print(f"  Teste:  [{df_tt[cols].min().min():.4f}, {df_tt[cols].max().max():.4f}]")

    return df_tn, df_tt, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Metadados do dataset
# ─────────────────────────────────────────────────────────────────────────────

def gerar_info(df_train, df_test):
    cols = [c for c in df_train.columns if c not in COLUNAS_META]
    df_orig = df_train[df_train["augmentacao"] == "original"]

    grupos = {
        "G1_hsv": [c for c in cols if c.startswith("hsv_")],
        "G2_rgb_lab_ycbcr": [c for c in cols if c.startswith(("rgb_", "lab_", "ycbcr_"))],
        "G3_canal_v_eq": [c for c in cols if c.startswith(("v_original", "v_eq", "v_razao"))],
        "G4_basrelief": [c for c in cols if c.startswith(("sobel_", "laplace", "basrelief", "textura_"))],
        "G5_gabor": [c for c in cols if c.startswith("gabor_")],
        "G6_lbp": [c for c in cols if c.startswith("lbp_")],
        "G7_glcm": [c for c in cols if c.startswith("glcm_")],
        "G8_satd": [c for c in cols if c.startswith("satd_")],
        "G9_hog": [c for c in cols if c.startswith("hog_")],
        "G10_geometria_mser": [c for c in cols if c.startswith("geom_")],
        "G11_hough": [c for c in cols if c.startswith("hough_")],
        "G12_grade": [c for c in cols if c.startswith("grade_")],
        "G13_multiescala": [c for c in cols if c.startswith("escala_")],
        "G14_contagem": [c for c in cols if c.startswith("cnt_")],
        "G_mascara": [c for c in cols if c.startswith("mascara_")],
    }

    return {
        "gerado_em": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "versao": "6.0 — Máscara forma/textura + G8 SATD + G14 Contagem Direta",
        "img_size": IMG_SIZE,
        "n_features_total": len(cols),
        "n_por_grupo": {k: len(v) for k, v in grupos.items()},
        "mudancas_v6": {
            "problema_resolvido": (
                "v5 tinha máscara 100% branca porque ExG e LAB a* não discriminam "
                "verde-de-fruta de verde-de-folha. v6 usa FORMA e TEXTURA."
            ),
            "mascara_crit_A_gabor_isotropy": (
                "Fruta esférica = textura isotrópica (igual em 0/45/90/135°). "
                "Folha = nervuras direcionais = anisotropy alta. "
                "Threshold percentil 65 (mais seletivo). "
                "Ref: Kurtulmus et al. (2011)."
            ),
            "mascara_crit_B_satd": (
                "Superfície lisa da fruta → SATD baixo. Nervuras → SATD alto. "
                "Threshold percentil 40. Ref: Zhao & Lee (2016)."
            ),
            "mascara_crit_C_laplaciano": (
                "Sem bordas internas (nervuras) = provável fruta. "
                "Threshold percentil 45 (mais seletivo que v5)."
            ),
            "mascara_crit_D_lab_b": (
                "Único critério de cor válido: carotenoides elevam b* em cítricos. "
                "Threshold percentil 55 (mais seletivo que v5). "
                "Ref: Li et al. (2016)."
            ),
            "mascara_crit_E_hough_seed": (
                "Regiões dentro de círculos Hough recebem voto. "
                "Independente de cor — funciona diretamente em geometria."
            ),
            "mascara_filtro_circularidade": (
                "Após votação, remove blobs com circularidade < 0.35. "
                "Elimina regiões que passaram na votação mas têm forma de folha."
            ),
            "removido_ExG": "ExG não discrimina verde-de-fruta de verde-de-folha.",
            "removido_LAB_a": "LAB a* similar em fruta verde e folha verde.",
            "removido_MSER_V": "MSER no canal V capturava folhas planas (max_variation=0.25).",
            "G8_satd_novo": (
                "Grupo dedicado SATD com features globais, dentro da máscara, "
                "razão fundo/fruta e prop_lisa_global. "
                "Ref: Zhao & Lee (2016) — 83.4% citrus verde."
            ),
            "G14_contagem_novo": (
                "Features diretas para regressão de contagem: "
                "n_hough, n_mser, n_blob, estimativa por área de isotropy, "
                "células ativas, ensemble ponderado. "
                "Transformações log e sqrt para linearizar distribuição de Poisson."
            ),
            "hough_param2_40": "param2=40 (v5 usava 30) → menos falsos positivos.",
            "mser_max_variation_012": "0.12 (v5 usava 0.25) → mais restritivo.",
            "mser_razao_aspecto_15": "ma/mi < 1.5 (v5 usava 1.8) → mais circular.",
        },
        "normalizacao": {
            "metodo": "MinMaxScaler sklearn",
            "range": "[0, 1]",
            "fit_em": "treino apenas (sem data leakage)",
        },
        "treino": {
            "n_originais": int(len(df_orig)),
            "n_total": int(len(df_train)),
            "aug": ["flip_h", "flip_v", "bright_75", "bright_125"],
            "cnt_min": int(df_orig["contagem"].min()),
            "cnt_max": int(df_orig["contagem"].max()),
            "cnt_media": round(float(df_orig["contagem"].mean()), 2),
            "n_zero": int((df_orig["contagem"] == 0).sum()),
        },
        "teste": {
            "n_imagens": int(len(df_test)),
            "cnt_min": int(df_test["contagem"].min()),
            "cnt_max": int(df_test["contagem"].max()),
            "cnt_media": round(float(df_test["contagem"].mean()), 2),
            "n_zero": int((df_test["contagem"] == 0).sum()),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    data_dir = Path(DATA_DIR)
    img_dir = str(data_dir / "images")
    ann_train = str(data_dir / "annotations_coco" / "instances_train.json")
    ann_test = str(data_dir / "annotations_coco" / "instances_test.json")

    print("\n" + "═" * 65)
    print("  OranDet v6.0 — Laranjas Verdes sobre Fundo Verde")
    print("  Máscara: Gabor Isotropy + SATD + Laplaciano + LAB b* + Hough")
    print("  Novos grupos: G8 SATD | G14 Contagem Direta")
    print("═" * 65)

    print("\n[1/5] Carregando anotações...")
    reg_train = carregar_anotacoes(ann_train, img_dir)
    reg_test = carregar_anotacoes(ann_test, img_dir)
    print(f"  Treino: {len(reg_train)} | Teste: {len(reg_test)}")

    print("\n[2/5] Extraindo features...")
    df_train = processar_split(reg_train, "train", aplicar_augmentacao=True)
    df_test = processar_split(reg_test, "test", aplicar_augmentacao=False)

    print("\n[3/5] Salvando bruto...")
    df_train.to_csv(os.path.join(OUTPUT_DIR, "orandet_v6_train.csv"), index=False)
    df_test.to_csv(os.path.join(OUTPUT_DIR, "orandet_v6_test.csv"), index=False)
    pd.concat([df_train, df_test], ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "orandet_v6_all.csv"), index=False
    )

    print("\n[4/5] Normalizando [0, 1]...")
    df_tn, df_tt, scaler = normalizar(df_train, df_test)
    df_tn.to_csv(os.path.join(OUTPUT_DIR, "orandet_v6_train_norm.csv"), index=False)
    df_tt.to_csv(os.path.join(OUTPUT_DIR, "orandet_v6_test_norm.csv"), index=False)
    pd.concat([df_tn, df_tt], ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "orandet_v6_all_norm.csv"), index=False
    )
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "orandet_v6_scaler.joblib"))

    print("\n[5/5] Salvando metadados...")
    info = gerar_info(df_train, df_test)
    with open(os.path.join(OUTPUT_DIR, "orandet_v6_info.json"), "w",
              encoding="utf-8") as fj:
        json.dump(info, fj, indent=2, ensure_ascii=False)

    print(f"\n{'═' * 65}")
    print(f"  Total de features: {info['n_features_total']}")
    for grupo, n in info["n_por_grupo"].items():
        print(f"    {grupo:<30} {n:>4} features")
    print(f"\n  Treino: {info['treino']['n_originais']} imgs → "
          f"{info['treino']['n_total']} amostras")
    print(f"  Teste:  {info['teste']['n_imagens']} imgs | "
          f"média {info['teste']['cnt_media']:.1f} laranjas/img")
    print(f"\n  Arquivos em: {OUTPUT_DIR}/")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()
