"""
REFERÊNCIAS PRINCIPAIS:
  Maldonado & Barbosa (2016) — Bas-relief + razão brilho vertical, citrus verde
  Kurtulmus et al. (2011) — Gabor isotropy, citrus sobre fundo verde
  Zhao & Lee (2016) — SATD, 83.4% acurácia citrus verde
  Okamoto & Lee (2009) — Chromaticidade Cr-Cb para citrus verde
  Hu (2018) — LBP + MSER + Hough hierárquico para citrus verde
  Frangi et al. (1998) — Hessiana multiscale blob enhancement
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
from skimage.feature import hog, local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "/Users/antonioreis/Downloads/dataverse_files"
OUTPUT_DIR = "./dataset_preparado_v80"
IMG_SIZE = 416

COLUNAS_META = [
    "image_id", "file_name", "split", "contagem",
    "contagem_log1p", "contagem_sqrt", "augmentacao",
]

# Thresholds para limpeza de features
VAR_THRESHOLD = 1e-5
CORR_THRESHOLD = 0.97

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PRIMITIVAS COMPARTILHADAS
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_skew(arr):
    val = skew(arr.flatten())
    return float(val) if np.isfinite(val) else 0.0


def _safe_kurtosis(arr):
    val = kurtosis(arr.flatten())
    return float(val) if np.isfinite(val) else 0.0


def _first_order(canal, prefixo):
    """12 estatísticas de primeira ordem compactas."""
    c = np.array(canal, dtype=np.float64).flatten()
    sufixos = [
        "mean", "std", "median", "entropy",
        "p10", "p90", "iqr",
        "skewness", "kurtosis", "energy", "uniformity", "mad",
    ]
    if len(c) == 0:
        return {f"{prefixo}_{s}": 0.0 for s in sufixos}

    norm = 255.0
    hist, _ = np.histogram(c, bins=256, range=(0, 256), density=False)
    prob = hist / (hist.sum() + 1e-9)
    prob_nz = prob[prob > 0]

    p10 = np.percentile(c, 10)
    p90 = np.percentile(c, 90)

    return {
        f"{prefixo}_mean": float(c.mean() / norm),
        f"{prefixo}_std": float(c.std() / norm),
        f"{prefixo}_median": float(np.median(c) / norm),
        f"{prefixo}_entropy": float(-np.sum(prob_nz * np.log2(prob_nz + 1e-10))),
        f"{prefixo}_p10": float(p10 / norm),
        f"{prefixo}_p90": float(p90 / norm),
        f"{prefixo}_iqr": float((np.percentile(c, 75) - np.percentile(c, 25)) / norm),
        f"{prefixo}_skewness": _safe_skew(c),
        f"{prefixo}_kurtosis": _safe_kurtosis(c),
        f"{prefixo}_energy": float(np.sum(c ** 2) / (norm ** 2 * c.size)),
        f"{prefixo}_uniformity": float(np.sum(prob ** 2)),
        f"{prefixo}_mad": float(np.mean(np.abs(c - c.mean())) / norm),
    }


def _first_order_mascara(canal, mascara, prefixo):
    """Aplica first_order apenas nos pixels dentro da máscara."""
    px = canal[mascara > 0]
    if len(px) == 0:
        return {f"{prefixo}_{s}": 0.0 for s in [
            "mean", "std", "median", "entropy", "p10", "p90", "iqr",
            "skewness", "kurtosis", "energy", "uniformity", "mad",
        ]}
    return _first_order(px, prefixo)


# ═══════════════════════════════════════════════════════════════════════════════
# MAPAS AUXILIARES (compartilhados entre máscara e features)
# ═══════════════════════════════════════════════════════════════════════════════

def _gabor_isotropy_map(gray):
    """Isotropia Gabor: fruta esférica = resposta uniforme em todas orientações."""
    orientacoes = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    respostas = []
    for theta in orientacoes:
        kernel = cv2.getGaborKernel(
            (21, 21), sigma=4.0, theta=theta,
            lambd=10.0, gamma=1.0, psi=0,
        )
        resp = np.abs(cv2.filter2D(gray, cv2.CV_64F, kernel))
        respostas.append(resp)

    stacked = np.stack(respostas, axis=0)
    mean_resp = stacked.mean(axis=0) + 1e-9
    std_resp = stacked.std(axis=0)
    isotropy = 1.0 - np.clip(std_resp / mean_resp, 0, 1)
    return isotropy.astype(np.float32)


def _satd_map(gray):
    """SATD: fruta lisa = baixo, nervura = alto."""
    gray_f = gray.astype(np.float32)
    blur5 = cv2.blur(gray_f, (5, 5))
    blur11 = cv2.blur(gray_f, (11, 11))
    return (np.abs(gray_f - blur5) + np.abs(gray_f - blur11)) / 2.0


def _laplacian_smooth_map(gray):
    """Laplaciano suavizado: folha = alto (nervuras), fruta = baixo."""
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap_abs = np.abs(lap).astype(np.float32)
    return cv2.GaussianBlur(lap_abs, (31, 31), 0)


def _hessian_convexity_map(gray):
    """
    Mapa de 'blob-ness' via Hessiana.
    Esfera: det>0 e trace>0. Folha plana: det≈0. Nervura: det<0.
    Ref: Frangi et al. (1998).
    """
    gray_f = gray.astype(np.float64)
    gray_blur = cv2.GaussianBlur(gray_f, (5, 5), 1.5)

    Dxx = cv2.Sobel(gray_blur, cv2.CV_64F, 2, 0, ksize=3)
    Dyy = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 2, ksize=3)
    Dxy = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 1, ksize=3)

    trace = Dxx + Dyy
    det = Dxx * Dyy - Dxy ** 2

    blobness = np.where(
        (det > 0) & (trace > 0),
        det / (trace ** 2 + 1e-9),
        0.0,
    ).astype(np.float32)

    return cv2.GaussianBlur(blobness, (15, 15), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# MÁSCARA v7.1 — CORREÇÕES CRÍTICAS
#
# 1. Adiciona Cr-Cb (Okamoto 2009) e Hessiana (Frangi 1998) como critérios
# 2. Votação 2/6 (era 3/5) — mais sensível a frutas em sombra
# 3. Fallback Gabor+Cr-Cb quando máscara < 0.5%
# 4. Filtro de circularidade ↑ 0.40 + solidity ≥ 0.75
# 5. Morfologia mais suave (kernel 5×5, menos iterações)
# ═══════════════════════════════════════════════════════════════════════════════

def construir_mascara_fruta_verde(img_bgr):
    """
    Máscara v7.1: 6 critérios (votação 2/6).
    A: Gabor Isotropy  |  B: SATD liso  |  C: Laplaciano baixo
    D: LAB b*          |  E: Hessiana blob-ness  |  F: Cr-Cb diff
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    # --- Critério A: Gabor Isotropy (Kurtulmus 2011) ---
    iso_map = _gabor_isotropy_map(gray)
    crit_A = (iso_map >= float(np.percentile(iso_map, 55))).astype(np.float32)

    # --- Critério B: SATD liso (Zhao & Lee 2016) ---
    satd_map_v = _satd_map(gray)
    crit_B = (satd_map_v <= float(np.percentile(satd_map_v, 50))).astype(np.float32)

    # --- Critério C: Laplaciano baixo (sem nervuras) ---
    lap_map = _laplacian_smooth_map(gray)
    crit_C = (lap_map <= float(np.percentile(lap_map, 55))).astype(np.float32)

    # --- Critério D: LAB b* (carotenoides) ---
    b_ch = lab[:, :, 2].astype(np.float32)
    crit_D = (b_ch >= float(np.percentile(b_ch, 50))).astype(np.float32)

    # --- Critério E: Hessiana blob-ness (Frangi 1998) — NOVO v7.1 ---
    hess_map = _hessian_convexity_map(gray)
    crit_E = (hess_map >= float(np.percentile(hess_map, 60))).astype(np.float32)

    # --- Critério F: Cr-Cb discrimination (Okamoto & Lee 2009) — NOVO v7.1 ---
    Cr = ycrcb[:, :, 1].astype(np.float32)
    Cb = ycrcb[:, :, 2].astype(np.float32)
    crcb_diff = Cr - Cb
    crit_F = (crcb_diff >= float(np.percentile(crcb_diff, 45))).astype(np.float32)

    # --- Votação 2/6 (mais permissiva que 3/5) ---
    voto = crit_A + crit_B + crit_C + crit_D + crit_E + crit_F
    mask_raw = (voto >= 2).astype(np.uint8) * 255

    # --- Morfologia suavizada ---
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, k3, iterations=1)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k5, iterations=2)

    # --- Filtro por circularidade + solidity (mais rigoroso) ---
    cnts, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara_filtrada = np.zeros_like(mascara)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        perim = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * area / (perim ** 2 + 1e-6)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)
        if circ >= 0.40 and solidity >= 0.75:
            cv2.drawContours(mascara_filtrada, [cnt], -1, 255, -1)

    prop = float(mascara_filtrada.sum()) / (255.0 * h * w)

    # --- FALLBACK: máscara muito pequena → usa apenas A+F (mais robustos) ---
    if prop < 0.005:
        fallback_mask = ((crit_A + crit_F) >= 1).astype(np.uint8) * 255
        fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_CLOSE, k5, iterations=2)
        cnts_fb, _ = cv2.findContours(fallback_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mascara_fb = np.zeros_like(mascara)
        for cnt in cnts_fb:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue
            perim = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * area / (perim ** 2 + 1e-6)
            if circ >= 0.35:
                cv2.drawContours(mascara_fb, [cnt], -1, 255, -1)
        prop_fb = float(mascara_fb.sum()) / (255.0 * h * w)
        if 0.005 <= prop_fb <= 0.45:
            mascara_filtrada = mascara_fb
            prop = prop_fb

    # --- Proteção final ---
    if prop < 0.002 or prop > 0.45:
        mascara_filtrada = np.zeros_like(mascara_filtrada)

    return mascara_filtrada


# ═══════════════════════════════════════════════════════════════════════════════
# GRUPOS DE FEATURES (G1-G16) — mantidos da v7
# ═══════════════════════════════════════════════════════════════════════════════

# G1 — HSV
def features_hsv(img_bgr, hsv):
    feats = {}
    for i, nome in enumerate(["H", "S", "V"]):
        hist = cv2.calcHist([hsv], [i], None, [16], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-7)
        for b, val in enumerate(hist):
            feats[f"hsv_hist_{nome}_b{b:02d}"] = float(val)
        feats.update(_first_order(hsv[:, :, i], f"hsv_{nome}"))
    return feats


# G2 — RGB + LAB + YCbCr
def features_rgb_lab(img_bgr):
    feats = {}
    for nome, idx in {"R": 2, "G": 1, "B": 0}.items():
        feats.update(_first_order(img_bgr[:, :, idx], f"rgb_{nome}"))

    R = img_bgr[:, :, 2].astype(np.float64) + 1e-7
    G = img_bgr[:, :, 1].astype(np.float64) + 1e-7
    B = img_bgr[:, :, 0].astype(np.float64) + 1e-7
    feats["rgb_razao_RG"] = float((R / G).mean())
    feats["rgb_razao_RB"] = float((R / B).mean())
    feats["rgb_razao_GB"] = float((G / B).mean())
    feats["rgb_ExG"] = float(((2 * G - R - B) / (R + G + B + 1e-7)).mean())

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    for nome, idx in {"L": 0, "a": 1, "b": 2}.items():
        feats.update(_first_order(lab[:, :, idx], f"lab_{nome}"))
    a_ch = lab[:, :, 1].astype(np.float64) + 1e-7
    b_ch = lab[:, :, 2].astype(np.float64) + 1e-7
    feats["lab_razao_a_b"] = float((a_ch / b_ch).mean())

    ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    for nome, idx in {"Y": 0, "Cr": 1, "Cb": 2}.items():
        feats.update(_first_order(ycbcr[:, :, idx], f"ycbcr_{nome}"))

    return feats


# G3 — Canal V equalizado
def features_canal_v_eq(hsv, mascara):
    feats = {}
    V = hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    V_eq = clahe.apply(V)

    feats.update(_first_order_mascara(V, mascara, "v_original"))
    feats.update(_first_order_mascara(V_eq, mascara, "v_eq"))

    hist = cv2.calcHist([V_eq], [0], None, [16], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-7)
    for b, val in enumerate(hist):
        feats[f"v_eq_hist_b{b:02d}"] = float(val)

    tem = mascara.sum() > 0
    v_fruta = float(V[mascara > 0].mean()) if tem else 0.0
    v_global = float(V.mean()) + 1e-6
    feats["v_razao_fruta_global"] = v_fruta / v_global

    return feats, V_eq


# G4 — Bas-relief Maldonado
def _bas_relief_map(V_eq):
    sobel_x = cv2.Sobel(V_eq, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x_abs = np.abs(sobel_x).astype(np.float32)
    lap = cv2.Laplacian(V_eq, cv2.CV_64F, ksize=3)
    lap_abs = np.abs(lap).astype(np.float32)
    bas = cv2.addWeighted(sobel_x_abs, 0.6, lap_abs, 0.4, 0)
    bas_blur = cv2.GaussianBlur(bas, (11, 11), 0)
    return cv2.normalize(bas_blur, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def _vertical_brightness_ratio(V, mascara=None):
    h = V.shape[0]
    mid = h // 2
    if mascara is not None and mascara.sum() > 0:
        upper = V[:mid, :][mascara[:mid, :] > 0]
        lower = V[mid:, :][mascara[mid:, :] > 0]
    else:
        upper = V[:mid, :].flatten()
        lower = V[mid:, :].flatten()
    if len(upper) == 0 or len(lower) == 0:
        return 1.0
    return (float(np.mean(upper)) + 1e-6) / (float(np.mean(lower)) + 1e-6)


def features_basrelief(V_eq, mascara):
    feats = {}
    V_blur = cv2.GaussianBlur(V_eq, (5, 5), 0)
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

    bas = _bas_relief_map(V_eq)
    feats.update(_first_order(bas, "basrelief_maldonado_global"))
    feats.update(_first_order_mascara(bas, mascara, "basrelief_maldonado_fruta"))

    V_uint8 = V_eq.astype(np.uint8)
    ratio_global = _vertical_brightness_ratio(V_uint8)
    ratio_fruta = _vertical_brightness_ratio(V_uint8, mascara) if mascara.sum() > 0 else ratio_global
    feats["brilho_razao_vertical_global"] = float(np.clip(ratio_global, 0, 5))
    feats["brilho_razao_vertical_fruta"] = float(np.clip(ratio_fruta, 0, 5))
    feats["brilho_razao_vertical_diff"] = float(ratio_fruta - ratio_global)

    return feats


# G5 — Gabor Circular
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

        stacked = np.stack(respostas, axis=0)
        mean_r = stacked.mean(axis=0) + 1e-9
        cv_entre = stacked.std(axis=0) / mean_r
        isotropy = 1.0 - np.clip(cv_entre, 0, 1)

        feats[f"gabor_lam{lam}_isotropy_mean"] = float(isotropy.mean())
        feats[f"gabor_lam{lam}_isotropy_std"] = float(isotropy.std())
        feats[f"gabor_lam{lam}_isotropy_p75"] = float(np.percentile(isotropy, 75))
        feats[f"gabor_lam{lam}_isotropy_p90"] = float(np.percentile(isotropy, 90))
        feats[f"gabor_lam{lam}_prop_isotropy_high"] = float((isotropy > 0.7).mean())

    return feats


# G6 — LBP
def features_lbp(gray):
    feats = {}
    for P, R_lbp, nome in [(8, 1, "s1"), (16, 2, "s2")]:
        lbp = local_binary_pattern(gray, P=P, R=R_lbp, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(lbp.flatten(), bins=n_bins, range=(0, n_bins), density=True)
        for b, val in enumerate(hist):
            feats[f"lbp_{nome}_b{b:02d}"] = float(val)
        feats[f"lbp_{nome}_mean"] = float(lbp.mean())
        feats[f"lbp_{nome}_std"] = float(lbp.std())
        h_nz = hist[hist > 0]
        feats[f"lbp_{nome}_entropy"] = float(-np.sum(h_nz * np.log2(h_nz + 1e-10)))
    return feats


# G7 — GLCM Haralick
def features_glcm(gray):
    feats = {}
    gray_q = (gray // 4).astype(np.uint8)
    angulos = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    for dist in [1, 3]:
        glcm = graycomatrix(gray_q, distances=[dist], angles=angulos,
                            levels=64, symmetric=True, normed=True)
        for prop in ["contrast", "correlation", "energy", "homogeneity", "dissimilarity"]:
            vals = graycoprops(glcm, prop)[0]
            feats[f"glcm_d{dist}_{prop}_mean"] = float(vals.mean())
            feats[f"glcm_d{dist}_{prop}_std"] = float(vals.std())

        gf = gray.astype(np.float64)
        sigma = gf.std()
        feats[f"glcm_d{dist}_img_smoothness"] = float(1.0 - 1.0 / (1.0 + sigma ** 2 + 1e-7))
        feats[f"glcm_d{dist}_img_skewness"] = _safe_skew(gf)
        feats[f"glcm_d{dist}_img_kurtosis"] = _safe_kurtosis(gf)

    return feats


# G8 — SATD
def features_satd(gray, mascara):
    feats = {}
    satd_map = _satd_map(gray)
    satd_255 = np.clip(satd_map / (satd_map.max() + 1e-9) * 255, 0, 255).astype(np.uint8)

    feats.update(_first_order(satd_255, "satd_global"))
    feats.update(_first_order_mascara(satd_255, mascara, "satd_fruta"))

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
    thr_liso = float(np.percentile(satd_map, 35))
    feats["satd_prop_lisa_global"] = float((satd_map <= thr_liso).mean())

    return feats


# G9 — HOG
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


# G10 — Geometria de contornos + MSER
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
                        ("aspect_ratio", aspect_ratios), ("circularity", circularities)]:
        arr = np.array(lista) if lista else np.array([0.0])
        feats[f"geom_{nome}_mean"] = float(arr.mean())
        feats[f"geom_{nome}_std"] = float(arr.std())
        feats[f"geom_{nome}_max"] = float(arr.max())
        feats[f"geom_{nome}_count"] = float(len(lista))

    n_circ = sum(1 for c in circularities if c >= 0.5)
    feats["geom_n_blobs_circulares"] = float(n_circ)
    feats["geom_prop_blobs_circulares"] = float(n_circ / (len(circularities) + 1e-6))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        mser = cv2.MSER_create(
            _delta=8, _min_area=400, _max_area=25000,
            _max_variation=0.12,
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
                    if mi > 0 and (ma / mi) < 1.5:
                        n_circ_mser += 1
    except Exception:
        n_total = n_circ_mser = 0

    feats["geom_mser_total"] = float(n_total)
    feats["geom_mser_circular"] = float(n_circ_mser)
    feats["geom_log_mser"] = float(np.log1p(n_circ_mser))

    return feats


# G11 — Hough Circles (v8.0: 6 faixas de raio + stats espaciais + overlap)
def features_hough_circles(img_bgr, mascara):
    """
    Hough expandido v8.0:
    - 6 faixas de raio não sobrepostas (antes: 3)
    - raio médio por faixa, densidade por área
    - distância média entre círculos e estimativa de sobreposição
    - compatibilidade retroativa: hough_pequeno/medio/grande_count preservados
    """
    feats = {}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    # 6 faixas não sobrepostas — captura diversidade de tamanho de frutos
    faixas_novas = [
        (8,  18, "f1_micro"),
        (18, 32, "f2_xpequeno"),
        (32, 48, "f3_pequeno"),
        (48, 68, "f4_medio"),
        (68, 95, "f5_grande"),
        (95, 135, "f6_xgrande"),
    ]
    total_circ = 0
    todos_radii = []
    todos_centros = []

    for rmin, rmax, nome in faixas_novas:
        min_dist = max(int(rmin * 1.2), 15)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=min_dist,
            param1=50, param2=38,
            minRadius=rmin, maxRadius=rmax,
        )
        n_v = int(len(circles[0])) if circles is not None else 0
        radii = [float(r) for _, _, r in circles[0]] if circles is not None else []
        centros = [(float(x), float(y)) for x, y, _ in circles[0]] if circles is not None else []

        feats[f"hough_{nome}_count"] = float(n_v)
        feats[f"hough_{nome}_raio_mean"] = float(np.mean(radii)) if radii else 0.0
        feats[f"hough_{nome}_raio_std"] = float(np.std(radii)) if radii else 0.0
        # densidade: círculos detectados por área de 100×100 px
        feats[f"hough_{nome}_density"] = float(n_v / (IMG_SIZE * IMG_SIZE / 1e4))
        total_circ += n_v
        todos_radii.extend(radii)
        todos_centros.extend(centros)

    # ── Compatibilidade retroativa (código legado usa esses nomes) ──────────
    feats["hough_pequeno_count"] = feats["hough_f1_micro_count"] + feats["hough_f2_xpequeno_count"]
    feats["hough_medio_count"]   = feats["hough_f3_pequeno_count"] + feats["hough_f4_medio_count"]
    feats["hough_grande_count"]  = feats["hough_f5_grande_count"] + feats["hough_f6_xgrande_count"]
    feats["hough_pequeno_raio_mean"] = feats["hough_f2_xpequeno_raio_mean"]
    feats["hough_medio_raio_mean"]   = feats["hough_f4_medio_raio_mean"]
    feats["hough_grande_raio_mean"]  = feats["hough_f5_grande_raio_mean"]

    feats["hough_total_estimado"] = float(total_circ)
    feats["hough_log_total"] = float(np.log1p(total_circ))
    feats["hough_sqrt_total"] = float(np.sqrt(total_circ))
    feats["hough_prop_pequenos"] = float(feats["hough_pequeno_count"] / (total_circ + 1e-7))
    feats["hough_prop_medios"]   = float(feats["hough_medio_count"] / (total_circ + 1e-7))
    feats["hough_prop_grandes"]  = float(feats["hough_grande_count"] / (total_circ + 1e-7))

    # ── Stats globais de raio ─────────────────────────────────────────────────
    if todos_radii:
        feats["hough_raio_global_mean"] = float(np.mean(todos_radii))
        feats["hough_raio_global_std"]  = float(np.std(todos_radii))
        feats["hough_raio_global_p25"]  = float(np.percentile(todos_radii, 25))
        feats["hough_raio_global_p75"]  = float(np.percentile(todos_radii, 75))
        feats["hough_raio_global_iqr"]  = feats["hough_raio_global_p75"] - feats["hough_raio_global_p25"]
    else:
        for k in ["mean", "std", "p25", "p75", "iqr"]:
            feats[f"hough_raio_global_{k}"] = 0.0

    # ── Distância média entre círculos detectados ─────────────────────────────
    if len(todos_centros) >= 2:
        centros_arr = np.array(todos_centros)
        dists = []
        n_c = len(centros_arr)
        for k in range(n_c):
            # só os 4 vizinhos mais próximos para controlar custo O(n²)
            for ll in range(k + 1, min(k + 5, n_c)):
                dists.append(float(np.linalg.norm(centros_arr[k] - centros_arr[ll])))
        feats["hough_dist_media"] = float(np.mean(dists))
        feats["hough_dist_std"]   = float(np.std(dists))
    else:
        feats["hough_dist_media"] = 0.0
        feats["hough_dist_std"]   = 0.0

    # ── Estimativa de sobreposição (proxy para oclusão) ───────────────────────
    n_overlap = 0
    if len(todos_centros) >= 2 and len(todos_radii) == len(todos_centros):
        centros_arr = np.array(todos_centros)
        radii_arr   = np.array(todos_radii)
        n_c = len(centros_arr)
        for k in range(n_c):
            for ll in range(k + 1, n_c):
                d = float(np.linalg.norm(centros_arr[k] - centros_arr[ll]))
                if d < (radii_arr[k] + radii_arr[ll]) * 0.85:
                    n_overlap += 1
    feats["hough_overlap_est"]     = float(n_overlap)
    feats["hough_overlap_log"]     = float(np.log1p(n_overlap))

    # ── Hough sobre região da máscara ─────────────────────────────────────────
    if mascara.sum() > 0:
        gray_m = gray.copy()
        gray_m[mascara == 0] = 0
        blur_m = cv2.GaussianBlur(gray_m, (9, 9), 2)
        circles_m = cv2.HoughCircles(blur_m, cv2.HOUGH_GRADIENT,
                                     dp=1.2, minDist=20, param1=50, param2=35,
                                     minRadius=8, maxRadius=135)
        n_m = int(len(circles_m[0])) if circles_m is not None else 0
    else:
        n_m = 0
    feats["hough_mascara_count"] = float(n_m)
    feats["hough_log_mascara"]   = float(np.log1p(n_m))

    return feats


# G12 — Grade espacial
def features_grade_espacial(hsv, mascara, grid=(4, 4)):
    feats = {}
    V = hsv[:, :, 2].astype(np.float64)
    H_img, W_img = mascara.shape
    gh, gw = H_img // grid[0], W_img // grid[1]

    densidades = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            y1, y2 = i * gh, (i + 1) * gh
            x1, x2 = j * gw, (j + 1) * gw
            dens = float(mascara[y1:y2, x1:x2].mean()) / 255.0
            brilho = float(V[y1:y2, x1:x2].mean()) / 255.0
            feats[f"grade_{i}_{j}_densidade"] = dens
            feats[f"grade_{i}_{j}_brilho"] = brilho
            densidades.append(dens)

    d = np.array(densidades)
    feats["grade_dens_mean"] = float(d.mean())
    feats["grade_dens_std"] = float(d.std())
    feats["grade_dens_max"] = float(d.max())
    feats["grade_n_celulas_ativas"] = float((d > 0.05).sum())
    feats["grade_frac_ativa"] = float((d > 0.05).mean())
    d_norm = d / (d.sum() + 1e-9)
    d_nz = d_norm[d_norm > 0]
    feats["grade_entropia_espacial"] = float(-np.sum(d_nz * np.log2(d_nz + 1e-10)))

    return feats


# G13 — Multi-escala
def features_multiescala(img_bgr):
    feats = {}

    escalas_data = {}

    for fator, nome in [(0.5, "escala_208"), (0.25, "escala_104")]:

        sz = (int(IMG_SIZE * fator), int(IMG_SIZE * fator))

        img_r = cv2.resize(
            img_bgr,
            sz,
            interpolation=cv2.INTER_AREA
        )

        hsv_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2HSV)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        mask_r = construir_mascara_fruta_verde(img_r)

        # ============================================================
        # HOG
        # ============================================================

        hog_v = hog(
            gray_r,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True,
            block_norm="L2-Hys"
        )

        feats[f"{nome}_hog_mean"] = float(hog_v.mean())
        feats[f"{nome}_hog_std"] = float(hog_v.std())

        # ============================================================
        # FFT / frequência
        # ============================================================

        fft = np.fft.fft2(gray_r)
        fft_shift = np.fft.fftshift(fft)

        mag = np.log1p(np.abs(fft_shift))

        feats[f"{nome}_fft_mean"] = float(mag.mean())
        feats[f"{nome}_fft_std"] = float(mag.std())
        feats[f"{nome}_fft_energy"] = float(np.mean(mag ** 2))

        # ============================================================
        # Laplacian
        # ============================================================

        lap = cv2.Laplacian(gray_r, cv2.CV_32F)

        feats[f"{nome}_lap_mean"] = float(np.mean(np.abs(lap)))
        feats[f"{nome}_lap_std"] = float(np.std(lap))

        # ============================================================
        # HSV simplificado
        # ============================================================

        for ci, cn in enumerate(["H", "S", "V"]):

            hist = cv2.calcHist(
                [hsv_r],
                [ci],
                None,
                [8],
                [0, 256]
            ).flatten()

            hist = hist / (hist.sum() + 1e-7)

            feats[f"{nome}_{cn}_entropy"] = float(
                -np.sum(hist * np.log2(hist + 1e-9))
            )

            feats[f"{nome}_{cn}_maxbin"] = float(hist.max())

        # ============================================================
        # máscara
        # ============================================================

        prop_fruta = float(mask_r.mean()) / 255.0

        feats[f"{nome}_prop_fruta"] = prop_fruta

        # ============================================================
        # blobs
        # ============================================================

        k3 = np.ones((3, 3), np.uint8)

        mask_c = cv2.morphologyEx(
            mask_r,
            cv2.MORPH_OPEN,
            k3
        )

        cnts, _ = cv2.findContours(
            mask_c,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        areas = []

        n_blobs = 0

        for cnt in cnts:

            area = cv2.contourArea(cnt)

            if area < 10:
                continue

            circ = (
                4 * np.pi * area /
                (cv2.arcLength(cnt, True) ** 2 + 1e-6)
            )

            if circ >= 0.35:
                n_blobs += 1
                areas.append(area)

        feats[f"{nome}_n_blobs"] = float(n_blobs)

        if len(areas) > 0:

            feats[f"{nome}_blob_area_mean"] = float(np.mean(areas))
            feats[f"{nome}_blob_area_std"] = float(np.std(areas))

        else:

            feats[f"{nome}_blob_area_mean"] = 0.0
            feats[f"{nome}_blob_area_std"] = 0.0

        # ============================================================
        # spatial pyramid simples (2x2)
        # ============================================================

        h, w = gray_r.shape

        idx = 0

        for y0, y1 in [(0, h//2), (h//2, h)]:
            for x0, x1 in [(0, w//2), (w//2, w)]:

                patch_mask = mask_r[y0:y1, x0:x1]

                prop_patch = float(patch_mask.mean()) / 255.0

                feats[f"{nome}_patch{idx}_prop"] = prop_patch

                idx += 1

        # ============================================================
        # Gabor isotropia
        # ============================================================

        for lam in [8, 15]:

            resps = []

            for theta in [
                0,
                np.pi / 4,
                np.pi / 2,
                3 * np.pi / 4
            ]:

                k = cv2.getGaborKernel(
                    (15, 15),
                    3.0,
                    theta,
                    float(lam),
                    1.0,
                    0
                )

                r = np.abs(
                    cv2.filter2D(gray_r, cv2.CV_64F, k)
                )

                resps.append(r)

            stk = np.stack(resps, axis=0)

            cv_ = stk.std(axis=0) / (stk.mean(axis=0) + 1e-9)

            iso = 1.0 - np.clip(cv_, 0, 1)

            feats[f"{nome}_gabor_iso_mean_{lam}"] = float(iso.mean())
            feats[f"{nome}_gabor_iso_std_{lam}"] = float(iso.std())

        # salvar para relações entre escalas
        escalas_data[nome] = {
            "prop": prop_fruta,
            "blobs": n_blobs,
            "hog_mean": hog_v.mean(),
            "fft_mean": mag.mean()
        }

    # ============================================================
    # RELAÇÕES ENTRE ESCALAS
    # ============================================================

    e208 = escalas_data["escala_208"]
    e104 = escalas_data["escala_104"]

    feats["multi_ratio_blobs"] = float(
        e208["blobs"] / (e104["blobs"] + 1e-6)
    )

    feats["multi_ratio_prop"] = float(
        e208["prop"] / (e104["prop"] + 1e-6)
    )

    feats["multi_delta_hog"] = float(
        e208["hog_mean"] - e104["hog_mean"]
    )

    feats["multi_delta_fft"] = float(
        e208["fft_mean"] - e104["fft_mean"]
    )

    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# G14 — Contagem Direta v7.1 — CORREÇÕES CRÍTICAS
#
# 1. Watershed adaptativo (kernel/threshold dependem do tamanho da máscara)
# 2. Proteção Hough: zera se máscara < 1% (evita falsos positivos em fundo)
# 3. Área isotropia recalibrada: raio médio estimado da própria máscara
# 4. Ensemble v7.1 com pesos adaptativos + truncamento de outliers
# 5. Ensemble v7 legado preservado para compatibilidade
# ═══════════════════════════════════════════════════════════════════════════════

def _watershed_count(gray, mascara):
    """
    Watershed adaptativo: erosão e threshold ajustados ao tamanho da máscara.
    Máscaras pequenas (< 2%) usam erosão suave para não desaparecer.
    """
    if mascara.sum() == 0:
        return 0

    prop = float(mascara.sum()) / (255.0 * mascara.shape[0] * mascara.shape[1])

    # Kernel adaptativo
    if prop < 0.02:
        kernel = np.ones((3, 3), np.uint8)
        iterations = 1
    else:
        kernel = np.ones((5, 5), np.uint8)
        iterations = 2

    eroded = cv2.erode(mascara, kernel, iterations=iterations)
    if eroded.sum() == 0:
        return 0

    dist = ndimage.distance_transform_edt(eroded)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold adaptativo: mais sensível para máscaras pequenas
    thr_pct = 40 if prop < 0.02 else 50
    thr_dist = float(np.percentile(dist[dist > 0], thr_pct)) if np.any(dist > 0) else 5
    _, markers_bin = cv2.threshold(dist_norm, int(thr_dist), 255, cv2.THRESH_BINARY)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        markers_bin.astype(np.uint8), connectivity=8
    )
    area_min = 15 if prop < 0.02 else 20
    n_valid = sum(
        1 for i in range(1, n_labels)
        if stats[i, cv2.CC_STAT_AREA] >= area_min
    )
    return n_valid


def features_contagem_direta(img_bgr, mascara, n_hough_total, n_hough_mascara,
                             n_mser_circular):
    """
    Consolida estimadores diretos de contagem com proteções v7.1.
    """
    feats = {}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]

    prop_mascara = float(mascara.sum()) / (255.0 * mascara.shape[0] * mascara.shape[1])

    # ── Estimador 1: Blob circular (filtros mais rigorosos) ──────────────────
    k3 = np.ones((3, 3), np.uint8)
    mask_c = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k3)
    cnts, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_blob = sum(
        1 for cnt in cnts
        if cv2.contourArea(cnt) >= 100 and
        4 * np.pi * cv2.contourArea(cnt) / (cv2.arcLength(cnt, True) ** 2 + 1e-6) >= 0.40
    )
    feats["cnt_blob_n"] = float(n_blob)
    feats["cnt_blob_log"] = float(np.log1p(n_blob))
    feats["cnt_blob_sqrt"] = float(np.sqrt(n_blob))

    # ── Estimador 2: Watershed adaptativo ──────────────────────────────
    n_watershed = _watershed_count(gray, mascara)
    feats["cnt_watershed_n"] = float(n_watershed)
    feats["cnt_watershed_log"] = float(np.log1p(n_watershed))
    feats["cnt_watershed_sqrt"] = float(np.sqrt(n_watershed))

    # ── Estimador 3: Hough (proteção anti-ruído) ──────────────────────────
    if prop_mascara < 0.01:
        n_hough_total = 0
        n_hough_mascara = 0
    feats["cnt_hough_n"] = float(n_hough_total)
    feats["cnt_hough_mascara_n"] = float(n_hough_mascara)
    feats["cnt_hough_log"] = float(np.log1p(n_hough_total))
    feats["cnt_hough_sqrt"] = float(np.sqrt(n_hough_total))

    # ── Estimador 4: MSER ──────────────────────────
    feats["cnt_mser_n"] = float(n_mser_circular)
    feats["cnt_mser_log"] = float(np.log1p(n_mser_circular))
    feats["cnt_mser_sqrt"] = float(np.sqrt(n_mser_circular))

    # ── Estimador 5: Área de isotropia (recalibrado por escala real) ────
    iso_map = _gabor_isotropy_map(gray)
    area_iso_alta = float((iso_map > 0.70).mean())

    # Estima raio médio a partir da máscara atual
    if mascara.sum() > 0:
        cnts_m, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas_m = [cv2.contourArea(c) for c in cnts_m if cv2.contourArea(c) > 0]
        if areas_m:
            r_medio = np.sqrt(np.median(areas_m) / np.pi)
            r_medio = float(np.clip(r_medio, 15, 60))
        else:
            r_medio = 30.0
    else:
        r_medio = 30.0

    n_est_area = area_iso_alta * IMG_SIZE * IMG_SIZE / (np.pi * r_medio ** 2 + 1e-6)
    feats["cnt_estimativa_area_iso"] = float(np.clip(n_est_area, 0, 200))
    feats["cnt_area_iso_prop"] = area_iso_alta
    feats["cnt_area_iso_rmedio"] = r_medio

    # ── Estimador 6: Bas-relief + razão vertical ────
    V_uint8 = V.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    V_eq = clahe.apply(V_uint8)
    bas = _bas_relief_map(V_eq)

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

    thr_bas = float(np.percentile(bas, 50))
    candidatos = (bas >= thr_bas) & (ratio_map >= 1.15)
    candidatos_u8 = (candidatos * 255).astype(np.uint8)
    cnts2, _ = cv2.findContours(candidatos_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_basrelief = sum(1 for cnt in cnts2 if cv2.contourArea(cnt) >= 150)
    feats["cnt_basrelief_n"] = float(n_basrelief)
    feats["cnt_basrelief_log"] = float(np.log1p(n_basrelief))
    feats["cnt_basrelief_sqrt"] = float(np.sqrt(n_basrelief))

    # ── Ensemble v7.1 (robusto: pesos adaptativos + truncamento) ───────────────────
    estimadores = np.array([
        n_hough_total,
        n_mser_circular,
        n_watershed,
        n_blob,
        n_basrelief,
        feats["cnt_estimativa_area_iso"]
    ], dtype=np.float32)

    # Mediana dos estimadores positivos como referência de escala
    pos_vals = estimadores[estimadores > 0]
    med = float(np.median(pos_vals)) if len(pos_vals) > 0 else 1.0

    # Reduz peso de Hough se for outlier >3× mediana
    w_hough = 0.05 if n_hough_total > 3 * max(med, 1.0) else 0.20
    # Reduz peso de área isotropia se for outlier
    w_area = 0.05 if feats["cnt_estimativa_area_iso"] > 3 * max(med, 1.0) else 0.10

    ensemble_v71 = (
        w_hough * n_hough_total +
        0.20 * n_mser_circular +
        0.20 * n_watershed +
        0.20 * n_blob +
        0.15 * n_basrelief +
        w_area * feats["cnt_estimativa_area_iso"]
    )
    w_sum = w_hough + 0.20 + 0.20 + 0.20 + 0.15 + w_area
    ensemble_v71 = ensemble_v71 / w_sum

    feats["cnt_ensemble_v71"] = float(ensemble_v71)
    feats["cnt_ensemble_v71_log"] = float(np.log1p(ensemble_v71))
    feats["cnt_ensemble_v71_sqrt"] = float(np.sqrt(max(0, ensemble_v71)))

    # ── Ensemble v7 legado (preservado para compatibilidade) ───────────────────
    ensemble_legacy = (
        0.25 * n_hough_total +
        0.20 * n_mser_circular +
        0.20 * n_watershed +
        0.15 * n_blob +
        0.10 * n_basrelief +
        0.10 * feats["cnt_estimativa_area_iso"]
    )
    feats["cnt_ensemble_v7"] = float(ensemble_legacy)
    feats["cnt_ensemble_v7_log"] = float(np.log1p(ensemble_legacy))
    feats["cnt_ensemble_v7_sqrt"] = float(np.sqrt(max(0, ensemble_legacy)))

    # Razão densidade
    n_celulas_ativas = max(float((mascara > 0).mean() * 16), 1e-6)
    feats["cnt_density_per_cell"] = float(ensemble_v71 / n_celulas_ativas)

    return feats


# G15 — Curvatura Hessiana
def features_curvatura_hessiana(gray, mascara):
    feats = {}
    for sigma in [1.5, 3.0, 5.0]:
        gray_s = cv2.GaussianBlur(gray.astype(np.float64), (0, 0), sigma)

        Dxx = cv2.Sobel(gray_s, cv2.CV_64F, 2, 0, ksize=3)
        Dyy = cv2.Sobel(gray_s, cv2.CV_64F, 0, 2, ksize=3)
        Dxy = cv2.Sobel(gray_s, cv2.CV_64F, 1, 1, ksize=3)

        trace = Dxx + Dyy
        det = Dxx * Dyy - Dxy ** 2

        blobness = np.where(
            (det > 0) & (trace > 0),
            det / (trace ** 2 + 1e-9),
            0.0,
        ).astype(np.float32)
        blobness_smooth = cv2.GaussianBlur(blobness, (15, 15), 0)
        blob_u8 = cv2.normalize(blobness_smooth, None, 0, 255,
                                cv2.NORM_MINMAX, cv2.CV_8U)

        s = int(sigma)
        feats[f"hessian_s{s}_blob_mean"] = float(blobness_smooth.mean())
        feats[f"hessian_s{s}_blob_std"] = float(blobness_smooth.std())
        feats[f"hessian_s{s}_blob_p75"] = float(np.percentile(blobness_smooth, 75))
        feats[f"hessian_s{s}_blob_p90"] = float(np.percentile(blobness_smooth, 90))
        feats[f"hessian_s{s}_prop_convex"] = float((det > 0).mean())
        feats[f"hessian_s{s}_prop_blob_high"] = float((blobness_smooth > np.percentile(blobness_smooth, 75)).mean())
        feats.update(_first_order_mascara(blob_u8, mascara, f"hessian_s{s}_fruta"))

        if mascara.sum() > 0:
            conv_fruta = float(det[mascara > 0].mean())
            conv_fundo = float(det[mascara == 0].mean()) if (mascara == 0).sum() > 0 else conv_fruta
            feats[f"hessian_s{s}_conv_razao"] = float(conv_fruta / (abs(conv_fundo) + 1e-9))
        else:
            feats[f"hessian_s{s}_conv_razao"] = 0.0

    return feats


# G16 — Chromaticidade Cr-Cb
def features_chromaticidade_crcb(img_bgr, mascara):
    feats = {}
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Cr = ycrcb[:, :, 1].astype(np.float64)
    Cb = ycrcb[:, :, 2].astype(np.float64)
    Y = ycrcb[:, :, 0].astype(np.float64)

    feats.update(_first_order(Cr, "cr_global"))
    feats.update(_first_order(Cb, "cb_global"))

    diff_crcb = Cr - Cb
    feats.update(_first_order(diff_crcb + 128, "crcb_diff"))

    razao_crcb = Cr / (Cb + 1e-7)
    feats["crcb_razao_mean"] = float(np.clip(razao_crcb, -5, 5).mean())
    feats["crcb_razao_std"] = float(np.clip(razao_crcb, -5, 5).std())

    feats.update(_first_order_mascara(Cr.astype(np.uint8), mascara, "cr_fruta"))
    feats.update(_first_order_mascara(Cb.astype(np.uint8), mascara, "cb_fruta"))

    if mascara.sum() > 0:
        cr_fruta = float(Cr[mascara > 0].mean())
        cr_fundo = float(Cr[mascara == 0].mean()) if (mascara == 0).sum() > 0 else cr_fruta
        cb_fruta = float(Cb[mascara > 0].mean())
        cb_fundo = float(Cb[mascara == 0].mean()) if (mascara == 0).sum() > 0 else cb_fruta
    else:
        cr_fruta = cr_fundo = float(Cr.mean())
        cb_fruta = cb_fundo = float(Cb.mean())

    feats["cr_contraste_fruta_fundo"] = float(cr_fruta - cr_fundo)
    feats["cb_contraste_fruta_fundo"] = float(cb_fruta - cb_fundo)
    feats["crcb_dist_fruta_fundo"] = float(
        np.sqrt((cr_fruta - cr_fundo) ** 2 + (cb_fruta - cb_fundo) ** 2)
    )

    Cr_u8 = np.clip(Cr, 0, 255).astype(np.uint8)
    Cb_u8 = np.clip(Cb, 0, 255).astype(np.uint8)
    hist2d, _, _ = np.histogram2d(
        Cr_u8.flatten(), Cb_u8.flatten(),
        bins=8, range=[[0, 256], [0, 256]],
    )
    hist2d_norm = hist2d / (hist2d.sum() + 1e-9)
    for i in range(8):
        for j in range(8):
            feats[f"crcb_hist2d_{i}_{j}"] = float(hist2d_norm[i, j])

    R = img_bgr[:, :, 2].astype(np.float64) + 1e-7
    G = img_bgr[:, :, 1].astype(np.float64) + 1e-7
    B = img_bgr[:, :, 0].astype(np.float64) + 1e-7
    total = R + G + B + 1e-7
    r_norm, g_norm, b_norm = R / total, G / total, B / total
    ExG = 2 * g_norm - r_norm - b_norm
    feats["exg_mean"] = float(ExG.mean())
    feats["exg_std"] = float(ExG.std())
    feats["exg_p75"] = float(np.percentile(ExG, 75))
    if mascara.sum() > 0:
        feats["exg_fruta_mean"] = float(ExG[mascara > 0].mean())
        feats["exg_contraste"] = float(ExG[mascara == 0].mean() - ExG[mascara > 0].mean()
                                       if (mascara == 0).sum() > 0 else 0.0)
    else:
        feats["exg_fruta_mean"] = feats["exg_mean"]
        feats["exg_contraste"] = 0.0

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline completo por imagem
# ─────────────────────────────────────────────────────────────────────────────
def _extrair_de_img(img_bgr):

    # ============================================================
    # Pré-processamento base
    # ============================================================

    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    mascara = construir_mascara_fruta_verde(img_bgr)

    f = {}

    # ============================================================
    # FEATURES MAIS IMPORTANTES (manter e fortalecer)
    # ============================================================

    # ------------------------------------------------------------
    # HOG / estrutura
    # ------------------------------------------------------------

    f.update(features_hog(gray))

    # ------------------------------------------------------------
    # MULTIESCALA (MUITO IMPORTANTE)
    # ------------------------------------------------------------

    f.update(features_multiescala(img_bgr))

    # ------------------------------------------------------------
    # LBP (muito forte no SVR)
    # ------------------------------------------------------------

    f.update(features_lbp(gray))

    # ------------------------------------------------------------
    # GLCM
    # ------------------------------------------------------------

    f.update(features_glcm(gray))

    # ------------------------------------------------------------
    # Gabor isotropia
    # ------------------------------------------------------------

    f.update(features_gabor(gray))

    # ------------------------------------------------------------
    # Hessian / blobness
    # ------------------------------------------------------------

    f.update(features_curvatura_hessiana(gray, mascara))

    # ------------------------------------------------------------
    # Hough
    # ------------------------------------------------------------

    feats_hough = features_hough_circles(img_bgr, mascara)

    f.update(feats_hough)

    n_hough_total = int(
        feats_hough.get("hough_total_estimado", 0)
    )

    n_hough_mascara = int(
        feats_hough.get("hough_mascara_count", 0)
    )

    # ============================================================
    # FEATURES SECUNDÁRIAS (reduzidas)
    # ============================================================

    # ------------------------------------------------------------
    # HSV
    # (manter porque XGBoost gosta)
    # ------------------------------------------------------------

    f.update(features_hsv(img_bgr, hsv))

    # ------------------------------------------------------------
    # RGB/LAB/YCbCr
    # ------------------------------------------------------------

    f.update(features_rgb_lab(img_bgr))

    # ------------------------------------------------------------
    # Canal V
    # ------------------------------------------------------------

    feats_v, V_eq = features_canal_v_eq(hsv, mascara)

    f.update(feats_v)

    # ------------------------------------------------------------
    # Bas-relief
    # ------------------------------------------------------------

    f.update(features_basrelief(V_eq, mascara))

    # ============================================================
    # FEATURES QUE DEVEM CONTINUAR
    # ============================================================

    # ------------------------------------------------------------
    # Contagem direta
    # ------------------------------------------------------------

    n_mser_circular = int(
        f.get("geom_mser_circular", 0)
    )

    f.update(
        features_contagem_direta(
            img_bgr,
            mascara,
            n_hough_total,
            n_hough_mascara,
            n_mser_circular
        )
    )

    # ------------------------------------------------------------
    # Grade espacial
    # ------------------------------------------------------------

    f.update(features_grade_espacial(hsv, mascara))

    # ============================================================
    # FFT / frequência (NOVO)
    # ============================================================

    fft = np.fft.fft2(gray)

    fft_shift = np.fft.fftshift(fft)

    mag = np.log1p(np.abs(fft_shift))

    f["fft_mean"] = float(mag.mean())
    f["fft_std"] = float(mag.std())
    f["fft_energy"] = float(np.mean(mag ** 2))

    # baixa frequência central
    h, w = mag.shape

    cy, cx = h // 2, w // 2

    low = mag[
        cy - 20:cy + 20,
        cx - 20:cx + 20
    ]

    f["fft_lowfreq_mean"] = float(low.mean())

    # alta frequência periférica
    high = mag.copy()

    high[
        cy - 20:cy + 20,
        cx - 20:cx + 20
    ] = 0

    f["fft_highfreq_mean"] = float(
        high.mean()
    )

    # ============================================================
    # Laplacian / detalhe estrutural (NOVO)
    # ============================================================

    lap = cv2.Laplacian(gray, cv2.CV_32F)

    abs_lap = np.abs(lap)

    f["lap_mean"] = float(abs_lap.mean())
    f["lap_std"] = float(abs_lap.std())
    f["lap_p90"] = float(np.percentile(abs_lap, 90))

    # ============================================================
    # Spatial pyramid simples (NOVO)
    # ============================================================

    h, w = gray.shape

    idx = 0

    for y0, y1 in [(0, h // 2), (h // 2, h)]:
        for x0, x1 in [(0, w // 2), (w // 2, w)]:

            patch_gray = gray[y0:y1, x0:x1]

            patch_mask = mascara[y0:y1, x0:x1]

            prop = float(
                patch_mask.mean()
            ) / 255.0

            f[f"spatial_{idx}_prop"] = prop

            gx = cv2.Sobel(
                patch_gray,
                cv2.CV_32F,
                1,
                0
            )

            gy = cv2.Sobel(
                patch_gray,
                cv2.CV_32F,
                0,
                1
            )

            grad = np.sqrt(gx ** 2 + gy ** 2)

            f[f"spatial_{idx}_grad_mean"] = float(
                grad.mean()
            )

            f[f"spatial_{idx}_grad_std"] = float(
                grad.std()
            )

            idx += 1

    # ============================================================
    # Features globais de máscara
    # ============================================================

    f["mascara_prop_fruta"] = float(
        mascara.mean()
    ) / 255.0

    contours, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    f["mascara_n_blobs"] = float(
        len(contours)
    )

    # ============================================================
    # Densidade estrutural (NOVO)
    # ============================================================

    prop = f["mascara_prop_fruta"]

    blobs = max(
        f["mascara_n_blobs"],
        1e-6
    )

    f["density_blob_ratio"] = float(
        prop / blobs
    )

    # ============================================================
    # Relação HOG/frequência (NOVO)
    # ============================================================

    hog_mean = float(
        f.get("hog_mean", 0.0)
    )

    fft_mean = float(
        f.get("fft_mean", 0.0)
    )

    f["hog_fft_ratio"] = float(
        hog_mean / (fft_mean + 1e-6)
    )

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

            cnt = reg["contagem"]
            feats = _extrair_de_img(img)
            linha = {
                "image_id": reg["image_id"],
                "file_name": reg["file_name"],
                "split": nome_split,
                "contagem": cnt,
                "contagem_log1p": float(np.log1p(cnt)),
                "contagem_sqrt": float(np.sqrt(cnt)),
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
                            "contagem": cnt,
                            "contagem_log1p": float(np.log1p(cnt)),
                            "contagem_sqrt": float(np.sqrt(cnt)),
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
            print(f"\n  [erro] {reg.get('file_name', '?' )}: {e}")

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  {nome_split}: {i + 1}/{total} "
                  f"| amostras: {len(linhas)} | erros: {erros}", end="\r")

    print(f"\n  {nome_split}: {len(linhas)} amostras, {erros} erros.")
    return pd.DataFrame(linhas)


# ─────────────────────────────────────────────────────────────────────────────
# Seleção automática de features
# ─────────────────────────────────────────────────────────────────────────────
def selecionar_features(df_train, df_test, var_thr=VAR_THRESHOLD, corr_thr=CORR_THRESHOLD):
    cols = [c for c in df_train.columns if c not in COLUNAS_META]
    removidas = []

    variancias = df_train[cols].var()
    cols_var_baixa = variancias[variancias < var_thr].index.tolist()
    removidas.extend([(c, "variancia_zero") for c in cols_var_baixa])
    cols = [c for c in cols if c not in cols_var_baixa]
    print(f"  [seleção] Removidas {len(cols_var_baixa)} features com variância < {var_thr}")

    corr_matrix = df_train[cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    cols_alta_corr = [
        col for col in upper.columns
        if any(upper[col] > corr_thr)
    ]
    removidas.extend([(c, "correlacao_alta") for c in cols_alta_corr])
    cols = [c for c in cols if c not in cols_alta_corr]
    print(f"  [seleção] Removidas {len(cols_alta_corr)} features com correlação > {corr_thr}")
    print(f"  [seleção] Features finais: {len(cols)}")

    colunas_finais = COLUNAS_META + cols
    return df_train[colunas_finais], df_test[colunas_finais], removidas, cols


# ─────────────────────────────────────────────────────────────────────────────
# Normalização
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
        print(f"  [aviso] {len(constantes)} colunas constantes após normalização")

    print(f"  Treino norm: [{df_tn[cols].min().min():.4f}, {df_tn[cols].max().max():.4f}]")
    print(f"  Teste norm:  [{df_tt[cols].min().min():.4f}, {df_tt[cols].max().max():.4f}]")

    return df_tn, df_tt, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Metadados
# ─────────────────────────────────────────────────────────────────────────────
def gerar_info(df_train, df_test, removidas, n_features_final):
    cols = [c for c in df_train.columns if c not in COLUNAS_META]
    df_orig = df_train[df_train["augmentacao"] == "original"]

    grupos = {
        "G1_hsv": [c for c in cols if c.startswith("hsv_")],
        "G2_rgb_lab_ycbcr": [c for c in cols if c.startswith(("rgb_", "lab_", "ycbcr_"))],
        "G3_canal_v_eq": [c for c in cols if c.startswith(("v_original", "v_eq", "v_razao"))],
        "G4_basrelief": [c for c in cols if c.startswith(("sobel_", "laplace", "basrelief", "textura_", "brilho_"))],
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
        "G15_curvatura": [c for c in cols if c.startswith("hessian_")],
        "G16_chromaticidade": [c for c in cols if c.startswith(("cr_", "cb_", "crcb_", "exg_"))],
        "G_mascara": [c for c in cols if c.startswith("mascara_")],
    }

    return {
        "gerado_em": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "versao": "7.1",
        "img_size": IMG_SIZE,
        "n_features_bruto": len([c for c in df_train.columns if c not in COLUNAS_META]),
        "n_features_apos_selecao": n_features_final,
        "n_removidas_variancia": sum(1 for _, r in removidas if r == "variancia_zero"),
        "n_removidas_correlacao": sum(1 for _, r in removidas if r == "correlacao_alta"),
        "n_por_grupo": {k: len(v) for k, v in grupos.items()},
        "targets_disponiveis": {
            "contagem": "valor raw (int) — XGBoost count:poisson",
            "contagem_log1p": "log1p(contagem) — SVR/MLP",
            "contagem_sqrt": "sqrt(contagem) — alternativa",
        },
        "correcoes_v71": {
            "mascara": (
                "6 critérios (A-F), votação 2/6, fallback Gabor+Cr-Cb, "
                "filtro circularidade ≥0.40 + solidity ≥0.75"
            ),
            "watershed": "Kernel e threshold adaptativos ao tamanho da máscara",
            "hough_protecao": "Zera Hough se máscara < 1% da imagem",
            "area_iso_recalibrada": "Raio médio estimado da própria máscara, não fixo 30px",
            "ensemble_v71": "Pesos adaptativos: reduz Hough/Área se outliers >3× mediana",
        },
        "recomendacao_modelo": {
            "melhor_opcao": "XGBoost com objective='count:poisson' ou LightGBM",
            "config_xgboost": {
                "objective": "count:poisson",
                "eval_metric": "mae",
                "max_depth": 6,
                "n_estimators": 500,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
            },
        },
        "normalizacao": {
            "metodo": "MinMaxScaler sklearn",
            "range": "[0, 1]",
            "fit_em": "treino apenas",
        },
        "arquivos_gerados": {
            "orandet_v71_train_raw.csv": "Treino sem normalização",
            "orandet_v71_test_raw.csv": "Teste sem normalização",
            "orandet_v71_train_norm.csv": "Treino normalizado [0,1]",
            "orandet_v71_test_norm.csv": "Teste normalizado [0,1]",
        },
        "treino": {
            "n_originais": int(len(df_orig)),
            "n_total_com_aug": int(len(df_train)),
            "aug": ["flip_h", "flip_v", "bright_75", "bright_125"],
            "cnt_min": int(df_orig["contagem"].min()),
            "cnt_max": int(df_orig["contagem"].max()),
            "cnt_media": round(float(df_orig["contagem"].mean()), 2),
            "cnt_mediana": round(float(df_orig["contagem"].median()), 2),
            "n_zero": int((df_orig["contagem"] == 0).sum()),
        },
        "teste": {
            "n_imagens": int(len(df_test)),
            "cnt_min": int(df_test["contagem"].min()),
            "cnt_max": int(df_test["contagem"].max()),
            "cnt_media": round(float(df_test["contagem"].mean()), 2),
            "cnt_mediana": round(float(df_test["contagem"].median()), 2),
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

    print("\n" + "═" * 70)
    print("  OranDet v7.1 — Correções de Máscara + Ensemble Robustificado")
    print("  Grupos: G1-G16 | Máscara 2/6 + Fallback | Ensemble v7.1 adaptativo")
    print("  Saída: 4 datasets (train/test × raw/norm)")
    print("═" * 70)

    print("\n[1/6] Carregando anotações...")
    reg_train = carregar_anotacoes(ann_train, img_dir)
    reg_test = carregar_anotacoes(ann_test, img_dir)
    print(f"  Treino: {len(reg_train)} | Teste: {len(reg_test)}")

    print("\n[2/6] Extraindo features (treino)...")
    df_train = processar_split(reg_train, "train", aplicar_augmentacao=False)

    print("\n[3/6] Extraindo features (teste)...")
    df_test = processar_split(reg_test, "test", aplicar_augmentacao=False)

    print("\n[4/6] Seleção automática de features...")
    df_train_sel, df_test_sel, removidas, cols_final = selecionar_features(df_train, df_test)

    print("\n[5/6] Salvando datasets RAW...")
    df_train_sel.to_csv(os.path.join(OUTPUT_DIR, "orandet_v71_train_raw.csv"), index=False)
    df_test_sel.to_csv(os.path.join(OUTPUT_DIR, "orandet_v71_test_raw.csv"), index=False)
    print(f"  Salvo: orandet_v71_train_raw.csv ({len(df_train_sel)} × {len(df_train_sel.columns)})")
    print(f"  Salvo: orandet_v71_test_raw.csv  ({len(df_test_sel)} × {len(df_test_sel.columns)})")

    print("\n[6/6] Normalizando [0,1] e salvando...")
    df_train_norm, df_test_norm, scaler = normalizar(df_train_sel, df_test_sel)
    df_train_norm.to_csv(os.path.join(OUTPUT_DIR, "orandet_v71_train_norm.csv"), index=False)
    df_test_norm.to_csv(os.path.join(OUTPUT_DIR, "orandet_v71_test_norm.csv"), index=False)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "orandet_v71_scaler.joblib"))
    print(f"  Salvo: orandet_v71_train_norm.csv ({len(df_train_norm)} × {len(df_train_norm.columns)})")
    print(f"  Salvo: orandet_v71_test_norm.csv  ({len(df_test_norm)} × {len(df_test_norm.columns)})")

    print("\n  Salvando metadados...")
    info = gerar_info(df_train_sel, df_test_sel, removidas, len(cols_final))
    with open(os.path.join(OUTPUT_DIR, "orandet_v71_info.json"), "w", encoding="utf-8") as fj:
        json.dump(info, fj, indent=2, ensure_ascii=False)
    joblib.dump(cols_final, os.path.join(OUTPUT_DIR, "orandet_v71_feature_cols.joblib"))

    print(f"\n{'═' * 70}")
    print(f"  Features: {info['n_features_bruto']} brutas → {info['n_features_apos_selecao']} após seleção")
    print(f"  Removidas: {info['n_removidas_variancia']} (var zero) + {info['n_removidas_correlacao']} (alta correlação)")
    print(f"\n  Por grupo (após seleção):")
    for grupo, n in info["n_por_grupo"].items():
        if n > 0:
            print(f"    {grupo:<28} {n:>4} features")
    print(f"\n  Treino: {info['treino']['n_originais']} imgs → {info['treino']['n_total_com_aug']} amostras (aug)")
    print(f"  Teste:  {info['teste']['n_imagens']} imgs | média {info['teste']['cnt_media']:.1f} laranjas/img")
    print(f"\n  Datasets gerados em: {OUTPUT_DIR}/")
    print(f"    ├─ orandet_v71_train_raw.csv   ← XGBoost/LightGBM")
    print(f"    ├─ orandet_v71_test_raw.csv    ← XGBoost/LightGBM")
    print(f"    ├─ orandet_v71_train_norm.csv  ← SVR/MLP")
    print(f"    └─ orandet_v71_test_norm.csv   ← SVR/MLP")
    print(f"\n  ⚠  Target recomendado:")
    print(f"     XGBoost/LightGBM → 'contagem' + objective='count:poisson'")
    print(f"     SVR/MLP          → 'contagem_log1p' (aplicar exp-1 na saída)")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()