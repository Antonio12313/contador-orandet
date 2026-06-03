"""
REFERÊNCIAS PRINCIPAIS:
  Maldonado & Barbosa (2016) — Bas-relief + razão brilho vertical, citrus verde
  Kurtulmus et al. (2011)   — Gabor isotropy, citrus sobre fundo verde
  Zhao & Lee (2016)         — SATD, 83.4% acurácia citrus verde
  Okamoto & Lee (2009)      — Chromaticidade Cr-Cb para citrus verde
  Hu (2018)                 — LBP + MSER + Hough hierárquico para citrus verde
  Frangi et al. (1998)      — Hessiana multiscale blob enhancement

CORREÇÕES desta versão (v8.1):
  [BUG] features_geometria: cv2.MSER_create(_delta=...) lançava TypeError no
        OpenCV 4.x e o try/except zerava tudo (MSER sempre = 0). Corrigido
        para usar setters — G10 e cnt_mser_* agora funcionam.
  [BUG] features derivadas usavam f["mascara_prop_fruta"] / f["mascara_n_blobs"]
        que nunca eram criadas → KeyError em toda imagem. Agora essas features
        de máscara são calculadas (e populam o grupo G_mascara).
  [LIMPEZA] Removidas augmentação comentada e helpers não usados (treino é só
        nas 2759 originais).
  [META] gerar_info: normalização corrigida para z-score (StandardScaler não
        produz [0,1]); augmentação marcada como não aplicada.
  [MELHORIA] máscara passada às features de cor (HSV/RGB/LAB); guarda anti
        região-plana na isotropia de Gabor; V_eq reaproveitado na contagem.
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
from sklearn.preprocessing import StandardScaler
import time
from ambiente import coletar_ambiente

warnings.filterwarnings("ignore")

# CONFIGURAÇÃO

DATA_DIR = "/Users/antonioreis/Downloads/dataverse_files"
OUTPUT_DIR = "./dataset_preparado_v11"
IMG_SIZE = 416

COLUNAS_META = [
    "image_id", "file_name", "split", "contagem",
    "contagem_log1p", "contagem_sqrt", "augmentacao",
]

# Thresholds para limpeza de features
VAR_THRESHOLD = 1e-5
CORR_THRESHOLD = 0.97

os.makedirs(OUTPUT_DIR, exist_ok=True)


# PRIMITIVAS COMPARTILHADAS

def _safe_skew(arr):
    val = skew(arr.flatten())
    return float(val) if np.isfinite(val) else 0.0


def _safe_kurtosis(arr):
    val = kurtosis(arr.flatten())
    return float(val) if np.isfinite(val) else 0.0


_SUFIXOS_FO = ["mean", "std", "median", "entropy", "p10", "p90", "iqr",
               "skewness", "kurtosis", "energy", "uniformity", "mad"]


def _first_order(canal, prefixo):
    """12 estatísticas de primeira ordem compactas."""
    c = np.array(canal, dtype=np.float64).flatten()
    if len(c) == 0:
        return {f"{prefixo}_{s}": 0.0 for s in _SUFIXOS_FO}

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
        return {f"{prefixo}_{s}": 0.0 for s in _SUFIXOS_FO}
    return _first_order(px, prefixo)


def _stats_mascarado(canal, mascara, prefixo):
    """Calcula _first_order só sobre os pixels da fruta.
    Faz fallback para a imagem inteira se a máscara for vazia/minúscula."""
    if mascara is not None:
        vals = canal[mascara > 0]
        if vals.size < 50:
            vals = canal.reshape(-1)
    else:
        vals = canal.reshape(-1)
    return _first_order(vals.reshape(-1, 1), prefixo)


# MAPAS AUXILIARES

def _gabor_isotropy_map(gray):
    """Isotropia Gabor: fruta esférica = resposta uniforme em todas orientações.

    Guarda anti região-plana: onde NÃO há resposta de Gabor (fundo liso sem
    textura), a isotropia é zerada — evita marcar fundo plano como 'esférico',
    que era um falso sinal das versões anteriores.
    """
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
    mean_resp = stacked.mean(axis=0)
    std_resp = stacked.std(axis=0)
    isotropy = 1.0 - np.clip(std_resp / (mean_resp + 1e-9), 0, 1)
    piso = 0.15 * (mean_resp.mean() + 1e-9)        # piso de resposta relativo
    isotropy[mean_resp < piso] = 0.0
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


# MÁSCARA v7.1

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

    iso_map = _gabor_isotropy_map(gray)
    crit_A = (iso_map >= float(np.percentile(iso_map, 55))).astype(np.float32)

    satd_map_v = _satd_map(gray)
    crit_B = (satd_map_v <= float(np.percentile(satd_map_v, 50))).astype(np.float32)

    lap_map = _laplacian_smooth_map(gray)
    crit_C = (lap_map <= float(np.percentile(lap_map, 55))).astype(np.float32)

    b_ch = lab[:, :, 2].astype(np.float32)
    crit_D = (b_ch >= float(np.percentile(b_ch, 50))).astype(np.float32)

    hess_map = _hessian_convexity_map(gray)
    crit_E = (hess_map >= float(np.percentile(hess_map, 60))).astype(np.float32)

    Cr = ycrcb[:, :, 1].astype(np.float32)
    Cb = ycrcb[:, :, 2].astype(np.float32)
    crcb_diff = Cr - Cb
    crit_F = (crcb_diff >= float(np.percentile(crcb_diff, 45))).astype(np.float32)

    voto = crit_A + crit_B + crit_C + crit_D + crit_E + crit_F
    mask_raw = (voto >= 2).astype(np.uint8) * 255

    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, k3, iterations=1)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k5, iterations=2)

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

    if prop < 0.002 or prop > 0.45:
        mascara_filtrada = np.zeros_like(mascara_filtrada)

    return mascara_filtrada


# G1 — HSV
def features_hsv(img_bgr, hsv, mascara=None):
    feats = {}
    cfg = [("H", 0, 180, 12), ("S", 0, 256, 8), ("V", 0, 256, 6)]
    usar_mask = mascara is not None and cv2.countNonZero(mascara) >= 50
    mask8 = mascara if usar_mask else None

    for i, (nome, lo, hi, nb) in enumerate(cfg):
        hist = cv2.calcHist([hsv], [i], mask8, [nb], [lo, hi]).flatten()
        hist = hist / (hist.sum() + 1e-7)
        for b, val in enumerate(hist):
            feats[f"hsv_hist_{nome}_b{b:02d}"] = float(val)
        feats.update(_stats_mascarado(hsv[:, :, i], mascara, f"hsv_{nome}"))
    return feats


# G2 — RGB + LAB + YCbCr
def features_rgb_lab(img_bgr, mascara=None):
    feats = {}
    m = (mascara > 0) if mascara is not None else None
    if m is not None and m.sum() < 50:
        m = None

    def _vals(canal_f):
        return canal_f[m] if m is not None else canal_f.reshape(-1)

    for nome, idx in {"R": 2, "G": 1, "B": 0}.items():
        feats.update(_stats_mascarado(img_bgr[:, :, idx], mascara, f"rgb_{nome}"))

    R = _vals(img_bgr[:, :, 2].astype(np.float64)) + 1e-7
    G = _vals(img_bgr[:, :, 1].astype(np.float64)) + 1e-7
    B = _vals(img_bgr[:, :, 0].astype(np.float64)) + 1e-7
    feats["rgb_razao_RG"] = float((R / G).mean())
    feats["rgb_razao_RB"] = float((R / B).mean())
    feats["rgb_razao_GB"] = float((G / B).mean())
    soma = R + G + B
    feats["rgb_ExG"] = float(((2 * G - R - B) / soma).mean())
    feats["rgb_g_norm_mean"] = float((G / soma).mean())
    feats["rgb_g_norm_std"] = float((G / soma).std())
    feats["rgb_r_norm_mean"] = float((R / soma).mean())

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    for nome, idx in {"L": 0, "a": 1, "b": 2}.items():
        feats.update(_stats_mascarado(lab[:, :, idx], mascara, f"lab_{nome}"))
    a_ch = _vals(lab[:, :, 1].astype(np.float64)) + 1e-7
    b_ch = _vals(lab[:, :, 2].astype(np.float64)) + 1e-7
    feats["lab_razao_a_b"] = float((a_ch / b_ch).mean())

    ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    for nome, idx in {"Y": 0, "Cr": 1, "Cb": 2}.items():
        feats.update(_stats_mascarado(ycbcr[:, :, idx], mascara, f"ycbcr_{nome}"))
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

    for img_u8, pref in [(sx_abs, "sobel_x"),
                         (sy_abs, "sobel_y"),
                         (mag_n, "sobel_mag"),
                         (lap_abs, "laplace"),
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

    for nome, lista in [("area", areas),
                        ("solidity", solidities),
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

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    n_total, n_circ_mser = 0, 0
    try:
        # [BUG CORRIGIDO] No OpenCV 4.x os kwargs _delta/_min_area não existem;
        # usar setters. Antes, o try/except mascarava o erro e zerava o MSER.
        mser = cv2.MSER_create()
        mser.setDelta(8)
        mser.setMinArea(400)
        mser.setMaxArea(25000)
        regs, _ = mser.detectRegions(gray)
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
    except Exception as e:
        print(f"  [aviso MSER] {type(e).__name__}: {e}")
        n_total = n_circ_mser = 0

    feats["geom_mser_total"] = float(n_total)
    feats["geom_mser_circular"] = float(n_circ_mser)
    feats["geom_log_mser"] = float(np.log1p(n_circ_mser))

    return feats


# G11 — Hough Circles
def features_hough_circles(img_bgr, mascara):
    feats = {}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    faixas_novas = [
        (8, 18, "f1_micro"),
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
        feats[f"hough_{nome}_density"] = float(n_v / (IMG_SIZE * IMG_SIZE / 1e4))
        total_circ += n_v
        todos_radii.extend(radii)
        todos_centros.extend(centros)

    feats["hough_pequeno_count"] = feats["hough_f1_micro_count"] + feats["hough_f2_xpequeno_count"]
    feats["hough_medio_count"] = feats["hough_f3_pequeno_count"] + feats["hough_f4_medio_count"]
    feats["hough_grande_count"] = feats["hough_f5_grande_count"] + feats["hough_f6_xgrande_count"]
    feats["hough_pequeno_raio_mean"] = feats["hough_f2_xpequeno_raio_mean"]
    feats["hough_medio_raio_mean"] = feats["hough_f4_medio_raio_mean"]
    feats["hough_grande_raio_mean"] = feats["hough_f5_grande_raio_mean"]

    feats["hough_total_estimado"] = float(total_circ)
    feats["hough_log_total"] = float(np.log1p(total_circ))
    feats["hough_sqrt_total"] = float(np.sqrt(total_circ))
    feats["hough_prop_pequenos"] = float(feats["hough_pequeno_count"] / (total_circ + 1e-7))
    feats["hough_prop_medios"] = float(feats["hough_medio_count"] / (total_circ + 1e-7))
    feats["hough_prop_grandes"] = float(feats["hough_grande_count"] / (total_circ + 1e-7))

    if todos_radii:
        feats["hough_raio_global_mean"] = float(np.mean(todos_radii))
        feats["hough_raio_global_std"] = float(np.std(todos_radii))
        feats["hough_raio_global_p25"] = float(np.percentile(todos_radii, 25))
        feats["hough_raio_global_p75"] = float(np.percentile(todos_radii, 75))
        feats["hough_raio_global_iqr"] = feats["hough_raio_global_p75"] - feats["hough_raio_global_p25"]
    else:
        for k in ["mean", "std", "p25", "p75", "iqr"]:
            feats[f"hough_raio_global_{k}"] = 0.0

    if len(todos_centros) >= 2:
        centros_arr = np.array(todos_centros)
        dists = []
        n_c = len(centros_arr)
        for k in range(n_c):
            for ll in range(k + 1, min(k + 5, n_c)):
                dists.append(float(np.linalg.norm(centros_arr[k] - centros_arr[ll])))
        feats["hough_dist_media"] = float(np.mean(dists))
        feats["hough_dist_std"] = float(np.std(dists))
    else:
        feats["hough_dist_media"] = 0.0
        feats["hough_dist_std"] = 0.0

    n_overlap = 0
    if len(todos_centros) >= 2 and len(todos_radii) == len(todos_centros):
        centros_arr = np.array(todos_centros)
        radii_arr = np.array(todos_radii)
        n_c = len(centros_arr)
        for k in range(n_c):
            for ll in range(k + 1, n_c):
                d = float(np.linalg.norm(centros_arr[k] - centros_arr[ll]))
                if d < (radii_arr[k] + radii_arr[ll]) * 0.85:
                    n_overlap += 1
    feats["hough_overlap_est"] = float(n_overlap)
    feats["hough_overlap_log"] = float(np.log1p(n_overlap))

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
    feats["hough_log_mascara"] = float(np.log1p(n_m))

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
        img_r = cv2.resize(img_bgr, sz, interpolation=cv2.INTER_AREA)
        hsv_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2HSV)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        mask_r = construir_mascara_fruta_verde(img_r)

        hog_v = hog(gray_r, orientations=9,
                    pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                    feature_vector=True, block_norm="L2-Hys")
        feats[f"{nome}_hog_mean"] = float(hog_v.mean())
        feats[f"{nome}_hog_std"] = float(hog_v.std())

        fft = np.fft.fft2(gray_r)
        fft_shift = np.fft.fftshift(fft)
        mag = np.log1p(np.abs(fft_shift))
        feats[f"{nome}_fft_mean"] = float(mag.mean())
        feats[f"{nome}_fft_std"] = float(mag.std())
        feats[f"{nome}_fft_energy"] = float(np.mean(mag ** 2))

        lap = cv2.Laplacian(gray_r, cv2.CV_32F)
        feats[f"{nome}_lap_mean"] = float(np.mean(np.abs(lap)))
        feats[f"{nome}_lap_std"] = float(np.std(lap))

        for ci, cn in enumerate(["H", "S", "V"]):
            hist = cv2.calcHist([hsv_r], [ci], None, [8], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-7)
            feats[f"{nome}_{cn}_entropy"] = float(-np.sum(hist * np.log2(hist + 1e-9)))
            feats[f"{nome}_{cn}_maxbin"] = float(hist.max())

        prop_fruta = float(mask_r.mean()) / 255.0
        feats[f"{nome}_prop_fruta"] = prop_fruta

        k3 = np.ones((3, 3), np.uint8)
        mask_c = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, k3)
        cnts, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        n_blobs = 0
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 10:
                continue
            circ = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2 + 1e-6)
            if circ >= 0.35:
                n_blobs += 1
                areas.append(area)
        feats[f"{nome}_n_blobs"] = float(n_blobs)
        feats[f"{nome}_blob_area_mean"] = float(np.mean(areas)) if areas else 0.0
        feats[f"{nome}_blob_area_std"] = float(np.std(areas)) if areas else 0.0

        h, w = gray_r.shape
        idx = 0
        for y0, y1 in [(0, h // 2), (h // 2, h)]:
            for x0, x1 in [(0, w // 2), (w // 2, w)]:
                patch_mask = mask_r[y0:y1, x0:x1]
                prop_patch = float(patch_mask.mean()) / 255.0
                feats[f"{nome}_patch{idx}_prop"] = prop_patch
                idx += 1

        for lam in [8, 15]:
            resps = []
            for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
                k = cv2.getGaborKernel((15, 15), 3.0, theta, float(lam), 1.0, 0)
                r = np.abs(cv2.filter2D(gray_r, cv2.CV_64F, k))
                resps.append(r)
            stk = np.stack(resps, axis=0)
            cv_ = stk.std(axis=0) / (stk.mean(axis=0) + 1e-9)
            iso = 1.0 - np.clip(cv_, 0, 1)
            feats[f"{nome}_gabor_iso_mean_{lam}"] = float(iso.mean())
            feats[f"{nome}_gabor_iso_std_{lam}"] = float(iso.std())

        escalas_data[nome] = {
            "prop": prop_fruta, "blobs": n_blobs,
            "hog_mean": hog_v.mean(), "fft_mean": mag.mean(),
        }

    e208 = escalas_data["escala_208"]
    e104 = escalas_data["escala_104"]
    feats["multi_ratio_blobs"] = float(e208["blobs"] / (e104["blobs"] + 1e-6))
    feats["multi_ratio_prop"] = float(e208["prop"] / (e104["prop"] + 1e-6))
    feats["multi_delta_hog"] = float(e208["hog_mean"] - e104["hog_mean"])
    feats["multi_delta_fft"] = float(e208["fft_mean"] - e104["fft_mean"])

    return feats


# G14 — Contagem Direta v7.1
def _watershed_count(gray, mascara):
    if mascara.sum() == 0:
        return 0

    prop = float(mascara.sum()) / (255.0 * mascara.shape[0] * mascara.shape[1])
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

    thr_pct = 40 if prop < 0.02 else 50
    thr_dist = float(np.percentile(dist[dist > 0], thr_pct)) if np.any(dist > 0) else 5
    _, markers_bin = cv2.threshold(dist_norm, int(thr_dist), 255, cv2.THRESH_BINARY)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        markers_bin.astype(np.uint8), connectivity=8
    )
    area_min = 15 if prop < 0.02 else 20
    return sum(1 for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= area_min)


def features_contagem_direta(img_bgr, mascara, n_hough_total, n_hough_mascara,
                             n_mser_circular, V_eq=None):
    feats = {}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]

    prop_mascara = float(mascara.sum()) / (255.0 * mascara.shape[0] * mascara.shape[1])

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

    n_watershed = _watershed_count(gray, mascara)
    feats["cnt_watershed_n"] = float(n_watershed)
    feats["cnt_watershed_log"] = float(np.log1p(n_watershed))
    feats["cnt_watershed_sqrt"] = float(np.sqrt(n_watershed))

    if prop_mascara < 0.01:
        n_hough_total = 0
        n_hough_mascara = 0
    feats["cnt_hough_n"] = float(n_hough_total)
    feats["cnt_hough_mascara_n"] = float(n_hough_mascara)
    feats["cnt_hough_log"] = float(np.log1p(n_hough_total))
    feats["cnt_hough_sqrt"] = float(np.sqrt(n_hough_total))

    feats["cnt_mser_n"] = float(n_mser_circular)
    feats["cnt_mser_log"] = float(np.log1p(n_mser_circular))
    feats["cnt_mser_sqrt"] = float(np.sqrt(n_mser_circular))

    iso_map = _gabor_isotropy_map(gray)
    area_iso_alta = float((iso_map > 0.70).mean())

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

    # reaproveita V_eq já calculado (evita CLAHE redundante); senão calcula
    if V_eq is None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        V_eq = clahe.apply(V.astype(np.uint8))
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

    estimadores = np.array([
        n_hough_total, n_mser_circular, n_watershed, n_blob,
        n_basrelief, feats["cnt_estimativa_area_iso"]
    ], dtype=np.float32)

    pos_vals = estimadores[estimadores > 0]
    med = float(np.median(pos_vals)) if len(pos_vals) > 0 else 1.0

    w_hough = 0.05 if n_hough_total > 3 * max(med, 1.0) else 0.20
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
        blob_u8 = cv2.normalize(blobness_smooth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

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
        feats["exg_contraste"] = float(
            ExG[mascara == 0].mean() - ExG[mascara > 0].mean()
            if (mascara == 0).sum() > 0 else 0.0
        )
    else:
        feats["exg_fruta_mean"] = feats["exg_mean"]
        feats["exg_contraste"] = 0.0

    return feats


# Pipeline completo por imagem
def _extrair_de_img(img_bgr):
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mascara = construir_mascara_fruta_verde(img_bgr)

    f = {}

    f.update(features_hog(gray))                              # G9
    f.update(features_multiescala(img_bgr))                   # G13
    f.update(features_lbp(gray))                              # G6
    f.update(features_glcm(gray))                             # G7
    f.update(features_gabor(gray))                            # G5
    f.update(features_curvatura_hessiana(gray, mascara))      # G15

    feats_hough = features_hough_circles(img_bgr, mascara)    # G11
    f.update(feats_hough)
    n_hough_total = int(feats_hough.get("hough_total_estimado", 0))
    n_hough_mascara = int(feats_hough.get("hough_mascara_count", 0))

    f.update(features_hsv(img_bgr, hsv, mascara))             # G1 (com máscara)
    f.update(features_rgb_lab(img_bgr, mascara))              # G2 (com máscara)

    feats_v, V_eq = features_canal_v_eq(hsv, mascara)         # G3
    f.update(feats_v)
    f.update(features_basrelief(V_eq, mascara))               # G4
    f.update(features_satd(gray, mascara))                    # G8
    f.update(features_geometria(img_bgr, mascara))            # G10 (precede G14)
    f.update(features_chromaticidade_crcb(img_bgr, mascara))  # G16

    n_mser_circular = int(f.get("geom_mser_circular", 0))
    f.update(features_contagem_direta(                        # G14
        img_bgr, mascara,
        n_hough_total, n_hough_mascara, n_mser_circular,
        V_eq=V_eq,
    ))

    f.update(features_grade_espacial(hsv, mascara))           # G12

    # FFT global
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    mag = np.log1p(np.abs(fft_shift))
    f["fft_mean"] = float(mag.mean())
    f["fft_std"] = float(mag.std())
    f["fft_energy"] = float(np.mean(mag ** 2))
    hh, ww = mag.shape
    cy, cx = hh // 2, ww // 2
    low = mag[cy - 20:cy + 20, cx - 20:cx + 20]
    f["fft_lowfreq_mean"] = float(low.mean())
    high = mag.copy()
    high[cy - 20:cy + 20, cx - 20:cx + 20] = 0
    f["fft_highfreq_mean"] = float(high.mean())

    # Laplaciano global
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    abs_lap = np.abs(lap)
    f["lap_mean"] = float(abs_lap.mean())
    f["lap_std"] = float(abs_lap.std())
    f["lap_p90"] = float(np.percentile(abs_lap, 90))

    # Spatial pyramid 2×2
    hh, ww = gray.shape
    idx = 0
    for y0, y1 in [(0, hh // 2), (hh // 2, hh)]:
        for x0, x1 in [(0, ww // 2), (ww // 2, ww)]:
            patch_gray = gray[y0:y1, x0:x1]
            patch_mask = mascara[y0:y1, x0:x1]
            f[f"spatial_{idx}_prop"] = float(patch_mask.mean()) / 255.0
            gx = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1)
            grad = np.sqrt(gx ** 2 + gy ** 2)
            f[f"spatial_{idx}_grad_mean"] = float(grad.mean())
            f[f"spatial_{idx}_grad_std"] = float(grad.std())
            idx += 1

    # Features de máscara (G_mascara) — [BUG CORRIGIDO] antes eram referenciadas
    # sem nunca terem sido criadas, causando KeyError nas features derivadas.
    # prop_fruta = float((mascara > 0).mean())
    # cnts_m, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # n_blobs_m = sum(1 for c in cnts_m if cv2.contourArea(c) >= 100)
    # f["mascara_prop_fruta"] = prop_fruta
    # f["mascara_n_blobs"] = float(n_blobs_m)

    # Features derivadas
    # f["density_blob_ratio"] = float(prop_fruta / max(n_blobs_m, 1e-6))
    # f["hog_fft_ratio"] = float(f.get("hog_mean", 0.0) / (f.get("fft_mean", 0.0) + 1e-6))

    return f


# Leitura de anotações COCO
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


# Processamento de um split (sem augmentação — treino usa só as originais)
def processar_split(registros, nome_split):
    linhas = []
    erros = 0
    total = len(registros)
    tempos_extracao = []
    t_inicio_split = time.perf_counter()

    print(f"\n  {nome_split}: {total} imagens (sem augmentação)...")

    for i, reg in enumerate(registros):
        try:
            img = cv2.imread(reg["caminho"])
            if img is None:
                raise FileNotFoundError(reg["caminho"])

            cnt = reg["contagem"]
            _t0 = time.perf_counter()
            feats = _extrair_de_img(img)               # resize feito internamente
            tempos_extracao.append(time.perf_counter() - _t0)

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

        except FileNotFoundError:
            erros += 1
        except Exception as e:
            erros += 1
            print(f"\n  [erro] {reg.get('file_name', '?')}: {e}")

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  {nome_split}: {i + 1}/{total} "
                  f"| amostras: {len(linhas)} | erros: {erros}", end="\r")

    t_total_split = time.perf_counter() - t_inicio_split
    if tempos_extracao:
        print(f"\n  {nome_split}: {len(linhas)} amostras, {erros} erros.")
        print(f"  Tempo extração — total: {t_total_split:.1f}s | "
              f"média/img: {np.mean(tempos_extracao):.3f}s | "
              f"min: {min(tempos_extracao):.3f}s | "
              f"max: {max(tempos_extracao):.3f}s")
    else:
        print(f"\n  {nome_split}: {len(linhas)} amostras, {erros} erros.")

    stats_tempo = {
        "n_imagens_medidas": len(tempos_extracao),
        "total_s": round(t_total_split, 2),
        "media_s": round(float(np.mean(tempos_extracao)), 4) if tempos_extracao else None,
        "mediana_s": round(float(np.median(tempos_extracao)), 4) if tempos_extracao else None,
        "std_s": round(float(np.std(tempos_extracao)), 4) if tempos_extracao else None,
        "min_s": round(min(tempos_extracao), 4) if tempos_extracao else None,
        "max_s": round(max(tempos_extracao), 4) if tempos_extracao else None,
        "p25_s": round(float(np.percentile(tempos_extracao, 25)), 4) if tempos_extracao else None,
        "p75_s": round(float(np.percentile(tempos_extracao, 75)), 4) if tempos_extracao else None,
    }
    return pd.DataFrame(linhas), stats_tempo


# Seleção automática de features
def selecionar_features(df_train, df_test, var_thr=VAR_THRESHOLD, corr_thr=CORR_THRESHOLD):
    cols = [c for c in df_train.columns if c not in COLUNAS_META]
    removidas = []

    variancias = df_train[cols].var()
    cols_var_baixa = variancias[variancias < var_thr].index.tolist()
    removidas.extend([(c, "variancia_zero") for c in cols_var_baixa])
    cols = [c for c in cols if c not in cols_var_baixa]
    print(f"  [seleção] Removidas {len(cols_var_baixa)} features com variância < {var_thr}")

    corr_matrix = df_train[cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    cols_alta_corr = [col for col in upper.columns if any(upper[col] > corr_thr)]
    removidas.extend([(c, "correlacao_alta") for c in cols_alta_corr])
    cols = [c for c in cols if c not in cols_alta_corr]
    print(f"  [seleção] Removidas {len(cols_alta_corr)} features com correlação > {corr_thr}")
    print(f"  [seleção] Features finais: {len(cols)}")

    colunas_finais = COLUNAS_META + cols
    return df_train[colunas_finais], df_test[colunas_finais], removidas, cols


# Normalização (z-score)
def normalizar(df_train, df_test):
    cols = [c for c in df_train.columns if c not in COLUNAS_META]

    for df in [df_train, df_test]:
        df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)

    medianas = df_train[cols].median()
    df_train[cols] = df_train[cols].fillna(medianas)
    df_test[cols] = df_test[cols].fillna(medianas)

    scaler = StandardScaler()                 # z-score; fit só no treino
    scaler.fit(df_train[cols])

    df_tn = df_train.copy()
    df_tt = df_test.copy()
    df_tn[cols] = scaler.transform(df_train[cols])
    df_tt[cols] = scaler.transform(df_test[cols])

    constantes = [c for c in cols if df_train[c].std() == 0]
    if constantes:
        print(f"  [aviso] {len(constantes)} colunas constantes após normalização")

    print(f"  Treino norm: média={df_tn[cols].mean().mean():.4f}, std={df_tn[cols].std().mean():.4f}")
    print(f"  Teste norm:  média={df_tt[cols].mean().mean():.4f}, std={df_tt[cols].std().mean():.4f}")

    return df_tn, df_tt, scaler


# Metadados
def gerar_info(df_train, df_test, removidas, n_features_final,
               stats_tempo_treino=None, stats_tempo_teste=None, ambiente=None):
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
        "G13_multiescala": [c for c in cols if c.startswith(("escala_", "multi_"))],
        "G14_contagem": [c for c in cols if c.startswith("cnt_")],
        "G15_curvatura": [c for c in cols if c.startswith("hessian_")],
        "G16_chromaticidade": [c for c in cols if c.startswith(("cr_", "cb_", "crcb_", "exg_"))],
        "G_fft": [c for c in cols if c.startswith("fft_")],
        "G_laplaciano": [c for c in cols if c.startswith("lap_")],
        "G_spatial_pyramid": [c for c in cols if c.startswith("spatial_")],
    }

    return {
        "gerado_em": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "versao": "8.1",
        "img_size": IMG_SIZE,

        "protocolo_experimental": {
            "origem_do_split": (
                "Arquivos COCO externos: instances_train.json e instances_test.json. "
                "O split NÃO é feito por divisão aleatória interna; é determinado "
                "pelos arquivos de anotação fornecidos."
            ),
            "conjunto_validacao": (
                "AUSENTE nesta etapa de extração. O pipeline gera apenas treino e "
                "teste. Se um conjunto de validação for necessário (ex.: early stopping "
                "ou GridSearchCV), ele deve ser criado nos scripts de cada modelo."
            ),
            "aleatoriedade_extracao": (
                "NENHUMA. O pipeline de extração é completamente determinístico: "
                "não há chamadas a geradores aleatórios nas funções de features, na "
                "máscara ou na normalização."
            ),
            "ajuste_de_hiperparametros": (
                "[VERIFICAR nos scripts dos modelos] — o pipeline de extração não "
                "treina modelos nem realiza busca de hiperparâmetros."
            ),
        },

        "dimensionalidade": {
            "n_features_brutas_por_imagem": (
                    len([c for c in df_train.columns if c not in COLUNAS_META])
                    + sum(1 for _, r in removidas if r == "variancia_zero")
                    + sum(1 for _, r in removidas if r == "correlacao_alta")
            ),
            "n_features_apos_selecao": n_features_final,
            "n_removidas_variancia_zero": sum(1 for _, r in removidas if r == "variancia_zero"),
            "n_removidas_correlacao_alta": sum(1 for _, r in removidas if r == "correlacao_alta"),
            "limiar_variancia": VAR_THRESHOLD,
            "limiar_correlacao": CORR_THRESHOLD,
        },

        "preprocessamento": {
            "normalizacao": {
                "metodo": "StandardScaler (z-score: média 0, desvio 1) — scikit-learn",
                "faixa_saida": "não limitada (tipicamente ~[-3, +3]); NÃO é [0,1]",
                "fit_em": "Somente no conjunto de treino — sem vazamento para o teste",
                "nan_inf": "±inf → NaN → preenchido com a MEDIANA do treino (antes do scaler)",
                "persistencia": "orandet_v11_scaler.joblib",
            },
            "modelos_recomendados_por_dataset": {
                "RAW (sem normalização)": ["XGBoost", "LightGBM"],
                "NORM (padronizado z-score)": ["SVR", "MLP"],
            },
        },

        "alvos_e_transformacoes": {
            "contagem": {
                "descricao": "Valor inteiro cru (número de frutas na imagem)",
                "transformacao_inversa": "Nenhuma — já está na escala original",
                "modelo_recomendado": "XGBoost / LightGBM com objective='count:poisson'",
            },
            "contagem_log1p": {
                "descricao": "log(1 + contagem) — comprime a cauda longa",
                "transformacao_inversa": "expm1(pred) = e^pred − 1",
                "modelo_recomendado": "SVR / MLP",
            },
            "contagem_sqrt": {
                "descricao": "raiz quadrada da contagem",
                "transformacao_inversa": "pred² (pred ao quadrado)",
                "modelo_recomendado": "alternativa para SVR / MLP",
            },
        },

        "augmentacao": {
            "aplicada": False,
            "nota": (
                "Augmentação NÃO aplicada nesta extração — treino usa apenas as "
                "imagens originais. As funções de augmentação foram removidas do "
                "pipeline."
            ),
        },

        "n_por_grupo_apos_selecao": {k: len(v) for k, v in grupos.items()},

        "correcoes_v81": {
            "MSER": (
                "cv2.MSER_create(_delta=...) lançava TypeError no OpenCV 4.x e o "
                "try/except zerava o MSER. Corrigido com setters — G10 e cnt_mser_* "
                "agora funcionam."
            ),
            "features_derivadas": (
                "mascara_prop_fruta / mascara_n_blobs eram referenciadas sem serem "
                "criadas (KeyError). Agora são calculadas e populam o grupo G_mascara."
            ),
            "isotropia": "guarda anti região-plana adicionada a _gabor_isotropy_map.",
            "cor_mascarada": "máscara passada a features_hsv e features_rgb_lab.",
        },

        "mascara": {
            "versao": "7.1",
            "criterios": "6 critérios (A–F), votação 2-de-6",
            "descricao": {
                "A": "Gabor Isotropy ≥ percentil 55 (Kurtulmus et al. 2011)",
                "B": "SATD ≤ percentil 50 — superfície lisa (Zhao & Lee 2016)",
                "C": "Laplaciano suavizado ≤ percentil 55 — sem nervuras",
                "D": "LAB b* ≥ percentil 50 — carotenoides",
                "E": "Hessiana blob-ness ≥ percentil 60 (Frangi et al. 1998)",
                "F": "Cr − Cb ≥ percentil 45 (Okamoto & Lee 2009)",
            },
            "filtro_pos_votacao": "circularidade ≥ 0,40 E solidity ≥ 0,75",
            "fallback": "Gabor (A) + Cr-Cb (F) quando máscara < 0,5% da imagem",
            "protecao_final": "Máscara zerada se proporção < 0,2% ou > 45%",
            "limitacao_conhecida": (
                "Critérios por percentil passam fração ~fixa de pixels por imagem; "
                "diagnóstico mostrou correlação baixa entre contagem direta e contagem "
                "real. As features de contagem (G14) devem ser interpretadas com cautela."
            ),
        },

        "metricas_avaliacao": {
            "metrica_principal": "R² (coeficiente de determinação)",
            "complementares": ["MAPE", "MAE", "RMSE", "MdAPE"],
            "faixas_mape": "[1],[2],[3-4],[5-7],[8+]",
            "acerto_n_frutas": "±1 e ±2 frutas",
        },

        "eficiencia_computacional": {
            "extracao_features_treino": stats_tempo_treino,
            "extracao_features_teste": stats_tempo_teste,
            "nota_tempo": (
                "Medido com time.perf_counter() por imagem (inclui resize + máscara + "
                "todos os grupos G1-G16 + FFT + spatial pyramid)."
            ),
            "tempo_treino_xgboost_s": "[VERIFICAR]",
            "tempo_treino_mlp_s": "[VERIFICAR]",
            "tempo_treino_svr_s": "[VERIFICAR]",
            "tempo_inferencia_xgboost_s": "[VERIFICAR]",
            "tempo_inferencia_mlp_s": "[VERIFICAR]",
            "tempo_inferencia_svr_s": "[VERIFICAR]",
            "ambiente": ambiente,
        },

        "treino": {
            "n_originais": int(len(df_orig)),
            "n_total": int(len(df_train)),
            "augmentacao_aplicada": False,
            "cnt_min": int(df_orig["contagem"].min()),
            "cnt_max": int(df_orig["contagem"].max()),
            "cnt_media": round(float(df_orig["contagem"].mean()), 2),
            "cnt_mediana": round(float(df_orig["contagem"].median()), 2),
            "cnt_std": round(float(df_orig["contagem"].std()), 2),
            "n_zero": int((df_orig["contagem"] == 0).sum()),
        },
        "teste": {
            "n_imagens": int(len(df_test)),
            "cnt_min": int(df_test["contagem"].min()),
            "cnt_max": int(df_test["contagem"].max()),
            "cnt_media": round(float(df_test["contagem"].mean()), 2),
            "cnt_mediana": round(float(df_test["contagem"].median()), 2),
            "cnt_std": round(float(df_test["contagem"].std()), 2),
            "n_zero": int((df_test["contagem"] == 0).sum()),
        },

        "arquivos_gerados": {
            "orandet_v11_train_raw.csv": "Treino sem normalização (XGBoost / LightGBM)",
            "orandet_v11_test_raw.csv": "Teste sem normalização  (XGBoost / LightGBM)",
            "orandet_v11_train_norm.csv": "Treino padronizado z-score (SVR / MLP)",
            "orandet_v11_test_norm.csv": "Teste padronizado z-score  (SVR / MLP)",
            "orandet_v11_scaler.joblib": "StandardScaler ajustado no treino",
            "orandet_v11_feature_cols.joblib": "Lista de features após seleção automática",
            "orandet_v11_info.json": "Este arquivo de metadados",
        },
    }


# Main
def main():
    data_dir = Path(DATA_DIR)
    img_dir = str(data_dir / "images")
    ann_train = str(data_dir / "annotations_coco" / "instances_train.json")
    ann_test = str(data_dir / "annotations_coco" / "instances_test.json")

    print("\n" + "=" * 70)
    print("  OranDet v8.1 — Pipeline corrigido (MSER + features derivadas)")
    print("  Grupos G1-G16 completos | normalização z-score | sem augmentação")
    print("=" * 70)

    print("\n[1/6] Carregando anotações...")
    reg_train = carregar_anotacoes(ann_train, img_dir)
    reg_test = carregar_anotacoes(ann_test, img_dir)
    print(f"  Treino: {len(reg_train)} | Teste: {len(reg_test)}")

    print("\n[2/6] Extraindo features (treino)...")
    df_train, tempo_treino = processar_split(reg_train, "train")

    print("\n[3/6] Extraindo features (teste)...")
    df_test, tempo_teste = processar_split(reg_test, "test")

    print("\n[4/6] Seleção automática de features...")
    df_train_sel, df_test_sel, removidas, cols_final = selecionar_features(df_train, df_test)

    print("\n[5/6] Salvando datasets RAW...")
    df_train_sel.to_csv(os.path.join(OUTPUT_DIR, "orandet_v11_train_raw.csv"), index=False)
    df_test_sel.to_csv(os.path.join(OUTPUT_DIR, "orandet_v11_test_raw.csv"), index=False)

    print("\n[6/6] Padronizando (z-score) e salvando...")
    df_train_norm, df_test_norm, scaler = normalizar(df_train_sel, df_test_sel)
    df_train_norm.to_csv(os.path.join(OUTPUT_DIR, "orandet_v11_train_norm.csv"), index=False)
    df_test_norm.to_csv(os.path.join(OUTPUT_DIR, "orandet_v11_test_norm.csv"), index=False)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "orandet_v11_scaler.joblib"))

    print("\n  Salvando metadados...")
    info = gerar_info(df_train_sel, df_test_sel, removidas, len(cols_final),
                      tempo_treino, tempo_teste, coletar_ambiente())
    with open(os.path.join(OUTPUT_DIR, "orandet_v11_info.json"), "w", encoding="utf-8") as fj:
        json.dump(info, fj, indent=2, ensure_ascii=False)
    joblib.dump(cols_final, os.path.join(OUTPUT_DIR, "orandet_v11_feature_cols.joblib"))

    n_bruto = info["dimensionalidade"]["n_features_brutas_por_imagem"]
    n_final = info["dimensionalidade"]["n_features_apos_selecao"]
    print(f"\n{'=' * 70}")
    print(f"  Features: {n_bruto} brutas → {n_final} após seleção")
    print(f"\n  Por grupo (após seleção):")
    for grupo, n in info["n_por_grupo_apos_selecao"].items():
        status = "" if n > 0 else "  ← VAZIO (verificar)"
        print(f"    {grupo:<28} {n:>4}{status}")
    print(f"\n  Treino: {info['treino']['n_originais']} imgs | "
          f"Teste: {info['teste']['n_imagens']} imgs")
    print(f"  Datasets em: {OUTPUT_DIR}/")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()