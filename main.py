"""
visualizar_pipeline_v6.py
─────────────────────────────────────────────────────────────────────────────
Visualiza o pipeline de extração de features v6.0 (laranjas VERDES).
Exporta dois PNGs:

  • <nome>_pipeline_etapas.png   — cada etapa em painel separado (6×5 = 30)
  • <nome>_pipeline_completo.png — visão geral compacta (4×7 = 28)

Mudanças v6 refletidas aqui:
  MÁSCARA COMPLETAMENTE REFEITA — 5 critérios de forma/textura:
    Crit A: Gabor Isotropy alto  (textura esférica ≠ nervura direcional)
    Crit B: SATD baixo           (superfície lisa vs nervurada)
    Crit C: Laplaciano baixo     (sem bordas internas)
    Crit D: LAB b* alto          (carotenoides — único critério de cor válido)
    Crit E: Hough seed map       (proximidade geométrica a círculos)

  REMOVIDOS da máscara:  ExG, LAB a*, MSER V (todos inúteis verde-sobre-verde)

  NOVOS GRUPOS DE FEATURES:
    G8  — SATD dedicado (global + fruta + razão fundo/fruta)
    G14 — Contagem direta (Hough + MSER + blob + ensemble)

  VISUALIZAÇÃO:
    Linha 1 — 5 critérios da máscara v6 (Gabor Iso, SATD, Lap, LABb, Hough seed)
    Linha 2 — Máscara final + mapa de votos + análise da máscara (SATD/Lap overlay)
    Linha 3 — G5 Gabor circular (orientações + isotropy λ10/λ20)
    Linha 4 — G8 SATD + G4 Bas-relief + G6 LBP + G9 HOG
    Linha 5 — G10 MSER (novos params) + G11 Hough + G12 Grade + Geometria
    Linha 6 — G14 Contagem direta overlay + augmentações + multi-escala

Dependências:
    pip install opencv-python scikit-image scipy numpy matplotlib
─────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO — edite CAMINHO para sua imagem
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 416
CAMINHO = "/Users/antonioreis/Downloads/dataverse_files/images/1408_2_1.jpg"


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTES INTERNOS DA MÁSCARA v6
# (cópias fiéis de extracao_features_v6.py para visualização)
# ─────────────────────────────────────────────────────────────────────────────

def _gabor_isotropy_map(gray):
    """
    Alta isotropia = textura igual em 0/45/90/135° = superfície esférica (fruta).
    Baixa isotropia = textura direcional = nervura de folha.
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
    isotropy = 1.0 - np.clip(std_resp / mean_resp, 0, 1)
    return isotropy.astype(np.float32)


def _satd_map(gray):
    """
    SATD: diferença local pixel vs vizinhança.
    Fruta lisa → baixo. Nervura de folha → alto.
    Ref: Zhao & Lee (2016).
    """
    gray_f = gray.astype(np.float32)
    blur5 = cv2.blur(gray_f, (5, 5))
    blur11 = cv2.blur(gray_f, (11, 11))
    return (np.abs(gray_f - blur5) + np.abs(gray_f - blur11)) / 2.0


def _laplacian_smooth_map(gray):
    """Bordas internas (nervuras) → alto. Superfície lisa (fruta) → baixo."""
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap_abs = np.abs(lap).astype(np.float32)
    return cv2.GaussianBlur(lap_abs, (31, 31), 0)


def _hough_seed_map(gray, img_size):
    """
    Campo de atração ao redor de círculos Hough detectados.
    Independente de cor — funciona em verde-sobre-verde.
    """
    seed = np.zeros((img_size, img_size), dtype=np.float32)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    for rmin, rmax in [(15, 40), (40, 80), (80, 120)]:
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=30,
            param1=50, param2=40,
            minRadius=rmin, maxRadius=rmax,
        )
        if circles is not None:
            for cx, cy, r in circles[0]:
                cv2.circle(seed, (int(cx), int(cy)), int(r * 1.1), 1.0, -1)
    return cv2.GaussianBlur(seed, (21, 21), 0)


def construir_mascara_v6(img_bgr):
    """
    Máscara v6 completa. Retorna (mascara_final, dict_crits_para_visualizacao).
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # ── Critério A: Gabor Isotropy ───────────────────────────────────────────
    iso_map = _gabor_isotropy_map(gray)
    thr_iso = float(np.percentile(iso_map, 65))
    crit_A = (iso_map >= thr_iso).astype(np.float32)

    # ── Critério B: SATD baixo ───────────────────────────────────────────────
    satd_map = _satd_map(gray)
    thr_satd = float(np.percentile(satd_map, 40))
    crit_B = (satd_map <= thr_satd).astype(np.float32)

    # ── Critério C: Laplaciano suavizado baixo ───────────────────────────────
    lap_map = _laplacian_smooth_map(gray)
    thr_lap = float(np.percentile(lap_map, 45))
    crit_C = (lap_map <= thr_lap).astype(np.float32)

    # ── Critério D: LAB b* alto ──────────────────────────────────────────────
    b_ch = lab[:, :, 2].astype(np.float32)
    thr_b = float(np.percentile(b_ch, 55))
    crit_D = (b_ch >= thr_b).astype(np.float32)

    # ── Critério E: Hough seed ───────────────────────────────────────────────
    hough_seed = _hough_seed_map(gray, img_bgr.shape[0])
    crit_E = (hough_seed > 0.05).astype(np.float32)

    # ── Votação 3/5 ──────────────────────────────────────────────────────────
    voto = crit_A + crit_B + crit_C + crit_D + crit_E
    mask_raw = (voto >= 3).astype(np.uint8) * 255

    # ── Morfologia ───────────────────────────────────────────────────────────
    k3 = np.ones((3, 3), np.uint8)
    k9 = np.ones((9, 9), np.uint8)
    mascara = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, k3, iterations=2)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k9, iterations=3)

    # ── Filtro de circularidade ───────────────────────────────────────────────
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

    # ── Validação ────────────────────────────────────────────────────────────
    prop = float(mascara_filtrada.sum()) / (255.0 * h * w)
    if prop < 0.003 or prop > 0.40:
        mascara_filtrada = np.zeros_like(mascara_filtrada)

    crits = {
        "crit_A_gabor_iso": _norm255(iso_map),
        "crit_B_satd": _norm255(-satd_map),  # invertido: claro=lisa=fruta
        "crit_C_lap": _norm255(-lap_map),  # invertido: claro=liso=fruta
        "crit_D_lab_b": lab[:, :, 2],
        "crit_E_hough_seed": _norm255(hough_seed),
        "voto_raw": np.clip(voto / 5 * 255, 0, 255).astype(np.uint8),
        "mascara_morfo": mascara,
    }
    return mascara_filtrada, crits, iso_map, satd_map, lap_map, hough_seed


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm255(arr):
    """Normaliza float array para uint8 [0, 255]."""
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)


def _ajusta_brilho(img, fator):
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h[:, :, 2] = np.clip(h[:, :, 2] * fator, 0, 255)
    return cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augmentacoes(img):
    return {
        "flip_h": cv2.flip(img, 1),
        "flip_v": cv2.flip(img, 0),
        "bright_75": _ajusta_brilho(img, 0.75),
        "bright_125": _ajusta_brilho(img, 1.25),
    }


def _overlay_mask(rgb_img, mask, cor=(255, 100, 0), alpha=0.45):
    """Sobrepõe máscara colorida sobre imagem RGB."""
    out = rgb_img.copy().astype(np.float32)
    m = mask > 0
    for c, v in enumerate(cor):
        out[:, :, c][m] = out[:, :, c][m] * (1 - alpha) + v * alpha
    return out.clip(0, 255).astype(np.uint8)


def _draw_hough_circles(rgb_img, gray, param2=40):
    """Desenha círculos Hough por faixa de raio com cores diferentes."""
    out = rgb_img.copy()
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    faixas = [
        (15, 40, (255, 220, 50), "pequeno"),
        (40, 80, (255, 120, 20), "médio"),
        (80, 120, (200, 50, 10), "grande"),
    ]
    total = 0
    for rmin, rmax, cor, nome in faixas:
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=30,
            param1=50, param2=param2,
            minRadius=rmin, maxRadius=rmax,
        )
        if circles is not None:
            for cx, cy, r in circles[0]:
                cv2.circle(out, (int(cx), int(cy)), int(r), cor, 2)
                cv2.circle(out, (int(cx), int(cy)), 3, (255, 255, 255), -1)
                total += 1
    return out, total


def _draw_mser(rgb_img, gray):
    """MSER com parâmetros v6 (max_variation=0.12, min_area=400)."""
    out = rgb_img.copy()
    n_circ = 0
    try:
        mser = cv2.MSER_create(
            _delta=8, _min_area=400, _max_area=25000,
            _max_variation=0.12, _min_diversity=0.20,
        )
        regs, _ = mser.detectRegions(gray)
        for pts in regs:
            if len(pts) < 400:
                continue
            hull = cv2.convexHull(pts.reshape(-1, 1, 2))
            ha = cv2.contourArea(hull)
            if ha > 0 and (len(pts) / ha) > 0.55 and len(pts) >= 5:
                ell = cv2.fitEllipse(pts.reshape(-1, 1, 2))
                _, (ma, mi), _ = ell
                if mi > 0 and (ma / mi) < 1.5:
                    cv2.polylines(out, [hull], True, (255, 60, 200), 2)
                    cv2.ellipse(out, ell, (100, 200, 255), 1)
                    n_circ += 1
    except Exception:
        pass
    return out, n_circ


def _draw_contagem_direta(rgb_img, gray, mascara):
    """
    Overlay G14 — Contagem Direta.
    Círculos Hough (laranja), blobs da máscara (verde), ensemble anotado.
    """
    out = rgb_img.copy()
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    # Hough
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=30, param1=50, param2=40,
        minRadius=30, maxRadius=90,
    )
    n_hough = 0
    if circles is not None:
        for cx, cy, r in circles[0]:
            cv2.circle(out, (int(cx), int(cy)), int(r), (255, 165, 0), 3)
            n_hough += 1

    # Blobs da máscara
    k3 = np.ones((3, 3), np.uint8)
    mask_c = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k3)
    cnts, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_blob = 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        perim = cv2.arcLength(cnt, True)
        if 4 * np.pi * area / (perim ** 2 + 1e-6) >= 0.40:
            cv2.drawContours(out, [cnt], -1, (50, 255, 100), 2)
            n_blob += 1

    iso_map = _gabor_isotropy_map(gray)
    est_iso = float((iso_map > 0.72).mean()) * IMG_SIZE * IMG_SIZE / 2500.0
    ensemble = 0.35 * n_hough + 0.20 * n_blob + 0.15 * est_iso

    cv2.putText(out, f"Hough: {n_hough}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2)
    cv2.putText(out, f"Blob:  {n_blob}", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 255, 100), 2)
    cv2.putText(out, f"Ens:   {ensemble:.1f}", (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE INTERMEDIÁRIO v6
# ─────────────────────────────────────────────────────────────────────────────

def pipeline_intermediario(img):
    res = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Máscara v6 ──────────────────────────────────────────────────────────
    mascara, crits, iso_map, satd_map, lap_map, hough_seed = construir_mascara_v6(img)
    res["mascara_final"] = mascara
    res["voto_raw"] = crits["voto_raw"]
    res["mascara_morfo"] = crits["mascara_morfo"]
    res["crit_A_gabor_iso"] = crits["crit_A_gabor_iso"]
    res["crit_B_satd_inv"] = crits["crit_B_satd"]
    res["crit_C_lap_inv"] = crits["crit_C_lap"]
    res["crit_D_lab_b"] = crits["crit_D_lab_b"]
    res["crit_E_hough_seed"] = crits["crit_E_hough_seed"]
    res["mascara_overlay"] = _overlay_mask(rgb, mascara, cor=(255, 120, 0), alpha=0.5)

    # SATD + Laplaciano combinados (escuro = candidato a fruta)
    satd_n = _norm255(satd_map)
    lap_n = _norm255(lap_map)
    res["satd_lap_combined"] = np.clip(
        satd_n.astype(np.int32) + lap_n.astype(np.int32), 0, 255
    ).astype(np.uint8)

    # ── G5 — Gabor circular (4 orientações λ10 + λ20 + isotropy) ───────────
    for lam in [10, 20]:
        pilha = []
        for theta, ang in zip([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], ["000", "045", "090", "135"]):
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta, float(lam), 1.0, 0)
            resp = np.abs(cv2.filter2D(gray, cv2.CV_64F, kernel))
            pilha.append(resp)
            res[f"gabor_lam{lam}_a{ang}"] = _norm255(resp)

        stacked = np.stack(pilha, axis=0)
        mean_r = stacked.mean(axis=0) + 1e-9
        cv_ = stacked.std(axis=0) / mean_r
        isotropy = 1.0 - np.clip(cv_, 0, 1)
        res[f"gabor_lam{lam}_isotropy"] = _norm255(isotropy)  # claro = fruta

    # ── G8 — SATD ────────────────────────────────────────────────────────────
    res["satd_global"] = _norm255(-satd_map)  # invertido: claro=liso
    satd_fruta = satd_map.copy();
    satd_fruta[mascara == 0] = 0
    res["satd_fruta"] = _norm255(-satd_fruta + satd_fruta.max())

    # ── G4 — Bas-relief ──────────────────────────────────────────────────────
    V = hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    V_eq = clahe.apply(V)
    V_blur = cv2.GaussianBlur(V_eq, (5, 5), 0)
    sx = cv2.Sobel(V_blur, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(V_blur, cv2.CV_64F, 0, 1, ksize=3)
    sx_abs = np.abs(sx).astype(np.uint8)
    lap_v = cv2.Laplacian(V_blur, cv2.CV_64F, ksize=3)
    lap_abs_v = np.abs(lap_v).astype(np.uint8)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    res["sobel_mag"] = np.clip(mag / (mag.max() + 1e-9) * 255, 0, 255).astype(np.uint8)
    res["laplace_v"] = lap_abs_v
    res["basrelief"] = cv2.addWeighted(sx_abs, 0.6, lap_abs_v, 0.4, 0)
    res["lap_suavizado"] = cv2.GaussianBlur(lap_abs_v, (21, 21), 0)

    # ── G6 — LBP ─────────────────────────────────────────────────────────────
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    res["lbp"] = _norm255(lbp)

    # ── G9 — HOG ─────────────────────────────────────────────────────────────
    img_128 = cv2.resize(gray, (128, 128))
    _, hog_img = hog(
        img_128, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), feature_vector=True,
        block_norm="L2-Hys", visualize=True,
    )
    res["hog"] = cv2.resize(
        np.clip(hog_img / (hog_img.max() + 1e-9) * 255, 0, 255).astype(np.uint8),
        (IMG_SIZE, IMG_SIZE),
    )

    # ── G10 — Geometria ──────────────────────────────────────────────────────
    k3 = np.ones((3, 3), np.uint8)
    mask_g = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k3)
    cnts, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    geo_vis = rgb.copy()
    for cnt in cnts:
        a = cv2.contourArea(cnt)
        if a < 100:
            continue
        p = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * a / (p ** 2 + 1e-6)
        cor = (0, 255, 128) if circ >= 0.50 else (255, 220, 30)
        cv2.drawContours(geo_vis, [cnt], -1, cor, 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(geo_vis, (x, y), (x + w, y + h), (200, 200, 200), 1)
        cv2.putText(geo_vis, f"{circ:.2f}", (x + 2, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, cor, 1)
    res["geometria"] = geo_vis

    # ── G10 — MSER v6 ────────────────────────────────────────────────────────
    mser_vis, n_mser = _draw_mser(rgb, gray)
    cv2.putText(mser_vis, f"MSER circ: {n_mser}",
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 60, 200), 2)
    res["mser_candidatos"] = mser_vis

    # ── G11 — Hough v6 (param2=40) ───────────────────────────────────────────
    hough_vis, total_h = _draw_hough_circles(rgb, gray, param2=40)
    cv2.putText(hough_vis, f"Total: {total_h}",
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 2)
    res["hough"] = hough_vis

    # ── G12 — Grade 4×4 ──────────────────────────────────────────────────────
    mf = mascara.astype(np.float64) / 255.0
    grade_vis = rgb.copy()
    gh, gw = IMG_SIZE // 4, IMG_SIZE // 4
    for i in range(4):
        for j in range(4):
            y1, y2 = i * gh, (i + 1) * gh
            x1, x2 = j * gw, (j + 1) * gw
            dens = mf[y1:y2, x1:x2].mean()
            overlay = grade_vis[y1:y2, x1:x2].copy()
            cor = np.array([255, 140, 0]) * dens + np.array([0, 55, 10]) * (1 - dens)
            grade_vis[y1:y2, x1:x2] = (overlay * 0.5 + cor * 0.5).clip(0, 255).astype(np.uint8)
            cv2.rectangle(grade_vis, (x1, y1), (x2 - 1, y2 - 1), (240, 240, 240), 1)
            cv2.putText(grade_vis, f"{dens:.2f}", (x1 + 3, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    res["grade"] = grade_vis

    # ── G14 — Contagem Direta overlay ────────────────────────────────────────
    res["contagem_direta"] = _draw_contagem_direta(rgb, gray, mascara)

    # ── Augmentações ─────────────────────────────────────────────────────────
    aug = augmentacoes(img)
    res["aug_flip_h"] = cv2.cvtColor(aug["flip_h"], cv2.COLOR_BGR2RGB)
    res["aug_bright_75"] = cv2.cvtColor(aug["bright_75"], cv2.COLOR_BGR2RGB)

    # ── Multi-escala ─────────────────────────────────────────────────────────
    img_half = cv2.resize(img, (208, 208), interpolation=cv2.INTER_AREA)
    res["escala_208"] = cv2.resize(
        cv2.cvtColor(img_half, cv2.COLOR_BGR2RGB),
        (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST,
    )

    return res, rgb, aug


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — renderiza um painel
# ─────────────────────────────────────────────────────────────────────────────

def _render_painel(ax, imagem, titulo, cmap, cor_borda):
    ax.set_facecolor("#111111")
    if imagem.ndim == 3:
        ax.imshow(imagem)
    else:
        ax.imshow(imagem, cmap=cmap, vmin=0, vmax=255)
    ax.set_title(titulo, color=cor_borda, fontsize=7.8, fontweight="bold",
                 pad=4, fontfamily="monospace")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(cor_borda)
        spine.set_linewidth(1.4)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 1 — Etapas separadas  (6 × 5 = 30 painéis)
# ─────────────────────────────────────────────────────────────────────────────

def figura_etapas(rgb, res):
    fig, axes = plt.subplots(6, 5, figsize=(26, 31))
    fig.patch.set_facecolor("#0d0d0d")

    C = {
        "mask": "#FF8C00",
        "res": "#00FF7F",
        "gabor": "#DA70D6",
        "text": "#00BFFF",
        "det": "#7CFC00",
        "cnt": "#FF69B4",
    }

    paineis = [
        # L1 — 5 critérios da máscara v6
        (rgb, "ORIGINAL", "gray", C["mask"]),
        (res["crit_A_gabor_iso"], "Crit A · Gabor Isotropy\n(claro=esférico=fruta)", "viridis", C["mask"]),
        (res["crit_B_satd_inv"], "Crit B · SATD inv.\n(claro=liso=fruta)", "plasma", C["mask"]),
        (res["crit_C_lap_inv"], "Crit C · Lap inv.\n(claro=sem nervura)", "hot", C["mask"]),
        (res["crit_D_lab_b"], "Crit D · LAB b*\n(único discriminador de cor)", "YlOrBr", C["mask"]),

        # L2 — resultado da máscara
        (res["crit_E_hough_seed"], "Crit E · Hough Seed\n(campo geométrico)", "hot", C["res"]),
        (res["voto_raw"], "Mapa de votos\n(branco = 5/5 critérios)", "hot", C["res"]),
        (res["mascara_morfo"], "Pós-morfologia\n(antes filtro circularidade)", "Greens", C["res"]),
        (res["mascara_final"], "MÁSCARA FINAL v6\n(+filtro circ. ≥ 0.35)", "Greens", C["res"]),
        (res["mascara_overlay"], "Overlay s/ original\n(laranja=fruta detectada)", "gray", C["res"]),

        # L3 — G5 Gabor λ10 (4 orientações + isotropy)
        (res["gabor_lam10_a000"], "G5 · Gabor λ10 θ0°\n(γ=1, circular)", "magma", C["gabor"]),
        (res["gabor_lam10_a045"], "G5 · Gabor λ10 θ45°", "magma", C["gabor"]),
        (res["gabor_lam10_a090"], "G5 · Gabor λ10 θ90°", "magma", C["gabor"]),
        (res["gabor_lam10_a135"], "G5 · Gabor λ10 θ135°", "magma", C["gabor"]),
        (res["gabor_lam10_isotropy"], "G5 · Isotropy λ10\n(claro=fruta esférica ★)", "viridis", C["gabor"]),

        # L4 — G8 SATD + G4 bas-relief + G6 LBP
        (res["satd_global"], "G8 · SATD inv. global\n(claro=superfície lisa)", "hot", C["text"]),
        (res["satd_fruta"], "G8 · SATD s/ máscara\n(fruta isolada)", "hot", C["text"]),
        (res["satd_lap_combined"], "G8+G4 · SATD+Lap\n(escuro=candidato fruta ★)", "hot", C["text"]),
        (res["basrelief"], "G4 · Bas-relief\n(Sobel×0.6 + Lap×0.4)", "copper", C["text"]),
        (res["lbp"], "G6 · LBP (P=8,R=1)\n(padrão local binário)", "viridis", C["text"]),

        # L5 — detecção
        (res["geometria"], "G10 · Geometria\n(verde=circ≥0.5, am=espúrio)", "gray", C["det"]),
        (res["mser_candidatos"], "G10 · MSER v6\n(max_var=0.12, roxo=circular)", "gray", C["det"]),
        (res["hough"], "G11 · Hough\n(param2=40, menos FP q/ v5 ★)", "gray", C["det"]),
        (res["grade"], "G12 · Grade 4×4\n(dens. máscara por célula)", "gray", C["det"]),
        (res["hog"], "G9 · HOG\n(gradientes orientados 9 bins)", "inferno", C["det"]),

        # L6 — G14 + augmentações + escala
        (res["contagem_direta"], "G14 · Contagem Direta ★\n(Hough+Blob+Ensemble)", "gray", C["cnt"]),
        (res["gabor_lam20_isotropy"], "G5 · Isotropy λ20\n(escala maior)", "viridis", C["cnt"]),
        (res["aug_flip_h"], "Aug · flip_h", "gray", C["cnt"]),
        (res["aug_bright_75"], "Aug · bright_75", "gray", C["cnt"]),
        (res["escala_208"], "G13 · Escala 208px\n(multi-escala)", "gray", C["cnt"]),
    ]

    for ax, (imagem, titulo, cmap, cor) in zip(axes.flat, paineis):
        _render_painel(ax, imagem, titulo, cmap, cor)

    fig.suptitle(
        "PIPELINE DE EXTRAÇÃO DE FEATURES — OranDet v6.0  ·  Laranjas Verdes sobre Fundo Verde\n"
        "Máscara v6 (forma/textura): Gabor Isotropy · SATD baixo · Laplaciano baixo · LAB b* · Hough Seed\n"
        "Removidos: ExG · LAB a* · MSER V  |  Novos grupos: G8 SATD  ·  G14 Contagem Direta  [★ = features chave]",
        color="#00FF7F", fontsize=10, fontweight="bold",
        y=0.999, fontfamily="monospace",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.974])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 2 — Visão completa  (4 × 7 = 28 painéis)
# ─────────────────────────────────────────────────────────────────────────────

def figura_completo(rgb, res):
    fig, axes = plt.subplots(4, 7, figsize=(32, 18))
    fig.patch.set_facecolor("#0a0a0a")

    C = ["#FF8C00", "#DA70D6", "#00BFFF", "#7CFC00"]
    labels = [
        "MÁSCARA v6\n5 CRITÉRIOS",
        "DISCRIMINADORES\nCHAVE",
        "PIPELINE\nTEXTURA",
        "DETECÇÃO\nE CONTAGEM",
    ]

    linhas = [
        [  # L1 — Critérios da máscara
            (rgb, "ORIGINAL", "gray", C[0]),
            (res["crit_A_gabor_iso"], "Crit A Gabor Iso", "viridis", C[0]),
            (res["crit_B_satd_inv"], "Crit B SATD inv\n(liso=fruta)", "plasma", C[0]),
            (res["crit_C_lap_inv"], "Crit C Lap inv\n(s/nervura)", "hot", C[0]),
            (res["crit_D_lab_b"], "Crit D LAB b*\n(carotenoide)", "YlOrBr", C[0]),
            (res["voto_raw"], "Votos (branco=5/5)", "hot", C[0]),
            (res["mascara_final"], "MÁSCARA FINAL v6", "Greens", C[0]),
        ],
        [  # L2 — Discriminadores chave
            (res["crit_E_hough_seed"], "Crit E Hough Seed", "hot", C[1]),
            (res["gabor_lam10_isotropy"], "Isotropy λ10 ★\n(CHAVE)", "viridis", C[1]),
            (res["gabor_lam20_isotropy"], "Isotropy λ20", "viridis", C[1]),
            (res["satd_global"], "SATD inv ★\n(CHAVE)", "hot", C[1]),
            (res["satd_lap_combined"], "SATD+Lap ★\n(escuro=fruta)", "hot", C[1]),
            (res["crit_D_lab_b"], "LAB b*\n(único cor válido)", "YlOrBr", C[1]),
            (res["mascara_overlay"], "Overlay resultado", "gray", C[1]),
        ],
        [  # L3 — Textura
            (res["gabor_lam10_a000"], "G5 Gabor λ10 θ0°", "magma", C[2]),
            (res["gabor_lam10_a090"], "G5 Gabor λ10 θ90°", "magma", C[2]),
            (res["basrelief"], "G4 Bas-relief", "copper", C[2]),
            (res["lap_suavizado"], "G4 Lap suav.\n(escuro=fruta)", "hot", C[2]),
            (res["lbp"], "G6 LBP (P=8,R=1)", "viridis", C[2]),
            (res["hog"], "G9 HOG", "inferno", C[2]),
            (res["satd_fruta"], "G8 SATD c/ máscara", "hot", C[2]),
        ],
        [  # L4 — Detecção e contagem
            (res["mser_candidatos"], "G10 MSER v6\n(max_var=0.12)", "gray", C[3]),
            (res["hough"], "G11 Hough\n(param2=40)", "gray", C[3]),
            (res["grade"], "G12 Grade 4×4", "gray", C[3]),
            (res["geometria"], "G10 Geometria", "gray", C[3]),
            (res["contagem_direta"], "G14 Contagem Direta ★", "gray", C[3]),
            (res["aug_flip_h"], "Aug flip_h", "gray", C[3]),
            (res["escala_208"], "G13 Escala 208px", "gray", C[3]),
        ],
    ]

    for li, (linha, label, cor_l) in enumerate(zip(linhas, labels, C)):
        for ci, (imagem, titulo, cmap, cor) in enumerate(linha):
            _render_painel(axes[li, ci], imagem, titulo, cmap, cor)
        fig.text(
            0.003, 0.875 - li * 0.248, label,
            color=cor_l, fontsize=8.5, fontweight="bold",
            fontfamily="monospace", rotation=90, va="center", ha="center",
        )

    fig.suptitle(
        "VISÃO COMPLETA — Pipeline OranDet v6.0  ·  Laranjas Verdes\n"
        "Máscara: Forma/Textura (não cor)  |  Removidos: ExG · LAB a* · MSER V  |  "
        "Novos: G8 SATD · G14 Contagem  [★ = features chave para contagem]",
        color="white", fontsize=11, fontweight="bold",
        y=1.005, fontfamily="monospace",
    )
    plt.tight_layout(rect=[0.02, 0, 1, 1])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    caminho = CAMINHO
    base = os.path.splitext(os.path.basename(caminho))[0]
    saida_etapas = f"{base}_pipeline_etapas.png"
    saida_completo = f"{base}_pipeline_completo.png"

    print(f"\n[1/4] Carregando: {caminho}")
    img = cv2.imread(caminho)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {caminho}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    print(f"      Shape: {img.shape}")

    print("[2/4] Calculando transformações v6.0...")
    res, rgb, aug = pipeline_intermediario(img)
    prop = float(res["mascara_final"].sum()) / (255.0 * IMG_SIZE * IMG_SIZE)
    status = "OK" if 0.003 < prop < 0.40 else f"ALERTA ({prop * 100:.1f}% fora de 0.3-40%)"
    print(f"      Máscara v6: {prop * 100:.2f}%  [{status}]")

    print("[3/4] Gerando figuras...")
    fig1 = figura_etapas(rgb, res)
    fig1.savefig(saida_etapas, dpi=120, bbox_inches="tight",
                 facecolor="#0d0d0d", edgecolor="none")
    plt.close(fig1)
    print(f"      ✓ {saida_etapas}")

    fig2 = figura_completo(rgb, res)
    fig2.savefig(saida_completo, dpi=120, bbox_inches="tight",
                 facecolor="#0a0a0a", edgecolor="none")
    plt.close(fig2)
    print(f"      ✓ {saida_completo}")

    print("\n[4/4] Pronto!\n")


if __name__ == "__main__":
    main()
