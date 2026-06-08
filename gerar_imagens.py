import json
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÕES
# ============================================================

EXTRATOR_PATH = Path("./extracao_final.py")

# Se quiser forçar uma imagem específica:
EXEMPLO_FILE_NAME = "1408_2_2.jpg"
# EXEMPLO_FILE_NAME = None

OUT_DIR = Path("./figuras_metodologia")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Para escolher imagens representativas
CONTAGEM_MIN = 3
CONTAGEM_MAX = 7


# ============================================================
# IMPORTA O SEU CÓDIGO REAL DE EXTRAÇÃO
# ============================================================

def carregar_extrator():
    if not EXTRATOR_PATH.exists():
        raise FileNotFoundError(
            f"Não encontrei {EXTRATOR_PATH}. "
            "Coloque este script na mesma pasta do seu extracao_final.py."
        )

    spec = importlib.util.spec_from_file_location("extracao_final", EXTRATOR_PATH)
    extrator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extrator)

    return extrator


extrator = carregar_extrator()

IMG_SIZE = extrator.IMG_SIZE
DATA_DIR = Path(extrator.DATA_DIR)

IMG_DIR = DATA_DIR / "images"
ANN_DIR = DATA_DIR / "annotations_coco"

TRAIN_ANN = ANN_DIR / "instances_train.json"
TEST_ANN = ANN_DIR / "instances_test.json"

DATASET_PREPARADO = Path(extrator.OUTPUT_DIR)
TRAIN_CSV_RAW = DATASET_PREPARADO / "orandet_v11_train_raw.csv"
TEST_CSV_RAW = DATASET_PREPARADO / "orandet_v11_test_raw.csv"


# ============================================================
# COCO
# ============================================================

def carregar_coco(caminho_json):
    with open(caminho_json, "r", encoding="utf-8") as f:
        return json.load(f)


def resolver_caminho_imagem(file_name):
    caminho = IMG_DIR / file_name
    if caminho.exists():
        return caminho

    encontrados = list(IMG_DIR.rglob(Path(file_name).name))
    if encontrados:
        return encontrados[0]

    return caminho


def carregar_registros_coco(caminho_json):
    coco = carregar_coco(caminho_json)

    imagens = {img["id"]: img for img in coco["images"]}
    anns_por_img = {img_id: [] for img_id in imagens}

    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        if image_id in anns_por_img:
            anns_por_img[image_id].append(ann)

    registros = []

    for image_id, img_info in imagens.items():
        file_name = img_info["file_name"]
        anns = anns_por_img.get(image_id, [])

        registros.append({
            "image_id": image_id,
            "file_name": file_name,
            "caminho": resolver_caminho_imagem(file_name),
            "contagem": len(anns),
            "annotations": anns,
            "width_coco": img_info.get("width"),
            "height_coco": img_info.get("height"),
        })

    return registros


def escolher_registro(registros):
    if EXEMPLO_FILE_NAME is not None:
        for r in registros:
            if Path(r["file_name"]).name == Path(EXEMPLO_FILE_NAME).name:
                return r
        raise ValueError(f"Imagem não encontrada no COCO: {EXEMPLO_FILE_NAME}")

    candidatos = [
        r for r in registros
        if CONTAGEM_MIN <= r["contagem"] <= CONTAGEM_MAX
        and r["caminho"].exists()
    ]

    if not candidatos:
        candidatos = [r for r in registros if r["caminho"].exists()]

    if not candidatos:
        raise FileNotFoundError("Nenhuma imagem encontrada.")

    return candidatos[len(candidatos) // 2]


def carregar_img_original(reg):
    img = cv2.imread(str(reg["caminho"]))
    if img is None:
        raise FileNotFoundError(f"Erro ao abrir imagem: {reg['caminho']}")
    return img


def preparar_img_pipeline(img_bgr):
    """
    Igual ao começo de _extrair_de_img:
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    """
    return cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))


def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ============================================================
# DEBUG DA MÁSCARA
# A máscara final vem da função REAL:
# extrator.construir_mascara_fruta_verde(img_bgr)
#
# A máscara bruta é reconstruída só para visualização, porque
# sua função real não retorna mask_raw.
# ============================================================

def mascara_bruta_por_votacao_compativel(img_bgr_416):
    gray = cv2.cvtColor(img_bgr_416, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img_bgr_416, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(img_bgr_416, cv2.COLOR_BGR2YCrCb)

    iso_map = extrator._gabor_isotropy_map(gray)
    crit_A = (iso_map >= float(np.percentile(iso_map, 55))).astype(np.float32)

    satd_map_v = extrator._satd_map(gray)
    crit_B = (satd_map_v <= float(np.percentile(satd_map_v, 50))).astype(np.float32)

    lap_map = extrator._laplacian_smooth_map(gray)
    crit_C = (lap_map <= float(np.percentile(lap_map, 55))).astype(np.float32)

    b_ch = lab[:, :, 2].astype(np.float32)
    crit_D = (b_ch >= float(np.percentile(b_ch, 50))).astype(np.float32)

    hess_map = extrator._hessian_convexity_map(gray)
    crit_E = (hess_map >= float(np.percentile(hess_map, 60))).astype(np.float32)

    Cr = ycrcb[:, :, 1].astype(np.float32)
    Cb = ycrcb[:, :, 2].astype(np.float32)
    crcb_diff = Cr - Cb
    crit_F = (crcb_diff >= float(np.percentile(crcb_diff, 45))).astype(np.float32)

    voto = crit_A + crit_B + crit_C + crit_D + crit_E + crit_F
    mask_raw = (voto >= 2).astype(np.uint8) * 255

    return mask_raw, voto


def desenhar_contornos(img_bgr_416, mascara):
    saida = img_bgr_416.copy()

    cnts, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(saida, cnts, -1, (0, 0, 255), 2)

    return saida


def overlay_mascara(img_bgr_416, mascara, alpha=0.25):
    overlay = img_bgr_416.copy()
    overlay[mascara > 0] = (0, 255, 0)

    saida = cv2.addWeighted(
        overlay,
        alpha,
        img_bgr_416,
        1 - alpha,
        0,
    )

    cnts, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(saida, cnts, -1, (0, 0, 255), 2)

    return saida


def proporcao_mascara(mascara):
    return float(cv2.countNonZero(mascara)) / float(mascara.shape[0] * mascara.shape[1])


def escolher_registro_mascara(registros):
    candidatos = [
        r for r in registros
        if CONTAGEM_MIN <= r["contagem"] <= CONTAGEM_MAX
        and r["caminho"].exists()
    ]

    if not candidatos:
        candidatos = [r for r in registros if r["caminho"].exists()]

    avaliados = []

    for reg in candidatos[:300]:
        try:
            img = preparar_img_pipeline(carregar_img_original(reg))
            mascara = extrator.construir_mascara_fruta_verde(img)
            prop = proporcao_mascara(mascara)

            # evita caso totalmente vazio ou máscara enorme demais
            if 0.005 <= prop <= 0.30:
                avaliados.append((prop, reg))
        except Exception:
            continue

    if not avaliados:
        return escolher_registro(registros)

    # pega caso mediano, não o melhor caso
    avaliados.sort(key=lambda x: x[0])
    return avaliados[len(avaliados) // 2][1]


# ============================================================
# FIG. 2 — COCO PARA CONTAGEM
# Compatível com o pipeline: imagem redimensionada para 416×416
# e bounding boxes escaladas para o mesmo tamanho.
# ============================================================

def gerar_figura_coco_contagem(reg):
    img_original = carregar_img_original(reg)
    h0, w0 = img_original.shape[:2]

    img_416 = preparar_img_pipeline(img_original)
    img_bbox = img_416.copy()

    sx = IMG_SIZE / w0
    sy = IMG_SIZE / h0

    for ann in reg["annotations"]:
        x, y, w, h = ann["bbox"]

        x1 = int(round(x * sx))
        y1 = int(round(y * sy))
        x2 = int(round((x + w) * sx))
        y2 = int(round((y + h) * sy))

        cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].imshow(bgr_to_rgb(img_416))
    axes[0].set_title("Imagem RGB 416×416")
    axes[0].axis("off")

    axes[1].imshow(bgr_to_rgb(img_bbox))
    axes[1].set_title(f"Caixas COCO | contagem = {reg['contagem']}")
    axes[1].axis("off")

    fig.tight_layout()

    out = OUT_DIR / "fig_coco_contagem.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Fig. 2 salva: {out}")


# ============================================================
# FIG. 3 — HISTOGRAMA DA CONTAGEM
# Usa CSV raw se existir; caso contrário, usa o COCO.
# ============================================================

def obter_contagens():
    if TRAIN_CSV_RAW.exists() and TEST_CSV_RAW.exists():
        df_train = pd.read_csv(TRAIN_CSV_RAW)
        df_test = pd.read_csv(TEST_CSV_RAW)

        if "augmentacao" in df_train.columns:
            df_train = df_train[df_train["augmentacao"] == "original"]

        return (
            df_train["contagem"].astype(int).to_numpy(),
            df_test["contagem"].astype(int).to_numpy(),
            "CSVs gerados pelo extrator",
        )

    train_regs = carregar_registros_coco(TRAIN_ANN)
    test_regs = carregar_registros_coco(TEST_ANN)

    return (
        np.array([r["contagem"] for r in train_regs], dtype=int),
        np.array([r["contagem"] for r in test_regs], dtype=int),
        "anotações COCO",
    )


def gerar_figura_dist_contagem():
    cont_train, cont_test, origem = obter_contagens()

    max_count = int(max(cont_train.max(), cont_test.max()))
    bins = np.arange(-0.5, max_count + 1.5, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    axes[0].hist(cont_train, bins=bins, edgecolor="black")
    axes[0].axvline(cont_train.mean(), linestyle="--", linewidth=1.5)
    axes[0].set_title(f"Treino | n = {len(cont_train)} | média = {cont_train.mean():.2f}")
    axes[0].set_xlabel("Número de frutos por imagem")
    axes[0].set_ylabel("Número de imagens")
    axes[0].set_xticks(range(0, max_count + 1))
    axes[0].set_xlim(-0.5, max_count + 0.5)

    axes[1].hist(cont_test, bins=bins, edgecolor="black")
    axes[1].axvline(cont_test.mean(), linestyle="--", linewidth=1.5)
    axes[1].set_title(f"Teste | n = {len(cont_test)} | média = {cont_test.mean():.2f}")
    axes[1].set_xlabel("Número de frutos por imagem")
    axes[1].set_ylabel("Número de imagens")
    axes[1].set_xticks(range(0, max_count + 1))
    axes[1].set_xlim(-0.5, max_count + 0.5)

    fig.suptitle(f"Distribuição da contagem por imagem ({origem})")
    fig.tight_layout()

    out = OUT_DIR / "fig_dist_contagem.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Fig. 3 salva: {out}")


# ============================================================
# FIG. 4 — MÁSCARA DE CANDIDATOS
# A máscara final é 100% gerada pela função real do extrator.
# ============================================================

def gerar_figura_mascara(reg):
    img_original = carregar_img_original(reg)
    img_416 = preparar_img_pipeline(img_original)

    mask_raw, voto = mascara_bruta_por_votacao_compativel(img_416)

    # ESTA é a chamada real do seu pipeline
    mask_final = extrator.construir_mascara_fruta_verde(img_416)

    contorno = desenhar_contornos(img_416, mask_final)
    overlay = overlay_mascara(img_416, mask_final)

    prop = proporcao_mascara(mask_final) * 100.0

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))

    axes[0].imshow(bgr_to_rgb(img_416))
    axes[0].set_title("Imagem RGB 416×416")
    axes[0].axis("off")

    axes[1].imshow(mask_raw, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Votação ≥ 2 de 6")
    axes[1].axis("off")

    axes[2].imshow(mask_final, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"Máscara final\nárea = {prop:.1f}%")
    axes[2].axis("off")

    axes[3].imshow(bgr_to_rgb(overlay))
    axes[3].set_title("Sobreposição final")
    axes[3].axis("off")

    fig.suptitle(
        f"{Path(reg['file_name']).name} | contagem COCO = {reg['contagem']}"
    )
    fig.tight_layout()

    out = OUT_DIR / "fig_mascara.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Fig. 4 salva: {out}")


# ============================================================
# FIG. 5 — BAS-RELIEF
# Usa exatamente:
# hsv -> features_canal_v_eq -> _bas_relief_map
# ============================================================

def gerar_figura_basrelief(reg):
    img_original = carregar_img_original(reg)
    img_416 = preparar_img_pipeline(img_original)

    hsv = cv2.cvtColor(img_416, cv2.COLOR_BGR2HSV)

    # Chamada real da máscara
    mascara = extrator.construir_mascara_fruta_verde(img_416)

    # Chamada real do G3, que devolve o V_eq usado no G4
    _, V_eq = extrator.features_canal_v_eq(hsv, mascara)

    # Chamada real do mapa bas-relief do seu código
    bas = extrator._bas_relief_map(V_eq)

    contorno = desenhar_contornos(img_416, mascara)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))

    axes[0].imshow(bgr_to_rgb(img_416))
    axes[0].set_title("Imagem RGB 416×416")
    axes[0].axis("off")

    axes[1].imshow(V_eq, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Canal V equalizado")
    axes[1].axis("off")

    axes[2].imshow(bas, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Bas-relief do código")
    axes[2].axis("off")

    axes[3].imshow(bgr_to_rgb(contorno))
    axes[3].set_title("Máscara usada no G4")
    axes[3].axis("off")

    fig.suptitle(
        f"{Path(reg['file_name']).name} | contagem COCO = {reg['contagem']}"
    )
    fig.tight_layout()

    out = OUT_DIR / "fig_basrelief.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Fig. 5 salva: {out}")


# ============================================================
# MAIN
# ============================================================

def main():
    registros_train = carregar_registros_coco(TRAIN_ANN)

    reg_base = escolher_registro(registros_train)
    reg_mascara = escolher_registro_mascara(registros_train)

    print("Imagem base escolhida:")
    print(f"  {reg_base['file_name']} | contagem = {reg_base['contagem']}")

    print("Imagem escolhida para máscara:")
    print(f"  {reg_mascara['file_name']} | contagem = {reg_mascara['contagem']}")

    gerar_figura_coco_contagem(reg_base)
    gerar_figura_dist_contagem()
    gerar_figura_mascara(reg_mascara)
    gerar_figura_basrelief(reg_base)

    print("\nFiguras geradas em:")
    print(f"  {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()