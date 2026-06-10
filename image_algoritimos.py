"""
gerar_figuras_resultados.py

Gera as figuras da secao de Resultados do TCC reaproveitando a logica dos tres
scripts de treino (XGBoost, MLP e SVR), sem re-treinar nem refazer a busca de
hiperparametros.

O que o script faz:
  1. importa os tres scripts de treino como modulos;
  2. usa a funcao carregar_dados() de cada um para obter o conjunto de teste
     com EXATAMENTE o mesmo pre-processamento usado no treino;
  3. carrega o modelo ja treinado e salvo em disco (.joblib) por cada script;
  4. gera duas figuras em DIR_FIGURAS:
       - fig_predito_vs_real.png : dispersao predito vs real (3 paineis)
       - fig_importancia_xgb.png : importancia dos atributos do XGBoost (ganho)

IMPORTANTE:
  - Rode este script no MESMO diretorio em que voce rodou os scripts de treino,
    porque eles usam caminhos relativos (dataset_preparado_v11/ e resultados_*/).
  - Os modelos .joblib precisam ja existir; rode os tres treinos antes.
  - Ajuste os caminhos em CONFIG se os nomes dos seus arquivos forem diferentes.
"""

import os
import sys
import importlib.util

import matplotlib
matplotlib.use("Agg")  # backend sem display, apenas para salvar PNG
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import r2_score

# ============================ CONFIG ============================
# Caminhos para os tres scripts de treino (ajuste conforme seus nomes de arquivo)
SCRIPT_XGB = "treinar_xgboost.py"
SCRIPT_MLP = "mpl_train.py"
SCRIPT_SVR = "svr_train.py"

# Nomes dos modelos salvos por cada script (dentro do OUTPUT_DIR de cada um)
MODELO_XGB = "xgb_v6_modelo.joblib"
MODELO_MLP = "mlp_v9_modelo.joblib"
MODELO_SVR = "svr_v3_modelo.joblib"

# Onde salvar as figuras
DIR_FIGURAS = "./figuras_tcc"

# Aparencia
DPI = 200
N_TOP_FEATURES = 15   # numero de atributos no grafico de importancia
JITTER = 0.06         # deslocamento horizontal leve (apenas visual) nos pontos
# ===============================================================

os.makedirs(DIR_FIGURAS, exist_ok=True)


def importar_modulo(caminho, nome):
    """Importa um arquivo .py como modulo, independente do nome do arquivo."""
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Script nao encontrado: {caminho}. Ajuste o caminho em CONFIG."
        )
    spec = importlib.util.spec_from_file_location(nome, caminho)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nome] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def carregar_modelo(output_dir, nome_modelo, nome_amigavel):
    caminho = os.path.join(output_dir, nome_modelo)
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Modelo do {nome_amigavel} nao encontrado em {caminho}. "
            f"Rode o script de treino do {nome_amigavel} antes."
        )
    return joblib.load(caminho)


def obter_predicoes():
    """Reaproveita carregar_dados() de cada modulo e o modelo salvo para devolver
    (y_real, y_pred) de cada modelo na escala original, mais a importancia do XGBoost."""
    resultados = {}

    # ---- XGBoost (alvo: contagem direta; sem transformacao inversa) ----
    print("\n>>> XGBoost")
    xgb = importar_modulo(SCRIPT_XGB, "treino_xgb")
    df_train_full, df_orig, df_test = xgb.carregar_dados()
    X_tr, y_tr, X_test, y_test, feat_cols, medianas = xgb.preparar_xy(df_train_full, df_test)
    modelo_xgb = carregar_modelo(xgb.OUTPUT_DIR, MODELO_XGB, "XGBoost")
    y_pred = np.maximum(modelo_xgb.predict(X_test), 0)
    resultados["XGBoost"] = (np.asarray(y_test, float), np.asarray(y_pred, float))

    # importancia (ganho) reaproveitando a propria logica do script
    imp, _ = xgb.diagnosticar_features(modelo_xgb, feat_cols)

    # ---- MLP (alvo: log1p; inversa expm1) ----
    print("\n>>> MLP")
    mlp = importar_modulo(SCRIPT_MLP, "treino_mlp")
    dados_mlp = mlp.carregar_dados()
    X_test_mlp, y_test_mlp = dados_mlp[5], dados_mlp[6]
    modelo_mlp = carregar_modelo(mlp.OUTPUT_DIR, MODELO_MLP, "MLP")
    y_pred_mlp = np.clip(np.expm1(modelo_mlp.predict(X_test_mlp)), 0, None)
    resultados["MLP"] = (np.asarray(y_test_mlp, float), np.asarray(y_pred_mlp, float))

    # ---- SVR (alvo: log1p; inversa expm1; pipeline com SelectKBest) ----
    print("\n>>> SVR")
    svr = importar_modulo(SCRIPT_SVR, "treino_svr")
    dados_svr = svr.carregar_dados()
    X_test_svr, y_test_svr = dados_svr[2], dados_svr[3]
    modelo_svr = carregar_modelo(svr.OUTPUT_DIR, MODELO_SVR, "SVR")
    y_pred_svr = np.clip(np.expm1(modelo_svr.predict(X_test_svr)), 0, None)
    resultados["SVR"] = (np.asarray(y_test_svr, float), np.asarray(y_pred_svr, float))

    return resultados, imp


def figura_predito_vs_real(resultados):
    ordem = ["XGBoost", "SVR", "MLP"]  # mesma ordem da secao de resultados
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    rng = np.random.default_rng(42)

    for ax, nome in zip(axes, ordem):
        y, yp = resultados[nome]
        lim = max(y.max(), yp.max()) + 0.5

        # dispersao com leve jitter horizontal (apenas visual; x e inteiro)
        x_plot = y + rng.uniform(-JITTER, JITTER, size=len(y))
        ax.scatter(x_plot, yp, alpha=0.35, s=18, color="steelblue", edgecolors="none")

        # reta y = x (predicao perfeita)
        ax.plot([0, lim], [0, lim], "r--", lw=1.3, label="Predicao perfeita")

        # reta de tendencia (regressao linear dos pontos)
        if len(y) > 1:
            a, b = np.polyfit(y, yp, 1)
            ax.plot([0, lim], [b, a * lim + b], color="darkorange", lw=1.3,
                    label=f"Tendencia (incl. {a:.2f})")

        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{nome}  (R\u00b2 = {r2_score(y, yp):.2f})")
        ax.set_xlabel("Contagem real")
        if nome == ordem[0]:
            ax.set_ylabel("Contagem predita")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    caminho = os.path.join(DIR_FIGURAS, "fig_predito_vs_real.png")
    fig.savefig(caminho, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ok] Figura salva: {caminho}")
    return caminho


def figura_importancia_xgb(imp):
    df = imp.head(N_TOP_FEATURES).iloc[::-1]  # inverte para o maior ficar no topo
    fig, ax = plt.subplots(figsize=(8, 6))
    cores = plt.cm.viridis(np.linspace(0.15, 0.9, len(df)))
    ax.barh(df["feature"], df["gain"], color=cores)
    ax.set_xlabel("Importancia (ganho)")
    ax.set_title(f"XGBoost \u2014 {N_TOP_FEATURES} atributos de maior ganho")
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["gain"], i, f"  {row['gain']:.3f}", va="center", fontsize=8)
    ax.margins(x=0.12)
    fig.tight_layout()
    caminho = os.path.join(DIR_FIGURAS, "fig_importancia_xgb.png")
    fig.savefig(caminho, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] Figura salva: {caminho}")
    return caminho


def main():
    resultados, imp = obter_predicoes()

    print("\n" + "=" * 60)
    print("  Resumo (conjunto de teste) — confira contra os JSONs")
    print("=" * 60)
    for nome, (y, yp) in resultados.items():
        print(f"  {nome:<8} R\u00b2 = {r2_score(y, yp):.3f}   n = {len(y)}")

    figura_predito_vs_real(resultados)
    figura_importancia_xgb(imp)
    print("\nConcluido.")


if __name__ == "__main__":
    main()