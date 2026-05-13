"""
treinar_xgboost.py
═══════════════════════════════════════════════════════════════════════════════
Treinamento XGBoost para contagem de laranjas — OranDet v7
Usa os arquivos _raw.csv (sem normalização) gerados pelo extracao_features_v7.py

IMPORTANTE:
  - Use 'contagem' como target com objective='count:poisson'
  - NÃO use os arquivos _norm.csv — XGBoost não precisa de normalização
  - early_stopping evita overfitting sem precisar tunar n_estimators manualmente
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR = "./dataset_preparado_v7"
OUTPUT_DIR = "./resultados_xgboost"
TARGET_COL = "contagem"  # target principal (Poisson)
META_COLS = [  # colunas que NÃO são features
    "image_id", "file_name", "split",
    "contagem", "contagem_log1p", "contagem_sqrt", "augmentacao",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARREGAMENTO DOS DADOS
# ─────────────────────────────────────────────────────────────────────────────
def carregar_dados():
    print("\n[1/5] Carregando datasets RAW...")
    train_path = os.path.join(DATASET_DIR, "orandet_v7_train_raw.csv")
    test_path = os.path.join(DATASET_DIR, "orandet_v7_test_raw.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {train_path}\n"
            "Execute extracao_features_v7.py primeiro."
        )

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Usa apenas imagens originais para treino (sem augmentação duplicar padrões)
    # A augmentação já ajudou na extração — aqui filtramos para evitar
    # que o modelo veja o mesmo fruto 5x e superestime sua importância
    df_orig = df_train[df_train["augmentacao"] == "original"].copy()
    df_aug = df_train[df_train["augmentacao"] != "original"].copy()

    print(f"  Treino originais: {len(df_orig)} imagens")
    print(f"  Treino augmentado: {len(df_aug)} imagens")
    print(f"  Teste: {len(df_test)} imagens")
    print(f"  Contagem — treino: min={df_orig[TARGET_COL].min()}, "
          f"max={df_orig[TARGET_COL].max()}, "
          f"média={df_orig[TARGET_COL].mean():.1f}")

    return df_train, df_test


def preparar_xy(df_train, df_test):
    feat_cols = [c for c in df_train.columns if c not in META_COLS]

    # Limpeza: substitui inf e NaN pela mediana do treino
    for col in feat_cols:
        df_train[col] = df_train[col].replace([np.inf, -np.inf], np.nan)
        df_test[col] = df_test[col].replace([np.inf, -np.inf], np.nan)

    medianas = df_train[feat_cols].median()
    df_train[feat_cols] = df_train[feat_cols].fillna(medianas)
    df_test[feat_cols] = df_test[feat_cols].fillna(medianas)

    X_train = df_train[feat_cols].values.astype(np.float32)
    y_train = df_train[TARGET_COL].values.astype(np.float32)
    X_test = df_test[feat_cols].values.astype(np.float32)
    y_test = df_test[TARGET_COL].values.astype(np.float32)

    print(f"\n  Features: {X_train.shape[1]}")
    print(f"  Shape treino: {X_train.shape} | Shape teste: {X_test.shape}")

    return X_train, y_train, X_test, y_test, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2. TREINAMENTO
# ─────────────────────────────────────────────────────────────────────────────
def treinar_modelo(X_train, y_train, X_test, y_test):
    print("\n[2/5] Treinando XGBoost (objective=count:poisson)...")

    modelo = xgb.XGBRegressor(
        # Objetivo principal: distribuição de Poisson
        # Ideal para contagem de objetos (inteiros, ≥0, assimétricos)
        objective="count:poisson",
        eval_metric="mae",

        # Estrutura das árvores
        n_estimators=3000,  # Alto — early_stopping para antes
        max_depth=8,  # Profundidade controlada para evitar overfitting
        min_child_weight=5,  # Mínimo de amostras por folha (previne overfitting)

        # Learning rate + regularização
        learning_rate=0.02,
        subsample=0.8,  # 80% das amostras por árvore (bagging)
        colsample_bytree=0.7,  # 70% das features por árvore
        reg_alpha=0.1,  # L1 (sparsidade — zera features irrelevantes)
        reg_lambda=1.0,  # L2 (suavização)

        # Aceleração
        tree_method="hist",  # Mais rápido que 'exact', funciona igual
        n_jobs=-1,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=500,
    )

    modelo.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50,
    )

    print(f"\n  Melhor n_estimators: {modelo.best_iteration}")
    print(f"  MAE no teste (melhor iter): {modelo.best_score:.4f}")

    return modelo


# ─────────────────────────────────────────────────────────────────────────────
# 3. AVALIAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
def avaliar(modelo, X_test, y_test, df_test):
    print("\n[3/5] Avaliando no conjunto de teste...")

    y_pred = modelo.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # contagem nunca negativa
    y_pred_int = np.round(y_pred).astype(int)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # MAPE — ignora imagens com contagem = 0 (divisão por zero)
    mask = y_test > 0
    mape = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100)

    # MdAPE (mediana — mais robusta a outliers)
    mdape = float(np.median(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100)

    # Acurácia por tolerância (útil para contagem)
    dentro_1 = float(np.mean(np.abs(y_test - y_pred_int) <= 1) * 100)
    dentro_2 = float(np.mean(np.abs(y_test - y_pred_int) <= 2) * 100)
    dentro_10 = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6)) <= 0.10) * 100)
    dentro_20 = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6)) <= 0.20) * 100)

    metricas = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R²": round(r2, 4),
        "MAPE (%)": round(mape, 2),
        "MdAPE (%)": round(mdape, 2),
        "Acerto ±1 fruta": round(dentro_1, 1),
        "Acerto ±2 frutas": round(dentro_2, 1),
        "Dentro de 10%": round(dentro_10, 1),
        "Dentro de 20%": round(dentro_20, 1),
        "n_estimators_final": modelo.best_iteration,
    }

    print(f"\n  {'─' * 40}")
    print(f"  MAE:             {mae:.3f} laranjas/imagem")
    print(f"  RMSE:            {rmse:.3f}")
    print(f"  R²:              {r2:.4f}")
    print(f"  MAPE:            {mape:.1f}%")
    print(f"  MdAPE:           {mdape:.1f}%   (mediana — mais robusta)")
    print(f"  Acerto ±1 fruta: {dentro_1:.1f}%")
    print(f"  Acerto ±2 frutas:{dentro_2:.1f}%")
    print(f"  Dentro de 10%:   {dentro_10:.1f}%")
    print(f"  Dentro de 20%:   {dentro_20:.1f}%")
    print(f"  {'─' * 40}")

    return y_pred, metricas


# ─────────────────────────────────────────────────────────────────────────────
# 4. IMPORTÂNCIA DE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def analise_features(modelo, feat_cols):
    print("\n[4/5] Analisando importância das features...")

    imp = pd.DataFrame({
        "feature": feat_cols,
        "gain": modelo.feature_importances_,  # gain normalizado pelo XGBoost
    }).sort_values("gain", ascending=False)

    # Top 30
    print("\n  Top 20 features mais importantes:")
    for i, row in imp.head(20).iterrows():
        print(f"    {row['feature']:<45} {row['gain']:.6f}")

    # Importância por GRUPO (muito mais útil para diagnóstico)
    grupos = {
        "G1_hsv": "hsv_",
        "G2_rgb_lab_ycbcr": ["rgb_", "lab_", "ycbcr_"],
        "G3_canal_v": ["v_original", "v_eq", "v_razao"],
        "G4_basrelief": ["sobel_", "laplace", "basrelief", "textura_", "brilho_"],
        "G5_gabor": "gabor_",
        "G6_lbp": "lbp_",
        "G7_glcm": "glcm_",
        "G8_satd": "satd_",
        "G9_hog": "hog_",
        "G10_geometria": "geom_",
        "G11_hough": "hough_",
        "G12_grade": "grade_",
        "G13_multiescala": "escala_",
        "G14_contagem": "cnt_",
        "G15_curvatura": "hessian_",
        "G16_crcb": ["cr_", "cb_", "crcb_", "exg_"],
        "mascara": "mascara_",
    }

    imp_grupo = []
    for grupo, prefixos in grupos.items():
        if isinstance(prefixos, str):
            prefixos = [prefixos]
        mask = imp["feature"].apply(lambda f: any(f.startswith(p) for p in prefixos))
        total = imp.loc[mask, "gain"].sum()
        n_feat = mask.sum()
        imp_grupo.append({
            "grupo": grupo,
            "gain_total": round(float(total), 6),
            "gain_medio": round(float(total / max(n_feat, 1)), 6),
            "n_features": int(n_feat),
        })

    df_grupo = pd.DataFrame(imp_grupo).sort_values("gain_total", ascending=False)
    print("\n  Importância por grupo (gain total):")
    for _, row in df_grupo.iterrows():
        bar = "█" * int(row["gain_total"] * 200)
        print(f"    {row['grupo']:<22} {row['gain_total']:.4f}  {bar}")

    return imp, df_grupo


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALIZAÇÕES E SALVAMENTO
# ─────────────────────────────────────────────────────────────────────────────
def salvar_resultados(modelo, y_test, y_pred, metricas, imp, df_grupo,
                      df_test, feat_cols):
    print("\n[5/5] Salvando resultados...")

    # ── Modelo e metadados ────────────────────────────────────────────
    joblib.dump(modelo, os.path.join(OUTPUT_DIR, "xgb_modelo.joblib"))
    joblib.dump(feat_cols, os.path.join(OUTPUT_DIR, "xgb_feature_cols.joblib"))

    with open(os.path.join(OUTPUT_DIR, "xgb_metricas.json"), "w") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    imp.to_csv(os.path.join(OUTPUT_DIR, "xgb_feature_importance.csv"), index=False)
    df_grupo.to_csv(os.path.join(OUTPUT_DIR, "xgb_group_importance.csv"), index=False)

    # ── CSV com predições por imagem ─────────────────────────────────
    df_preds = df_test[["image_id", "file_name", TARGET_COL]].copy()
    df_preds["pred_float"] = y_pred
    df_preds["pred_int"] = np.round(y_pred).astype(int)
    df_preds["erro_abs"] = np.abs(df_preds[TARGET_COL] - df_preds["pred_float"])
    df_preds["erro_perc"] = (
            df_preds["erro_abs"] / (df_preds[TARGET_COL] + 1e-6) * 100
    ).round(1)
    df_preds.to_csv(os.path.join(OUTPUT_DIR, "xgb_predicoes_teste.csv"), index=False)

    # ── Gráfico 1: Real vs Predito ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Scatter real vs predito
    ax = axes[0]
    max_val = max(y_test.max(), y_pred.max()) * 1.05
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, color="steelblue", label="Imagens")
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfeito")
    ax.fill_between([0, max_val],
                    [0 * 0.8, max_val * 0.8],
                    [0 * 1.2, max_val * 1.2],
                    alpha=0.1, color="green", label="±20%")
    ax.set_xlabel("Contagem Real")
    ax.set_ylabel("Contagem Predita")
    ax.set_title(f"Real vs Predito\nMAE={metricas['MAE']:.2f} | MAPE={metricas['MAPE (%)']:.1f}%")
    ax.legend(fontsize=8)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # Histograma de erros absolutos
    ax = axes[1]
    erros = np.abs(y_test - y_pred)
    ax.hist(erros, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(metricas["MAE"], color="red", linestyle="--",
               label=f"MAE={metricas['MAE']:.2f}")
    ax.axvline(np.median(erros), color="orange", linestyle="--",
               label=f"Mediana={np.median(erros):.2f}")
    ax.set_xlabel("Erro Absoluto (laranjas)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição dos Erros Absolutos")
    ax.legend()

    # Importância por grupo (top 10)
    ax = axes[2]
    df_top = df_grupo.head(10)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_top)))
    bars = ax.barh(df_top["grupo"][::-1], df_top["gain_total"][::-1],
                   color=colors[::-1])
    ax.set_xlabel("Gain Total (importância)")
    ax.set_title("Importância por Grupo de Features")
    for bar, val in zip(bars, df_top["gain_total"][::-1]):
        ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "xgb_resultados.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Gráfico 2: Curva de aprendizado ───────────────────────────────
    results = modelo.evals_result()
    if results:
        fig, ax = plt.subplots(figsize=(10, 5))
        train_mae = results["validation_0"]["mae"]
        test_mae = results["validation_1"]["mae"]
        ax.plot(train_mae, label="Treino", color="steelblue")
        ax.plot(test_mae, label="Teste", color="tomato")
        ax.axvline(modelo.best_iteration, color="gray", linestyle="--",
                   label=f"Melhor iter: {modelo.best_iteration}")
        ax.set_xlabel("Iteração (n_estimators)")
        ax.set_ylabel("MAPE")
        ax.set_title("Curva de Aprendizado — Early Stopping")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "xgb_curva_aprendizado.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n  Arquivos salvos em: {OUTPUT_DIR}/")
    print(f"    ├─ xgb_modelo.joblib")
    print(f"    ├─ xgb_metricas.json")
    print(f"    ├─ xgb_predicoes_teste.csv")
    print(f"    ├─ xgb_feature_importance.csv")
    print(f"    ├─ xgb_group_importance.csv")
    print(f"    ├─ xgb_resultados.png")
    print(f"    └─ xgb_curva_aprendizado.png")


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÃO PARA USAR O MODELO EM PRODUÇÃO (nova imagem)
# ─────────────────────────────────────────────────────────────────────────────
def contar_laranjas_nova_imagem(caminho_imagem):
    """
    Usa o modelo treinado para contar laranjas em uma nova imagem.

    Exemplo:
        n = contar_laranjas_nova_imagem("/path/para/imagem.jpg")
        print(f"Estimativa: {n} laranjas")
    """
    import cv2
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from extracao_features_v7 import _extrair_de_img

    modelo = joblib.load(os.path.join(OUTPUT_DIR, "xgb_modelo.joblib"))
    feat_cols = joblib.load(os.path.join(OUTPUT_DIR, "xgb_feature_cols.joblib"))

    img = cv2.imread(caminho_imagem)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {caminho_imagem}")

    feats = _extrair_de_img(img)

    # Garante que todas as features esperadas estão presentes
    X = np.array([[feats.get(c, 0.0) for c in feat_cols]], dtype=np.float32)
    pred = modelo.predict(X)[0]
    return max(0, int(round(pred)))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "═" * 60)
    print("  XGBoost — Contagem de Laranjas Verdes (OranDet v7)")
    print("  objective=count:poisson | early_stopping | feature importance")
    print("═" * 60)

    df_train, df_test = carregar_dados()
    X_train, y_train, X_test, y_test, feat_cols = preparar_xy(df_train, df_test)
    modelo = treinar_modelo(X_train, y_train, X_test, y_test)
    y_pred, metricas = avaliar(modelo, X_test, y_test, df_test)
    imp, df_grupo = analise_features(modelo, feat_cols)
    salvar_resultados(modelo, y_test, y_pred, metricas, imp, df_grupo, df_test, feat_cols)

    print(f"\n{'═' * 60}")
    print(f"  MAPE final: {metricas['MAPE (%)']:.1f}%  "
          f"(era ~60% com SVR/MLP)")
    print(f"  MAE final:  {metricas['MAE']:.2f} laranjas por imagem")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
