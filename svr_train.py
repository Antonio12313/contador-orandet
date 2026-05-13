
import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
DATASET_DIR = "dataset_preparado_v9"
OUTPUT_DIR  = "./modelos_svr"

# Arquivos gerados pelo script de extração ajustado ao estilo Maldonado
TRAIN_CSV   = os.path.join(DATASET_DIR, "orandet_v9_train_norm.csv")
TEST_CSV    = os.path.join(DATASET_DIR, "orandet_v9_test_norm.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colunas que NÃO são features — só metadados
COLUNAS_META = ["image_id", "file_name", "split", "contagem", "augmentacao", "contagem_log", "contagem_total"]

# Número de features a selecionar antes do SVR.
# Com o pipeline novo, o conjunto de features ficou mais rico.
# Mantém a busca restrita o suficiente para evitar dimensionalidade excessiva,
# mas com candidatos mais adequados ao volume atual de colunas.
K_FEATURES_PADRAO = 160


# ─────────────────────────────────────────────
# 1. CARREGAMENTO DOS DADOS
# ─────────────────────────────────────────────

def carregar_dados():
    print("\n[1/6] Carregando dados normalizados...")

    df_train_full = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    assert isinstance(df_train_full, pd.DataFrame)
    assert isinstance(df_test, pd.DataFrame)

    print(f"  Treino (com aug): {len(df_train_full)} amostras")
    print(f"  Teste:            {len(df_test)} amostras")

    # Separa features de metadados
    feat_cols = [c for c in df_train_full.columns if c not in COLUNAS_META]
    print(f"  Features disponíveis: {len(feat_cols)}")

    # X e y do treino (inclui augmentadas)
    X_train = df_train_full[feat_cols].values
    y_train = df_train_full["contagem"].values.astype(np.float64)

    # X e y do teste (apenas originais)
    X_test  = df_test[feat_cols].values
    y_test  = df_test["contagem"].values.astype(np.float64)

    # Metadados do teste para análise por imagem
    meta_test = df_test[["image_id", "file_name", "contagem"]].copy()

    print(f"\n  Distribuição de contagens — TREINO (originais):")
    df_orig = df_train_full[df_train_full["augmentacao"] == "original"]
    print(
        f"    min={df_orig['contagem'].min():.0f}  "
        f"max={df_orig['contagem'].max():.0f}  "
        f"média={df_orig['contagem'].mean():.1f}  "
        f"mediana={df_orig['contagem'].median():.1f}"
    )

    print(f"  Distribuição de contagens — TESTE:")
    print(
        f"    min={df_test['contagem'].min():.0f}  "
        f"max={df_test['contagem'].max():.0f}  "
        f"média={df_test['contagem'].mean():.1f}  "
        f"mediana={df_test['contagem'].median():.1f}"
    )

    # Sanity check: NaN / Inf residuais
    nan_train = np.isnan(X_train).sum() + np.isinf(X_train).sum()
    nan_test  = np.isnan(X_test).sum()  + np.isinf(X_test).sum()
    if nan_train > 0 or nan_test > 0:
        print(f"  [AVISO] NaN/Inf residuais: treino={nan_train}, teste={nan_test}")
        print(f"  Substituindo pela mediana do treino como fallback...")
        medianas   = np.nanmedian(X_train, axis=0)
        idx_nan_tr = np.where(np.isnan(X_train) | np.isinf(X_train))
        idx_nan_te = np.where(np.isnan(X_test)  | np.isinf(X_test))
        X_train[idx_nan_tr] = medianas[idx_nan_tr[1]]
        X_test[idx_nan_te]  = medianas[idx_nan_te[1]]

    return X_train, y_train, X_test, y_test, meta_test, feat_cols


# ─────────────────────────────────────────────
# 2. PIPELINE: SelectKBest + SVR
# ─────────────────────────────────────────────

def construir_pipeline(k_features=K_FEATURES_PADRAO):
    return Pipeline([
        ("selector", SelectKBest(score_func=f_regression, k=k_features)),
        ("svr",      SVR()),
    ])


# ─────────────────────────────────────────────
# 3. GRID SEARCH — busca os melhores hiperparâmetros
# ─────────────────────────────────────────────

def _candidatos_k(n_features):
    candidatos = [80, 120, 160, 240, 320]
    candidatos = [k for k in candidatos if k < n_features]
    if not candidatos:
        candidatos = [max(1, min(100, n_features - 1))]
    return candidatos


def grid_search(pipeline, X_train, y_train):
    print("\n[3/6] Grid Search com validação cruzada (5-fold)...")
    print("      Isso pode demorar alguns minutos...\n")

    n_features = X_train.shape[1]
    candidatos_k = _candidatos_k(n_features)

    param_grid = [
        # Kernel RBF — principal candidato
        {
            "selector__k":   candidatos_k,
            "svr__kernel":   ["rbf"],
            "svr__C":        [1, 10, 100],
            "svr__epsilon":  [0.5, 1.0, 2.0],
            "svr__gamma":    ["scale", "auto"],
        },
        # Kernel Linear — baseline
        {
            "selector__k":  candidatos_k,
            "svr__kernel":  ["linear"],
            "svr__C":       [0.1, 1, 10],
            "svr__epsilon": [0.5, 1.0, 2.0],
        },
        # Kernel Poly — captura interações
        {
            "selector__k":   [k for k in candidatos_k if k <= max(80, n_features // 2)] or candidatos_k[:2],
            "svr__kernel":   ["poly"],
            "svr__C":        [1, 10],
            "svr__epsilon":  [0.5, 1.0],
            "svr__degree":   [2, 3],
            "svr__gamma":    ["scale"],
            "svr__coef0":    [0, 1],
        },
    ]

    gs = GridSearchCV(
        estimator  = pipeline,
        param_grid = param_grid,
        scoring    = "neg_mean_absolute_error",
        cv         = 5,
        n_jobs     = -1,
        refit      = True,
        verbose    = 1,
        return_train_score = True,
    )

    # Grid Search somente com imagens originais, evitando leakage entre augmentações
    df_train_full = pd.read_csv(TRAIN_CSV)
    df_orig      = df_train_full[df_train_full["augmentacao"] == "original"].copy()
    feat_cols    = [c for c in df_orig.columns if c not in COLUNAS_META]
    X_orig       = df_orig[feat_cols].values
    y_orig       = df_orig["contagem"].values.astype(np.float64)

    print(f"  Grid Search em {len(X_orig)} amostras originais (sem augmentadas)")
    total_configs = (
        len(candidatos_k) * 3 * 3 * 2 +
        len(candidatos_k) * 3 * 3 +
        max(1, len([k for k in candidatos_k if k <= max(80, n_features // 2)] or candidatos_k[:2])) * 2 * 2 * 2 * 1 * 2
    )
    print(f"  Features totais: {n_features}")
    print(f"  Candidatos de k: {candidatos_k}")
    print(f"  Total aproximado de combinações: {total_configs} configs × 5 folds\n")

    gs.fit(X_orig, y_orig)

    print(f"\n  Melhor MAE (CV): {-gs.best_score_:.4f} laranjas")
    print(f"  Melhores params: {gs.best_params_}")

    return gs


# ─────────────────────────────────────────────
# 4. TREINO FINAL com os melhores hiperparâmetros
# ─────────────────────────────────────────────

def treino_final(gs, X_train, y_train):
    print("\n[4/6] Treino final com todos os dados de treino (aug incluída)...")

    melhor_pipeline = gs.best_estimator_
    melhor_pipeline.fit(X_train, y_train)

    print(f"  Modelo treinado com {len(X_train)} amostras (originais + augmentadas)")
    return melhor_pipeline


# ─────────────────────────────────────────────
# 5. AVALIAÇÃO NO TESTE
# ─────────────────────────────────────────────

def avaliar(modelo, X_test, y_test, meta_test):
    print("\n[5/6] Avaliando no conjunto de teste...")

    y_pred_raw = modelo.predict(X_test)

    # Pós-processamento: contagem não pode ser negativa
    y_pred_clip    = np.clip(y_pred_raw, 0, None)
    y_pred_int     = np.round(y_pred_clip).astype(int)

    mae   = mean_absolute_error(y_test, y_pred_clip)
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred_clip))
    r2    = r2_score(y_test, y_pred_clip)
    mape  = np.mean(np.abs((y_test - y_pred_clip) / (y_test + 1e-7))) * 100

    mae_int  = mean_absolute_error(y_test, y_pred_int)
    rmse_int = np.sqrt(mean_squared_error(y_test, y_pred_int))
    r2_int   = r2_score(y_test, y_pred_int)

    print(f"\n  ── Métricas (predição contínua) ──────────────────────")
    print(f"  MAE:   {mae:.4f}  laranjas/imagem")
    print(f"  RMSE:  {rmse:.4f} laranjas/imagem")
    print(f"  R²:    {r2:.4f}")
    print(f"  MAPE:  {mape:.2f}%")

    print(f"\n  ── Métricas (predição arredondada para inteiro) ──────")
    print(f"  MAE:   {mae_int:.4f}  laranjas/imagem")
    print(f"  RMSE:  {rmse_int:.4f} laranjas/imagem")
    print(f"  R²:    {r2_int:.4f}")

    print(f"\n  ── Erro médio por faixa de contagem ──────────────────")
    faixas = [(0, 5, "0–5"), (6, 15, "6–15"), (16, 30, "16–30"), (31, 999, "31+")]
    for lo, hi, label in faixas:
        idx = np.where((y_test >= lo) & (y_test <= hi))[0]
        if len(idx) > 0:
            mae_f = mean_absolute_error(y_test[idx], y_pred_clip[idx])
            print(f"    [{label:>6} laranjas]  n={len(idx):>4}  MAE={mae_f:.3f}")

    resultado = meta_test.copy().reset_index(drop=True)
    resultado["pred_continua"]  = y_pred_clip
    resultado["pred_inteiro"]   = y_pred_int
    resultado["erro_absoluto"]  = np.abs(y_test - y_pred_clip)
    resultado["erro_relativo%"] = np.abs(y_test - y_pred_clip) / (y_test + 1e-7) * 100

    metricas = {
        "mae_continuo":  round(float(mae),      4),
        "rmse_continuo": round(float(rmse),     4),
        "r2_continuo":   round(float(r2),       4),
        "mape":          round(float(mape),     2),
        "mae_inteiro":   round(float(mae_int),  4),
        "rmse_inteiro":  round(float(rmse_int), 4),
        "r2_inteiro":    round(float(r2_int),   4),
        "n_test":        int(len(y_test)),
    }

    return resultado, metricas


# ─────────────────────────────────────────────
# 6. ANÁLISE DE FEATURES SELECIONADAS
# ─────────────────────────────────────────────

def analisar_features(modelo, feat_cols):
    """Mostra quais features foram selecionadas pelo SelectKBest."""
    selector = modelo.named_steps["selector"]
    mask     = selector.get_support()
    scores   = selector.scores_

    feats_selecionadas = [(feat_cols[i], scores[i])
                          for i in range(len(feat_cols)) if mask[i]]
    feats_selecionadas.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  ── Top 20 features selecionadas (por F-score) ────────")
    for nome, score in feats_selecionadas[:20]:
        print(f"    {nome:<45} F={score:.1f}")

    # Contagem por grupo de features — atualizada para o pipeline novo
    grupos = {
        "G1_hsv":        "hsv_",
        "G2_rgb_lab":    ["rgb_", "lab_", "ycbcr_"],
        "G3_v_eq":       ["v_original", "v_eq"],
        "G4_basrelief":  ["sobel_", "laplace", "basrelief", "textura_"],
        "G5_gabor":      "gabor_",
        "G6_lbp":        "lbp_",
        "G7_glcm":       "glcm_",
        "G8_satd":       "satd_",
        "G9_hog":        "hog_",
        "G10_geom_mser": "geom_",
        "G11_hough":     "hough_",
        "G12_grade":     "grade_",
        "G13_escala":    "escala_",
        "G14_contagem":  "cnt_",
        "G_mascara":     "mascara_",
    }

    print(f"\n  ── Features selecionadas por grupo ───────────────────")
    nomes_sel = [f for f, _ in feats_selecionadas]
    for grupo, prefixo in grupos.items():
        if isinstance(prefixo, list):
            n = sum(1 for f in nomes_sel if any(f.startswith(p) for p in prefixo))
        else:
            n = sum(1 for f in nomes_sel if f.startswith(prefixo))
        if n > 0:
            print(f"    {grupo:<20} {n:>3} features")

    return feats_selecionadas


# ─────────────────────────────────────────────
# 7. SALVAR ARTEFATOS
# ─────────────────────────────────────────────

def salvar(modelo, gs, resultado, metricas, feats_selecionadas, feat_cols):
    print("\n[6/6] Salvando artefatos...")

    joblib.dump(modelo, os.path.join(OUTPUT_DIR, "svr_modelo.joblib"))
    print("  svr_modelo.joblib")

    resultado.to_csv(
        os.path.join(OUTPUT_DIR, "svr_predicoes_teste.csv"), index=False
    )
    print("  svr_predicoes_teste.csv")

    df_feats = pd.DataFrame(feats_selecionadas, columns=["feature", "f_score"])
    df_feats.to_csv(
        os.path.join(OUTPUT_DIR, "svr_features_selecionadas.csv"), index=False
    )
    print("  svr_features_selecionadas.csv")

    df_gs = pd.DataFrame(gs.cv_results_)
    df_gs.sort_values("rank_test_score").to_csv(
        os.path.join(OUTPUT_DIR, "svr_grid_search_resultados.csv"), index=False
    )
    print("  svr_grid_search_resultados.csv")

    relatorio = {
        "gerado_em":      datetime.now().strftime("%d/%m/%Y %H:%M"),
        "modelo":         "SVR (sklearn)",
        "dataset":        "OranDet (Embrapa eContaFruto) — features v6 ajustadas ao estilo Maldonado",
        "pipeline":       "SelectKBest(f_regression) → SVR",
        "melhores_params": gs.best_params_,
        "mae_cv":         round(float(-gs.best_score_), 4),
        "metricas_teste": metricas,
        "n_features_entrada":    len(feat_cols),
        "n_features_selecionadas": len(feats_selecionadas),
        "top10_features":  [f for f, _ in feats_selecionadas[:10]],
        "pos_processamento": "clip(0, inf) + round para inteiro",
    }
    with open(os.path.join(OUTPUT_DIR, "svr_relatorio.json"), "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    print("  svr_relatorio.json")

    return relatorio


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "═" * 65)
    print("  SVR — CONTAGEM DE LARANJAS | OranDet v6")
    print("  Pipeline: SelectKBest(f_regression, k) → SVR(kernel)")
    print("  Dataset: features ajustadas ao estilo Maldonado")
    print("═" * 65)

    # 1. Carrega dados
    X_train, y_train, X_test, y_test, meta_test, feat_cols = carregar_dados()

    # 2. Constrói pipeline
    print("\n[2/6] Construindo pipeline SelectKBest → SVR...")
    pipeline = construir_pipeline()
    print(f"  Pipeline: SelectKBest(k={K_FEATURES_PADRAO}) → SVR")
    print(f"  Busca de hiperparâmetros: GridSearchCV 5-fold")

    # 3. Grid Search (só em originais para evitar leakage de CV)
    gs = grid_search(pipeline, X_train, y_train)

    # 4. Treino final (com augmentadas)
    modelo_final = treino_final(gs, X_train, y_train)

    # 5. Avaliação no teste
    resultado, metricas = avaliar(modelo_final, X_test, y_test, meta_test)

    # 6. Análise de features
    feats_sel = analisar_features(modelo_final, feat_cols)

    # 7. Salva tudo
    relatorio = salvar(modelo_final, gs, resultado, metricas, feats_sel, feat_cols)

    print(f"\n{'═'*65}")
    print(f"  RESULTADO FINAL — SVR no conjunto de teste")
    print(f"{'─'*65}")
    print(f"  Melhores hiperparâmetros:")
    for k, v in gs.best_params_.items():
        print(f"    {k:<25} = {v}")
    print(f"\n  MAE  (CV treino):          {-gs.best_score_:.4f} laranjas")
    print(f"  MAE  (teste, contínuo):    {metricas['mae_continuo']:.4f} laranjas/img")
    print(f"  RMSE (teste, contínuo):    {metricas['rmse_continuo']:.4f} laranjas/img")
    print(f"  R²   (teste, contínuo):    {metricas['r2_continuo']:.4f}")
    print(f"  MAE  (teste, arredondado): {metricas['mae_inteiro']:.4f} laranjas/img")
    print(f"\n  Arquivos salvos em: {OUTPUT_DIR}")
    print(f"    svr_modelo.joblib")
    print(f"    svr_predicoes_teste.csv")
    print(f"    svr_features_selecionadas.csv")
    print(f"    svr_grid_search_resultados.csv")
    print(f"    svr_relatorio.json")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
