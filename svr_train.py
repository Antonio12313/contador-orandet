import os
import json
import time
import warnings
import importlib.metadata
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# CONFIGURAÇÃO
DATASET_DIR = "dataset_preparado_v11"
OUTPUT_DIR = "./resultados_svr_v11"

TRAIN_CSV = os.path.join(DATASET_DIR, "orandet_v11_train_norm.csv")
TEST_CSV = os.path.join(DATASET_DIR, "orandet_v11_test_norm.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

COLUNAS_META = [
    "image_id", "file_name", "split",
    "contagem", "contagem_log1p", "contagem_sqrt", "augmentacao",
]

TARGET_COL = "contagem_log1p"
TARGET_INVERSA = "expm1"

# Repetições da medição de tempo de inferência — alinhado com XGBoost (N=10)
INFERENCIA_REPS = 10

K_FEATURES_PADRAO = 160

# ─── FAIXAS DE CONTAGEM ────────────────────────────────────────────────────────
# Convenção dos três JSON de resultados: [1], [2], [3-4], [5-7], [8+]
# A faixa [0] é mantida internamente para diagnóstico, mas não entra na
# tabela comparativa entre modelos (alinhamento com XGBoost e MLP).
FAIXAS_DEF = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 999)]
FAIXAS_LABEL = ["0", "1", "2", "3-4", "5-7", "8+"]


# 1. CARREGAMENTO

def carregar_dados():
    print("\n[1/6] Carregando dados normalizados...")

    df_train_full = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    feat_cols = [c for c in df_train_full.columns if c not in COLUNAS_META]
    print(f"  Treino (com aug): {len(df_train_full)} amostras")
    print(f"  Teste:            {len(df_test)} amostras")
    print(f"  Features entrada: {len(feat_cols)}")
    print(f"  Target:           {TARGET_COL}  (inversa: {TARGET_INVERSA})")

    X_train = df_train_full[feat_cols].values
    y_train = df_train_full[TARGET_COL].values.astype(np.float64)
    X_test = df_test[feat_cols].values
    y_test = df_test["contagem"].values.astype(np.float64)  # escala original
    y_test_t = df_test[TARGET_COL].values.astype(np.float64)  # escala log1p
    meta_test = df_test[["image_id", "file_name", "contagem"]].copy()

    print(f"\n  Distribuição (treino — originais):")
    df_orig = df_train_full[df_train_full["augmentacao"] == "original"]
    print(f"    min={df_orig['contagem'].min():.0f}  max={df_orig['contagem'].max():.0f}"
          f"  média={df_orig['contagem'].mean():.1f}  mediana={df_orig['contagem'].median():.1f}")
    print(f"  Distribuição (teste):")
    print(f"    min={df_test['contagem'].min():.0f}  max={df_test['contagem'].max():.0f}"
          f"  média={df_test['contagem'].mean():.1f}  mediana={df_test['contagem'].median():.1f}")

    nan_tr = np.isnan(X_train).sum() + np.isinf(X_train).sum()
    nan_te = np.isnan(X_test).sum() + np.isinf(X_test).sum()
    if nan_tr > 0 or nan_te > 0:
        print(f"  [AVISO] NaN/Inf: treino={nan_tr}, teste={nan_te} — substituindo por mediana")
        medianas = np.nanmedian(X_train, axis=0)
        for X, idx in [(X_train, np.where(np.isnan(X_train) | np.isinf(X_train))),
                       (X_test, np.where(np.isnan(X_test) | np.isinf(X_test)))]:
            X[idx] = medianas[idx[1]]

    return X_train, y_train, X_test, y_test, y_test_t, meta_test, feat_cols


# 2. PIPELINE

def construir_pipeline(k_features=K_FEATURES_PADRAO):
    return Pipeline([
        ("selector", SelectKBest(score_func=f_regression, k=k_features)),
        ("svr", SVR()),
    ])


# 3. GRID SEARCH — apenas originais para evitar leakage de CV entre augmentações

def _candidatos_k(n_features):
    candidatos = [80, 120, 160, 240, 320]
    candidatos = [k for k in candidatos if k < n_features]
    return candidatos or [max(1, min(100, n_features - 1))]


def grid_search(pipeline, X_train, y_train):
    print("\n[3/6] Grid Search com validação cruzada (5-fold, random_state=42)...")

    df_train_full = pd.read_csv(TRAIN_CSV)
    df_orig = df_train_full[df_train_full["augmentacao"] == "original"].copy()
    feat_cols = [c for c in df_orig.columns if c not in COLUNAS_META]
    X_orig = df_orig[feat_cols].values
    y_orig = df_orig[TARGET_COL].values.astype(np.float64)

    n_features = X_train.shape[1]
    candidatos_k = _candidatos_k(n_features)

    param_grid = [
        {
            "selector__k": candidatos_k,
            "svr__kernel": ["rbf"],
            "svr__C": [1, 10, 100],
            "svr__epsilon": [0.5, 1.0, 2.0],
            "svr__gamma": ["scale", "auto"],
        },
        {
            "selector__k": candidatos_k,
            "svr__kernel": ["linear"],
            "svr__C": [0.1, 1, 10],
            "svr__epsilon": [0.5, 1.0, 2.0],
        },
        {
            "selector__k": [k for k in candidatos_k if k <= max(80, n_features // 2)]
                           or candidatos_k[:2],
            "svr__kernel": ["poly"],
            "svr__C": [1, 10],
            "svr__epsilon": [0.5, 1.0],
            "svr__degree": [2, 3],
            "svr__gamma": ["scale"],
            "svr__coef0": [0, 1],
        },
    ]

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=True,
    )

    print(f"  Grid Search em {len(X_orig)} amostras originais (sem augmentadas)")
    print(f"  Scoring: neg_MAE em escala log1p  |  Candidatos k: {candidatos_k}\n")

    _t0 = time.perf_counter()
    gs.fit(X_orig, y_orig)
    tempo_gs = round(time.perf_counter() - _t0, 2)

    print(f"\n  Melhor MAE-CV (escala log1p): {-gs.best_score_:.4f}")
    print(f"  Melhores params: {gs.best_params_}")
    print(f"  Tempo grid search: {tempo_gs}s")

    return gs, param_grid, candidatos_k, tempo_gs


# 4. TREINO FINAL

def treino_final(gs, X_train, y_train):
    print("\n[4/6] Treino final (originais + augmentadas)...")

    _t0 = time.perf_counter()
    melhor_pipeline = gs.best_estimator_
    melhor_pipeline.fit(X_train, y_train)
    tempo_treino = round(time.perf_counter() - _t0, 2)

    print(f"  Treinado com {len(X_train)} amostras | tempo: {tempo_treino}s")
    return melhor_pipeline, tempo_treino


# 5. AVALIAÇÃO — inclui tempo de inferência por imagem (NOVO)

def _medir_inferencia(modelo, X_test, n_repeticoes=INFERENCIA_REPS):
    """
    Mede o tempo médio de inferência por imagem.
    Faz n_repeticoes passagens completas pelo conjunto de teste e
    retorna a mediana (mais robusta a jitter do sistema).
    Metodologia alinhada à do XGBoost (N=10) para comparação direta.

    Retorna
    -------
    tempo_por_imagem_ms : float
        Mediana do tempo de inferência por imagem, em milissegundos.
    tempo_total_ms : float
        Tempo total para processar todo o conjunto de teste (1 passagem).
    """
    n = len(X_test)

    # warmup — evita penalidade de inicialização na primeira chamada
    _ = modelo.predict(X_test[:min(5, n)])

    tempos = []
    for _ in range(n_repeticoes):
        t0 = time.perf_counter()
        modelo.predict(X_test)
        tempos.append(time.perf_counter() - t0)

    mediana_total_s = float(np.median(tempos))
    por_imagem_ms = round(mediana_total_s / n * 1000, 4)
    total_ms = round(mediana_total_s * 1000, 2)

    print(f"\n  Tempo de inferência (mediana de {n_repeticoes} passes):")
    print(f"    Total ({n} imgs):  {total_ms:.2f} ms")
    print(f"    Por imagem:        {por_imagem_ms:.4f} ms")

    return por_imagem_ms, total_ms


def avaliar(modelo, X_train, y_train_log, X_test, y_test, y_test_log, meta_test):
    print("\n[5/6] Avaliando...")

    # ── Tempo de inferência por imagem ────────────────────────────────────────
    tempo_inf_por_img_ms, tempo_inf_total_ms = _medir_inferencia(modelo, X_test)

    # ── Predição no teste ─────────────────────────────────────────────────────
    y_pred_log = modelo.predict(X_test)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    y_pred_int = np.round(y_pred).astype(int)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    mask = y_test > 0
    mape = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100)
    mdape = float(np.median(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100)

    dentro_1 = float(np.mean(np.abs(y_test - y_pred_int) <= 1) * 100)
    dentro_2 = float(np.mean(np.abs(y_test - y_pred_int) <= 2) * 100)
    dentro_20 = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6)) <= 0.20) * 100)

    # ── Predição no treino (gap) ──────────────────────────────────────────────
    y_pred_tr_log = modelo.predict(X_train)
    y_pred_tr = np.clip(np.expm1(y_pred_tr_log), 0, None)
    y_train_orig = np.expm1(y_train_log)

    mae_tr = mean_absolute_error(y_train_orig, y_pred_tr)
    rmse_tr = np.sqrt(mean_squared_error(y_train_orig, y_pred_tr))
    r2_tr = r2_score(y_train_orig, y_pred_tr)
    mask_tr = y_train_orig > 0
    mape_tr = float(np.mean(np.abs(
        (y_train_orig[mask_tr] - y_pred_tr[mask_tr]) / y_train_orig[mask_tr]
    )) * 100)

    print(f"\n  {'Métrica':<20} {'Treino':>10} {'Teste':>10}")
    print(f"  {'─' * 42}")
    print(f"  {'MAE':<20} {mae_tr:>10.3f} {mae:>10.3f}")
    print(f"  {'RMSE':<20} {rmse_tr:>10.3f} {rmse:>10.3f}")
    print(f"  {'R²':<20} {r2_tr:>10.4f} {r2:>10.4f}")
    print(f"  {'MAPE':<20} {mape_tr:>9.1f}% {mape:>9.1f}%")
    print(f"  {'─' * 42}")
    print(f"  MdAPE (teste):      {mdape:.1f}%")
    print(f"  ±1 fruta:           {dentro_1:.1f}%")
    print(f"  ±2 frutas:          {dentro_2:.1f}%")
    print(f"  Dentro de 20%:      {dentro_20:.1f}%")
    print(f"  Inf. por imagem:    {tempo_inf_por_img_ms:.4f} ms")

    # ── Análise por faixa ─────────────────────────────────────────────────────
    # Convenção idêntica à dos outros dois modelos: [1],[2],[3-4],[5-7],[8+]
    # A faixa [0] é incluída apenas para diagnóstico — não entra na tabela
    # comparativa entre modelos (não é ponto de comparação entre SVR/MLP/XGB).
    print(f"\n  MAPE por faixa:")
    print(f"  {'Faixa':<8} {'N':>5} {'MAE':>6} {'MAPE%':>7} {'±1':>7} {'Bias':>7}")
    print(f"  {'─' * 42}")

    metricas_f = []
    for (lo, hi), label in zip(FAIXAS_DEF, FAIXAS_LABEL):
        idx = (y_test >= lo) & (y_test <= hi)
        n = idx.sum()
        if n == 0:
            continue
        yt, yp = y_test[idx], y_pred[idx]
        mae_f = float(np.mean(np.abs(yt - yp)))
        mf = yt > 0
        mape_f = float(np.mean(np.abs((yt[mf] - yp[mf]) / yt[mf])) * 100) if mf.sum() > 0 else 0.0
        a1 = float(np.mean(np.abs(yt - np.round(yp)) <= 1) * 100)
        bias = float(np.mean(yp - yt))
        flag = " ← PROBLEMA" if mape_f > 60 else ""
        print(f"  {label:<8} {n:>5} {mae_f:>6.2f} {mape_f:>6.1f}%  {a1:>5.1f}%  {bias:>+6.2f}{flag}")
        metricas_f.append({
            "faixa": label,
            "n": int(n),
            "mae": round(mae_f, 3),
            "mape": round(mape_f, 1),
            "acerto_pm1": round(a1, 1),
            "bias": round(bias, 3),
        })

    resultado = meta_test.copy().reset_index(drop=True)
    resultado["pred_continua"] = y_pred
    resultado["pred_inteiro"] = y_pred_int
    resultado["erro_absoluto"] = np.abs(y_test - y_pred)
    resultado["erro_relativo_pct"] = np.abs(y_test - y_pred) / (y_test + 1e-6) * 100

    metricas = {
        "treino": {
            "MAE": round(mae_tr, 4),
            "RMSE": round(rmse_tr, 4),
            "R2": round(r2_tr, 4),
            "MAPE": round(mape_tr, 2),
        },
        "teste": {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "MAPE": round(mape, 2),
            "MdAPE": round(mdape, 2),
            "gap_MAPE_MdAPE": round(mape - mdape, 2),
            "acerto_pm1": round(dentro_1, 1),
            "acerto_pm2": round(dentro_2, 1),
            "dentro_20pct": round(dentro_20, 1),
        },
        "gap_treino_teste": {
            "MAE_ratio": round(mae / (mae_tr + 1e-9), 2),
            "MAPE_delta": round(mape - mape_tr, 2),
            "R2_delta": round(r2 - r2_tr, 4),
        },
        "por_faixa": metricas_f,
        # tempos para uso na seção eficiência do JSON
        "_inferencia_ms": {
            "por_imagem": tempo_inf_por_img_ms,
            "total_teste": tempo_inf_total_ms,
        },
    }
    return resultado, metricas


# 6. ANÁLISE DE FEATURES

def analisar_features(modelo, feat_cols):
    selector = modelo.named_steps["selector"]
    mask = selector.get_support()
    scores = selector.scores_

    feats_sel = sorted(
        [(feat_cols[i], scores[i]) for i in range(len(feat_cols)) if mask[i]],
        key=lambda x: x[1], reverse=True,
    )

    print(f"\n  Top 20 features selecionadas (F-score):")
    for nome, score in feats_sel[:20]:
        print(f"    {nome:<45} F={score:.1f}")

    grupos = {
        "G1_hsv": ["hsv_"],
        "G2_rgb_lab_ycbcr": ["rgb_", "lab_", "ycbcr_"],
        "G3_v_eq": ["v_original", "v_eq", "v_razao"],
        "G4_basrelief": ["sobel_", "laplace", "basrelief", "textura_", "brilho_"],
        "G5_gabor": ["gabor_"],
        "G6_lbp": ["lbp_"],
        "G7_glcm": ["glcm_"],
        "G8_satd": ["satd_"],
        "G9_hog": ["hog_"],
        "G10_geometria": ["geom_"],
        "G11_hough": ["hough_"],
        "G12_grade": ["grade_"],
        "G13_multiescala": ["escala_", "multi_"],
        "G14_contagem": ["cnt_"],
        "G15_curvatura": ["hessian_"],
        "G16_crcb": ["cr_", "cb_", "crcb_", "exg_"],
        "G_mascara": ["mascara_"],
        "G_fft": ["fft_"],
        "G_spatial": ["spatial_"],
        "G_derivadas": ["density_", "hog_fft"],
    }

    print(f"\n  Features selecionadas por grupo:")
    nomes_sel = [f for f, _ in feats_sel]
    grupo_counts = {}
    for grupo, prefixos in grupos.items():
        n = sum(1 for f in nomes_sel if any(f.startswith(p) for p in prefixos))
        grupo_counts[grupo] = n
        if n > 0:
            print(f"    {grupo:<22} {n:>3}")

    return feats_sel, grupo_counts


# 7. SALVAMENTO

def salvar(modelo, gs, param_grid, candidatos_k, resultado, metricas,
           feats_sel, grupo_counts, feat_cols,
           tempo_gs, tempo_treino):
    print("\n[6/6] Salvando artefatos...")

    joblib.dump(modelo, os.path.join(OUTPUT_DIR, "svr_v3_modelo.joblib"))
    resultado.to_csv(os.path.join(OUTPUT_DIR, "svr_v3_predicoes_teste.csv"), index=False)

    df_feats = pd.DataFrame(feats_sel, columns=["feature", "f_score"])
    df_feats.to_csv(os.path.join(OUTPUT_DIR, "svr_v3_features_selecionadas.csv"), index=False)

    df_gs = pd.DataFrame(gs.cv_results_)
    df_gs.sort_values("rank_test_score").to_csv(
        os.path.join(OUTPUT_DIR, "svr_v3_grid_search_resultados.csv"), index=False
    )

    try:
        versao_sklearn = importlib.metadata.version("scikit-learn")
    except Exception:
        import sklearn
        versao_sklearn = sklearn.__version__

    cv_results = gs.cv_results_
    best_idx = gs.best_index_
    cv_por_fold = []
    for fold in range(5):
        chave = f"split{fold}_test_score"
        if chave in cv_results:
            cv_por_fold.append({
                "fold": fold + 1,
                "mae_log": round(float(-cv_results[chave][best_idx]), 4),
            })

    mae_cv_scores = [-cv_results[f"split{f}_test_score"][best_idx]
                     for f in range(5)
                     if f"split{f}_test_score" in cv_results]

    # ── extrair tempos de inferência salvos em metricas ───────────────────────
    inf = metricas.pop("_inferencia_ms")  # remove da estrutura interna

    # ── dados de treino para o JSON ───────────────────────────────────────────
    df_tr = pd.read_csv(TRAIN_CSV)
    n_treino_total = int(len(df_tr))
    n_treino_originais = int(len(df_tr[df_tr["augmentacao"] == "original"]))

    saida = {

        # ── Protocolo ─────────────────────────────────────────────────────────
        "protocolo": {
            "dataset_treino": "orandet_v11_train_norm.csv (normalizado)",
            "dataset_teste": "orandet_v11_test_norm.csv  (normalizado)",
            "n_amostras_treino": n_treino_total,
            "n_amostras_treino_originais": n_treino_originais,
            # NOVO: declaração explícita para comparação entre modelos
            "n_amostras_treino_gridsearch": n_treino_originais,
            "n_amostras_teste": 306,
            "treino_final": "Originais + augmentadas (todas as amostras de treino)",
            "n_features_entrada": len(feat_cols),
            # NOVO: n_features_selecionadas é o k escolhido pelo GridSearch
            "n_features_selecionadas": len(feats_sel),
            "nota_features": (
                f"DIFERENÇA DE PIPELINE: o SVR opera em um espaço de features "
                f"REDUZIDO de {len(feats_sel)} features (via SelectKBest com "
                f"f_regression), enquanto XGBoost e MLP usam as {len(feat_cols)} "
            ),
            "alvo_modelo": TARGET_COL,
        },

        # ── Integridade / Leakage ─────────────────────────────────────────────
        # NOVO: seção explícita para auditoria e comparação com XGBoost
        "integridade_experimento": {
            "leakage_teste_no_cv": False,
            "descricao": (
                "O conjunto de teste NÃO foi usado em nenhum momento "
                "durante a busca de hiperparâmetros (GridSearchCV usa apenas "
                "as amostras originais de treino)"
            ),
            "comparabilidade_cv_entre_modelos": (
                "NÃO comparável diretamente: "
                "MLP otimizou MAPE na escala original; "
                "SVR otimizou MAE na escala log1p; "
                "Os CVs servem para diagnóstico interno de cada modelo."
            ),
        },

        # ── Pipeline ──────────────────────────────────────────────────────────
        "pipeline": {
            "etapa_1": "SelectKBest(score_func=f_regression)",
            "etapa_2": "SVR",
            "nota": "SelectKBest usa correlação linear (F-test) como critério de seleção",
        },

        # ── Hiperparâmetros ───────────────────────────────────────────────────
        "hiperparametros_finais": {
            **gs.best_params_,
            "random_state_cv": 42,
        },
        "grade_busca": {
            "candidatos_k": candidatos_k,
            "rbf": {"C": [1, 10, 100], "epsilon": [0.5, 1.0, 2.0], "gamma": ["scale", "auto"]},
            "linear": {"C": [0.1, 1, 10], "epsilon": [0.5, 1.0, 2.0]},
            "poly": {"C": [1, 10], "epsilon": [0.5, 1.0], "degree": [2, 3],
                     "gamma": ["scale"], "coef0": [0, 1]},
            "scoring": "neg_mean_absolute_error (escala log1p)",
            "cv_folds": 5,
            "n_jobs": -1,
        },

        # ── Validação cruzada ─────────────────────────────────────────────────
        "validacao_cruzada": {
            "n_folds": 5,
            "random_state": 42,
            "amostras": "originais apenas",
            "mae_cv_mean_log": round(float(-gs.best_score_), 4),
            "mae_cv_std_log": round(float(np.std(mae_cv_scores)), 4) if mae_cv_scores else None,
            "por_fold_log": cv_por_fold,
            "nota_escala": (
                "MAE em escala log1p — não comparável com MAE do XGBoost "
                "(escala original) nem com MAPE do MLP. "
                "Ver campo integridade_experimento.comparabilidade_cv_entre_modelos."
            ),
        },

        # ── Resultados ────────────────────────────────────────────────────────
        "resultados_treino": metricas["treino"],
        "resultados_teste": metricas["teste"],
        "gap_treino_teste": metricas["gap_treino_teste"],

        # ── Por faixa ─────────────────────────────────────────────────────────
        # Convenção: [1],[2],[3-4],[5-7],[8+] — alinhada com XGBoost e MLP
        # A faixa [0] aparece apenas para diagnóstico interno.
        "resultados_por_faixa": metricas["por_faixa"],
        "nota_faixas": (
            "Faixas [1],[2],[3-4],[5-7],[8+] — convenção adotada nos três modelos. "
            "Faixa [0] incluída apenas para diagnóstico; "
            "viés sistemático esperado: superestimação em [1] e subestimação em [8+] "
            "(regressão à média)."
        ),

        # ── Features ─────────────────────────────────────────────────────────
        "features": {
            "n_entrada": len(feat_cols),
            "n_selecionadas": len(feats_sel),
            "top10": [f for f, _ in feats_sel[:10]],
            "por_grupo": grupo_counts,
        },

        # ── Eficiência ────────────────────────────────────────────────────────
        # NOVO: tempo de inferência por imagem (mediana de 5 passes, com warmup)
        "eficiencia": {
            "tempo_grid_search_s": tempo_gs,
            "tempo_treino_final_s": tempo_treino,
            "inferencia_por_imagem_ms": inf["por_imagem"],
            "inferencia_total_teste_ms": inf["total_teste"],
            "n_imagens_teste": 306,
            "metodo_medicao_inferencia": (
                f"mediana de {INFERENCIA_REPS} passagens completas pelo conjunto "
                "de teste, com warmup de 5 amostras antes das medições "
                "(N alinhado com o XGBoost para comparação direta)"
            ),
            "versao_sklearn": versao_sklearn,
        },
    }

    with open(os.path.join(OUTPUT_DIR, "svr_v11_relatorio.json"), "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)

    print(f"  Arquivos em: {OUTPUT_DIR}/")
    print(f"    ├─ svr_v3_modelo.joblib")
    print(f"    ├─ svr_v11_relatorio.json")
    print(f"    ├─ svr_v3_predicoes_teste.csv")
    print(f"    ├─ svr_v3_features_selecionadas.csv")
    print(f"    └─ svr_v3_grid_search_resultados.csv")

    return saida


# MAIN

def main():
    print("  SVR v3 — CONTAGEM DE LARANJAS | OranDet v11")
    print("  Pipeline: SelectKBest(f_regression) → SVR")
    print("  Target: contagem_log1p  |  Inversa: expm1")

    X_train, y_train, X_test, y_test, y_test_log, meta_test, feat_cols = carregar_dados()

    print("\n[2/6] Construindo pipeline...")
    pipeline = construir_pipeline()

    gs, param_grid, candidatos_k, tempo_gs = grid_search(pipeline, X_train, y_train)

    modelo_final, tempo_treino = treino_final(gs, X_train, y_train)

    resultado, metricas = avaliar(
        modelo_final,
        X_train, y_train,
        X_test, y_test, y_test_log,
        meta_test,
    )

    feats_sel, grupo_counts = analisar_features(modelo_final, feat_cols)

    relatorio = salvar(
        modelo_final, gs, param_grid, candidatos_k,
        resultado, metricas,
        feats_sel, grupo_counts, feat_cols,
        tempo_gs, tempo_treino,
    )

    print(f"\n{'═' * 65}")
    print(f"  Melhores hiperparâmetros:")
    for k, v in gs.best_params_.items():
        print(f"    {k:<25} = {v}")
    print(f"\n  MAE CV (log1p):         {-gs.best_score_:.4f}")
    print(f"  MAE treino:             {metricas['treino']['MAE']:.4f} laranjas/img")
    print(f"  MAE teste:              {metricas['teste']['MAE']:.4f} laranjas/img")
    print(f"  RMSE teste:             {metricas['teste']['RMSE']:.4f}")
    print(f"  R² teste:               {metricas['teste']['R2']:.4f}")
    print(f"  MAPE teste:             {metricas['teste']['MAPE']:.1f}%")
    print(f"  MdAPE teste:            {metricas['teste']['MdAPE']:.1f}%")
    print(f"  ±1 fruta:               {metricas['teste']['acerto_pm1']:.1f}%")
    print(f"  ±2 frutas:              {metricas['teste']['acerto_pm2']:.1f}%")
    print(f"  Gap MAE ratio:          {metricas['gap_treino_teste']['MAE_ratio']:.2f}x")
    infer = relatorio["eficiencia"]
    print(f"  Inf. por imagem:        {infer['inferencia_por_imagem_ms']:.4f} ms")
    print(f"  Tempo GS:               {tempo_gs}s")
    print(f"  Tempo treino final:     {tempo_treino}s")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()
