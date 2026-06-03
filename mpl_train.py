import warnings

warnings.filterwarnings("ignore")

import json
import time
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import loguniform

DATASET_DIR = "dataset_preparado_v11"
TRAIN_CSV = f"{DATASET_DIR}/orandet_v11_train_norm.csv"
TEST_CSV = f"{DATASET_DIR}/orandet_v11_test_norm.csv"
OUTPUT_DIR = "./resultados_mlp_v11"

COLUNAS_META = [
    "image_id", "file_name", "split",
    "contagem", "contagem_log1p", "contagem_sqrt", "augmentacao",
]
SEED = 42
np.random.seed(SEED)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ── MÉTRICA PARA GRID SEARCH (escala log → original via expm1) ───────────────

def mape_original_scale(y_true_log, y_pred_log, epsilon=1.0):
    """MAPE na escala original — usado no CV onde o target ainda está em log1p."""
    y_true = np.expm1(np.array(y_true_log, dtype=np.float64))
    y_pred = np.expm1(np.array(y_pred_log, dtype=np.float64))
    y_pred = np.maximum(y_pred, 0.0)
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


mape_scorer = make_scorer(mape_original_scale, greater_is_better=False)

PARAM_DIST = {
    "hidden_layer_sizes": [
        (64,), (128,), (256,),
        (128, 64), (256, 128), (256, 64), (512, 128),
        (256, 128, 64), (128, 64, 32), (256, 128, 32),
        (128, 128, 128), (64, 64, 64),
    ],
    "activation": ["relu", "tanh"],
    "solver": ["adam"],
    "alpha": loguniform(1e-4, 5e-2),
    "learning_rate": ["adaptive"],
    "learning_rate_init": loguniform(1e-4, 1e-3),
    "max_iter": [500],
    "n_iter_no_change": [20],
    "random_state": [SEED],
}

PARAM_DIST_DOC = {
    "hidden_layer_sizes": [
        "(64,)", "(128,)", "(256,)",
        "(128, 64)", "(256, 128)", "(256, 64)", "(512, 128)",
        "(256, 128, 64)", "(128, 64, 32)", "(256, 128, 32)",
        "(128, 128, 128)", "(64, 64, 64)",
    ],
    "activation": ["relu", "tanh"],
    "solver": ["adam"],
    "alpha": "loguniform(1e-4, 5e-2)",
    "learning_rate": ["adaptive"],
    "learning_rate_init": "loguniform(1e-4, 1e-3)",
    "max_iter": [500],
    "n_iter_no_change": [20],
    "random_state": [SEED],
}


# ── CARREGAMENTO ──────────────────────────────────────────────────────────────

def carregar_dados():
    print("[1/5] Carregando dados...")

    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    feat_cols = [c for c in df_train.columns if c not in COLUNAS_META]

    # CV apenas com originais — evita leakage entre aug e original
    df_orig = df_train[df_train["augmentacao"] == "original"].copy()
    X_cv = df_orig[feat_cols].values.astype(np.float32)
    y_cv = df_orig["contagem_log1p"].values.astype(np.float64)

    # Treino final com augmentação completa (originais + augmentadas)
    # NOTA METODOLÓGICA: o XGBoost foi treinado apenas com originais (2.759).
    # O MLP usa originais + augmentadas (33.108). Essa diferença de pipeline
    # é declarada como limitação na comparação entre modelos — ver campo
    # "limitacao_comparacao" no JSON de saída.
    X_full = df_train[feat_cols].values.astype(np.float32)
    y_full = df_train["contagem_log1p"].values.astype(np.float64)
    y_full_orig = df_train["contagem"].values.astype(np.float64)

    # Teste
    X_test = df_test[feat_cols].values.astype(np.float32)
    y_test = df_test["contagem"].values.astype(np.float64)

    nomes_test = df_test["file_name"].values if "file_name" in df_test.columns else None
    ids_test = df_test["image_id"].values if "image_id" in df_test.columns else None

    print(f"  Features          : {len(feat_cols)}")
    print(f"  Treino CV (orig)  : {X_cv.shape[0]} imagens")
    print(f"  Treino full+aug   : {X_full.shape[0]} amostras")
    print(f"  Teste             : {X_test.shape[0]} imagens")
    print(f"  Target            : log1p(contagem)  |  inversa: expm1(pred)")
    print(f"  Contagem teste    : min={y_test.min():.0f}  max={y_test.max():.0f}"
          f"  média={y_test.mean():.1f}")

    return (X_cv, y_cv, X_full, y_full, y_full_orig,
            X_test, y_test, feat_cols, nomes_test, ids_test)


# ── BUSCA DE HIPERPARÂMETROS ──────────────────────────────────────────────────

def rodar_grid_search(X_cv, y_cv):
    print(f"\n[2/5] HalvingRandomSearchCV — MLPRegressor")
    print(f"  Candidatos iniciais : 60")
    print(f"  CV                  : 5-fold (KFold shuffle, random_state={SEED})")
    print(f"  Early stopping      : ativo no CV (n_iter_no_change=20)")
    print(f"  Target              : log1p(y)  |  Score: MAPE escala original\n")

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    mlp = MLPRegressor(early_stopping=True, validation_fraction=0.1)

    hs = HalvingRandomSearchCV(
        estimator=mlp,
        param_distributions=PARAM_DIST,
        n_candidates=60,
        factor=3,
        scoring=mape_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=False,
        random_state=SEED,
        min_resources="exhaust",
    )

    _t0 = time.perf_counter()
    hs.fit(X_cv, y_cv)
    tempo_gs = round(time.perf_counter() - _t0, 2)

    res = pd.DataFrame(hs.cv_results_)
    res = res.sort_values("mean_test_score", ascending=False).reset_index(drop=True)
    res["mape_cv"] = -res["mean_test_score"]
    res["mape_cv_std"] = res["std_test_score"]

    melhores_params = hs.best_params_.copy()
    melhor_mape_cv = -hs.best_score_

    for k in ("early_stopping", "validation_fraction"):
        melhores_params.pop(k, None)

    print(f"\n  Melhor MAPE CV  : {melhor_mape_cv:.2f}%")
    print(f"  Arquitetura     : {melhores_params['hidden_layer_sizes']}")
    print(f"  Ativação        : {melhores_params['activation']}")
    print(f"  Alpha           : {melhores_params['alpha']:.5f}")
    print(f"  LR init         : {melhores_params['learning_rate_init']:.5f}")
    print(f"  Tempo busca     : {tempo_gs}s")

    return melhores_params, melhor_mape_cv, res, tempo_gs


# ── TREINO FINAL + AVALIAÇÃO ──────────────────────────────────────────────────

def treinar_e_avaliar(melhores_params, X_full, y_full_log, y_full_orig,
                      X_test, y_test):
    print("\n[3/5] Treinando modelo final (max_iter=2000, sem early_stopping)...")

    params_final = {
        **melhores_params,
        "max_iter": 2000,
        "n_iter_no_change": 50,
        "early_stopping": False,
    }
    modelo = MLPRegressor(**params_final)

    _t0 = time.perf_counter()
    modelo.fit(X_full, y_full_log)
    tempo_treino = round(time.perf_counter() - _t0, 2)
    print(f"  Tempo treino final: {tempo_treino}s")

    # ── Predição no teste ─────────────────────────────────────────────────────
    y_pred_log = modelo.predict(X_test)
    y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)
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

    # ── Predição no treino (gap de generalização) ─────────────────────────────
    y_pred_tr_log = modelo.predict(X_full)
    y_pred_tr = np.clip(np.expm1(y_pred_tr_log), 0.0, None)

    mae_tr = mean_absolute_error(y_full_orig, y_pred_tr)
    rmse_tr = np.sqrt(mean_squared_error(y_full_orig, y_pred_tr))
    r2_tr = r2_score(y_full_orig, y_pred_tr)
    mask_tr = y_full_orig > 0
    mape_tr = float(np.mean(np.abs(
        (y_full_orig[mask_tr] - y_pred_tr[mask_tr]) / y_full_orig[mask_tr]
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
    }
    return modelo, y_pred, y_pred_int, metricas, tempo_treino


# ── INFERÊNCIA — tempo por imagem ─────────────────────────────────────────────

def medir_inferencia(modelo, X_test, n_repeticoes=50):
    """
    Mede tempo de inferência por imagem com n_repeticoes passagens completas
    pelo conjunto de teste e reporta média e desvio por amostra.
    Metodologia idêntica à do XGBoost para comparação direta.
    """
    print(f"\n[4/5] Medindo tempo de inferência ({n_repeticoes} repetições)...")

    tempos = []
    for _ in range(n_repeticoes):
        t0 = time.perf_counter()
        _ = modelo.predict(X_test)
        tempos.append(time.perf_counter() - t0)

    tempos = np.array(tempos)
    n_amostras = X_test.shape[0]

    ms_por_img_mean = round(float(tempos.mean() / n_amostras * 1000), 4)
    ms_por_img_std = round(float(tempos.std() / n_amostras * 1000), 4)
    ms_total_mean = round(float(tempos.mean() * 1000), 2)

    print(f"  Amostras por batch     : {n_amostras}")
    print(f"  Tempo médio total      : {ms_total_mean} ms")
    print(f"  Tempo médio por imagem : {ms_por_img_mean} ms  ±{ms_por_img_std} ms")

    return {
        "n_repeticoes": n_repeticoes,
        "n_amostras_batch": n_amostras,
        "tempo_total_medio_ms": ms_total_mean,
        "tempo_por_imagem_ms": ms_por_img_mean,
        "tempo_por_imagem_std_ms": ms_por_img_std,
        "nota": (
            "Inferência em batch sobre o conjunto de teste completo. "
            "Não inclui extração de features — apenas forward pass do MLP. "
            "Comparável ao campo equivalente do XGBoost e SVR."
        ),
    }


# ── DIAGNÓSTICO POR FAIXA ─────────────────────────────────────────────────────

def diagnostico_por_faixa(y_test, y_pred):
    """
    Faixas: [0], [1], [2], [3-4], [5-7], [8+] — consistentes com XGBoost e SVR.
    """
    print("\n  MAPE por faixa de contagem:")
    print(f"  {'Faixa':<8} {'N':>5} {'MAE':>6} {'MAPE%':>7} {'±1':>7} {'Bias':>7}")
    print(f"  {'─' * 42}")

    faixas_def = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 999)]
    faixas_label = ["0", "1", "2", "3-4", "5-7", "8+"]
    metricas_f = []

    for (lo, hi), label in zip(faixas_def, faixas_label):
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
        print(f"  {label:<8} {int(n):>5} {mae_f:>6.2f} {mape_f:>6.1f}%  {a1:>5.1f}%  {bias:>+6.2f}{flag}")
        metricas_f.append({
            "faixa": label,
            "n": int(n),
            "mae": round(mae_f, 3),
            "mape": round(mape_f, 1),
            "acerto_pm1": round(a1, 1),
            "bias": round(bias, 3),
        })

    return metricas_f


# ── RELATÓRIO POR IMAGEM ──────────────────────────────────────────────────────

def gerar_relatorio_imagens(y_test, y_pred, y_pred_int, nomes_test, ids_test):
    n = len(y_test)
    df = pd.DataFrame({
        "image_id": ids_test if ids_test is not None else range(n),
        "file_name": nomes_test if nomes_test is not None else [f"img_{i}" for i in range(n)],
        "contagem_real": y_test.astype(int),
        "pred_continua": np.round(y_pred, 2),
        "pred_arredondada": y_pred_int.astype(int),
        "erro_abs": np.abs(y_test - y_pred_int).astype(int),
        "erro_perc": np.round(
            np.abs(y_test - y_pred) / np.maximum(np.abs(y_test), 1.0) * 100, 2
        ),
    })
    return df.sort_values("erro_perc", ascending=False).reset_index(drop=True)


def imprimir_amostras(df_pred, n=15):
    print(f"\n  Top {n} maiores erros:")
    print(f"  {'Arquivo':<35} {'Real':>6} {'Pred':>6} {'ErrAbs':>7} {'Err%':>7}")
    print("  " + "─" * 65)
    for _, row in df_pred.head(n).iterrows():
        nome = str(row["file_name"])[-35:]
        print(f"  {nome:<35} {int(row['contagem_real']):>6} "
              f"{int(row['pred_arredondada']):>6} "
              f"{int(row['erro_abs']):>7} "
              f"{row['erro_perc']:>6.1f}%")


def imprimir_tabela_tcc(res_grid, n=10):
    top = res_grid.head(n)
    print(f"\n  Top-{n} arquiteturas:")
    print(f"  {'#':<3} {'Arquitetura':<28} {'Ativ.':<6} {'Alpha':<10}"
          f"  {'LR':<10} {'MAPE CV':>9} {'±std':>8}")
    print("  " + "─" * 78)
    for i, (_, row) in enumerate(top.iterrows()):
        alpha = row.get("param_alpha", "n/d")
        lr = row.get("param_learning_rate_init", "n/d")
        alpha_s = f"{float(alpha):.5f}" if isinstance(alpha, (int, float, np.floating)) else str(alpha)
        lr_s = f"{float(lr):.5f}" if isinstance(lr, (int, float, np.floating)) else str(lr)
        print(f"  {i + 1:<3} {str(row['param_hidden_layer_sizes']):<28} "
              f"{str(row['param_activation']):<6} "
              f"{alpha_s:<10}  {lr_s:<10} "
              f"{row['mape_cv']:>8.2f}%  ±{row['mape_cv_std']:.2f}%")


# ── CAPTURA DE AMBIENTE ───────────────────────────────────────────────────────

def capturar_ambiente():
    """Captura CPU, RAM e versões — necessário para relatar eficiência com honestidade."""
    info = {
        "sistema": platform.system(),
        "maquina": platform.machine(),
        "processador": platform.processor() or "n/d",
        "python": platform.python_version(),
    }
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / 1e9, 1)
        info["cpu_contagem_logica"] = psutil.cpu_count(logical=True)
    except ImportError:
        info["ram_total_gb"] = "psutil não instalado"
        info["cpu_contagem_logica"] = "psutil não instalado"
    return info


# ── SALVAR — JSON completo para reprodutibilidade IEEE ────────────────────────

def salvar(modelo, melhores_params, melhor_mape_cv, mape_cv_std,
           metricas, metricas_faixa, inferencia, res_grid, df_pred, feat_cols,
           n_treino_full, n_treino_orig, n_teste,
           tempo_gs, tempo_treino, ambiente):
    print("\n[5/5] Salvando artefatos...")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    joblib.dump(modelo, f"{OUTPUT_DIR}/mlp_v9_modelo.joblib")
    res_grid.to_csv(f"{OUTPUT_DIR}/mlp_v9_grid_todas_arquiteturas.csv", index=False)
    df_pred.to_csv(f"{OUTPUT_DIR}/mlp_v9_predicoes_por_imagem.csv", index=False)

    # ── Versões ───────────────────────────────────────────────────────────────
    try:
        versao_sklearn = importlib.metadata.version("scikit-learn")
    except Exception:
        import sklearn;
        versao_sklearn = sklearn.__version__
    try:
        versao_scipy = importlib.metadata.version("scipy")
    except Exception:
        import scipy;
        versao_scipy = scipy.__version__

    # ── CV por fold do melhor candidato ───────────────────────────────────────
    cv_por_fold = []
    if not res_grid.empty:
        best_row = res_grid.iloc[0]
        for fold in range(5):
            chave = f"split{fold}_test_score"
            if chave in best_row.index:
                cv_por_fold.append({
                    "fold": fold + 1,
                    "mape_cv": round(float(-best_row[chave]), 4),
                })

    # ── Serializar melhores_params ────────────────────────────────────────────
    melhores_params_serial = {}
    for k, v in melhores_params.items():
        if isinstance(v, tuple):
            melhores_params_serial[k] = str(v)
        elif isinstance(v, (np.floating, np.integer)):
            melhores_params_serial[k] = float(v) if isinstance(v, np.floating) else int(v)
        else:
            melhores_params_serial[k] = v

    # ── JSON ──────────────────────────────────────────────────────────────────
    saida = {

        # ── Protocolo ─────────────────────────────────────────────────────────
        "protocolo": {
            "dataset_treino": "orandet_v11_train_norm.csv (normalizado)",
            "dataset_teste": "orandet_v11_test_norm.csv  (normalizado)",
            "n_amostras_treino_total": int(n_treino_full),
            "n_amostras_treino_originais": int(n_treino_orig),
            "n_amostras_teste": int(n_teste),
            "subset_busca": "Originais apenas (sem augmentadas) — evita leakage no CV",
            "treino_final": "Originais",
            "n_features": len(feat_cols),
            "alvo_modelo": "contagem_log1p",
            "transformacao_inversa": "np.expm1(pred) — converte log1p para escala original",
            "avaliacao_escala": "contagem direta (inteiro) após expm1",
            "normalizacao": "StandardScaler ajustado no treino (orandet_v11_scaler.joblib)",
            "conjunto_validacao_dedicado": (
                "Implícito: durante CV, early_stopping=True usa validation_fraction=0.1 "
                "internamente em cada fold. Treino final: early_stopping=False."
            ),
        },


        # ── Arquitetura ───────────────────────────────────────────────────────
        "arquitetura": {
            "tipo": "MLPRegressor (scikit-learn)",
            "hidden_layer_sizes": str(melhores_params["hidden_layer_sizes"]),
            "activation": melhores_params["activation"],
            "solver": melhores_params["solver"],
        },

        # ── Hiperparâmetros finais ────────────────────────────────────────────
        "hiperparametros_finais": {
            **melhores_params_serial,
            "max_iter_treino_final": 2000,
            "n_iter_no_change_treino_final": 50,
            "early_stopping_treino_final": False,
            "max_iter_cv": 500,
            "n_iter_no_change_cv": 20,
            "early_stopping_cv": True,
            "validation_fraction_cv": 0.1,
        },

        # ── Espaço de busca ───────────────────────────────────────────────────
        "espaco_de_busca": PARAM_DIST_DOC,

        # ── Busca de hiperparâmetros ──────────────────────────────────────────
        "busca_hiperparametros": {
            "metodo": "HalvingRandomSearchCV",
            "n_candidatos_inicial": 60,
            "factor": 3,
            "min_resources": "exhaust",
            "scoring": "neg_mape_escala_original (via make_scorer)",
            "cv_folds": 5,
            "kfold_shuffle": True,
            "kfold_random_state": SEED,
            "amostras_busca": "originais apenas",
        },

        "validacao_cruzada": {
            "n_folds": 5,
            "random_state": SEED,
            "mape_cv_mean": round(float(melhor_mape_cv), 4),
            "mape_cv_std": round(float(mape_cv_std), 4) if mape_cv_std is not None else None,
            "por_fold": cv_por_fold,
        },

        # ── Resultados ────────────────────────────────────────────────────────
        "resultados_treino": metricas["treino"],
        "resultados_teste": metricas["teste"],
        "gap_treino_teste": metricas["gap_treino_teste"],
        "resultados_por_faixa": metricas_faixa,

        # ── Eficiência ────────────────────────────────────────────────────────
        "eficiencia": {
            "tempo_grid_search_s": tempo_gs,
            "tempo_treino_final_s": tempo_treino,
            "inferencia": inferencia,  # ← novo: tempo por imagem
            "versao_sklearn": versao_sklearn,
            "versao_scipy": versao_scipy,
        },

        # ── Ambiente de execução ──────────────────────────────────────────────
        "ambiente": ambiente,  # ← novo: CPU/RAM/OS

        # ── Timestamp ─────────────────────────────────────────────────────────
        "timestamp": ts,
    }

    with open(f"{OUTPUT_DIR}/mlp_v9_relatorio.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)

    print(f"  Arquivos em: {OUTPUT_DIR}/")
    print(f"    ├─ mlp_v9_modelo.joblib")
    print(f"    ├─ mlp_v9_relatorio.json          ← protocolo + grade + métricas completas")
    print(f"    ├─ mlp_v9_grid_todas_arquiteturas.csv")
    print(f"    └─ mlp_v9_predicoes_por_imagem.csv")

    return saida


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 65)
    print("  MLP v9 — Contagem de Laranjas Verdes (OranDet v11)")
    print("  Target: log1p(y)  |  Inversa: expm1")
    print("  Busca: HalvingRandomSearchCV (60 candidatos, fator 3)")
    print("═" * 65)

    (X_cv, y_cv,
     X_full, y_full_log, y_full_orig,
     X_test, y_test,
     feat_cols, nomes_test, ids_test) = carregar_dados()

    melhores_params, melhor_mape_cv, res_grid, tempo_gs = rodar_grid_search(X_cv, y_cv)
    mape_cv_std = res_grid.iloc[0]["mape_cv_std"] if not res_grid.empty else None

    modelo, y_pred, y_pred_int, metricas, tempo_treino = treinar_e_avaliar(
        melhores_params, X_full, y_full_log, y_full_orig, X_test, y_test
    )

    # ── NOVO: inferência e ambiente ───────────────────────────────────────────
    inferencia = medir_inferencia(modelo, X_test, n_repeticoes=50)
    ambiente = capturar_ambiente()

    metricas_faixa = diagnostico_por_faixa(y_test, y_pred)

    df_pred = gerar_relatorio_imagens(
        y_test, y_pred, y_pred_int, nomes_test, ids_test
    )

    imprimir_tabela_tcc(res_grid, n=10)
    imprimir_amostras(df_pred, n=15)

    relatorio = salvar(
        modelo, melhores_params, melhor_mape_cv, mape_cv_std,
        metricas, metricas_faixa, inferencia, res_grid, df_pred, feat_cols,
        n_treino_full=len(X_full), n_treino_orig=len(X_cv), n_teste=len(X_test),
        tempo_gs=tempo_gs, tempo_treino=tempo_treino, ambiente=ambiente,
    )

    print(f"\n{'═' * 65}")
    print(f"  Arquitetura         : {melhores_params['hidden_layer_sizes']}")
    print(f"  Ativação            : {melhores_params['activation']}")
    print(f"  MAPE CV             : {melhor_mape_cv:.2f}%")
    print(f"  MAE teste           : {metricas['teste']['MAE']:.3f}")
    print(f"  RMSE teste          : {metricas['teste']['RMSE']:.3f}")
    print(f"  R² teste            : {metricas['teste']['R2']:.4f}")
    print(f"  MAPE teste          : {metricas['teste']['MAPE']:.1f}%")
    print(f"  MdAPE teste         : {metricas['teste']['MdAPE']:.1f}%")
    print(f"  ±1 fruta            : {metricas['teste']['acerto_pm1']:.1f}%")
    print(f"  ±2 frutas           : {metricas['teste']['acerto_pm2']:.1f}%")
    print(f"  Gap MAE ratio       : {metricas['gap_treino_teste']['MAE_ratio']:.2f}x")
    print(f"  Tempo busca         : {tempo_gs}s")
    print(f"  Tempo treino final  : {tempo_treino}s")
    print(f"  Inferência/imagem   : {inferencia['tempo_por_imagem_ms']} ms"
          f"  ±{inferencia['tempo_por_imagem_std_ms']} ms")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()
