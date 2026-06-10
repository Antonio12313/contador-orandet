import warnings

warnings.filterwarnings("ignore")

import json
import time
import platform
import importlib.metadata
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import loguniform

# CONFIGURACAO
DATASET_DIR = "dataset_preparado_v11"
TRAIN_CSV = f"{DATASET_DIR}/orandet_v11_train_norm.csv"
TEST_CSV = f"{DATASET_DIR}/orandet_v11_test_norm.csv"
OUTPUT_DIR = "./resultados_mlp_v12"

COLUNAS_META = [
    "image_id", "file_name", "split",
    "contagem", "contagem_log1p", "contagem_sqrt", "augmentacao",
]
SEED = 42
np.random.seed(SEED)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

FAIXAS_DEF = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 999)]
FAIXAS_LABEL = ["0", "1", "2", "3-4", "5-7", "8+"]


# METRICA DA BUSCA — R2 na escala original

def r2_original_scale(y_true_log, y_pred_log):
    """R2 na escala original. O alvo esta em log1p; aplica-se expm1 antes de medir,
    para que a busca otimize a metrica principal do trabalho."""
    y_true = np.expm1(np.asarray(y_true_log, dtype=np.float64))
    y_pred = np.clip(np.expm1(np.asarray(y_pred_log, dtype=np.float64)), 0.0, None)
    return float(r2_score(y_true, y_pred))


r2_scorer = make_scorer(r2_original_scale, greater_is_better=True)

# Espaco de busca: arquiteturas menores e regularizacao mais forte do que antes,
# para conter o sobreajuste (na versao anterior o treino chegava a R2 ~ 0,998).
PARAM_DIST = {
    "hidden_layer_sizes": [
        (32,), (64,), (128,), (256,),
        (64, 32), (128, 64), (256, 128),
        (128, 64, 32), (64, 32, 16),
    ],
    "activation": ["relu", "tanh"],
    "alpha": loguniform(1e-3, 1e0),
    "learning_rate_init": loguniform(1e-4, 1e-2),
}

PARAM_DIST_DOC = {
    "hidden_layer_sizes": [
        "(32,)", "(64,)", "(128,)", "(256,)",
        "(64, 32)", "(128, 64)", "(256, 128)",
        "(128, 64, 32)", "(64, 32, 16)",
    ],
    "activation": ["relu", "tanh"],
    "alpha": "loguniform(1e-3, 1e0)",
    "learning_rate_init": "loguniform(1e-4, 1e-2)",
    "solver": ["adam"],
    "max_iter_cv": 500,
    "n_iter_no_change_cv": 20,
    "early_stopping_cv": True,
    "validation_fraction_cv": 0.1,
}


# CARREGAMENTO

def carregar_dados():
    print("[1/5] Carregando dados...")
    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    feat_cols = [c for c in df_train.columns if c not in COLUNAS_META]

    # CV apenas com originais — evita leakage entre augmentadas e originais
    df_orig = df_train[df_train["augmentacao"] == "original"].copy()
    X_cv = df_orig[feat_cols].values.astype(np.float32)
    y_cv = df_orig["contagem_log1p"].values.astype(np.float64)

    # Treino final: sem augmentacao no pipeline atual, X_full == originais
    X_full = df_train[feat_cols].values.astype(np.float32)
    y_full = df_train["contagem_log1p"].values.astype(np.float64)
    y_full_orig = df_train["contagem"].values.astype(np.float64)

    X_test = df_test[feat_cols].values.astype(np.float32)
    y_test = df_test["contagem"].values.astype(np.float64)

    nomes_test = df_test["file_name"].values if "file_name" in df_test.columns else None
    ids_test = df_test["image_id"].values if "image_id" in df_test.columns else None

    print(f"  Features          : {len(feat_cols)}")
    print(f"  Treino CV (orig)  : {X_cv.shape[0]} imagens")
    print(f"  Treino final      : {X_full.shape[0]} amostras")
    print(f"  Teste             : {X_test.shape[0]} imagens")
    print(f"  Target            : log1p(contagem)  |  inversa: expm1(pred)")
    print(f"  Scoring da busca  : R2 na escala original (metrica principal)")

    return (X_cv, y_cv, X_full, y_full, y_full_orig,
            X_test, y_test, feat_cols, nomes_test, ids_test)


# BUSCA DE HIPERPARAMETROS

def rodar_busca(X_cv, y_cv):
    print("\n[2/5] HalvingRandomSearchCV — MLPRegressor (scoring = R2)")
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # Estimador base: early stopping ja ativo no CV, para selecionar modelos que
    # generalizam, e nao que memorizam o treino.
    mlp = MLPRegressor(
        solver="adam",
        max_iter=500,
        n_iter_no_change=20,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED,
    )

    hs = HalvingRandomSearchCV(
        estimator=mlp,
        param_distributions=PARAM_DIST,
        n_candidates=60,
        factor=3,
        scoring=r2_scorer,
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

    res = pd.DataFrame(hs.cv_results_).sort_values(
        "mean_test_score", ascending=False).reset_index(drop=True)
    res["r2_cv"] = res["mean_test_score"]
    res["r2_cv_std"] = res["std_test_score"]

    melhores = dict(hs.best_params_)
    melhor_r2_cv = float(hs.best_score_)
    r2_cv_std = float(res.iloc[0]["std_test_score"]) if not res.empty else None

    print(f"\n  Melhor R2 CV    : {melhor_r2_cv:.4f}")
    print(f"  Arquitetura     : {melhores['hidden_layer_sizes']}")
    print(f"  Ativacao        : {melhores['activation']}")
    print(f"  Alpha           : {melhores['alpha']:.5f}")
    print(f"  LR init         : {melhores['learning_rate_init']:.5f}")
    print(f"  Tempo busca     : {tempo_gs}s")

    return melhores, melhor_r2_cv, r2_cv_std, res, tempo_gs


# TREINO FINAL + AVALIACAO

def treinar_e_avaliar(melhores, X_full, y_full_log, X_test, y_test):
    print("\n[3/5] Treino final (early stopping ativo — controla sobreajuste)...")

    params_final = {
        **melhores,
        "solver": "adam",
        "max_iter": 1000,
        "n_iter_no_change": 30,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "random_state": SEED,
    }
    modelo = MLPRegressor(**params_final)

    _t0 = time.perf_counter()
    modelo.fit(X_full, y_full_log)
    tempo_treino = round(time.perf_counter() - _t0, 2)
    print(f"  Tempo treino final: {tempo_treino}s  |  iteracoes: {modelo.n_iter_}")

    def _metricas(y_true, y_pred):
        y_pred = np.clip(y_pred, 0.0, None)
        y_pred_int = np.round(y_pred).astype(int)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mask = y_true > 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        mdape = float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        d1 = float(np.mean(np.abs(y_true - y_pred_int) <= 1) * 100)
        d2 = float(np.mean(np.abs(y_true - y_pred_int) <= 2) * 100)
        d20 = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6)) <= 0.20) * 100)
        return mae, rmse, r2, mape, mdape, d1, d2, d20, y_pred, y_pred_int

    # Teste
    y_pred = np.clip(np.expm1(modelo.predict(X_test)), 0.0, None)
    (mae, rmse, r2, mape, mdape, d1, d2, d20, y_pred, y_pred_int) = _metricas(y_test, y_pred)

    # Treino (escala original)
    y_pred_tr = np.clip(np.expm1(modelo.predict(X_full)), 0.0, None)
    y_tr_orig = np.expm1(y_full_log)
    mae_tr = mean_absolute_error(y_tr_orig, y_pred_tr)
    rmse_tr = np.sqrt(mean_squared_error(y_tr_orig, y_pred_tr))
    r2_tr = r2_score(y_tr_orig, y_pred_tr)
    mask_tr = y_tr_orig > 0
    mape_tr = float(np.mean(np.abs((y_tr_orig[mask_tr] - y_pred_tr[mask_tr]) / y_tr_orig[mask_tr])) * 100)

    print(f"\n  {'Metrica':<14}{'Treino':>10}{'Teste':>10}")
    print(f"  {'-' * 34}")
    print(f"  {'R2':<14}{r2_tr:>10.4f}{r2:>10.4f}")
    print(f"  {'MAE':<14}{mae_tr:>10.3f}{mae:>10.3f}")
    print(f"  {'RMSE':<14}{rmse_tr:>10.3f}{rmse:>10.3f}")
    print(f"  {'MAPE':<14}{mape_tr:>9.1f}%{mape:>9.1f}%")
    print(f"  MdAPE (teste): {mdape:.1f}%  |  +-1: {d1:.1f}%  +-2: {d2:.1f}%")

    metricas = {
        "treino": {"MAE": round(mae_tr, 4), "RMSE": round(rmse_tr, 4),
                   "R2": round(r2_tr, 4), "MAPE": round(mape_tr, 2)},
        "teste": {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4),
                  "MAPE": round(mape, 2), "MdAPE": round(mdape, 2),
                  "gap_MAPE_MdAPE": round(mape - mdape, 2),
                  "acerto_pm1": round(d1, 1), "acerto_pm2": round(d2, 1),
                  "dentro_20pct": round(d20, 1)},
        "gap_treino_teste": {"MAE_ratio": round(mae / (mae_tr + 1e-9), 2),
                             "MAPE_delta": round(mape - mape_tr, 2),
                             "R2_delta": round(r2 - r2_tr, 4)},
    }
    return modelo, y_pred, y_pred_int, metricas, tempo_treino


# DIAGNOSTICO POR FAIXA

def diagnostico_por_faixa(y_test, y_pred):
    out = []
    for (lo, hi), label in zip(FAIXAS_DEF, FAIXAS_LABEL):
        idx = (y_test >= lo) & (y_test <= hi)
        n = int(idx.sum())
        if n == 0:
            continue
        yt, yp = y_test[idx], y_pred[idx]
        mae = float(np.mean(np.abs(yt - yp)))
        mf = yt > 0
        mape = float(np.mean(np.abs((yt[mf] - yp[mf]) / yt[mf])) * 100) if mf.sum() > 0 else 0.0
        a1 = float(np.mean(np.abs(yt - np.round(yp)) <= 1) * 100)
        bias = float(np.mean(yp - yt))
        out.append({"faixa": label, "n": n, "mae": round(mae, 3),
                    "mape": round(mape, 1), "acerto_pm1": round(a1, 1),
                    "bias": round(bias, 3)})
    return out


# INFERENCIA

def medir_inferencia(modelo, X_test, n_repeticoes=50):
    n = len(X_test)
    _ = modelo.predict(X_test[:min(5, n)])  # warmup
    tempos = []
    for _ in range(n_repeticoes):
        t0 = time.perf_counter()
        modelo.predict(X_test)
        tempos.append(time.perf_counter() - t0)
    tempos = np.array(tempos)
    total_ms = float(np.mean(tempos)) * 1000
    return {
        "n_repeticoes": n_repeticoes,
        "n_amostras_batch": int(n),
        "tempo_total_medio_ms": round(total_ms, 4),
        "tempo_por_imagem_ms": round(total_ms / n, 4),
        "tempo_por_imagem_std_ms": round(float(np.std(tempos)) * 1000 / n, 4),
        "nota": ("Inferencia em batch sobre o conjunto de teste completo, com "
                 "warmup de 5 amostras. Nao inclui extracao de atributos — "
                 "apenas o forward pass do MLP."),
    }


def gerar_relatorio_imagens(y_test, y_pred, y_pred_int, nomes, ids):
    return pd.DataFrame({
        "image_id": ids if ids is not None else np.arange(len(y_test)),
        "file_name": nomes if nomes is not None else "",
        "contagem": np.round(y_test).astype(int),
        "pred_continua": np.round(y_pred, 3),
        "pred_inteiro": y_pred_int,
        "erro_absoluto": np.round(np.abs(y_test - y_pred), 3),
        "erro_relativo_pct": np.round(np.abs(y_test - y_pred) / (y_test + 1e-6) * 100, 1),
    })


def capturar_ambiente():
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
        info["ram_total_gb"] = "psutil nao instalado"
        info["cpu_contagem_logica"] = "psutil nao instalado"
    return info


# SALVAMENTO

def salvar(modelo, melhores, melhor_r2_cv, r2_cv_std, metricas, metricas_faixa,
           inferencia, res_grid, df_pred, feat_cols,
           n_treino_full, n_treino_orig, n_teste,
           tempo_gs, tempo_treino, ambiente):
    print("\n[5/5] Salvando artefatos...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    joblib.dump(modelo, f"{OUTPUT_DIR}/mlp_modelo.joblib")
    res_grid.to_csv(f"{OUTPUT_DIR}/mlp_grid_todas_arquiteturas.csv", index=False)
    df_pred.to_csv(f"{OUTPUT_DIR}/mlp_predicoes_por_imagem.csv", index=False)

    try:
        versao_sklearn = importlib.metadata.version("scikit-learn")
    except Exception:
        import sklearn
        versao_sklearn = sklearn.__version__
    try:
        versao_scipy = importlib.metadata.version("scipy")
    except Exception:
        import scipy
        versao_scipy = scipy.__version__

    cv_por_fold = []
    if not res_grid.empty:
        best_row = res_grid.iloc[0]
        for fold in range(5):
            chave = f"split{fold}_test_score"
            if chave in best_row.index:
                cv_por_fold.append({"fold": fold + 1,
                                    "r2_cv": round(float(best_row[chave]), 4)})

    melhores_serial = {}
    for k, v in melhores.items():
        if isinstance(v, tuple):
            melhores_serial[k] = str(v)
        elif isinstance(v, (np.floating, np.integer)):
            melhores_serial[k] = float(v) if isinstance(v, np.floating) else int(v)
        else:
            melhores_serial[k] = v

    saida = {
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
            "avaliacao_escala": "contagem direta (inteiro) apos expm1",
            "normalizacao": "StandardScaler ajustado no treino (orandet_v11_scaler.joblib)",
            "conjunto_validacao_dedicado": (
                "Early stopping ativo tanto no CV quanto no treino final "
                "(validation_fraction=0.1). O modelo final para quando o R2 de "
                "validacao interna deixa de melhorar, o que controla o sobreajuste."
            ),
        },
        "arquitetura": {
            "tipo": "MLPRegressor (scikit-learn)",
            "hidden_layer_sizes": str(melhores["hidden_layer_sizes"]),
            "activation": melhores["activation"],
            "solver": "adam",
        },
        "hiperparametros_finais": {
            **melhores_serial,
            "solver": "adam",
            "max_iter_treino_final": 1000,
            "n_iter_no_change_treino_final": 30,
            "early_stopping_treino_final": True,
            "validation_fraction_treino_final": 0.1,
            "max_iter_cv": 500,
            "n_iter_no_change_cv": 20,
            "early_stopping_cv": True,
            "validation_fraction_cv": 0.1,
        },
        "espaco_de_busca": PARAM_DIST_DOC,
        "busca_hiperparametros": {
            "metodo": "HalvingRandomSearchCV",
            "n_candidatos_inicial": 60,
            "factor": 3,
            "min_resources": "exhaust",
            "scoring": "R2 na escala original (via make_scorer) — metrica principal do TCC",
            "cv_folds": 5,
            "kfold_shuffle": True,
            "kfold_random_state": SEED,
            "amostras_busca": "originais apenas",
        },
        "validacao_cruzada": {
            "n_folds": 5,
            "random_state": SEED,
            "scoring": "R2 na escala original (apos expm1)",
            "r2_cv_mean": round(float(melhor_r2_cv), 4),
            "r2_cv_std": round(float(r2_cv_std), 4) if r2_cv_std is not None else None,
            "por_fold": cv_por_fold,
        },
        "resultados_treino": metricas["treino"],
        "resultados_teste": metricas["teste"],
        "gap_treino_teste": metricas["gap_treino_teste"],
        "resultados_por_faixa": metricas_faixa,
        "eficiencia": {
            "tempo_grid_search_s": tempo_gs,
            "tempo_treino_final_s": tempo_treino,
            "inferencia": inferencia,
            "versao_sklearn": versao_sklearn,
            "versao_scipy": versao_scipy,
        },
        "ambiente": ambiente,
        "timestamp": ts,
    }

    with open(f"{OUTPUT_DIR}/mlp_relatorio.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    return saida


def imprimir_top_cv(res_grid, n=8):
    if res_grid.empty:
        return
    print(f"\n  Top {n} configuracoes por R2 CV:")
    print(f"  {'arquitetura':<18}{'ativ.':<7}{'alpha':<11}{'lr_init':<11}{'R2 CV':>8}")
    print(f"  {'-' * 56}")
    for _, row in res_grid.head(n).iterrows():
        arq = str(row.get("param_hidden_layer_sizes", ""))
        ativ = str(row.get("param_activation", ""))
        alpha = float(row.get("param_alpha", float("nan")))
        lr = float(row.get("param_learning_rate_init", float("nan")))
        print(f"  {arq:<18}{ativ:<7}{alpha:<11.5f}{lr:<11.5f}{row['r2_cv']:>8.4f}")


# MAIN

def main():
    print("\n" + "=" * 65)
    print("  MLP — Contagem de Laranjas Verdes (OranDet)")
    print("  Target: log1p(y)  |  Inversa: expm1  |  Scoring: R2")
    print("  Busca: HalvingRandomSearchCV (60 candidatos, fator 3)")
    print("=" * 65)

    (X_cv, y_cv, X_full, y_full_log, y_full_orig,
     X_test, y_test, feat_cols, nomes_test, ids_test) = carregar_dados()

    melhores, melhor_r2_cv, r2_cv_std, res_grid, tempo_gs = rodar_busca(X_cv, y_cv)

    modelo, y_pred, y_pred_int, metricas, tempo_treino = treinar_e_avaliar(
        melhores, X_full, y_full_log, X_test, y_test
    )

    inferencia = medir_inferencia(modelo, X_test, n_repeticoes=50)
    ambiente = capturar_ambiente()
    metricas_faixa = diagnostico_por_faixa(y_test, y_pred)
    df_pred = gerar_relatorio_imagens(y_test, y_pred, y_pred_int, nomes_test, ids_test)

    imprimir_top_cv(res_grid, n=8)

    salvar(modelo, melhores, melhor_r2_cv, r2_cv_std, metricas, metricas_faixa,
           inferencia, res_grid, df_pred, feat_cols,
           n_treino_full=len(X_full), n_treino_orig=len(X_cv), n_teste=len(X_test),
           tempo_gs=tempo_gs, tempo_treino=tempo_treino, ambiente=ambiente)

    print(f"\n{'=' * 65}")
    print(f"  Arquitetura     : {melhores['hidden_layer_sizes']}")
    print(f"  R2 CV           : {melhor_r2_cv:.4f}")
    print(f"  R2 treino       : {metricas['treino']['R2']:.4f}")
    print(f"  R2 teste        : {metricas['teste']['R2']:.4f}")
    print(f"  Gap R2 (teste-treino): {metricas['gap_treino_teste']['R2_delta']:.4f}")
    print(f"  MAPE teste      : {metricas['teste']['MAPE']:.1f}%")
    print(f"  MAE teste       : {metricas['teste']['MAE']:.3f}")
    print(f"  +-1 fruta       : {metricas['teste']['acerto_pm1']:.1f}%")
    print(f"  Tempo busca     : {tempo_gs}s  |  treino final: {tempo_treino}s")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
