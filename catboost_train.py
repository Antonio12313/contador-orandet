import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from itertools import product

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
#
# CatBoost usa orandet_v8_train_RAW (sem normalização).
# Diferente do MLP/SVR que precisam de features em [0,1], árvores de decisão
# trabalham com ordenação dos valores — normalizar não muda os splits e só
# adiciona uma etapa desnecessária.
# ══════════════════════════════════════════════════════════════════════════════

DATASET_DIR = "dataset_preparado_v8"
TRAIN_CSV = f"{DATASET_DIR}/orandet_v8_train_raw.csv"
TEST_CSV = f"{DATASET_DIR}/orandet_v8_test_raw.csv"
OUTPUT_DIR = "./resultados_catboost_v8"

COLUNAS_META = ["image_id", "file_name", "split", "contagem", "contagem_log", "augmentacao"]

SEED = 42
np.random.seed(SEED)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAPE — sempre avaliado na escala ORIGINAL (não no log)
#
# Mesmo critério do MLP e SVR: treina em log(1+y), avalia expm1(pred) vs
# contagem real. Isso garante comparabilidade direta entre os 3 modelos
# na tabela final do TCC.
#
# epsilon=1.0: proteção contra divisão por zero para imagens com contagem=0
# no treino (4 imagens conforme orandet_v8_info.json).
# ══════════════════════════════════════════════════════════════════════════════

def mape_original_scale(y_true_log, y_pred_log, epsilon=1.0):
    """MAPE na escala original a partir de predições em log(1+y)."""
    y_true = np.expm1(np.array(y_true_log, dtype=np.float64))
    y_pred = np.expm1(np.array(y_pred_log, dtype=np.float64))
    y_pred = np.maximum(y_pred, 0.0)
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DOS DADOS
# ══════════════════════════════════════════════════════════════════════════════

def carregar_dados():
    print("[1/4] Carregando dados...")

    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    feat_cols = [c for c in df_train.columns if c not in COLUNAS_META]

    # CV apenas com originais — evita data leakage:
    # imagens augmentadas da mesma foto não podem estar em treino e validação
    # simultaneamente durante o cross-validation.
    df_orig = df_train[df_train["augmentacao"] == "original"].copy()
    X_cv = df_orig[feat_cols].values.astype(np.float32)
    y_cv = df_orig["contagem_log"].values.astype(np.float64)  # log(1+y)

    # Treino final: dataset completo com todas as augmentações
    X_full = df_train[feat_cols].values.astype(np.float32)
    y_full = df_train["contagem_log"].values.astype(np.float64)  # log(1+y)

    # Teste: target na escala ORIGINAL para avaliação comparável com MLP/SVR
    X_test = df_test[feat_cols].values.astype(np.float32)
    y_test = df_test["contagem"].values.astype(np.float64)  # escala original

    nomes_test = df_test["file_name"].values if "file_name" in df_test.columns else None
    ids_test = df_test["image_id"].values if "image_id" in df_test.columns else None

    print(f"  Features (v8)     : {len(feat_cols)}")
    print(f"  Treino CV (orig)  : {X_cv.shape[0]} imagens")
    print(f"  Treino full+aug   : {X_full.shape[0]} amostras")
    print(f"  Teste             : {X_test.shape[0]} imagens")
    print(f"  Target treino     : log(1+contagem), "
          f"min={y_cv.min():.3f}  max={y_cv.max():.3f}")
    print(f"  Contagem teste    : min={y_test.min():.0f}  "
          f"max={y_test.max():.0f}  média={y_test.mean():.1f}")

    return X_cv, y_cv, X_full, y_full, X_test, y_test, feat_cols, nomes_test, ids_test


# ══════════════════════════════════════════════════════════════════════════════
# GRID DE HIPERPARÂMETROS — CatBoostRegressor
#
# Por que esses parâmetros?
#
# iterations (nº de árvores):
#   Mais árvores = mais capacidade, mas controlado pelo early stopping interno
#   (od_type="Iter", od_wait=50). Na prática o modelo para antes de esgotar
#   as iterações quando o CV interno não melhora. Valores altos (2000) são
#   seguros com early stopping ativo.
#
# depth (profundidade das árvores):
#   CatBoost usa "oblivious trees" (árvores simétricas): profundidade 6 = 64
#   folhas, que é suficiente para capturar interações entre features de
#   contagem (Hough × MSER × grade) sem memorizar o treino.
#   depth=4 é mais conservador e indicado quando o dataset é pequeno.
#
# learning_rate:
#   Taxa menor + mais árvores geralmente ganha, mas com early stopping
#   não precisa ir abaixo de 0.03. Taxa 0.3 é agressiva e pode ser boa
#   com regularização forte.
#
# l2_leaf_reg (regularização L2 nas folhas):
#   Features de contagem são correlacionadas (Hough, MSER, grade estimam
#   coisas parecidas). L2 alto evita que o modelo deposite todo o peso
#   numa única feature. Range 1–10 cobre de suave a forte.
#
# subsample (fração de amostras por árvore):
#   Bagging interno — reduz variância. 0.8 é o padrão robusto.
#   1.0 = sem bagging (mais determinístico, pode overfitar levemente).
#
# colsample_bylevel (fração de features por nível da árvore):
#   Similar ao max_features do Random Forest. 0.8 introduz aleatoriedade
#   sem sacrificar muito da informação disponível. Ajuda especialmente
#   quando há features redundantes (nosso caso: 162 features correlacionadas).
#
# od_type="Iter" + od_wait=50:
#   Early stopping interno do CatBoost: para se MAPE no fold de validação
#   não melhorar em 50 iterações consecutivas. Mais eficiente que usar
#   max_iter fixo porque adapta ao dataset.
# ══════════════════════════════════════════════════════════════════════════════

PARAM_GRID = {
    "iterations": [500, 1000, 2000],
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.1, 0.3],
    "l2_leaf_reg": [1, 3, 10],
    "subsample": [0.8, 1.0],
    "colsample_bylevel": [0.8, 1.0],
}

# Parâmetros fixos — iguais em todas as combinações
PARAMS_FIXOS = {
    "loss_function": "RMSE",  # RMSE no espaço log(1+y) é equivalente a
    # minimizar erro relativo na escala original
    "eval_metric": "RMSE",
    "od_type": "Iter",  # early stopping por iteração
    "od_wait": 50,  # para após 50 iterações sem melhora
    "random_seed": SEED,
    "verbose": 0,  # sem output por iteração (silencioso)
    "thread_count": -1,  # usa todos os núcleos disponíveis
    "allow_writing_files": False,  # não cria arquivos temporários do CatBoost
}


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION MANUAL
#
# Por que não usar GridSearchCV do sklearn?
#   O CatBoostRegressor tem um scorer interno (eval_metric + od_wait) que
#   precisa de um eval_set para funcionar. O GridSearchCV do sklearn não
#   passa o eval_set automaticamente — o early stopping fica desativado e
#   o modelo treina sempre até iterations máximas (lento e com overfit).
#
#   A solução é fazer o CV manualmente: para cada fold, passamos o conjunto
#   de validação como eval_set e deixamos o early stopping funcionar.
#   Isso deixa o grid ~3–5× mais rápido e o MAPE do CV mais realista.
# ══════════════════════════════════════════════════════════════════════════════

def rodar_grid_search(X_cv, y_cv):
    # Conta combinações
    chaves = list(PARAM_GRID.keys())
    valores = list(PARAM_GRID.values())
    combos = list(product(*valores))
    n_comb = len(combos)

    print(f"\n[2/4] Grid Search manual — CatBoostRegressor (v8)")
    print(f"  Combinações  : {n_comb}")
    print(f"  CV           : 5-fold (KFold embaralhado)")
    print(f"  Early stop   : od_wait=50 iterações")
    print(f"  Score        : MAPE escala original (expm1)\n")

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(X_cv, y_cv))

    resultados = []

    for idx_combo, combo in enumerate(combos):
        params_combo = dict(zip(chaves, combo))
        params_full = {**params_combo, **PARAMS_FIXOS}

        mapes_fold = []

        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            X_tr, y_tr = X_cv[tr_idx], y_cv[tr_idx]
            X_val, y_val = X_cv[val_idx], y_cv[val_idx]

            modelo = CatBoostRegressor(**params_full)
            modelo.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                use_best_model=True,  # usa o checkpoint com menor eval_metric
                verbose=False,
            )

            y_pred_log = modelo.predict(X_val)
            mape_f = mape_original_scale(y_val, y_pred_log)
            mapes_fold.append(mape_f)

        mape_medio = float(np.mean(mapes_fold))
        mape_std = float(np.std(mapes_fold))

        resultados.append({
            **{f"param_{k}": v for k, v in params_combo.items()},
            "mape_cv": mape_medio,
            "mape_cv_std": mape_std,
        })

        # Progresso a cada 10 combinações
        if (idx_combo + 1) % 10 == 0 or (idx_combo + 1) == n_comb:
            print(f"  [{idx_combo + 1:>3}/{n_comb}]  "
                  f"iter={params_combo['iterations']}  "
                  f"depth={params_combo['depth']}  "
                  f"lr={params_combo['learning_rate']}  "
                  f"l2={params_combo['l2_leaf_reg']}  "
                  f"→ MAPE CV={mape_medio:.2f}% ±{mape_std:.2f}%")

    res = pd.DataFrame(resultados).sort_values("mape_cv").reset_index(drop=True)

    melhor = res.iloc[0]
    melhores_params = {
        k.replace("param_", ""): melhor[k]
        for k in res.columns if k.startswith("param_")
    }
    melhor_mape_cv = melhor["mape_cv"]

    print(f"\n  Melhor MAPE CV  : {melhor_mape_cv:.2f}%")
    print(f"  iterations      : {melhores_params['iterations']}")
    print(f"  depth           : {melhores_params['depth']}")
    print(f"  learning_rate   : {melhores_params['learning_rate']}")
    print(f"  l2_leaf_reg     : {melhores_params['l2_leaf_reg']}")
    print(f"  subsample       : {melhores_params['subsample']}")
    print(f"  colsample_bylevel: {melhores_params['colsample_bylevel']}")

    return melhores_params, melhor_mape_cv, res


# ══════════════════════════════════════════════════════════════════════════════
# TREINO FINAL + AVALIAÇÃO NO TESTE
#
# Fluxo do target log:
#   treina em log(1+y)  →  prediz em log  →  expm1(pred) = contagem estimada
#   avalia MAPE(y_test_original, expm1(pred))
#
# use_best_model=False no treino final: sem eval_set disponível, treinamos
# todas as iterations dos melhores parâmetros (o early stopping no CV já
# selecionou um número razoável de árvores via iterations).
# ══════════════════════════════════════════════════════════════════════════════

def treinar_e_avaliar(melhores_params, X_full, y_full, X_test, y_test):
    print("\n[3/4] Treinando modelo final (treino completo + augmentação)...")

    params_final = {**melhores_params, **PARAMS_FIXOS}
    # No treino final não há eval_set, então desativamos o early stopping
    # e treinamos até as iterations definidas pelo grid.
    params_final["od_type"] = "IncToDec"  # desativa early stopping
    params_final["od_wait"] = 0

    modelo = CatBoostRegressor(**params_final)
    modelo.fit(X_full, y_full, verbose=False)

    # Converte predição de volta para escala original
    y_pred_log = modelo.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # e^pred - 1
    y_pred = np.maximum(y_pred, 0.0)  # garante não-negativo
    y_pred_arr = np.round(y_pred)

    # y_test já está na escala original
    mape_v = mape_original_scale(np.log1p(y_test), y_pred_log)
    mae_v = mean_absolute_error(y_test, y_pred)
    rmse_v = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_v = r2_score(y_test, y_pred)

    print(f"\n  ══ Resultado no conjunto de teste ═══════════")
    print(f"  MAPE  : {mape_v:.2f}%    ← métrica principal")
    print(f"  MAE   : {mae_v:.4f}  laranjas")
    print(f"  RMSE  : {rmse_v:.4f}  laranjas")
    print(f"  R²    : {r2_v:.4f}")
    print(f"  ═════════════════════════════════════════════")

    metricas = {"mape": mape_v, "mae": mae_v, "rmse": rmse_v, "r2": r2_v}
    return modelo, y_pred, y_pred_arr, metricas


# ══════════════════════════════════════════════════════════════════════════════
# IMPORTÂNCIA DAS FEATURES
#
# CatBoost calcula importância de features nativamente via PredictionValuesChange:
# mede quanto cada feature muda a predição em média ao longo de todas as árvores.
# Útil para o TCC para mostrar quais grupos de features (Gabor, MSER, Hough,
# grade, multiescala) o modelo mais usa.
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_feature_importance(modelo, feat_cols, n=20):
    importancias = modelo.get_feature_importance()
    df_imp = pd.DataFrame({
        "feature": feat_cols,
        "importancia": importancias,
    }).sort_values("importancia", ascending=False).reset_index(drop=True)

    print(f"\n  ══ Top-{n} features mais importantes ════════════════════════")
    print(f"  {'#':<4} {'Feature':<40} {'Importância':>12}")
    print("  " + "─" * 60)
    for i, row in df_imp.head(n).iterrows():
        print(f"  {i + 1:<4} {row['feature']:<40} {row['importancia']:>11.4f}")
    print("  " + "═" * 60)

    return df_imp


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNÓSTICO POR FAIXA DE CONTAGEM
# ══════════════════════════════════════════════════════════════════════════════

def diagnostico_por_faixa(y_test, y_pred):
    """MAPE médio por faixa de contagem real — identifica onde o modelo erra."""
    df = pd.DataFrame({"real": y_test, "pred": y_pred})
    df["faixa"] = pd.cut(df["real"], bins=[0, 1, 2, 3, 5, 9, 99],
                         labels=["=1", "=2", "=3", "4–5", "6–9", "≥10"])
    print("\n  ══ MAPE por faixa de contagem ═══════════════")
    for faixa, grp in df.groupby("faixa", observed=True):
        mape_f = np.mean(np.abs(grp["real"] - grp["pred"]) /
                         np.maximum(grp["real"], 1.0)) * 100
        vies = grp["pred"].mean() - grp["real"].mean()
        sinal = "↑" if vies > 0.1 else ("↓" if vies < -0.1 else "≈")
        print(f"  Contagem {faixa:>4} : n={len(grp):>3}  "
              f"MAPE={mape_f:>6.1f}%  "
              f"viés={vies:>+.2f} {sinal}")
    print("  ═════════════════════════════════════════════")


# ══════════════════════════════════════════════════════════════════════════════
# RELATÓRIO POR IMAGEM
# ══════════════════════════════════════════════════════════════════════════════

def gerar_relatorio_imagens(y_test, y_pred, y_pred_arr, nomes_test, ids_test):
    n = len(y_test)
    df = pd.DataFrame({
        "image_id": ids_test if ids_test is not None else range(n),
        "file_name": nomes_test if nomes_test is not None else [f"img_{i}" for i in range(n)],
        "contagem_real": y_test.astype(int),
        "pred_continua": np.round(y_pred, 2),
        "pred_arredondada": y_pred_arr.astype(int),
        "erro_abs": np.abs(y_test - y_pred_arr).astype(int),
        "erro_perc": np.round(
            np.abs(y_test - y_pred) / np.maximum(np.abs(y_test), 1.0) * 100, 2
        ),
    })
    return df.sort_values("erro_perc", ascending=False).reset_index(drop=True)


def imprimir_amostras(df_pred, n=15):
    print(f"\n  ══ Top {n} maiores erros ═════════════════════════════════════")
    print(f"  {'Arquivo':<35} {'Real':>6} {'Pred':>6} {'ErrAbs':>7} {'Err%':>7}")
    print("  " + "─" * 65)
    for _, row in df_pred.head(n).iterrows():
        nome = str(row["file_name"])[-35:]
        print(f"  {nome:<35} {int(row['contagem_real']):>6} "
              f"{int(row['pred_arredondada']):>6} "
              f"{int(row['erro_abs']):>7} "
              f"{row['erro_perc']:>6.1f}%")
    print("  " + "═" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# TABELA TOP-10 PARA O TCC
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_tabela_tcc(res_grid, n=10):
    top = res_grid.head(n)
    print(f"\n  ══ Top-{n} combinações CatBoost — TCC ══════════════════════════")
    print(f"  {'#':<3} {'iter':>5} {'depth':>6} {'lr':>6} "
          f"{'l2':>5} {'sub':>5} {'col':>5} {'MAPE CV':>9} {'±std':>7}")
    print("  " + "─" * 72)
    for i, (_, row) in enumerate(top.iterrows()):
        print(f"  {i + 1:<3} "
              f"{int(row['param_iterations']):>5}  "
              f"{int(row['param_depth']):>5}  "
              f"{row['param_learning_rate']:>5}  "
              f"{row['param_l2_leaf_reg']:>4}  "
              f"{row['param_subsample']:>4}  "
              f"{row['param_colsample_bylevel']:>4}  "
              f"{row['mape_cv']:>8.2f}%  "
              f"±{row['mape_cv_std']:.2f}%")
    print("  " + "═" * 72)


# ══════════════════════════════════════════════════════════════════════════════
# SALVAR RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def salvar(modelo, melhores_params, melhor_mape_cv,
           metricas, res_grid, df_pred, df_imp, feat_cols):
    print("\n[4/4] Salvando resultados...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Modelo serializado (.cbm é o formato nativo do CatBoost, mais robusto
    # que joblib para este modelo específico — preserva metadados internos)
    p_modelo = f"{OUTPUT_DIR}/catboost_v8_{ts}.cbm"
    modelo.save_model(p_modelo)

    # Grid completo de resultados
    p_grid = f"{OUTPUT_DIR}/grid_todas_combinacoes_{ts}.csv"
    res_grid.to_csv(p_grid, index=False)

    # Predições por imagem
    p_pred = f"{OUTPUT_DIR}/predicoes_por_imagem_{ts}.csv"
    df_pred.to_csv(p_pred, index=False)

    # Importância de features
    p_imp = f"{OUTPUT_DIR}/feature_importance_{ts}.csv"
    df_imp.to_csv(p_imp, index=False)

    # Resumo (para tabela comparativa do TCC: MLP × SVR × CatBoost × XGBoost)
    resumo = pd.DataFrame([{
        "timestamp": ts,
        "modelo": "CatBoost",
        "dataset": "v8",
        "target": "log(1+contagem)",
        "n_features": len(feat_cols),
        "melhor_iterations": melhores_params["iterations"],
        "melhor_depth": melhores_params["depth"],
        "melhor_learning_rate": melhores_params["learning_rate"],
        "melhor_l2_leaf_reg": melhores_params["l2_leaf_reg"],
        "melhor_subsample": melhores_params["subsample"],
        "melhor_colsample": melhores_params["colsample_bylevel"],
        "mape_cv_perc": round(melhor_mape_cv, 4),
        "mape_teste_perc": round(metricas["mape"], 4),
        "mae_teste": round(metricas["mae"], 4),
        "rmse_teste": round(metricas["rmse"], 4),
        "r2_teste": round(metricas["r2"], 4),
    }])
    p_resumo = f"{OUTPUT_DIR}/resumo_{ts}.csv"
    resumo.to_csv(p_resumo, index=False)

    print(f"  Modelo        → {p_modelo}")
    print(f"  Grid results  → {p_grid}")
    print(f"  Predições     → {p_pred}")
    print(f"  Importâncias  → {p_imp}")
    print(f"  Resumo        → {p_resumo}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 65)
    print("  CatBoost v8 — Contagem de Laranjas Verdes (OranDet/Embrapa)")
    print("  Target: log(1+y) | Features: 162 focadas em contagem")
    print("  Problema: laranja verde em fundo de folhas verdes")
    print("  Dataset: features RAW (sem normalização — árvores não precisam)")
    print("═" * 65)

    (X_cv, y_cv,
     X_full, y_full,
     X_test, y_test,
     feat_cols,
     nomes_test, ids_test) = carregar_dados()

    melhores_params, melhor_mape_cv, res_grid = rodar_grid_search(X_cv, y_cv)

    modelo, y_pred, y_pred_arr, metricas = treinar_e_avaliar(
        melhores_params, X_full, y_full, X_test, y_test
    )

    df_pred = gerar_relatorio_imagens(
        y_test, y_pred, y_pred_arr, nomes_test, ids_test
    )

    df_imp = imprimir_feature_importance(modelo, feat_cols, n=20)
    diagnostico_por_faixa(y_test, y_pred)
    imprimir_tabela_tcc(res_grid, n=10)
    imprimir_amostras(df_pred, n=15)

    salvar(modelo, melhores_params, melhor_mape_cv,
           metricas, res_grid, df_pred, df_imp, feat_cols)

    print(f"\n{'═' * 65}")
    print(f"  MAPE teste final : {metricas['mape']:.2f}%")
    print(f"  iterations       : {melhores_params['iterations']}")
    print(f"  depth            : {melhores_params['depth']}")
    print(f"  learning_rate    : {melhores_params['learning_rate']}")
    print(f"  l2_leaf_reg      : {melhores_params['l2_leaf_reg']}")
    print(f"  Arquivos salvos em: {OUTPUT_DIR}/")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()
