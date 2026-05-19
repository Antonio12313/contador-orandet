import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.experimental import enable_halving_search_cv          # ← OBRIGATÓRIO antes do import abaixo
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import loguniform

DATASET_DIR = "dataset_preparado_v80"
TRAIN_CSV   = f"{DATASET_DIR}/orandet_v71_train_norm.csv"
TEST_CSV    = f"{DATASET_DIR}/orandet_v71_test_norm.csv"
OUTPUT_DIR  = "./resultados_mlp_v8"

COLUNAS_META = [
    "image_id", "file_name", "split",
    "contagem", "contagem_log1p", "contagem_sqrt", "augmentacao",
    "contagem_log", "contagem_total"
]
SEED = 42
np.random.seed(SEED)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICA
# ══════════════════════════════════════════════════════════════════════════════

def mape_original_scale(y_true_log, y_pred_log, epsilon=1.0):
    """Avalia MAPE na escala original, mesmo que treino seja em log."""
    y_true = np.expm1(np.array(y_true_log, dtype=np.float64))
    y_pred = np.expm1(np.array(y_pred_log, dtype=np.float64))
    y_pred = np.maximum(y_pred, 0.0)
    denom  = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

mape_scorer = make_scorer(mape_original_scale, greater_is_better=False)

# ══════════════════════════════════════════════════════════════════════════════
# ESPAÇO DE BUSCA
# Usamos loguniform para alpha e lr_init — amostragem contínua em escala log,
# mais eficiente que uma grade discreta de 3-4 valores.
# ══════════════════════════════════════════════════════════════════════════════

PARAM_DIST = {
    "hidden_layer_sizes": [
        (64,), (128,), (256,),
        (128, 64), (256, 128), (256, 64), (512, 128),
        (256, 128, 64), (128, 64, 32), (256, 128, 32),
        (128, 128, 128), (64, 64, 64),
    ],
    "activation"         : ["relu", "tanh"],
    "solver"             : ["adam"],
    "alpha"              : loguniform(1e-4, 5e-2),
    "learning_rate"      : ["adaptive"],
    "learning_rate_init" : loguniform(1e-4, 1e-3),
    "max_iter"           : [500],          # early_stopping cuida do resto no CV
    "n_iter_no_change"   : [20],
    "random_state"       : [SEED],
}

# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DOS DADOS
# ══════════════════════════════════════════════════════════════════════════════

def carregar_dados():
    print("[1/4] Carregando dados...")

    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    feat_cols = [c for c in df_train.columns if c not in COLUNAS_META]

    # CV apenas com originais — evita data leakage entre aug e original
    df_orig = df_train[df_train["augmentacao"] == "original"].copy()
    X_cv    = df_orig[feat_cols].values.astype(np.float32)
    y_cv    = df_orig["contagem_log1p"].values.astype(np.float64)   # log(1+y)

    # Treino final com augmentação completa
    X_full  = df_train[feat_cols].values.astype(np.float32)
    y_full  = df_train["contagem_log1p"].values.astype(np.float64)  # log(1+y)

    # Teste: target na escala ORIGINAL para avaliação final
    X_test  = df_test[feat_cols].values.astype(np.float32)
    y_test  = df_test["contagem"].values.astype(np.float64)

    nomes_test = df_test["file_name"].values  if "file_name" in df_test.columns else None
    ids_test   = df_test["image_id"].values   if "image_id"  in df_test.columns else None

    print(f"  Features (v8)     : {len(feat_cols)}")
    print(f"  Treino CV (orig)  : {X_cv.shape[0]} imagens")
    print(f"  Treino full+aug   : {X_full.shape[0]} amostras")
    print(f"  Teste             : {X_test.shape[0]} imagens")
    print(f"  Target treino     : log(1+contagem), min={y_cv.min():.3f} max={y_cv.max():.3f}")
    print(f"  Contagem teste    : min={y_test.min():.0f} max={y_test.max():.0f} "
          f"média={y_test.mean():.1f}")

    return X_cv, y_cv, X_full, y_full, X_test, y_test, feat_cols, nomes_test, ids_test

# ══════════════════════════════════════════════════════════════════════════════
# BUSCA DE HIPERPARÂMETROS
#
# HalvingRandomSearchCV: começa com poucos recursos (dados), elimina os piores
# candidatos progressivamente e concentra o orçamento nos mais promissores.
# Resultado: ~10-15x mais rápido que GridSearchCV com perda mínima de qualidade.
#
# early_stopping=True no estimador: cada fold para antes de max_iter se não
# melhorar por n_iter_no_change épocas — evita 500 iterações desnecessárias.
# ══════════════════════════════════════════════════════════════════════════════

def rodar_grid_search(X_cv, y_cv):
    print(f"\n[2/4] HalvingRandomSearchCV — MLPRegressor (v8)")
    print(f"  Candidatos iniciais : 60  (era 288 no GridSearch)")
    print(f"  CV                  : 5-fold (KFold embaralhado)")
    print(f"  Early stopping      : ativo no CV (n_iter_no_change=20)")
    print(f"  Target CV           : log(1+y)  |  Score: MAPE escala original\n")

    cv  = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # early_stopping=True: para automaticamente quando a loss de validação
    # não melhora — reduz ~3-4x o tempo por fold sem perder qualidade
    mlp = MLPRegressor(
        early_stopping=True,
        validation_fraction=0.1,
    )

    hs = HalvingRandomSearchCV(
        estimator=mlp,
        param_distributions=PARAM_DIST,
        n_candidates=60,          # candidatos na 1ª rodada
        factor=3,                 # a cada rodada mantém 1/3 dos melhores
        scoring=mape_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=False,
        random_state=SEED,
        min_resources="exhaust",  # usa todos os dados na rodada final
    )
    hs.fit(X_cv, y_cv)

    res = pd.DataFrame(hs.cv_results_)
    res = res.sort_values("mean_test_score", ascending=False).reset_index(drop=True)
    res["mape_cv"]     = -res["mean_test_score"]
    res["mape_cv_std"] =  res["std_test_score"]

    melhores_params = hs.best_params_.copy()
    melhor_mape_cv  = -hs.best_score_

    # Remove params exclusivos do CV (treino final não usa validação interna)
    for k in ("early_stopping", "validation_fraction"):
        melhores_params.pop(k, None)

    print(f"\n  Melhor MAPE CV  : {melhor_mape_cv:.2f}%")
    print(f"  Arquitetura     : {melhores_params['hidden_layer_sizes']}")
    print(f"  Ativação        : {melhores_params['activation']}")
    print(f"  Alpha           : {melhores_params['alpha']:.5f}")
    print(f"  LR init         : {melhores_params['learning_rate_init']:.5f}")

    return melhores_params, melhor_mape_cv, res

# ══════════════════════════════════════════════════════════════════════════════
# TREINO FINAL + AVALIAÇÃO
#
# Fluxo do target log:
#   treina em log(1+y)  →  prediz em log  →  expm1(pred) = contagem estimada
#   avalia MAPE(y_test_original, expm1(pred))
#
# No treino final: max_iter maior (2000) e sem early_stopping — usa todos os
# dados de augmentação sem reservar fração de validação.
# ══════════════════════════════════════════════════════════════════════════════

def treinar_e_avaliar(melhores_params, X_full, y_full, X_test, y_test):
    print("\n[3/4] Treinando modelo final (max_iter=2000, sem early_stopping)...")

    params_final = {
        **melhores_params,
        "max_iter"        : 2000,
        "n_iter_no_change": 50,
        "early_stopping"  : False,   # treina em todos os dados, sem val. interna
    }
    modelo = MLPRegressor(**params_final)
    modelo.fit(X_full, y_full)

    # Converte predição de volta para escala original
    y_pred_log = modelo.predict(X_test)
    y_pred     = np.expm1(y_pred_log)
    y_pred     = np.maximum(y_pred, 0.0)
    y_pred_arr = np.round(y_pred)

    mape_v = mape_original_scale(np.log1p(y_test), y_pred_log)
    mae_v  = mean_absolute_error(y_test, y_pred)
    rmse_v = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_v   = r2_score(y_test, y_pred)

    print(f"\n  ══ Resultado no conjunto de teste ═══════════")
    print(f"  MAPE  : {mape_v:.2f}%    ← métrica principal")
    print(f"  MAE   : {mae_v:.4f}  laranjas")
    print(f"  RMSE  : {rmse_v:.4f}  laranjas")
    print(f"  R²    : {r2_v:.4f}")
    print(f"  ═════════════════════════════════════════════")

    metricas = {"mape": mape_v, "mae": mae_v, "rmse": rmse_v, "r2": r2_v}
    return modelo, y_pred, y_pred_arr, metricas

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNÓSTICO POR FAIXA DE CONTAGEM
# ══════════════════════════════════════════════════════════════════════════════

def diagnostico_por_faixa(y_test, y_pred):
    """Imprime MAPE médio por faixa de contagem real."""
    df = pd.DataFrame({"real": y_test, "pred": y_pred})
    df["faixa"] = pd.cut(df["real"], bins=[0, 1, 2, 3, 5, 9, 99],
                         labels=["=1", "=2", "=3", "4–5", "6–9", "≥10"])
    print("\n  ══ MAPE por faixa de contagem ═══════════════")
    for faixa, grp in df.groupby("faixa", observed=True):
        mape_f = np.mean(np.abs(grp["real"] - grp["pred"]) /
                         np.maximum(grp["real"], 1.0)) * 100
        vies   = grp["pred"].mean() - grp["real"].mean()
        sinal  = "↑" if vies > 0.1 else ("↓" if vies < -0.1 else "≈")
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
        "image_id"        : ids_test   if ids_test   is not None else range(n),
        "file_name"       : nomes_test if nomes_test is not None else [f"img_{i}" for i in range(n)],
        "contagem_real"   : y_test.astype(int),
        "pred_continua"   : np.round(y_pred, 2),
        "pred_arredondada": y_pred_arr.astype(int),
        "erro_abs"        : np.abs(y_test - y_pred_arr).astype(int),
        "erro_perc"       : np.round(
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
    print(f"\n  ══ Top-{n} arquiteturas — TCC ══════════════════════════════════")
    print(f"  {'#':<3} {'Arquitetura':<28} {'Ativ.':<6} {'Alpha':<8} "
          f"{'LR':<7} {'MAPE CV':>9} {'±std':>7}")
    print("  " + "─" * 72)
    for i, (_, row) in enumerate(top.iterrows()):
        print(f"  {i+1:<3} {str(row['param_hidden_layer_sizes']):<28} "
              f"{str(row['param_activation']):<6} "
              f"{str(row['param_alpha']):<8} "
              f"{str(row['param_learning_rate_init']):<7} "
              f"{row['mape_cv']:>8.2f}%  "
              f"±{row['mape_cv_std']:.2f}%")
    print("  " + "═" * 72)

# ══════════════════════════════════════════════════════════════════════════════
# SALVAR
# ══════════════════════════════════════════════════════════════════════════════

def salvar(modelo, melhores_params, melhor_mape_cv,
           metricas, res_grid, df_pred, feat_cols):
    print("\n[4/4] Salvando resultados...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    joblib.dump(modelo, f"{OUTPUT_DIR}/mlp_v8_{ts}.joblib")
    res_grid.to_csv(f"{OUTPUT_DIR}/grid_todas_arquiteturas_{ts}.csv", index=False)
    df_pred.to_csv(f"{OUTPUT_DIR}/predicoes_por_imagem_{ts}.csv",     index=False)

    resumo = pd.DataFrame([{
        "timestamp"         : ts,
        "dataset"           : "v8",
        "target"            : "log(1+contagem)",
        "n_features"        : len(feat_cols),
        "melhor_arquitetura": str(melhores_params["hidden_layer_sizes"]),
        "melhor_ativacao"   : melhores_params["activation"],
        "melhor_alpha"      : melhores_params["alpha"],
        "melhor_lr_init"    : melhores_params["learning_rate_init"],
        "mape_cv_perc"      : round(melhor_mape_cv, 4),
        "mape_teste_perc"   : round(metricas["mape"], 4),
        "mae_teste"         : round(metricas["mae"], 4),
        "rmse_teste"        : round(metricas["rmse"], 4),
        "r2_teste"          : round(metricas["r2"], 4),
    }])
    resumo.to_csv(f"{OUTPUT_DIR}/resumo_{ts}.csv", index=False)
    print(f"  Arquivos salvos em: {OUTPUT_DIR}/")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 65)
    print("  MLP v8 — Contagem de Laranjas Verdes (OranDet/Embrapa)")
    print("  Target: log(1+y) | Features: 162 focadas em contagem")
    print("  Busca: HalvingRandomSearchCV (10-15x mais rápido que Grid)")
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

    diagnostico_por_faixa(y_test, y_pred)
    imprimir_tabela_tcc(res_grid, n=10)
    imprimir_amostras(df_pred, n=15)

    salvar(modelo, melhores_params, melhor_mape_cv,
           metricas, res_grid, df_pred, feat_cols)

    print(f"\n{'═' * 65}")
    print(f"  MAPE teste final : {metricas['mape']:.2f}%")
    print(f"  Arquitetura      : {melhores_params['hidden_layer_sizes']}")
    print(f"  Ativação         : {melhores_params['activation']}")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()