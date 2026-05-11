import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    make_scorer,
)

DATASET_DIR = "./dataset_preparado_v6"
TRAIN_CSV   = f"{DATASET_DIR}/orandet_v6_train.csv"
TEST_CSV    = f"{DATASET_DIR}/orandet_v6_test.csv"
OUTPUT_DIR  = "./resultados_mlp"

COLUNAS_META = ["image_id", "file_name", "split", "contagem", "augmentacao"]

SEED = 42
np.random.seed(SEED)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAPE — métrica principal do TCC
#
# Por que MAPE e não MSE/MAE?
#   Em contagem, errar 3 frutas numa imagem com 5 (erro 60%) é muito pior
#   do que errar 3 frutas numa imagem com 50 (erro 6%).
#   O MAPE captura essa proporção relativa, tornando a métrica interpretável
#   independente do tamanho da árvore/quantidade de frutas na cena.
#
# epsilon=1.0: evita divisão por zero nas imagens com contagem=0.
# Estratégia recomendada por Mario Filho (2023) para contagens baixas.
# ══════════════════════════════════════════════════════════════════════════════

def mape(y_true, y_pred, epsilon=1.0):
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    denominador = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denominador) * 100.0)

# GridSearchCV minimiza → scorer retorna negativo do MAPE
mape_scorer = make_scorer(mape, greater_is_better=False)


# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DOS DADOS
# ══════════════════════════════════════════════════════════════════════════════

def carregar_dados():
    print("[1/4] Carregando dados...")

    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    feat_cols = [c for c in df_train.columns if c not in COLUNAS_META]

    # ── CV usa só originais: evita que imagens augmentadas de uma mesma foto
    #    caiam em treino e validação ao mesmo tempo (data leakage no CV)
    df_orig = df_train[df_train["augmentacao"] == "original"].copy()
    X_cv = df_orig[feat_cols].values.astype(np.float32)
    y_cv = df_orig["contagem"].values.astype(np.float64)

    # ── Fit final usa treino completo com augmentação
    X_full = df_train[feat_cols].values.astype(np.float32)
    y_full = df_train["contagem"].values.astype(np.float64)

    X_test = df_test[feat_cols].values.astype(np.float32)
    y_test = df_test["contagem"].values.astype(np.float64)

    # Nomes dos arquivos de teste (para relatório por imagem)
    nomes_test = df_test["file_name"].values if "file_name" in df_test.columns else None
    ids_test   = df_test["image_id"].values  if "image_id"  in df_test.columns else None

    print(f"  Features          : {len(feat_cols)}")
    print(f"  Treino (CV)       : {X_cv.shape[0]} imagens originais")
    print(f"  Treino (full+aug) : {X_full.shape[0]} amostras")
    print(f"  Teste             : {X_test.shape[0]} imagens")
    print(f"  Contagem treino   : min={y_cv.min():.0f}  max={y_cv.max():.0f}  "
          f"média={y_cv.mean():.1f}  std={y_cv.std():.1f}")
    print(f"  Contagem teste    : min={y_test.min():.0f}  max={y_test.max():.0f}  "
          f"média={y_test.mean():.1f}  std={y_test.std():.1f}")

    return X_cv, y_cv, X_full, y_full, X_test, y_test, feat_cols, nomes_test, ids_test


# ══════════════════════════════════════════════════════════════════════════════
# GRID DE ARQUITETURAS MLP
#
# Baseado nos scripts do orientador (iris-tunning-grid.py e ISBI-2019),
# adaptado para REGRESSÃO com sklearn MLPRegressor.
#
# Famílias testadas:
#   Rasa   — 1 camada: rápido, baseline
#   Média  — 2 camadas: equilibrio bias/variância
#   Funda  — 3–4 camadas: mais expressiva, risco de overfit em dataset pequeno
#   Pirâmide — camadas decrescentes: padrão comum em visão computacional
#
# Nota sobre o problema verde-sobre-verde:
#   Features de textura/forma (Gabor, SATD, bas-relief) têm distribuições
#   não-lineares complexas. Arquiteturas mais fundas podem aprender
#   interações entre esses features que modelos rasos não capturam.
# ══════════════════════════════════════════════════════════════════════════════

PARAM_GRID = {
    # Arquiteturas: cada tupla define (n_neurônios_camada1, camada2, ...)
    "hidden_layer_sizes": [
        # Rasa — 1 camada (baseline simples)
        (64,),
        (128,),
        (256,),
        (512,),
        # 2 camadas — pirâmide decrescente
        (256, 128),
        (512, 256),
        (512, 128),
        (256, 64),
        (128, 64),
        # 3 camadas — pirâmide
        (512, 256, 128),
        (256, 128, 64),
        (512, 256, 64),
        (256, 128, 32),
        # 3 camadas — uniforme
        (256, 256, 256),
        (128, 128, 128),
        # 4 camadas — mais funda
        (512, 256, 128, 64),
        (256, 256, 128, 64),
        (256, 128, 64, 32),
    ],

    # Função de ativação (igual ao orientador: relu e tanh)
    "activation": ["relu", "tanh"],

    # Solver: adam é o padrão robusto para datasets de tamanho médio
    "solver": ["adam"],

    # Regularização L2 — importante pois features são altamente correlacionadas
    # (ex.: Gabor isotropy e SATD capturam coisas similares de ângulos diferentes)
    "alpha": [1e-4, 1e-3, 1e-2],

    # Learning rate adaptativa: reduz automaticamente se loss estagna
    "learning_rate": ["adaptive"],

    # Taxa inicial (igual faixa do ISBI-2019)
    "learning_rate_init": [1e-3, 5e-4],

    # Épocas: suficiente para convergir com early stopping interno do sklearn
    "max_iter": [500],

    # Reprodutibilidade
    "random_state": [SEED],
}


# ══════════════════════════════════════════════════════════════════════════════
# GRIDSEARCH
# ══════════════════════════════════════════════════════════════════════════════

def rodar_grid_search(X_cv, y_cv):
    n_comb = 1
    for v in PARAM_GRID.values():
        n_comb *= len(v)

    print(f"\n[2/4] GridSearchCV — MLPRegressor")
    print(f"  Combinações  : {n_comb}")
    print(f"  CV           : 5-fold (KFold embaralhado)")
    print(f"  Scorer       : MAPE (menor = melhor)\n")

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    mlp = MLPRegressor()

    gs = GridSearchCV(
        estimator=mlp,
        param_grid=PARAM_GRID,
        scoring=mape_scorer,
        cv=cv,
        n_jobs=-1,        # usa todos os núcleos do Mac
        verbose=1,
        refit=False,      # retreinamos manualmente no conjunto completo
        return_train_score=False,
    )
    gs.fit(X_cv, y_cv)

    # Organiza resultados
    res = pd.DataFrame(gs.cv_results_)
    res = res.sort_values("mean_test_score", ascending=False).reset_index(drop=True)
    res["mape_cv"]     = -res["mean_test_score"]   # positivo = MAPE em %
    res["mape_cv_std"] =  res["std_test_score"]

    melhores_params  = gs.best_params_
    melhor_mape_cv   = -gs.best_score_

    print(f"\n  Melhor MAPE CV  : {melhor_mape_cv:.2f}%")
    print(f"  Melhor arquitetura: {melhores_params['hidden_layer_sizes']}  "
          f"| ativação: {melhores_params['activation']}  "
          f"| alpha: {melhores_params['alpha']}  "
          f"| lr_init: {melhores_params['learning_rate_init']}")

    return melhores_params, melhor_mape_cv, res


# ══════════════════════════════════════════════════════════════════════════════
# TREINO FINAL + AVALIAÇÃO NO TESTE
# ══════════════════════════════════════════════════════════════════════════════

def treinar_e_avaliar(melhores_params, X_full, y_full, X_test, y_test):
    print("\n[3/4] Treinando modelo final (treino completo com augmentação)...")

    modelo = MLPRegressor(**melhores_params)
    modelo.fit(X_full, y_full)

    y_pred = modelo.predict(X_test)
    y_pred = np.maximum(y_pred, 0.0)  # contagem não pode ser negativa
    y_pred_arred = np.round(y_pred)   # versão inteira para exibição

    mape_v  = mape(y_test, y_pred)
    mae_v   = mean_absolute_error(y_test, y_pred)
    rmse_v  = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_v    = r2_score(y_test, y_pred)

    print(f"\n  ══ Resultado no conjunto de teste ══════════")
    print(f"  MAPE  : {mape_v:.2f}%    ← métrica principal")
    print(f"  MAE   : {mae_v:.4f}  laranjas")
    print(f"  RMSE  : {rmse_v:.4f}  laranjas")
    print(f"  R²    : {r2_v:.4f}")
    print(f"  ════════════════════════════════════════════")

    metricas = {"mape": mape_v, "mae": mae_v, "rmse": rmse_v, "r2": r2_v}
    return modelo, y_pred, y_pred_arred, metricas


# ══════════════════════════════════════════════════════════════════════════════
# RELATÓRIO POR IMAGEM
# Mostra predição × real para cada imagem do teste
# ══════════════════════════════════════════════════════════════════════════════

def gerar_relatorio_imagens(y_test, y_pred, y_pred_arred, nomes_test, ids_test):
    """
    Gera DataFrame com:
      - nome do arquivo
      - contagem real (anotação do dataset)
      - predição contínua do MLP
      - predição arredondada (inteiro)
      - erro absoluto e erro percentual por imagem
    """
    n = len(y_test)
    df = pd.DataFrame({
        "image_id"        : ids_test   if ids_test   is not None else range(n),
        "file_name"       : nomes_test if nomes_test is not None else [f"img_{i}" for i in range(n)],
        "contagem_real"   : y_test.astype(int),
        "pred_continua"   : np.round(y_pred, 2),
        "pred_arredondada": y_pred_arred.astype(int),
        "erro_abs"        : np.abs(y_test - y_pred_arred).astype(int),
        "erro_perc"       : np.round(
            np.abs(y_test - y_pred) / np.maximum(np.abs(y_test), 1.0) * 100, 2
        ),
    })
    df = df.sort_values("erro_perc", ascending=False).reset_index(drop=True)
    return df


def imprimir_amostras(df_pred, n=15):
    print(f"\n  ══ Predições por imagem (top {n} maiores erros) ══════════════════")
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
    print(f"\n  ══ Top-{n} arquiteturas MLP — para o TCC ══════════════════════════════")
    print(f"  {'#':<3} {'Arquitetura':<28} {'Ativ.':<6} {'Alpha':<8} "
          f"{'LR':<7} {'MAPE CV':>9} {'±std':>7}")
    print("  " + "─" * 72)
    for i, (_, row) in enumerate(top.iterrows()):
        arq  = str(row["param_hidden_layer_sizes"])
        ativ = str(row["param_activation"])
        alph = str(row["param_alpha"])
        lr   = str(row["param_learning_rate_init"])
        print(f"  {i+1:<3} {arq:<28} {ativ:<6} {alph:<8} "
              f"{lr:<7} {row['mape_cv']:>8.2f}%  ±{row['mape_cv_std']:.2f}%")
    print("  " + "═" * 72)


# ══════════════════════════════════════════════════════════════════════════════
# SALVAR TUDO
# ══════════════════════════════════════════════════════════════════════════════

def salvar(modelo, melhores_params, melhor_mape_cv,
           metricas, res_grid, df_pred, feat_cols):
    print("\n[4/4] Salvando resultados...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Modelo serializado
    p_modelo = f"{OUTPUT_DIR}/mlp_melhor_{ts}.joblib"
    joblib.dump(modelo, p_modelo)

    # Grid completo
    p_grid = f"{OUTPUT_DIR}/grid_todas_arquiteturas_{ts}.csv"
    res_grid.to_csv(p_grid, index=False)

    # Predições por imagem
    p_pred = f"{OUTPUT_DIR}/predicoes_por_imagem_{ts}.csv"
    df_pred.to_csv(p_pred, index=False)

    # Resumo único (para tabela do TCC)
    resumo = pd.DataFrame([{
        "timestamp"               : ts,
        "n_features"              : len(feat_cols),
        "melhor_arquitetura"      : str(melhores_params["hidden_layer_sizes"]),
        "melhor_ativacao"         : melhores_params["activation"],
        "melhor_alpha"            : melhores_params["alpha"],
        "melhor_lr_init"          : melhores_params["learning_rate_init"],
        "mape_cv_perc"            : round(melhor_mape_cv, 4),
        "mape_teste_perc"         : round(metricas["mape"], 4),
        "mae_teste"               : round(metricas["mae"], 4),
        "rmse_teste"              : round(metricas["rmse"], 4),
        "r2_teste"                : round(metricas["r2"], 4),
    }])
    p_resumo = f"{OUTPUT_DIR}/resumo_{ts}.csv"
    resumo.to_csv(p_resumo, index=False)

    print(f"  Modelo        → {p_modelo}")
    print(f"  Grid results  → {p_grid}")
    print(f"  Predições     → {p_pred}")
    print(f"  Resumo        → {p_resumo}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 65)
    print("  MLP — Contagem de Laranjas Verdes em Imagens")
    print("  OranDet (Embrapa) | Regressão | Métrica: MAPE")
    print("  Problema: laranja verde em fundo de folhas verdes")
    print("═" * 65)

    (X_cv, y_cv,
     X_full, y_full,
     X_test, y_test,
     feat_cols,
     nomes_test, ids_test) = carregar_dados()

    melhores_params, melhor_mape_cv, res_grid = rodar_grid_search(X_cv, y_cv)

    modelo, y_pred, y_pred_arred, metricas = treinar_e_avaliar(
        melhores_params, X_full, y_full, X_test, y_test
    )

    df_pred = gerar_relatorio_imagens(
        y_test, y_pred, y_pred_arred, nomes_test, ids_test
    )

    imprimir_tabela_tcc(res_grid, n=10)
    imprimir_amostras(df_pred, n=15)

    salvar(modelo, melhores_params, melhor_mape_cv,
           metricas, res_grid, df_pred, feat_cols)

    print(f"\n{'═' * 65}")
    print(f"  Resultado final — MAPE teste: {metricas['mape']:.2f}%")
    print(f"  Arquitetura: {melhores_params['hidden_layer_sizes']}")
    print(f"  Ativação   : {melhores_params['activation']}")
    print(f"  Arquivos salvos em: {OUTPUT_DIR}/")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()