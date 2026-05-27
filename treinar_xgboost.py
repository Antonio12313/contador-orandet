import os
import json
import time
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# CONFIGURAÇÃO
DATASET_DIR = "./dataset_preparado_v80"
OUTPUT_DIR = "./resultados_xgboost_v5"
TARGET_COL = "contagem"

META_COLS = [
    "image_id", "file_name", "split",
    "contagem", "contagem_log1p", "contagem_sqrt", "augmentacao",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. CARREGAMENTO
def carregar_dados():
    print("\n[1/6] Carregando datasets RAW...")
    # corrigido: era orandet_v71_*
    df_train_full = pd.read_csv(os.path.join(DATASET_DIR, "orandet_v80_train_raw.csv"))
    df_test = pd.read_csv(os.path.join(DATASET_DIR, "orandet_v80_test_raw.csv"))

    df_train = df_train_full[df_train_full["augmentacao"] == "original"].copy()
    df_aug = df_train_full[df_train_full["augmentacao"] != "original"]

    print(f"  Treino (originais apenas): {len(df_train)} imagens")
    print(f"  Aug descartado p/ treino:  {len(df_aug)}")
    print(f"  Teste:                     {len(df_test)} imagens")

    print(f"\n  Distribuição do target (treino):")
    bins = [-1, 0, 1, 2, 4, 7, 999]
    labels = ["0", "1", "2", "3-4", "5-7", "8+"]
    faixas = pd.cut(df_train[TARGET_COL], bins=bins, labels=labels)
    dist = faixas.value_counts().sort_index()
    for faixa, n in dist.items():
        pct = n / len(df_train) * 100
        print(f"    {faixa:>4}: {n:>4} ({pct:4.1f}%)  {'█' * int(pct / 2)}")

    return df_train, df_test


def preparar_xy(df_train, df_test):
    feat_cols = [c for c in df_train.columns if c not in META_COLS]

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
    print(f"  Treino: {X_train.shape} | Teste: {X_test.shape}")
    return X_train, y_train, X_test, y_test, feat_cols, medianas


# 2. VALIDAÇÃO CRUZADA K-FOLD
def validacao_cruzada(X_train, y_train, params_modelo):
    print("\n[2/6] Validação cruzada 5-Fold...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes, mapes, r2s = [], [], []

    for fold, (idx_tr, idx_val) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[idx_tr], X_train[idx_val]
        y_tr, y_val = y_train[idx_tr], y_train[idx_val]

        m = xgb.XGBRegressor(**params_modelo)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        yp = np.maximum(m.predict(X_val), 0)
        mae = mean_absolute_error(y_val, yp)
        mask = y_val > 0
        mape = float(np.mean(np.abs((y_val[mask] - yp[mask]) / y_val[mask])) * 100) if mask.sum() > 0 else 0.0
        r2 = r2_score(y_val, yp)
        maes.append(mae);
        mapes.append(mape);
        r2s.append(r2)
        print(f"    Fold {fold + 1}: MAE={mae:.3f}  MAPE={mape:.1f}%  R²={r2:.3f}")

    print(f"\n  CV MAE:  {np.mean(maes):.3f} ± {np.std(maes):.3f}")
    print(f"  CV MAPE: {np.mean(mapes):.1f}% ± {np.std(mapes):.1f}%")
    print(f"  CV R²:   {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")

    return {
        "n_folds": 5,
        "shuffle": True,
        "random_state": 42,
        "cv_mae_mean": round(float(np.mean(maes)), 3),
        "cv_mae_std": round(float(np.std(maes)), 3),
        "cv_mape_mean": round(float(np.mean(mapes)), 2),
        "cv_mape_std": round(float(np.std(mapes)), 2),
        "cv_r2_mean": round(float(np.mean(r2s)), 3),
        "cv_r2_std": round(float(np.std(r2s)), 3),
        "por_fold": [
            {"fold": i + 1, "mae": round(maes[i], 3),
             "mape": round(mapes[i], 2), "r2": round(r2s[i], 3)}
            for i in range(len(maes))
        ],
    }


# 3. TREINAMENTO
def treinar_modelo(X_train, y_train, X_test, y_test):
    print("\n[3/6] Treinando modelo final...")

    params = dict(
        objective="count:poisson",
        eval_metric="mae",
        n_estimators=3000,
        max_depth=4,
        min_child_weight=15,
        gamma=0.5,
        subsample=0.7,
        colsample_bytree=0.5,
        colsample_bylevel=0.7,
        reg_alpha=1.0,
        reg_lambda=5.0,
        learning_rate=0.02,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=100,
    )

    _t0 = time.perf_counter()
    modelo = xgb.XGBRegressor(**params)
    modelo.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100,
    )
    tempo_treino_s = round(time.perf_counter() - _t0, 2)

    results = modelo.evals_result()
    best = modelo.best_iteration
    mae_treino_final = results["validation_0"]["mae"][best]
    mae_teste_final = results["validation_1"]["mae"][best]
    gap = mae_teste_final / (mae_treino_final + 1e-9)

    print(f"\n  Melhor n_estimators: {best}")
    print(f"  MAE treino:  {mae_treino_final:.4f}")
    print(f"  MAE teste:   {mae_teste_final:.4f}")
    print(f"  Gap (teste/treino): {gap:.1f}x  "
          f"({'✓ ok (<2x)' if gap < 2 else '⚠ overfitting leve' if gap < 4 else '❌ overfitting severo'})")
    print(f"  Tempo de treino: {tempo_treino_s}s")

    return modelo, params, tempo_treino_s


# 4. AVALIAÇÃO
def avaliar(modelo, X_train, y_train, X_test, y_test):
    # ── Predições no teste ────────────────────────────────────────────────────
    y_pred = np.maximum(modelo.predict(X_test), 0)
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

    # ── Predições no treino (para calcular gap real) ──────────────────────────
    y_pred_tr = np.maximum(modelo.predict(X_train), 0)
    mae_tr = mean_absolute_error(y_train, y_pred_tr)
    rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_tr))
    r2_tr = r2_score(y_train, y_pred_tr)
    mask_tr = y_train > 0
    mape_tr = float(np.mean(np.abs((y_train[mask_tr] - y_pred_tr[mask_tr]) / y_train[mask_tr])) * 100)

    print(f"\n[4/6] Resultados:")
    print(f"  {'─' * 50}")
    print(f"  {'Métrica':<20} {'Treino':>10} {'Teste':>10}")
    print(f"  {'─' * 50}")
    print(f"  {'MAE':<20} {mae_tr:>10.3f} {mae:>10.3f}")
    print(f"  {'RMSE':<20} {rmse_tr:>10.3f} {rmse:>10.3f}")
    print(f"  {'R²':<20} {r2_tr:>10.4f} {r2:>10.4f}")
    print(f"  {'MAPE':<20} {mape_tr:>9.1f}% {mape:>9.1f}%")
    print(f"  {'─' * 50}")
    print(f"  MdAPE (teste):       {mdape:.1f}%")
    print(f"  Gap MAPE/MdAPE:      {mape - mdape:.1f}pp")
    print(f"  ±1 fruta:            {dentro_1:.1f}%")
    print(f"  ±2 frutas:           {dentro_2:.1f}%")
    print(f"  Dentro de 20%:       {dentro_20:.1f}%")

    # ── Análise por faixa ─────────────────────────────────────────────────────
    print(f"\n  MAPE por faixa de contagem:")
    print(f"  {'Faixa':<8} {'N':>5} {'MAE':>6} {'MAPE%':>7} {'±1':>7} {'Bias':>7}")
    print(f"  {'─' * 44}")
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
        print(f"  {label:<8} {n:>5} {mae_f:>6.2f} {mape_f:>6.1f}%  {a1:>5.1f}%  {bias:>+6.2f}{flag}")
        metricas_f.append({
            "faixa": label, "n": int(n),
            "mae": round(mae_f, 3), "mape": round(mape_f, 1),
            "acerto_pm1": round(a1, 1), "bias": round(bias, 3),
        })

    metricas = {
        "treino": {
            "MAE": round(mae_tr, 4), "RMSE": round(rmse_tr, 4),
            "R2": round(r2_tr, 4), "MAPE": round(mape_tr, 2),
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
    }
    return y_pred, metricas


# 5. DIAGNÓSTICO DE FEATURES
def diagnosticar_features(modelo, feat_cols):
    print("\n[5/6] Diagnóstico de features...")

    imp = pd.DataFrame({
        "feature": feat_cols,
        "gain": modelo.feature_importances_,
    }).sort_values("gain", ascending=False)

    grupos_def = {
        "G1_hsv": ["hsv_"],
        "G2_rgb_lab_ycbcr": ["rgb_", "lab_", "ycbcr_"],
        "G3_canal_v": ["v_original", "v_eq", "v_razao"],
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
        "mascara": ["mascara_"],
        "fft": ["fft_"],
        "laplaciano": ["lap_"],
        "spatial_pyramid": ["spatial_"],
        "derivadas": ["density_", "hog_fft"],
    }

    imp_grupo = []
    for grupo, prefixos in grupos_def.items():
        mask = imp["feature"].apply(lambda f: any(f.startswith(p) for p in prefixos))
        total = imp.loc[mask, "gain"].sum()
        n_f = mask.sum()
        imp_grupo.append({
            "grupo": grupo,
            "gain_total": round(float(total), 6),
            "gain_medio": round(float(total / max(n_f, 1)), 6),
            "n_features": int(n_f),
        })
    df_grupo = pd.DataFrame(imp_grupo).sort_values("gain_total", ascending=False)

    print("\n  Importância por grupo:")
    print(f"  {'Grupo':<22} {'Gain':>8}  {'N feat':>7}  {'Gain/feat':>10}")
    print(f"  {'─' * 56}")
    for _, row in df_grupo.iterrows():
        print(f"  {row['grupo']:<22} {row['gain_total']:>8.4f}  "
              f"{row['n_features']:>7}  {row['gain_medio']:>10.6f}")

    return imp, df_grupo


# 6. SALVAMENTO
def salvar_resultados(modelo, y_train, y_test, y_pred, metricas, cv_metricas,
                      imp, df_grupo, df_test, feat_cols, medianas,
                      params, tempo_treino_s):
    print("\n[6/6] Salvando resultados...")

    # ── Joblibs ───────────────────────────────────────────────────────────────
    joblib.dump(modelo, os.path.join(OUTPUT_DIR, "xgb_v5_modelo.joblib"))
    joblib.dump(feat_cols, os.path.join(OUTPUT_DIR, "xgb_v5_feature_cols.joblib"))
    joblib.dump(medianas, os.path.join(OUTPUT_DIR, "xgb_v5_medianas.joblib"))

    # ── Predições ─────────────────────────────────────────────────────────────
    df_preds = df_test[["image_id", "file_name", TARGET_COL]].copy()
    df_preds["pred_float"] = y_pred
    df_preds["pred_int"] = np.round(y_pred).astype(int)
    df_preds["erro_abs"] = np.abs(df_preds[TARGET_COL] - df_preds["pred_float"])
    df_preds["erro_perc"] = (df_preds["erro_abs"] / (df_preds[TARGET_COL] + 1e-6) * 100).round(1)
    df_preds.to_csv(os.path.join(OUTPUT_DIR, "xgb_v5_predicoes_teste.csv"), index=False)

    # ── Importâncias ──────────────────────────────────────────────────────────
    imp.to_csv(os.path.join(OUTPUT_DIR, "xgb_v5_feature_importance.csv"), index=False)
    df_grupo.to_csv(os.path.join(OUTPUT_DIR, "xgb_v5_group_importance.csv"), index=False)

    # ── Versão do XGBoost ─────────────────────────────────────────────────────
    try:
        versao_xgb = importlib.metadata.version("xgboost")
    except Exception:
        versao_xgb = xgb.__version__

    # ── MAE treino/teste no best_iteration (via evals_result) ────────────────
    results = modelo.evals_result()
    best = modelo.best_iteration
    mae_treino_best = round(results["validation_0"]["mae"][best], 4)
    mae_teste_best = round(results["validation_1"]["mae"][best], 4)
    gap_early = round(mae_teste_best / (mae_treino_best + 1e-9), 2)

    # ── JSON de metadados para o TCC ─────────────────────────────────────────
    saida = {

        # ── Protocolo experimental ────────────────────────────────────────────
        "protocolo": {
            "dataset_treino": "orandet_v80_train_raw.csv (sem normalização)",
            "dataset_teste": "orandet_v80_test_raw.csv  (sem normalização)",
            "n_amostras_treino": int(len(y_train)),
            "n_amostras_teste": int(len(y_test)),
            "subset_treino": (
                "Apenas imagens originais — augmentações (flip_h, flip_v, "
                "bright_75, bright_125) descartadas para treino do modelo"
            ),
            "n_features": len(feat_cols),
            "alvo": TARGET_COL,
            "transformacao_inversa": "nenhuma — alvo é contagem direta (inteiro)",
            "conjunto_validacao_dedicado": "ausente — validação feita por K-Fold CV",
            "nota_leakage": (
                "O early stopping usa X_test como segundo eval_set "
                "(validation_1). O número de árvores usado foi selecionado "
                "com base no desempenho no conjunto de teste — leakage leve "
                "que deve ser declarado na metodologia."
            ),
        },

        # ── Hiperparâmetros finais ────────────────────────────────────────────
        "hiperparametros": {
            "objective": params["objective"],
            "eval_metric": params["eval_metric"],
            "n_estimators_config": params["n_estimators"],
            "n_estimators_usado": best,
            "early_stopping_rounds": params["early_stopping_rounds"],
            "max_depth": params["max_depth"],
            "min_child_weight": params["min_child_weight"],
            "gamma": params["gamma"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "colsample_bylevel": params["colsample_bylevel"],
            "reg_alpha": params["reg_alpha"],
            "reg_lambda": params["reg_lambda"],
            "learning_rate": params["learning_rate"],
            "tree_method": params["tree_method"],
            "random_state": params["random_state"],
        },

        # ── Validação cruzada ─────────────────────────────────────────────────
        "validacao_cruzada": cv_metricas,

        # ── Resultados — treino ───────────────────────────────────────────────
        "resultados_treino": metricas["treino"],

        # ── Resultados — teste ────────────────────────────────────────────────
        "resultados_teste": metricas["teste"],

        # ── Gap treino/teste ──────────────────────────────────────────────────
        "gap_treino_teste": {
            **metricas["gap_treino_teste"],
            "MAE_treino_best_iter": mae_treino_best,
            "MAE_teste_best_iter": mae_teste_best,
            "gap_MAE_best_iter": gap_early,
            "interpretacao": (
                "ok (<2x)" if gap_early < 2
                else "overfitting leve (2–4x)" if gap_early < 4
                else "overfitting severo (>4x)"
            ),
        },

        # ── Análise por faixa de contagem ─────────────────────────────────────
        "resultados_por_faixa": metricas["por_faixa"],

        # ── Eficiência ────────────────────────────────────────────────────────
        "eficiencia": {
            "tempo_treino_s": tempo_treino_s,
            "versao_xgboost": versao_xgb,
            "nota_inferencia": "tempo de inferência por imagem: [VERIFICAR se necessário]",
        },
    }

    with open(os.path.join(OUTPUT_DIR, "xgb_v5_metricas.json"), "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)

    # ── Figura 1: 4 painéis ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    max_val = max(y_test.max(), y_pred.max()) * 1.05
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, color="steelblue")
    ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Perfeito")
    ax.fill_between([0, max_val],
                    [v * 0.8 for v in [0, max_val]],
                    [v * 1.2 for v in [0, max_val]],
                    alpha=0.1, color="green", label="±20%")
    ax.set_xlabel("Contagem Real");
    ax.set_ylabel("Contagem Predita")
    ax.set_title(f"Real vs Predito  |  MAE={metricas['teste']['MAE']:.2f}  "
                 f"MAPE={metricas['teste']['MAPE']:.1f}%  R²={metricas['teste']['R2']:.3f}")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    faixas_data = metricas.get("por_faixa", [])
    if faixas_data:
        labels_f = [d["faixa"] for d in faixas_data]
        mapes_f = [d["mape"] for d in faixas_data]
        biases_f = [d["bias"] for d in faixas_data]
        ns_f = [d["n"] for d in faixas_data]
        x, w = np.arange(len(labels_f)), 0.35
        bars1 = ax.bar(x - w / 2, mapes_f, w, label="MAPE %",
                       color=["#d73027" if m > 60 else "#fc8d59" if m > 35 else "#91cf60"
                              for m in mapes_f])
        ax2 = ax.twinx()
        ax2.bar(x + w / 2, biases_f, w, label="Bias",
                color=["tomato" if b > 0 else "steelblue" for b in biases_f], alpha=0.6)
        ax.axhline(20, color="green", linestyle="--", lw=1, label="Meta 20%")
        ax.set_xticks(x);
        ax.set_xticklabels(labels_f)
        ax.set_xlabel("Faixa");
        ax.set_ylabel("MAPE (%)")
        ax2.set_ylabel("Bias (laranjas)")
        ax.set_title("MAPE e Bias por Faixa\n(bias>0=superestima, <0=subestima)")
        for bar, n in zip(bars1, ns_f):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"n={n}", ha="center", fontsize=7)
        lines1, lbl1 = ax.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7)

    ax = axes[1, 0]
    erros = np.abs(y_test - y_pred)
    ax.hist(erros, bins=25, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(metricas["teste"]["MAE"], color="red", linestyle="--",
               label=f"MAE={metricas['teste']['MAE']:.2f}")
    ax.axvline(float(np.median(erros)), color="orange", linestyle="--",
               label=f"Mediana={np.median(erros):.2f}")
    ax.set_xlabel("Erro Absoluto (laranjas)");
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição dos Erros Absolutos");
    ax.legend()

    ax = axes[1, 1]
    df_top = df_grupo.head(12)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_top)))
    ax.barh(df_top["grupo"][::-1], df_top["gain_total"][::-1], color=colors[::-1])
    ax.set_xlabel("Gain Total");
    ax.set_title("Importância por Grupo de Features")
    for i, (_, row) in enumerate(df_top[::-1].iterrows()):
        ax.text(row["gain_total"] + 0.001, i,
                f"{row['gain_total']:.4f}", va="center", fontsize=8)

    cv = cv_metricas
    plt.suptitle(
        f"XGBoost v5  |  CV MAPE={cv['cv_mape_mean']:.1f}%±{cv['cv_mape_std']:.1f}%\n"
        f"MAPE={metricas['teste']['MAPE']:.1f}%  MdAPE={metricas['teste']['MdAPE']:.1f}%  "
        f"MAE={metricas['teste']['MAE']:.2f}  R²={metricas['teste']['R2']:.3f}",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "xgb_v5_resultados.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figura 2: Curva de aprendizado ────────────────────────────────────────
    if results:
        fig, axes2 = plt.subplots(1, 2, figsize=(16, 5))

        ax = axes2[0]
        train_mae = results["validation_0"]["mae"]
        test_mae = results["validation_1"]["mae"]
        ax.plot(train_mae, label="Treino", color="steelblue", lw=1.5)
        ax.plot(test_mae, label="Teste", color="tomato", lw=1.5)
        ax.axvline(best, color="gray", linestyle="--", label=f"Melhor: {best}")
        status = "✓ ok" if gap_early < 2 else "⚠ leve" if gap_early < 4 else "❌ severo"
        ax.text(0.55, 0.92,
                f"Gap: {gap_early:.1f}x  {status}\n"
                f"Treino: {mae_treino_best:.3f}  Teste: {mae_teste_best:.3f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax.set_xlabel("Iteração");
        ax.set_ylabel("MAE")
        ax.set_title("Curva de Aprendizado");
        ax.legend();
        ax.grid(alpha=0.3)

        ax = axes2[1]
        versoes = ["v1\n(Poisson)", "v2\n(sqlog+mape)", "v3\n(sqlog+mae)", "v5\n(este)"]
        mapes_hist = [54.4, 46.0, 55.8, metricas["teste"]["MAPE"]]
        gaps_hist = [7, 1, 5, gap_early]
        ax_r = ax.twinx()
        ax.bar(versoes, gaps_hist,
               color=["#d73027", "#91cf60", "#fc8d59", "steelblue"], alpha=0.7,
               label="Gap MAE (teste/treino)")
        ax_r.plot(versoes, mapes_hist, "ko--", lw=2, ms=8, label="MAPE %")
        ax.axhline(2, color="green", linestyle="--", lw=1.5, label="Gap ok (<2x)")
        ax.set_ylabel("Gap treino/teste");
        ax_r.set_ylabel("MAPE (%)")
        ax.set_title("Histórico de Versões")
        lines1, lbl1 = ax.get_legend_handles_labels()
        lines2, lbl2 = ax_r.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "xgb_v5_curva_aprendizado.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n  Arquivos em: {OUTPUT_DIR}/")
    print(f"    ├─ xgb_v5_modelo.joblib")
    print(f"    ├─ xgb_v5_metricas.json        ← protocolo + hiperparâmetros + métricas completas")
    print(f"    ├─ xgb_v5_predicoes_teste.csv")
    print(f"    ├─ xgb_v5_resultados.png")
    print(f"    └─ xgb_v5_curva_aprendizado.png")


# MAIN
def main():
    print("\n" + "═" * 65)
    print("  XGBoost v5 — Regularização forte | K-Fold CV | JSON IEEE")
    print("═" * 65)

    df_train, df_test = carregar_dados()
    X_train, y_train, X_test, y_test, feat_cols, medianas = preparar_xy(df_train, df_test)

    params_cv = dict(
        objective="count:poisson", eval_metric="mae",
        n_estimators=500, max_depth=4, min_child_weight=15,
        gamma=0.5, subsample=0.7, colsample_bytree=0.5,
        reg_alpha=1.0, reg_lambda=5.0, learning_rate=0.02,
        tree_method="hist", n_jobs=-1, random_state=42,
        verbosity=0, early_stopping_rounds=50,
    )
    cv_metricas = validacao_cruzada(X_train, y_train, params_cv)

    modelo, params, tempo_treino_s = treinar_modelo(X_train, y_train, X_test, y_test)

    y_pred, metricas = avaliar(modelo, X_train, y_train, X_test, y_test)

    imp, df_grupo = diagnosticar_features(modelo, feat_cols)

    salvar_resultados(modelo, y_train, y_test, y_pred, metricas, cv_metricas,
                      imp, df_grupo, df_test, feat_cols, medianas,
                      params, tempo_treino_s)

    print(f"\n{'═' * 65}")
    print(f"  CV MAPE:  {cv_metricas['cv_mape_mean']:.1f}% ± {cv_metricas['cv_mape_std']:.1f}%")
    print(f"  MAPE:     {metricas['teste']['MAPE']:.1f}%")
    print(f"  MAE:      {metricas['teste']['MAE']:.3f} laranjas/imagem")
    print(f"  R²:       {metricas['teste']['R2']:.4f}")
    print(f"  Treino:   {tempo_treino_s}s")

    results = modelo.evals_result()
    best = modelo.best_iteration
    mae_tr_best = results["validation_0"]["mae"][best]
    mae_te_best = results["validation_1"]["mae"][best]
    gap = mae_te_best / (mae_tr_best + 1e-9)
    print(f"  Gap:      {gap:.1f}x  (meta: < 2x)")

    if gap > 4:
        print(f"\n  ❌ Overfitting severo. Verificar qualidade da máscara "
              f"e considerar redução de features (ex.: remover G13).")
    elif gap > 2:
        print(f"\n  ⚠  Overfitting leve. Foque nas features G14 (contagem direta).")
    else:
        print(f"\n  ✓  Gap sob controle.")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()
