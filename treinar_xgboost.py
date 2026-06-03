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
from sklearn.model_selection import KFold, train_test_split

# CONFIGURAÇÃO
DATASET_DIR = "./dataset_preparado_v11"
OUTPUT_DIR = "./resultados_xgboost_v11"
TARGET_COL = "contagem"

META_COLS = [
    "image_id", "file_name", "split",
    "contagem", "contagem_log1p", "contagem_sqrt", "augmentacao",
]

# Configurações de protocolo
SEED = 42
VAL_INTERNA_FRAC = 0.15  # 15% do treino vira validação interna
INFERENCIA_REPS = 10  # número de execuções para média de tempo de inferência

os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. CARREGAMENTO — agora COM augmentadas para igualar MLP/SVR

def carregar_dados():
    print("\n[1/7] Carregando datasets RAW...")
    df_train_full = pd.read_csv(os.path.join(DATASET_DIR, "orandet_v11_train_raw.csv"))
    df_test = pd.read_csv(os.path.join(DATASET_DIR, "orandet_v11_test_raw.csv"))

    df_orig = df_train_full[df_train_full["augmentacao"] == "original"].copy()
    df_aug = df_train_full[df_train_full["augmentacao"] != "original"].copy()

    print(f"  Treino total (orig + aug): {len(df_train_full)} amostras")
    print(f"    └─ originais:            {len(df_orig)} imagens")
    print(f"    └─ augmentadas:          {len(df_aug)} amostras")
    print(f"  Teste:                     {len(df_test)} imagens")

    print(f"\n  Distribuição do target (treino — originais):")
    bins = [-1, 0, 1, 2, 4, 7, 999]
    labels = ["0", "1", "2", "3-4", "5-7", "8+"]
    faixas = pd.cut(df_orig[TARGET_COL], bins=bins, labels=labels)
    dist = faixas.value_counts().sort_index()
    for faixa, n in dist.items():
        pct = n / len(df_orig) * 100
        print(f"    {faixa:>4}: {n:>4} ({pct:4.1f}%)  {'█' * int(pct / 2)}")

    return df_train_full, df_orig, df_test


def preparar_xy(df_train_full, df_test):
    feat_cols = [c for c in df_train_full.columns if c not in META_COLS]

    for col in feat_cols:
        df_train_full[col] = df_train_full[col].replace([np.inf, -np.inf], np.nan)
        df_test[col] = df_test[col].replace([np.inf, -np.inf], np.nan)

    medianas = df_train_full[feat_cols].median()
    df_train_full[feat_cols] = df_train_full[feat_cols].fillna(medianas)
    df_test[feat_cols] = df_test[feat_cols].fillna(medianas)

    X_train_full = df_train_full[feat_cols].values.astype(np.float32)
    y_train_full = df_train_full[TARGET_COL].values.astype(np.float32)
    X_test = df_test[feat_cols].values.astype(np.float32)
    y_test = df_test[TARGET_COL].values.astype(np.float32)

    print(f"\n  Features: {X_train_full.shape[1]}")
    print(f"  Treino completo: {X_train_full.shape} | Teste: {X_test.shape}")
    return X_train_full, y_train_full, X_test, y_test, feat_cols, medianas


# 2. VALIDAÇÃO CRUZADA K-FOLD — apenas em originais para evitar leakage

def validacao_cruzada(df_orig, feat_cols, params_modelo):
    print("\n[2/7] Validação cruzada 5-Fold (originais apenas — evita leakage)...")

    X = df_orig[feat_cols].values.astype(np.float32)
    y = df_orig[TARGET_COL].values.astype(np.float32)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    maes, mapes, r2s = [], [], []

    for fold, (idx_tr, idx_val) in enumerate(kf.split(X)):
        X_tr, X_val = X[idx_tr], X[idx_val]
        y_tr, y_val = y[idx_tr], y[idx_val]

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
        "random_state": SEED,
        "amostras_cv": "originais apenas (sem augmentadas — evita leakage entre folds)",
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


# 3. SPLIT DE VALIDAÇÃO INTERNA — elimina o vazamento do early stopping

def split_validacao_interna(X_train_full, y_train_full, df_train_full):
    """
    Cria split estratificado por faixa de contagem dentro do treino.
    O eval_set do early stopping passa a ser este conjunto de validação,
    não o teste. Estratificação garante que faixas raras (8+) apareçam
    em ambos os subsets.
    """
    print(f"\n[3/7] Criando split de validação interna ({int(VAL_INTERNA_FRAC * 100)}% do treino)...")

    # Faixas para estratificação (mesmas usadas na avaliação)
    bins = [-1, 0, 1, 2, 4, 7, 999]
    labels = ["0", "1", "2", "3-4", "5-7", "8+"]
    faixas = pd.cut(df_train_full[TARGET_COL], bins=bins, labels=labels)

    # train_test_split estratificado
    idx_tr, idx_val = train_test_split(
        np.arange(len(X_train_full)),
        test_size=VAL_INTERNA_FRAC,
        random_state=SEED,
        stratify=faixas,
    )

    X_tr, y_tr = X_train_full[idx_tr], y_train_full[idx_tr]
    X_val, y_val = X_train_full[idx_val], y_train_full[idx_val]

    print(f"  Treino efetivo:    {len(X_tr)} amostras ({(1 - VAL_INTERNA_FRAC) * 100:.0f}%)")
    print(f"  Validação interna: {len(X_val)} amostras ({VAL_INTERNA_FRAC * 100:.0f}%)")
    print(f"  Estratificação:    por faixa de contagem (0, 1, 2, 3-4, 5-7, 8+)")
    print(f"  random_state:      {SEED}")

    return X_tr, y_tr, X_val, y_val, idx_tr, idx_val


# 4. TREINAMENTO — early stopping agora usa validação interna, não teste

def treinar_modelo(X_tr, y_tr, X_val, y_val):
    print("\n[4/7] Treinando modelo final (early stopping sem vazamento)...")

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
        random_state=SEED,
        verbosity=0,
        early_stopping_rounds=100,
    )

    _t0 = time.perf_counter()
    modelo = xgb.XGBRegressor(**params)
    # eval_set: (treino efetivo, validação interna) — TESTE NÃO ENTRA AQUI
    modelo.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=100,
    )
    tempo_treino_s = round(time.perf_counter() - _t0, 2)

    results = modelo.evals_result()
    best = modelo.best_iteration
    mae_tr_final = results["validation_0"]["mae"][best]
    mae_val_final = results["validation_1"]["mae"][best]
    gap_tr_val = mae_val_final / (mae_tr_final + 1e-9)

    print(f"\n  Melhor n_estimators: {best}")
    print(f"  MAE treino efetivo:  {mae_tr_final:.4f}")
    print(f"  MAE validação int.:  {mae_val_final:.4f}")
    print(f"  Gap (val/treino):    {gap_tr_val:.2f}x  "
          f"({'✓ ok (<2x)' if gap_tr_val < 2 else '⚠ overfitting leve' if gap_tr_val < 4 else '❌ overfitting severo'})")
    print(f"  Tempo de treino:     {tempo_treino_s}s")
    print(f"  ✓ Teste NÃO foi usado para early stopping — sem vazamento")

    return modelo, params, tempo_treino_s


# 5. AVALIAÇÃO — agora também mede tempo de inferência

def avaliar(modelo, X_train_full, y_train_full, X_test, y_test):
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

    # ── Predições no treino completo (originais + aug) ────────────────────────
    y_pred_tr = np.maximum(modelo.predict(X_train_full), 0)
    mae_tr = mean_absolute_error(y_train_full, y_pred_tr)
    rmse_tr = np.sqrt(mean_squared_error(y_train_full, y_pred_tr))
    r2_tr = r2_score(y_train_full, y_pred_tr)
    mask_tr = y_train_full > 0
    mape_tr = float(np.mean(np.abs(
        (y_train_full[mask_tr] - y_pred_tr[mask_tr]) / y_train_full[mask_tr]
    )) * 100)

    print(f"\n[5/7] Resultados:")
    print(f"  {'Métrica':<20} {'Treino':>10} {'Teste':>10}")
    print(f"  {'─' * 42}")
    print(f"  {'MAE':<20} {mae_tr:>10.3f} {mae:>10.3f}")
    print(f"  {'RMSE':<20} {rmse_tr:>10.3f} {rmse:>10.3f}")
    print(f"  {'R²':<20} {r2_tr:>10.4f} {r2:>10.4f}")
    print(f"  {'MAPE':<20} {mape_tr:>9.1f}% {mape:>9.1f}%")
    print(f"  {'─' * 42}")
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


# 6. MEDIÇÃO DE TEMPO DE INFERÊNCIA — N=10 execuções

def medir_tempo_inferencia(modelo, X_test, n_reps=INFERENCIA_REPS):
    """
    Mede tempo de inferência em N=10 execuções.
    Reporta total da batch, por imagem, primeira execução (cache cold).
    """
    print(f"\n[6/7] Medindo tempo de inferência ({n_reps} execuções)...")

    n_imagens = len(X_test)
    tempos = []

    # Primeira execução tende a ser mais lenta (cache cold, JIT compilation)
    _t0 = time.perf_counter()
    _ = modelo.predict(X_test)
    t_primeiro = time.perf_counter() - _t0
    tempos.append(t_primeiro)

    # Execuções subsequentes
    for _ in range(n_reps - 1):
        _t0 = time.perf_counter()
        _ = modelo.predict(X_test)
        tempos.append(time.perf_counter() - _t0)

    tempos_arr = np.array(tempos)
    tempo_total_mean = float(np.mean(tempos_arr))
    tempo_total_std = float(np.std(tempos_arr))
    tempo_por_img = tempo_total_mean / n_imagens
    tempo_por_img_ms = tempo_por_img * 1000

    print(f"  Batch total ({n_imagens} imagens): {tempo_total_mean * 1000:.2f} ± {tempo_total_std * 1000:.2f} ms")
    print(f"  Por imagem:                       {tempo_por_img_ms:.4f} ms ({tempo_por_img * 1e6:.1f} µs)")
    print(f"  Primeira execução (cache cold):   {t_primeiro * 1000:.2f} ms")
    print(f"  Execuções subsequentes (média):   {np.mean(tempos_arr[1:]) * 1000:.2f} ms")

    return {
        "n_execucoes": n_reps,
        "n_imagens_batch": int(n_imagens),
        "tempo_batch_total_ms": round(tempo_total_mean * 1000, 3),
        "tempo_batch_std_ms": round(tempo_total_std * 1000, 3),
        "tempo_por_imagem_ms": round(tempo_por_img_ms, 4),
        "tempo_por_imagem_us": round(tempo_por_img * 1e6, 1),
        "tempo_primeira_exec_ms": round(t_primeiro * 1000, 3),
        "tempo_subsequentes_ms": round(float(np.mean(tempos_arr[1:])) * 1000, 3),
        "nota": (
            "Medido com time.perf_counter() em batch predict. "
            "Tempo por imagem = tempo da batch / n_imagens. "
            "A primeira execução tende a ser mais lenta (cache cold)."
        ),
    }


# 7. DIAGNÓSTICO DE FEATURES

def diagnosticar_features(modelo, feat_cols):
    print("\n[7/7] Diagnóstico de features e salvamento...")

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


# SALVAMENTO

def salvar_resultados(modelo, X_test, y_train_full, y_test, y_pred,
                      metricas, cv_metricas, tempo_inferencia,
                      imp, df_grupo, df_test, feat_cols, medianas,
                      params, tempo_treino_s,
                      n_orig, n_aug, n_tr_efetivo, n_val_interna,
                      idx_tr, idx_val):
    # ── Joblibs + CSVs ────────────────────────────────────────────────────────
    joblib.dump(modelo, os.path.join(OUTPUT_DIR, "xgb_v6_modelo.joblib"))
    joblib.dump(feat_cols, os.path.join(OUTPUT_DIR, "xgb_v6_feature_cols.joblib"))
    joblib.dump(medianas, os.path.join(OUTPUT_DIR, "xgb_v6_medianas.joblib"))

    df_preds = df_test[["image_id", "file_name", TARGET_COL]].copy()
    df_preds["pred_float"] = y_pred
    df_preds["pred_int"] = np.round(y_pred).astype(int)
    df_preds["erro_abs"] = np.abs(df_preds[TARGET_COL] - df_preds["pred_float"])
    df_preds["erro_perc"] = (df_preds["erro_abs"] / (df_preds[TARGET_COL] + 1e-6) * 100).round(1)
    df_preds.to_csv(os.path.join(OUTPUT_DIR, "xgb_v6_predicoes_teste.csv"), index=False)

    imp.to_csv(os.path.join(OUTPUT_DIR, "xgb_v6_feature_importance.csv"), index=False)
    df_grupo.to_csv(os.path.join(OUTPUT_DIR, "xgb_v6_group_importance.csv"), index=False)

    try:
        versao_xgb = importlib.metadata.version("xgboost")
    except Exception:
        versao_xgb = xgb.__version__

    results = modelo.evals_result()
    best = modelo.best_iteration
    mae_tr_best = round(results["validation_0"]["mae"][best], 4)
    mae_val_best = round(results["validation_1"]["mae"][best], 4)
    gap_tr_val = round(mae_val_best / (mae_tr_best + 1e-9), 2)

    # ── JSON ──────────────────────────────────────────────────────────────────
    saida = {
        # ── Protocolo experimental ────────────────────────────────────────────
        "protocolo": {
            "dataset_treino": "orandet_v11_train_raw.csv (sem normalização)",
            "dataset_teste": "orandet_v11_test_raw.csv  (sem normalização)",
            "n_amostras_treino_total": int(len(y_train_full)),
            "n_amostras_treino_originais": int(n_orig),
            "n_amostras_treino_augmentadas": int(n_aug),
            "n_amostras_treino_efetivo": int(n_tr_efetivo),
            "n_amostras_validacao_interna": int(n_val_interna),
            "n_amostras_teste": int(len(y_test)),
            "subset_treino": (
                "Originais + augmentadas (igualando MLP e SVR para comparação justa). "
                "Mudança vs. v5: o XGBoost agora vê o mesmo conjunto de treino que os "
                "outros dois modelos."
            ),
            "n_features": len(feat_cols),
            "alvo": TARGET_COL,
            "transformacao_inversa": "nenhuma — alvo é contagem direta (inteiro)",
        },

        # ── Protocolo anti-vazamento ──────────────────────────────────────────
        "protocolo_anti_vazamento": {
            "problema_v5": (
                "Na v5, o early stopping usava o conjunto de teste como eval_set, "
                "selecionando n_estimators com base no desempenho no teste — "
                "vazamento leve declarado."
            ),
            "correcao_v6": (
                "Criado split estratificado interno do treino: "
                f"{int((1 - VAL_INTERNA_FRAC) * 100)}% treino efetivo + "
                f"{int(VAL_INTERNA_FRAC * 100)}% validação interna. "
                "O eval_set do early stopping agora é a validação interna, "
                "NÃO o conjunto de teste. Teste permanece intocado até "
                "avaliação final."
            ),
            "fracao_validacao_interna": VAL_INTERNA_FRAC,
            "estratificacao": "por faixa de contagem (0, 1, 2, 3-4, 5-7, 8+)",
            "random_state": SEED,
            "metodo": "sklearn.model_selection.train_test_split com stratify",
        },

        # ── Hiperparâmetros finais ────────────────────────────────────────────
        "hiperparametros": {
            "objective": params["objective"],
            "eval_metric": params["eval_metric"],
            "n_estimators_config": params["n_estimators"],
            "n_estimators_usado": best,
            "early_stopping_rounds": params["early_stopping_rounds"],
            "early_stopping_em": "validação interna (não no teste)",
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

        # ── Validação cruzada (em originais) ──────────────────────────────────
        "validacao_cruzada": cv_metricas,

        # ── Resultados — treino ───────────────────────────────────────────────
        "resultados_treino": metricas["treino"],

        # ── Resultados — teste ────────────────────────────────────────────────
        "resultados_teste": metricas["teste"],

        # ── Gap treino/teste ──────────────────────────────────────────────────
        "gap_treino_teste": {
            **metricas["gap_treino_teste"],
            "MAE_treino_eff_best_iter": mae_tr_best,
            "MAE_validacao_int_best_iter": mae_val_best,
            "gap_MAE_val_vs_treino": gap_tr_val,
            "interpretacao_val_vs_treino": (
                "ok (<2x)" if gap_tr_val < 2
                else "overfitting leve (2–4x)" if gap_tr_val < 4
                else "overfitting severo (>4x)"
            ),
        },

        # ── Análise por faixa ─────────────────────────────────────────────────
        "resultados_por_faixa": metricas["por_faixa"],

        # ── Eficiência ────────────────────────────────────────────────────────
        "eficiencia": {
            "tempo_treino_s": tempo_treino_s,
            "tempo_inferencia": tempo_inferencia,
            "versao_xgboost": versao_xgb,
        },
    }

    with open(os.path.join(OUTPUT_DIR, "xgb_v11_metricas.json"), "w", encoding="utf-8") as f:
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
        ax.set_title("MAPE e Bias por Faixa")
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
    ax.set_xlabel("Erro Absoluto");
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição dos Erros Absolutos");
    ax.legend()

    ax = axes[1, 1]
    df_top = df_grupo.head(12)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_top)))
    ax.barh(df_top["grupo"][::-1], df_top["gain_total"][::-1], color=colors[::-1])
    ax.set_xlabel("Gain Total");
    ax.set_title("Importância por Grupo")
    for i, (_, row) in enumerate(df_top[::-1].iterrows()):
        ax.text(row["gain_total"] + 0.001, i,
                f"{row['gain_total']:.4f}", va="center", fontsize=8)

    cv = cv_metricas
    plt.suptitle(
        f"XGBoost v6  |  Sem vazamento | Treino com aug | CV MAPE={cv['cv_mape_mean']:.1f}%\n"
        f"MAPE={metricas['teste']['MAPE']:.1f}%  MdAPE={metricas['teste']['MdAPE']:.1f}%  "
        f"MAE={metricas['teste']['MAE']:.2f}  R²={metricas['teste']['R2']:.3f}",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "xgb_v6_resultados.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figura 2: Curva de aprendizado treino vs validação interna ────────────
    if results:
        fig, ax = plt.subplots(figsize=(10, 5))
        train_mae = results["validation_0"]["mae"]
        val_mae = results["validation_1"]["mae"]
        ax.plot(train_mae, label="Treino efetivo", color="steelblue", lw=1.5)
        ax.plot(val_mae, label="Validação interna", color="tomato", lw=1.5)
        ax.axvline(best, color="gray", linestyle="--", label=f"Melhor: {best}")
        status = "✓ ok" if gap_tr_val < 2 else "⚠ leve" if gap_tr_val < 4 else "❌ severo"
        ax.text(0.55, 0.92,
                f"Gap val/treino: {gap_tr_val:.1f}x  {status}\n"
                f"Treino: {mae_tr_best:.3f}  Val: {mae_val_best:.3f}\n"
                f"✓ Sem vazamento — teste não usado",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax.set_xlabel("Iteração");
        ax.set_ylabel("MAE")
        ax.set_title("Curva de Aprendizado — Treino vs Validação Interna")
        ax.legend();
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "xgb_v6_curva_aprendizado.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n  Arquivos em: {OUTPUT_DIR}/")
    print(f"    ├─ xgb_v6_modelo.joblib")
    print(f"    ├─ xgb_v11_metricas.json        ← protocolo + anti-vazamento + tempos")
    print(f"    ├─ xgb_v6_predicoes_teste.csv")
    print(f"    ├─ xgb_v6_resultados.png")
    print(f"    └─ xgb_v6_curva_aprendizado.png")


# MAIN

def main():
    print("\n" + "═" * 65)
    print("  XGBoost v6 — Sem vazamento | Treino com augmentadas | Tempo inferência")
    print("═" * 65)

    df_train_full, df_orig, df_test = carregar_dados()
    X_train_full, y_train_full, X_test, y_test, feat_cols, medianas = preparar_xy(
        df_train_full, df_test
    )

    # CV em originais apenas (evita leakage entre folds)
    params_cv = dict(
        objective="count:poisson", eval_metric="mae",
        n_estimators=500, max_depth=4, min_child_weight=15,
        gamma=0.5, subsample=0.7, colsample_bytree=0.5,
        reg_alpha=1.0, reg_lambda=5.0, learning_rate=0.02,
        tree_method="hist", n_jobs=-1, random_state=SEED,
        verbosity=0, early_stopping_rounds=50,
    )
    cv_metricas = validacao_cruzada(df_orig, feat_cols, params_cv)

    # Split de validação interna — elimina vazamento do early stopping
    X_tr, y_tr, X_val, y_val, idx_tr, idx_val = split_validacao_interna(
        X_train_full, y_train_full, df_train_full
    )

    # Treino final — agora eval_set é validação interna, não teste
    modelo, params, tempo_treino_s = treinar_modelo(X_tr, y_tr, X_val, y_val)

    # Avaliação no teste
    y_pred, metricas = avaliar(modelo, X_train_full, y_train_full, X_test, y_test)

    # Tempo de inferência
    tempo_inferencia = medir_tempo_inferencia(modelo, X_test, n_reps=INFERENCIA_REPS)

    # Diagnóstico de features
    imp, df_grupo = diagnosticar_features(modelo, feat_cols)

    # Salvar tudo
    salvar_resultados(
        modelo, X_test, y_train_full, y_test, y_pred,
        metricas, cv_metricas, tempo_inferencia,
        imp, df_grupo, df_test, feat_cols, medianas,
        params, tempo_treino_s,
        n_orig=len(df_orig),
        n_aug=len(df_train_full) - len(df_orig),
        n_tr_efetivo=len(X_tr),
        n_val_interna=len(X_val),
        idx_tr=idx_tr, idx_val=idx_val,
    )

    print(f"\n{'═' * 65}")
    print(f"  CV MAPE:     {cv_metricas['cv_mape_mean']:.1f}% ± {cv_metricas['cv_mape_std']:.1f}%")
    print(f"  MAPE teste:  {metricas['teste']['MAPE']:.1f}%")
    print(f"  MAE teste:   {metricas['teste']['MAE']:.3f} laranjas/imagem")
    print(f"  R² teste:    {metricas['teste']['R2']:.4f}")
    print(f"  ±1 fruta:    {metricas['teste']['acerto_pm1']:.1f}%")
    print(f"  ±2 frutas:   {metricas['teste']['acerto_pm2']:.1f}%")
    print(f"  Treino:      {tempo_treino_s}s")
    print(f"  Inferência:  {tempo_inferencia['tempo_por_imagem_ms']:.3f} ms/imagem")
    print(f"  ✓ Sem vazamento | ✓ Treino com aug | ✓ Tempo inferência medido")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()
