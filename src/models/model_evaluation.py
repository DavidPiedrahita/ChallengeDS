
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ─────────────────────────────────────────────────────────────
# 1. SCORING — extract probabilities from any binary classifier/Pipeline
# ─────────────────────────────────────────────────────────────

def extract_fraud_probabilities(model, X) -> np.ndarray:
    """
    Returns P(fraude) para cada fila de X. Funciona con cualquier estimador
    o Pipeline sklearn-compatible (usa predict_proba si existe, si no,
    normaliza decision_function a [0, 1]).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        span = raw.max() - raw.min()
        return (raw - raw.min()) / (span + 1e-9)

    raise ValueError("El modelo no tiene predict_proba ni decision_function.")


def extract_feature_importance(pipeline, top_n: int = 10, normalize: bool = True) -> pd.DataFrame:
    """
    Extrae las `top_n` features más importantes de un Pipeline ya ajustado
    con steps "preprocessor" -> "post_encode" -> "clf".

    Los nombres de feature se reconstruyen encadenando
    `preprocessor.get_feature_names_out()` (nombres de negocio, p.ej.
    "o_target_enc") con `post_encode.get_feature_names_out(...)` (que
    expande las categóricas por el OneHotEncoder, p.ej. "cat__p_Y"), ya
    que son esos nombres expandidos los que el clasificador realmente ve.

    Importancia según el tipo de clasificador:
        - Árboles (RandomForest/LightGBM/XGBoost/CatBoost): `feature_importances_`.
        - LogisticRegression: `|coef_|` (magnitud del coeficiente en log-odds).
          No es directamente comparable entre modelos ni entre features de
          distinta escala -- se usa solo como proxy de importancia relativa
          dentro del propio modelo.

    Args:
        normalize: si es True (default), reescala las importancias para que
            sumen 100 (sobre TODAS las features, antes de recortar a
            `top_n`), quedando como "% del total". Los árboles ya suman ~1
            de por sí, pero `|coef_|` de la Logística no tiene una escala
            comparable entre modelos sin esto.

    Returns:
        DataFrame con columnas ["feature", "importance"], ordenado
        descendente, con `top_n` filas. Si `normalize=True`, "importance"
        está en % (0-100).
    """
    prep = pipeline.named_steps["preprocessor"]
    post = pipeline.named_steps["post_encode"]
    clf = pipeline.named_steps["clf"]

    baseline_names = list(prep.get_feature_names_out())
    feature_names = list(post.get_feature_names_out(baseline_names))

    if hasattr(clf, "feature_importances_"):
        importances = np.asarray(clf.feature_importances_)
    elif hasattr(clf, "coef_"):
        importances = np.abs(np.asarray(clf.coef_)).ravel()
    else:
        raise ValueError(f"{type(clf).__name__} no expone feature_importances_ ni coef_.")

    if normalize:
        total = importances.sum()
        if total > 0:
            importances = importances / total * 100.0

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ═══════════════════════════════════════════════════════════════
#  COMPUTE LAYER — datos puros, sin gráficos, fácil de testear
# ═══════════════════════════════════════════════════════════════

def summarize_pr_curve(y_true, y_scores) -> dict:
    """
    Curva Precision-Recall, AUC-PR, ROC-AUC y el threshold que maximiza F1.
    """
    from sklearn.metrics import precision_recall_curve

    y_true = np.asarray(y_true)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    best_idx = int(np.argmax(f1))

    return dict(
        precision=precision,
        recall=recall,
        thresholds=thresholds,
        f1=f1,
        auc_pr=auc_pr,
        roc_auc=roc_auc,
        best_f1_threshold=float(thresholds[best_idx]),
        best_f1=float(f1[best_idx]),
        best_idx=best_idx,
        baseline=float(y_true.mean()),
    )


def build_threshold_report(y_true, y_scores, threshold: float) -> dict:
    """Classification report (sklearn) evaluado en un threshold fijo."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_scores) >= threshold).astype(int)

    return dict(
        report_dict=classification_report(y_true, y_pred, digits=4, output_dict=True),
        report_str=classification_report(y_true, y_pred, digits=4),
        threshold=threshold,
        y_pred=y_pred,
    )


def show_threshold_report(stats: dict, model_name: str = "Modelo") -> None:
    """Imprime el resultado de build_threshold_report()."""
    print(f"\n{model_name} — Classification Report (threshold = {stats['threshold']:.4f})")
    print("─" * 62)
    print(stats["report_str"])


def sweep_business_profit(
    y_true,
    y_scores,
    monto=None,
    margin: float = 0.25,
    fraud_loss: float = 1.00,
    n_thresholds: int = 300,
) -> dict:
    """
    Barre thresholds y calcula la métrica de negocio J en cada uno:

        J = -fraud_loss * sum(monto en FN) - margin * sum(monto en FP)

    Si `monto` es None, usa conteo (todas las transacciones con monto=1)
    en vez de monto real.
    """
    y_true   = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    if monto is None:
        amounts      = np.ones(len(y_true), dtype=float)
        amount_label = "(count proxy)"
    else:
        amounts      = np.asarray(monto).astype(float)
        amount_label = "($)"

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    j_values   = np.empty_like(thresholds, dtype=float)

    for i, t in enumerate(thresholds):
        y_pred = (y_scores >= t).astype(int)   # 1 => block, 0 => approve

        fp = (y_pred == 1) & (y_true == 0)     # blocked legit  (-margin)
        fn = (y_pred == 0) & (y_true == 1)     # approved fraud (-loss)

        j_values[i] = (
            - margin     * amounts[fp].sum()
            - fraud_loss * amounts[fn].sum()
        )

    best_idx = int(np.argmax(j_values))

    return dict(
        thresholds=thresholds,
        j_values=j_values,
        best_j_threshold=float(thresholds[best_idx]),
        best_j=float(j_values[best_idx]),
        best_idx=best_idx,
        amount_label=amount_label,
        margin=margin,
        fraud_loss=fraud_loss,
    )


def compute_business_j(y_true, y_pred, monto=None, margin: float = 0.25, fraud_loss: float = 1.00) -> float:
    """J en un único threshold ya aplicado (y_pred binario), sin barrido."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    amounts = np.ones(len(y_true), dtype=float) if monto is None else np.asarray(monto).astype(float)

    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    return -fraud_loss * amounts[fn_mask].sum() - margin * amounts[fp_mask].sum()


def summarize_confusion_matrix(y_true, y_scores, threshold: float) -> dict:
    """Matriz de confusión y anotaciones de celda en un threshold dado."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_scores) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    labels = ["Legítima (0)", "Fraude (1)"]
    z_text = [
        [f"{cm[i, j]}<br>({cm[i, j] / cm[i].sum():.1%})" for j in range(2)]
        for i in range(2)
    ]

    tn, fp, fn, tp = cm.ravel()
    return dict(
        cm=cm,
        z_text=z_text,
        labels=labels,
        threshold=threshold,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        precision=float(tp / (tp + fp + 1e-9)),
        recall=float(tp / (tp + fn + 1e-9)),
    )


def sweep_fbeta_score(y_true, y_scores, beta: float = 2.0, n_thresholds: int = 300) -> dict:
    """Barre thresholds y calcula F-beta en cada uno (beta=2 pondera recall 4x más que precision)."""
    y_true_arr = np.asarray(y_true)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    fbeta_vals = np.empty(len(thresholds))

    for i, t in enumerate(thresholds):
        y_pred = (np.asarray(y_scores) >= t).astype(int)
        tp = ((y_pred == 1) & (y_true_arr == 1)).sum()
        fp = ((y_pred == 1) & (y_true_arr == 0)).sum()
        fn = ((y_pred == 0) & (y_true_arr == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        fbeta_vals[i] = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-9)

    best_idx = int(np.argmax(fbeta_vals))
    return dict(
        thresholds=thresholds,
        fbeta_vals=fbeta_vals,
        beta=beta,
        best_fbeta_threshold=float(thresholds[best_idx]),
        best_fbeta=float(fbeta_vals[best_idx]),
        best_idx=best_idx,
    )


def evaluate_fitted_models(
    fitted_pipelines: dict,
    X_test,
    y_test,
    monto=None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Tabla comparativa de métricas para varios modelos/Pipelines ya ajustados.

    Args:
        fitted_pipelines: dict {nombre_modelo: pipeline_o_modelo_fitteado}.
        X_test, y_test: partición de evaluación.
        monto: montos de cada transacción de X_test (para J); si es None, usa conteo.
        threshold: threshold de decisión para Precision/Recall/J (default 0.5).

    Returns:
        DataFrame indexado por nombre de modelo, con AUC y PR-AUC
        (independientes de threshold), Precision/Recall/J($) al `threshold`
        fijo dado, y Precision/Recall/"TH óptimo"/"J óptimo ($)" evaluados
        en el threshold que maximiza J (vía `sweep_business_profit`) --
        estos últimos no tienen por qué coincidir con los del threshold fijo.
    """
    rows = []
    for name, model in fitted_pipelines.items():
        scores = extract_fraud_probabilities(model, X_test)
        y_pred = (scores >= threshold).astype(int)

        profit_sweep = sweep_business_profit(y_test, scores, monto=monto)
        best_t = profit_sweep["best_j_threshold"]
        y_pred_opt = (scores >= best_t).astype(int)

        rows.append({
            "modelo": name,
            "AUC": roc_auc_score(y_test, scores),
            "PR-AUC": average_precision_score(y_test, scores),
            "Precision (TH fijo)": precision_score(y_test, y_pred),
            "Recall (TH fijo)": recall_score(y_test, y_pred),
            "J ($)": compute_business_j(y_test, y_pred, monto=monto),
            "J ($) block all": profit_sweep["j_values"][0],
            "TH óptimo": best_t,
            "Precision (TH óptimo)": precision_score(y_test, y_pred_opt),
            "Recall (TH óptimo)": recall_score(y_test, y_pred_opt),
            "J óptimo ($)": profit_sweep["best_j"],
        })
    return pd.DataFrame(rows).set_index("modelo").round(4)


# ═══════════════════════════════════════════════════════════════
#  PLOT LAYER — recibe el dict de una función compute/summarize/sweep
# ═══════════════════════════════════════════════════════════════

def plot_precision_recall(stats: dict, model_name: str = "Modelo", highlight_thresholds=None, ax=None) -> None:
    """
    Grafica la curva PR a partir del resultado de summarize_pr_curve().

    Si `ax` se pasa (p.ej. un subplot de una grilla), dibuja ahí y no crea
    figura propia ni llama a plt.show() -- lo maneja el caller. Si no,
    crea su propia figura y se muestra de forma independiente (comportamiento
    original).
    """
    precision, recall, thresholds = stats["precision"], stats["recall"], stats["thresholds"]
    best_idx, auc_pr = stats["best_idx"], stats["auc_pr"]
    best_f1, best_t, baseline = stats["best_f1"], stats["best_f1_threshold"], stats["baseline"]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(recall, precision, color="steelblue", linewidth=2, label=f"{model_name} (AUC-PR={auc_pr:.4f})")
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label=f"Baseline aleatorio ({baseline:.3f})")
    ax.scatter([recall[best_idx]], [precision[best_idx]], color="crimson", s=140, marker="*", zorder=5,
               label=f"Mejor F1={best_f1:.4f} (t={best_t:.3f})")

    colors = ["darkorange", "green", "purple", "brown"]
    for idx, (t, label) in enumerate(highlight_thresholds or []):
        nearest = int(np.argmin(np.abs(thresholds - t)))
        ax.scatter([recall[nearest]], [precision[nearest]], color=colors[idx % len(colors)], s=90, marker="D",
                   zorder=5, label=f"{label} (t={t:.3f})")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model_name} — Curva Precision-Recall")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    if standalone:
        fig.tight_layout()
        plt.show()


def plot_business_profit(stats: dict, model_name: str = "Modelo", highlight_thresholds=None, ax=None) -> None:
    """Grafica J vs threshold a partir del resultado de sweep_business_profit(). Ver `plot_precision_recall` para `ax`."""
    thresholds, j_values = stats["thresholds"], stats["j_values"]
    best_t, best_j, amount_label = stats["best_j_threshold"], stats["best_j"], stats["amount_label"]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(thresholds, j_values, color="steelblue", linewidth=2, label="J esperado")
    ax.axvline(best_t, color="crimson", linestyle="--", linewidth=1.5, label=f"J-óptimo t={best_t:.3f}")

    colors = ["darkorange", "green", "purple"]
    for idx, (t, label) in enumerate(highlight_thresholds or []):
        ax.axvline(t, color=colors[idx % len(colors)], linestyle=":", linewidth=1.5, label=f"{label} (t={t:.3f})")

    ax.set_xlabel("Threshold de decisión")
    ax.set_ylabel(f"J esperado {amount_label}")
    ax.set_title(f"{model_name} — J (impacto de negocio) vs Threshold {amount_label}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    if standalone:
        fig.tight_layout()
        plt.show()
    print(f"Mejor threshold por J: {best_t:.4f} -> J = {best_j:,.2f}")


def plot_confusion_heatmap(stats: dict, model_name: str = "Modelo", ax=None) -> None:
    """Grafica la matriz de confusión a partir de summarize_confusion_matrix(). Ver `plot_precision_recall` para `ax`."""
    cm, labels, threshold = stats["cm"], stats["labels"], stats["threshold"]
    tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]

    annot = np.array([
        [f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.1%})" for j in range(2)]
        for i in range(2)
    ])

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 4.2))

    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"{model_name} — Matriz de confusión (threshold = {threshold:.3f})")

    if standalone:
        fig.tight_layout()
        plt.show()
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}  |  Precision={stats['precision']:.4f}  Recall={stats['recall']:.4f}")


def plot_fbeta_curve(stats: dict, model_name: str = "Modelo", highlight_thresholds=None, ax=None) -> None:
    """Grafica F-beta vs threshold a partir de sweep_fbeta_score(). Ver `plot_precision_recall` para `ax`."""
    thresholds, fbeta_vals, beta = stats["thresholds"], stats["fbeta_vals"], stats["beta"]
    best_t, best_fbeta = stats["best_fbeta_threshold"], stats["best_fbeta"]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(thresholds, fbeta_vals, color="mediumseagreen", linewidth=2, label=f"F{beta}")
    ax.axvline(best_t, color="crimson", linestyle="--", linewidth=1.5, label=f"Mejor F{beta} t={best_t:.3f}")

    colors = ["steelblue", "darkorange", "purple"]
    for idx, (t, label) in enumerate(highlight_thresholds or []):
        ax.axvline(t, color=colors[idx % len(colors)], linestyle=":", linewidth=1.5, label=f"{label} (t={t:.3f})")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Threshold de decisión")
    ax.set_ylabel(f"F{beta}")
    ax.set_title(f"{model_name} — F{beta} vs Threshold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    if standalone:
        fig.tight_layout()
        plt.show()
    print(f"Mejor threshold F{beta}: {best_t:.4f} -> F{beta} = {best_fbeta:.4f}")


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str = "Modelo", ax=None, as_pct: bool = True) -> None:
    """
    Grafica un barh horizontal a partir del resultado de extract_feature_importance().
    Ver `plot_precision_recall` para `ax`.

    Args:
        as_pct: si es True (default), formatea el eje como "%" y anota cada
            barra con su valor -- asume que `importance_df["importance"]`
            ya viene en escala 0-100 (el default de `extract_feature_importance`).
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    ordered = importance_df.iloc[::-1]  # el más importante queda arriba en barh
    bars = ax.barh(ordered["feature"], ordered["importance"], color="steelblue")
    ax.set_xlabel("Importancia (%)" if as_pct else "Importancia")
    ax.set_title(f"{model_name} — Top {len(importance_df)} features")
    ax.grid(True, alpha=0.3, axis="x")

    if as_pct:
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)

    if standalone:
        fig.tight_layout()
        plt.show()


# ═══════════════════════════════════════════════════════════════
#  ARTIFACTOS — serialización del modelo/pipeline entrenado
# ═══════════════════════════════════════════════════════════════

# Métodos nativos de serialización por tipo de clasificador (cuando existen).
# Si el clasificador del Pipeline no coincide con ninguna de estas clases
# (p.ej. LogisticRegression, RandomForestClassifier), solo se guarda el
# Pipeline completo vía joblib.
_NATIVE_SAVERS = {
    "LGBMClassifier": lambda clf, path: clf.booster_.save_model(str(path.with_suffix(".txt"))),
    "XGBClassifier": lambda clf, path: clf.save_model(str(path.with_suffix(".json"))),
    "CatBoostClassifier": lambda clf, path: clf.save_model(str(path.with_suffix(".cbm"))),
}


def save_model_artifact(pipeline, model_name: str, artifacts_dir) -> dict:
    """
    Guarda los artefactos de un Pipeline (preprocessor + post_encode + clf)
    ya ajustado, en `artifacts_dir/model_name/`.

    Siempre guarda el Pipeline completo vía joblib (`pipeline.joblib`) --
    es lo que se necesita para aplicar el modelo a datos nuevos sin
    reconstruir el preprocesamiento a mano. Además, si el paso final del
    Pipeline es LightGBM/XGBoost/CatBoost, guarda también el clasificador
    en su formato nativo (más portable/estable entre versiones que pickle),
    junto a un `metadata.json` con el tipo real de cada artefacto.

    Args:
        pipeline: sklearn Pipeline ya ajustado (con step final "clf").
        model_name: nombre del modelo (usado como nombre de subcarpeta).
        artifacts_dir: carpeta base de artefactos (se crea si no existe).

    Returns:
        dict con las rutas de los artefactos guardados.
    """
    model_dir = Path(artifacts_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    pipeline_path = model_dir / "pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    saved_paths["pipeline"] = str(pipeline_path)

    clf = pipeline.named_steps.get("clf") if hasattr(pipeline, "named_steps") else None
    clf_type = type(clf).__name__ if clf is not None else None

    if clf_type in _NATIVE_SAVERS:
        native_path = model_dir / "model"
        _NATIVE_SAVERS[clf_type](clf, native_path)
        saved_paths["native"] = str(next(model_dir.glob("model.*")))

    metadata = {
        "model_name": model_name,
        "classifier_type": clf_type,
        "artifacts": saved_paths,
    }
    metadata_path = model_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    saved_paths["metadata"] = str(metadata_path)

    return saved_paths


def load_all_model_artifacts(artifacts_dir) -> dict:
    """
    Descubre y carga todos los Pipelines guardados en `artifacts_dir`
    (una subcarpeta por modelo, cada una con su `pipeline.joblib`, tal como
    los deja `save_model_artifact`).

    Args:
        artifacts_dir: carpeta base de artefactos (p.ej. `03_model_selection/artifacts`).

    Returns:
        dict {nombre_modelo: Pipeline ya ajustado}, en orden alfabético de subcarpeta.
    """
    artifacts_dir = Path(artifacts_dir)
    pipelines = {}
    for model_dir in sorted(artifacts_dir.iterdir()):
        pipeline_path = model_dir / "pipeline.joblib"
        if model_dir.is_dir() and pipeline_path.exists():
            pipelines[model_dir.name] = joblib.load(pipeline_path)
    return pipelines
