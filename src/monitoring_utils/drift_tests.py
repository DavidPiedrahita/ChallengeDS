"""Métricas de drift poblacional entre dos periodos de tiempo (referencia vs. actual) para una misma feature."""

import numpy as np
import pandas as pd
from scipy import stats

_MISSING_LABEL = "__MISSING__"
_OTHER_LABEL = "__OTHER__"


def ks_drift(reference: pd.Series, current: pd.Series) -> dict:
    """
    KS (Kolmogorov-Smirnov) entre la distribución de referencia y la actual
    de una variable continua, ignorando nulos.
    """
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()

    if ref.size == 0 or cur.size == 0:
        return {"ks_stat": np.nan, "ks_p_value": np.nan}

    result = stats.ks_2samp(ref, cur)
    return {"ks_stat": float(result.statistic), "ks_p_value": float(result.pvalue)}


def ks_drift_label(ks_stat: float) -> str:
    """Traduce un estadístico KS a una etiqueta cualitativa de severidad de drift."""
    if pd.isna(ks_stat):
        return "sin datos"
    if ks_stat < 0.1:
        return "sin drift"
    if ks_stat < 0.2:
        return "drift leve"
    return "drift significativo"


def psi_categorical(reference: pd.Series, current: pd.Series, max_categories: int = 20) -> float:
    """
    Population Stability Index (PSI) entre la distribución categórica de
    referencia y la actual. Los nulos se tratan como una categoría propia y
    las categorías poco frecuentes en la referencia se agrupan en
    "__OTHER__", para evitar explosión combinatoria en variables de alta
    cardinalidad.
    """
    ref = reference.astype(object).where(reference.notna(), _MISSING_LABEL)
    cur = current.astype(object).where(current.notna(), _MISSING_LABEL)

    top_categories = ref.value_counts().head(max_categories - 1).index
    ref_grouped = ref.where(ref.isin(top_categories), _OTHER_LABEL)
    cur_grouped = cur.where(cur.isin(top_categories), _OTHER_LABEL)

    categories = sorted(set(ref_grouped.unique()) | set(cur_grouped.unique()))
    ref_pct = ref_grouped.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    cur_pct = cur_grouped.value_counts(normalize=True).reindex(categories, fill_value=0.0)

    # suavizado (+eps) para evitar log(0) cuando una categoría solo aparece en un periodo
    eps = 1e-4
    ref_pct = ref_pct + eps
    cur_pct = cur_pct + eps

    return float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())


def psi_label(psi: float) -> str:
    """Traduce un PSI a una etiqueta cualitativa de severidad de drift (convención estándar de la industria)."""
    if pd.isna(psi):
        return "sin datos"
    if psi < 0.1:
        return "sin drift"
    if psi < 0.25:
        return "drift moderado"
    return "drift significativo"


def category_proportions(series: pd.Series, top_categories, other_label: str = _OTHER_LABEL) -> pd.Series:
    """
    Proporción de cada categoría en `top_categories` dentro de `series`
    (nulos como categoría propia, el resto agrupado en `other_label`).
    Útil para graficar la evolución de la distribución categórica en el
    tiempo con un eje de categorías consistente entre periodos.
    """
    filled = series.astype(object).where(series.notna(), _MISSING_LABEL)
    grouped = filled.where(filled.isin(top_categories), other_label)
    return grouped.value_counts(normalize=True).reindex(list(top_categories) + [other_label], fill_value=0.0)
