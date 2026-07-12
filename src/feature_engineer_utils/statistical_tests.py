"""
Pruebas estadísticas de asociación entre cada feature y el target binario `fraude`.

Una función por prueba (nombre autoexplicativo, retorna un dict con el/los
estadísticos y su interpretación), más una clase `FeatureStatTests` que las
orquesta según el tipo de variable:

- Continuas (`kind="continuous"`): Mann-Whitney U, KS, correlación
  punto-biserial, ROC-AUC univariado (derivado del propio estadístico U de
  Mann-Whitney, sin depender de scikit-learn, que no está instalado en este
  proyecto), d de Cohen, y Wald sobre una regresión logística univariada.
- Categóricas (`kind="categorical"`): Chi-cuadrado de independencia, V de
  Cramér, Information Value (IV) con tabla de Weight of Evidence (WoE), y
  likelihood-ratio test sobre una regresión logística con dummies.

Todas las pruebas categóricas agrupan categorías con nulos como su propia
categoría ("__MISSING__", no se descartan filas) y colapsan categorías raras
en "__OTHER__" cuando la cardinalidad excede el límite configurado, para
evitar explosión combinatoria en variables de alta cardinalidad (p.ej. `j`).
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError


_MISSING_LABEL = "__MISSING__"
_OTHER_LABEL = "__OTHER__"


# ----------------------------
# Helpers internos
# ----------------------------

def _xy_dropna(data: pd.DataFrame, column: str, target_column: str) -> tuple[np.ndarray, np.ndarray]:
    """Extrae (x, y) para una variable continua, descartando filas con x nulo."""
    sub = data[[column, target_column]].dropna(subset=[column])
    x = sub[column].astype(float).to_numpy()
    y = sub[target_column].astype(int).to_numpy()
    return x, y


def _fillna_categorical(series: pd.Series) -> pd.Series:
    """Convierte nulos en una categoría explícita en vez de descartar filas."""
    return series.astype(object).where(series.notna(), _MISSING_LABEL)


def _collapse_rare_categories(series: pd.Series, max_categories: int) -> pd.Series:
    """Colapsa las categorías menos frecuentes en `_OTHER_LABEL` si hay más de `max_categories`."""
    counts = series.value_counts()
    if counts.size <= max_categories:
        return series
    keep = set(counts.iloc[: max_categories - 1].index)
    return series.where(series.isin(keep), _OTHER_LABEL)


def _auc_interpretation(auc_abs: float) -> str:
    """Traduce un AUC absoluto a una etiqueta cualitativa de fuerza de separación."""
    if auc_abs < 0.6:
        return "casi aleatorio"
    if auc_abs < 0.7:
        return "separación débil"
    if auc_abs < 0.8:
        return "separación moderada"
    return "separación fuerte"


def _cohens_d_interpretation(abs_d: float) -> str:
    """Traduce un d de Cohen absoluto a una etiqueta cualitativa de tamaño del efecto."""
    if abs_d < 0.2:
        return "despreciable"
    if abs_d < 0.5:
        return "pequeño"
    if abs_d < 0.8:
        return "mediano"
    return "grande"


def _iv_interpretation(iv: float) -> str:
    """Traduce un Information Value a una etiqueta cualitativa de poder predictivo."""
    if iv < 0.02:
        return "inútil"
    if iv < 0.10:
        return "débil"
    if iv < 0.30:
        return "medio"
    if iv < 0.50:
        return "fuerte"
    return "sospechoso, revisar leakage"


def _cramers_v_interpretation(v: float) -> str:
    """Traduce una V de Cramér a una etiqueta cualitativa de fuerza de asociación."""
    if v < 0.1:
        return "despreciable"
    if v < 0.3:
        return "débil"
    if v < 0.5:
        return "moderada"
    return "fuerte"


# ----------------------------
# Pruebas para variables CONTINUAS
# ----------------------------

def mann_whitney_test(data: pd.DataFrame, column: str, target_column: str) -> dict:
    """
    Mann-Whitney U: compara la distribución de `column` entre fraude=0 y
    fraude=1 sin asumir normalidad.
    """
    x, y = _xy_dropna(data, column, target_column)
    group0, group1 = x[y == 0], x[y == 1]

    if group0.size == 0 or group1.size == 0:
        return {"mannwhitney_stat": np.nan, "mannwhitney_p_value": np.nan}

    statistic, p_value = stats.mannwhitneyu(group0, group1, alternative="two-sided")
    return {"mannwhitney_stat": float(statistic), "mannwhitney_p_value": float(p_value)}


def ks_statistic_test(data: pd.DataFrame, column: str, target_column: str) -> dict:
    """
    KS (Kolmogorov-Smirnov): máxima distancia entre las CDF empíricas de
    ambos grupos. A mayor KS, mejor separación.
    """
    x, y = _xy_dropna(data, column, target_column)
    group0, group1 = x[y == 0], x[y == 1]

    if group0.size == 0 or group1.size == 0:
        return {"ks_stat": np.nan, "ks_p_value": np.nan}

    result = stats.ks_2samp(group0, group1)
    return {"ks_stat": float(result.statistic), "ks_p_value": float(result.pvalue)}


def point_biserial_correlation(data: pd.DataFrame, column: str, target_column: str) -> dict:
    """
    Correlación punto-biserial entre `column` (continua) y el target binario
    (equivalente a Pearson cuando una variable es binaria).
    """
    x, y = _xy_dropna(data, column, target_column)

    if np.unique(y).size < 2 or np.std(x) == 0:
        return {"point_biserial_corr": np.nan, "point_biserial_p_value": np.nan}

    result = stats.pointbiserialr(y, x)
    return {"point_biserial_corr": float(result.correlation), "point_biserial_p_value": float(result.pvalue)}


def _auc_from_xy(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """AUC (y su versión absoluta) de un score continuo `x` contra un target binario `y`.

    Se calcula a partir del estadístico U de Mann-Whitney (AUC = U / (n0*n1)),
    lo que evita depender de scikit-learn (no instalado en este proyecto).
    """
    group0, group1 = x[y == 0], x[y == 1]
    if group0.size == 0 or group1.size == 0:
        return np.nan, np.nan

    u_statistic, _ = stats.mannwhitneyu(group1, group0, alternative="two-sided")
    auc = float(u_statistic / (group0.size * group1.size))
    return auc, max(auc, 1 - auc)


def roc_auc_univariate(data: pd.DataFrame, column: str, target_column: str) -> dict:
    """
    ROC-AUC usando `column` sola como score de un clasificador.
    Se reporta `auc_abs = max(auc, 1-auc)` para que la dirección no afecte
    la magnitud reportada.
    """
    x, y = _xy_dropna(data, column, target_column)
    auc, auc_abs = _auc_from_xy(x, y)
    if np.isnan(auc):
        return {"auc": np.nan, "auc_abs": np.nan, "auc_interpretation": None}
    return {"auc": auc, "auc_abs": auc_abs, "auc_interpretation": _auc_interpretation(auc_abs)}


def categorical_auc_via_woe(
    data: pd.DataFrame, column: str, target_column: str, iv_max_bins: int = 20
) -> dict:
    """
    AUC de una variable categórica, calculado tras codificarla con su propio
    Weight of Evidence (WoE) por categoría (mismo criterio de agrupamiento de
    categorías raras que `information_value`). El WoE es un score continuo,
    lo que permite comparar el AUC de una categórica directamente contra el
    AUC de una variable continua (misma escala 0.5-1 en `auc_abs`).
    """
    table = _woe_table(data, column, target_column, iv_max_bins)
    woe_map = table.set_index("categoria")["woe"]

    series = _collapse_rare_categories(_fillna_categorical(data[column]), iv_max_bins)
    x = series.map(woe_map).astype(float).to_numpy()
    y = data[target_column].astype(int).to_numpy()

    auc, auc_abs = _auc_from_xy(x, y)
    if np.isnan(auc):
        return {"auc_woe": np.nan, "auc_woe_abs": np.nan, "auc_woe_interpretation": None}
    return {
        "auc_woe": auc, "auc_woe_abs": auc_abs,
        "auc_woe_interpretation": _auc_interpretation(auc_abs),
    }


def cohens_d(data: pd.DataFrame, column: str, target_column: str) -> dict:
    """
    d de Cohen: diferencia de medias entre fraude=1 y fraude=0, dividida
    por la desviación estándar combinada (pooled).
    """
    x, y = _xy_dropna(data, column, target_column)
    group0, group1 = x[y == 0], x[y == 1]

    n0, n1 = group0.size, group1.size
    if n0 < 2 or n1 < 2:
        return {"cohens_d": np.nan, "cohens_d_interpretation": None}

    var0, var1 = group0.var(ddof=1), group1.var(ddof=1)
    pooled_std = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2))

    if pooled_std == 0:
        return {"cohens_d": np.nan, "cohens_d_interpretation": None}

    d = float((group1.mean() - group0.mean()) / pooled_std)
    return {"cohens_d": d, "cohens_d_interpretation": _cohens_d_interpretation(abs(d))}


def logistic_wald_test_continuous(data: pd.DataFrame, column: str, target_column: str) -> dict:
    """
    Regresión logística univariada (`column` como único predictor): reporta
    el coeficiente (beta), el odds ratio y el p-value de Wald del predictor.
    Detecta separación perfecta / inestabilidad numérica del optimizador.
    """
    x, y = _xy_dropna(data, column, target_column)

    if np.unique(y).size < 2 or np.std(x) == 0:
        return {
            "logit_beta": np.nan, "logit_odds_ratio": np.nan,
            "logit_p_value": np.nan, "logit_separation_detected": True,
        }

    X = sm.add_constant(x)
    try:
        result = sm.Logit(y, X).fit(disp=0)
    except (PerfectSeparationError, np.linalg.LinAlgError):
        return {
            "logit_beta": np.nan, "logit_odds_ratio": np.nan,
            "logit_p_value": np.nan, "logit_separation_detected": True,
        }

    beta = float(result.params[1])
    separation_detected = bool(not result.mle_retvals.get("converged", True) or abs(beta) > 20)

    return {
        "logit_beta": beta,
        "logit_odds_ratio": float(np.exp(beta)),
        "logit_p_value": float(result.pvalues[1]),
        "logit_separation_detected": separation_detected,
    }


# ----------------------------
# Pruebas para variables CATEGÓRICAS
# ----------------------------

def chi_square_test(data: pd.DataFrame, column: str, target_column: str) -> dict:
    """
    Chi-cuadrado de independencia sobre la tabla de contingencia
    `column` x `target_column`.
    """
    series = _fillna_categorical(data[column])
    table = pd.crosstab(series, data[target_column])
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    return {"chi2_stat": float(chi2), "chi2_p_value": float(p_value), "chi2_dof": int(dof)}


def cramers_v(data: pd.DataFrame, column: str, target_column: str) -> dict:
    """
    V de Cramér: tamaño del efecto asociado al Chi-cuadrado, normalizado
    entre 0 y 1.
    """
    series = _fillna_categorical(data[column])
    table = pd.crosstab(series, data[target_column])
    chi2, _, _, _ = stats.chi2_contingency(table)

    n = table.to_numpy().sum()
    r, k = table.shape
    denom = n * (min(r, k) - 1)

    if denom <= 0:
        return {"cramers_v": np.nan, "cramers_v_interpretation": None}

    v = float(np.sqrt(chi2 / denom))
    return {"cramers_v": v, "cramers_v_interpretation": _cramers_v_interpretation(v)}


def information_value(
    data: pd.DataFrame, column: str, target_column: str, iv_max_bins: int = 20
) -> dict:
    """
    Information Value (IV) vía Weight of Evidence (WoE): para cada categoría
    calcula WoE = ln(%buenos / %malos); el IV es la suma ponderada de esos
    WoE. Colapsa categorías raras en "__OTHER__" si hay más de `iv_max_bins`
    categorías, para evitar explosión combinatoria (p.ej. en `j`).
    """
    table = _woe_table(data, column, target_column, iv_max_bins)
    iv = float(table["iv_contribution"].sum())
    return {"iv": iv, "iv_interpretation": _iv_interpretation(iv)}


def _woe_table(
    data: pd.DataFrame, column: str, target_column: str, iv_max_bins: int = 20
) -> pd.DataFrame:
    """Tabla de WoE/IV por categoría (usada por `information_value` y `woe_table`)."""
    series = _fillna_categorical(data[column])
    series = _collapse_rare_categories(series, iv_max_bins)
    y = data[target_column]

    total_good = int((y == 0).sum())
    total_bad = int((y == 1).sum())

    grouped = pd.DataFrame({"category": series, "target": y}).groupby("category", observed=True)["target"].agg(
        n_total="count", n_bad="sum"
    )
    grouped["n_good"] = grouped["n_total"] - grouped["n_bad"]

    # suavizado (+0.5) para evitar log(0) cuando una categoría no tiene buenos o malos
    pct_good = (grouped["n_good"] + 0.5) / (total_good + 0.5 * len(grouped))
    pct_bad = (grouped["n_bad"] + 0.5) / (total_bad + 0.5 * len(grouped))

    grouped["pct_good"] = pct_good
    grouped["pct_bad"] = pct_bad
    grouped["woe"] = np.log(pct_good / pct_bad)
    grouped["iv_contribution"] = (pct_good - pct_bad) * grouped["woe"]

    grouped = grouped.reset_index().rename(columns={"category": "categoria"})
    return grouped[["categoria", "n_total", "n_good", "n_bad", "pct_good", "pct_bad", "woe", "iv_contribution"]]


def woe_table(
    data: pd.DataFrame, column: str, target_column: str, iv_max_bins: int = 20
) -> pd.DataFrame:
    """Expone la tabla completa de WoE por categoría (no solo el IV agregado)."""
    return _woe_table(data, column, target_column, iv_max_bins)


def logistic_dummies_test_categorical(
    data: pd.DataFrame, column: str, target_column: str, lg_max_categories: int = 20
) -> dict:
    """
    Codifica `column` con one-hot (colapsando categorías raras si hay más de
    `lg_max_categories`), ajusta una regresión logística completa vs. un
    modelo nulo (solo intercepto), y compara ambos con un likelihood-ratio
    test (chi-cuadrado).
    """
    series = _fillna_categorical(data[column])
    series = _collapse_rare_categories(series, lg_max_categories)
    y = data[target_column].astype(int)

    dummies = pd.get_dummies(series, drop_first=True, dtype=float)

    if dummies.shape[1] == 0:
        return {"lr_stat": np.nan, "lr_p_value": np.nan, "lr_dof": 0, "lr_separation_detected": True}

    X_full = sm.add_constant(dummies)
    X_null = sm.add_constant(pd.DataFrame(index=y.index))

    try:
        model_full = sm.Logit(y, X_full).fit(disp=0)
        model_null = sm.Logit(y, X_null).fit(disp=0)
    except (PerfectSeparationError, np.linalg.LinAlgError):
        # separación (cuasi-)perfecta: alguna categoría rara concentra una
        # sola clase del target y la matriz de diseño queda singular.
        return {
            "lr_stat": np.nan, "lr_p_value": np.nan,
            "lr_dof": dummies.shape[1], "lr_separation_detected": True,
        }

    lr_stat = 2 * (model_full.llf - model_null.llf)
    dof = dummies.shape[1]
    p_value = float(stats.chi2.sf(lr_stat, dof))
    separation_detected = bool(not model_full.mle_retvals.get("converged", True))

    return {
        "lr_stat": float(lr_stat), "lr_p_value": p_value, "lr_dof": dof,
        "lr_separation_detected": separation_detected,
    }


class FeatureStatTests:
    """
    Orquesta las pruebas estadísticas de asociación entre cada feature y el
    target, según el tipo de variable (`kind="continuous"` o `"categorical"`).
    """

    def __init__(self, data: pd.DataFrame):
        """Guarda el DataFrame base sobre el que se ejecutarán las pruebas."""
        self.data = data

    def compute(
        self,
        column: str,
        target_column: str,
        kind: str = "continuous",
        iv_max_bins: int = 20,
        lg_max_categories: int = 20,
        include_woe_table: bool = False,
    ) -> dict:
        """Ejecuta el conjunto de pruebas correspondiente al tipo de variable (`kind`) y devuelve los resultados en un solo dict."""
        if kind == "continuous":
            return self._compute_continuous(column, target_column)
        if kind == "categorical":
            return self._compute_categorical(
                column, target_column, iv_max_bins, lg_max_categories, include_woe_table
            )
        raise ValueError("kind must be 'continuous' or 'categorical'.")

    def _compute_continuous(self, column: str, target_column: str) -> dict:
        """Ejecuta todas las pruebas para variables continuas (Mann-Whitney, KS, punto-biserial, ROC-AUC, Cohen's d y Wald)."""
        result = {"variable": column}
        result.update(mann_whitney_test(self.data, column, target_column))
        result.update(ks_statistic_test(self.data, column, target_column))
        result.update(point_biserial_correlation(self.data, column, target_column))
        result.update(roc_auc_univariate(self.data, column, target_column))
        result.update(cohens_d(self.data, column, target_column))
        result.update(logistic_wald_test_continuous(self.data, column, target_column))
        return result

    def _compute_categorical(
        self,
        column: str,
        target_column: str,
        iv_max_bins: int,
        lg_max_categories: int,
        include_woe_table: bool,
    ) -> dict:
        """Ejecuta todas las pruebas para variables categóricas (Chi-cuadrado, V de Cramér, IV, AUC vía WoE y likelihood-ratio test)."""
        result = {"variable": column}
        result.update(chi_square_test(self.data, column, target_column))
        result.update(cramers_v(self.data, column, target_column))
        result.update(information_value(self.data, column, target_column, iv_max_bins))
        result.update(categorical_auc_via_woe(self.data, column, target_column, iv_max_bins))
        result.update(
            logistic_dummies_test_categorical(self.data, column, target_column, lg_max_categories)
        )

        if include_woe_table:
            table = woe_table(self.data, column, target_column, iv_max_bins)
            table.insert(0, "variable", column)
            result["woe_table"] = table

        return result
