import numpy as np
import pandas as pd
from typing import Tuple, List


# ----------------------------
# Helpers internos de numpy
# ----------------------------

def _values(data: pd.DataFrame, column: str) -> np.ndarray:
    """Array de numpy crudo que respalda una columna (sin el wrapping de dtype de pandas)."""
    return data[column].to_numpy()


def _is_missing(array: np.ndarray) -> np.ndarray:
    """Máscara vectorizada de valores faltantes, válida para arrays numéricos y de tipo object."""
    if array.dtype.kind in "fc":
        return np.isnan(array)
    if array.dtype.kind in "iub":
        return np.zeros(array.shape, dtype=bool)
    # dtype object/string: se usa el chequeo de nulos vectorizado (a nivel C)
    # de pandas en vez de un loop elemento a elemento en Python.
    return pd.isna(array)


def _to_float(array: np.ndarray) -> np.ndarray:
    """Convierte a float64, replicando el comportamiento de pd.to_numeric(errors='coerce')."""
    if array.dtype.kind in "fiu":
        return array.astype(float)
    out = np.full(array.shape, np.nan, dtype=float)
    for i, v in enumerate(array):
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def _to_datetime64(array: np.ndarray) -> np.ndarray:
    """Convierte a datetime64[ns]; los valores inválidos/no parseables quedan como NaT."""
    if np.issubdtype(array.dtype, np.datetime64):
        return array.astype("datetime64[ns]")
    try:
        return array.astype("datetime64[ns]")
    except (ValueError, TypeError):
        pass
    out = np.empty(array.shape, dtype="datetime64[ns]")
    for i, v in enumerate(array):
        try:
            out[i] = np.datetime64(v, "ns")
        except (ValueError, TypeError):
            out[i] = np.datetime64("NaT")
    return out

# ----------------------------
# Features de Log, Percentile and bines
# ----------------------------


def build_log_feature(data: pd.DataFrame, column: str, new_column: str, offset: float = 0.0) -> pd.Series:
    """
    Aplica la transformación log1p (con un desplazamiento opcional previo).
    Si, tras el desplazamiento, quedan ceros o negativos, esas filas
    retornarán NaN (se requiere x>-1).

    Args:
        data (pd.DataFrame): Dataframe de entrada.
        column (str): Columna original.
        new_column (str): Nombre de la nueva feature.
        offset (float): Constante a sumar antes de log1p (0.0 por defecto,
            igual que antes). Útil para columnas con valores <= -1,
            p.ej. offset = -min(x) + eps.

    Returns:
        pd.Series: Feature transformada con logaritmo.
    """
    x = _to_float(_values(data, column)) + offset
    x = np.where(x > -1, x, np.nan)  # log1p requiere x > -1
    result = np.log1p(x)
    return pd.Series(result, index=data.index, name=new_column)


def build_top_percentile_flag(
    data: pd.DataFrame, column: str, quantile: float, new_column: str, direction: str = "top"
) -> pd.Series:
    """
    Marca los valores del percentil superior (o inferior, con `direction="bottom"`).

    Args:
        data (pd.DataFrame): Dataframe de entrada.
        column (str): Nombre de la columna.
        quantile (float): Umbral de percentil (p.ej. 0.99).
        new_column (str): Nombre de la nueva feature.
        direction (str): "top" (>= percentil, default, igual que antes) o
            "bottom" (<= percentil).

    Returns:
        pd.Series: Flag binario.
    """
    x = _to_float(_values(data, column))
    threshold = np.nanquantile(x, quantile)

    direction = direction.lower().strip()
    if direction == "top":
        flag = (x >= threshold).astype(int)
    elif direction == "bottom":
        flag = (x <= threshold).astype(int)
    else:
        raise ValueError("direction must be 'top' or 'bottom'.")

    return pd.Series(flag, index=data.index, name=new_column)


# ----------------------------
# Features de Missings and Zeros Flags
# ----------------------------


def build_missing_flag(data: pd.DataFrame, column: str, new_column: str) -> pd.Series:
    """
    Crea un indicador binario de valor faltante.

    Args:
        data (pd.DataFrame): Dataframe de entrada.
        column (str): Nombre de la columna original.
        new_column (str): Nombre de la nueva feature.

    Returns:
        pd.Series: Indicador binario (1 si es faltante, 0 en caso contrario).
    """
    mask = _is_missing(_values(data, column))
    return pd.Series(mask.astype(int), index=data.index, name=new_column)


def build_zero_flag(data: pd.DataFrame, column: str, new_column: str, value: float = 0) -> pd.Series:
    """
    Crea un indicador binario de valor cero (o de cualquier valor dado por `value`).

    Args:
        data (pd.DataFrame): Dataframe de entrada.
        column (str): Nombre de la columna original.
        new_column (str): Nombre de la nueva feature.
        value (float): Valor a marcar (0 por defecto, igual que antes).

    Returns:
        pd.Series: Indicador binario (1 si el valor == value).
    """
    arr = _values(data, column)
    flag = (arr == value).astype(int)
    return pd.Series(flag, index=data.index, name=new_column)


# ----------------------------
# Features de fecha/hora
# ----------------------------

def build_hour(data: pd.DataFrame, column: str, new_column: str) -> pd.Series:
    """
    Extrae la hora del día desde una columna tipo fecha/hora.

    Args:
        data: Dataframe de entrada.
        column: Columna de fecha/hora.
        new_column: Nombre de la feature de salida.

    Returns:
        Hora del día (0..23) como Int64 (nullable).
    """
    dt = _to_datetime64(_values(data, column))
    valid = ~np.isnat(dt)
    hours = np.full(dt.shape, np.nan)
    hours[valid] = dt[valid].astype("datetime64[h]").astype(np.int64) % 24
    return pd.Series(hours, index=data.index, name=new_column).astype("Int64")


def build_weekday(data: pd.DataFrame, column: str, new_column: str) -> pd.Series:
    """
    Extrae el día de la semana (Lun=0..Dom=6) desde una columna tipo fecha/hora.

    Args:
        data: Dataframe de entrada.
        column: Columna de fecha/hora.
        new_column: Nombre de la feature de salida.

    Returns:
        Día de la semana como Int64 (nullable).
    """
    dt = _to_datetime64(_values(data, column))
    valid = ~np.isnat(dt)
    weekday = np.full(dt.shape, np.nan)
    days = dt[valid].astype("datetime64[D]").astype(np.int64)
    weekday[valid] = (days + 3) % 7  # el epoch (1970-01-01) fue jueves
    return pd.Series(weekday, index=data.index, name=new_column).astype("Int64")


def build_is_weekend_flag(
    data: pd.DataFrame, column: str, new_column: str, off_days: Tuple[int, ...] = (5, 6)
) -> pd.Series:
    """
    Indicador de fin de semana desde una columna tipo fecha/hora.

    Args:
        data: Dataframe de entrada.
        column: Columna de fecha/hora.
        new_column: Nombre de la feature de salida.
        off_days: Días (Lun=0..Dom=6) que cuentan como fin de semana
            ((5, 6) = Sáb/Dom por defecto, igual que antes).

    Returns:
        Indicador binario (1 si el día cae en off_days).
    """
    dt = _to_datetime64(_values(data, column))
    valid = ~np.isnat(dt)
    is_weekend = np.zeros(dt.shape, dtype=int)
    days = dt[valid].astype("datetime64[D]").astype(np.int64)
    weekday = (days + 3) % 7
    is_weekend[valid] = np.isin(weekday, off_days).astype(int)
    return pd.Series(is_weekend, index=data.index, name=new_column)


def build_hour_cyclic_sin_cos(
    data: pd.DataFrame,
    column: str,
    new_column: str,
    component: str = "sin",
    cycle_length: float = 24.0,
) -> pd.Series:
    """
    Codificación cíclica para la hora del día, retornando un único componente (sin o cos).

    Args:
        data: Dataframe de entrada.
        column: Columna de fecha/hora.
        new_column: Nombre de la feature de salida.
        component: Qué componente retornar: "sin" (default) o "cos".
        cycle_length: Periodo del ciclo (24 horas por defecto).

    Returns:
        pd.Series: Codificación cíclica sin o cos de la hora.
    """
    dt = _to_datetime64(_values(data, column))
    valid = ~np.isnat(dt)
    hour = np.full(dt.shape, np.nan)
    hour[valid] = dt[valid].astype("datetime64[h]").astype(np.int64) % 24
    angle = 2.0 * np.pi * hour / cycle_length

    component = component.lower().strip()
    if component == "sin":
        out = np.sin(angle)
    elif component == "cos":
        out = np.cos(angle)
    else:
        raise ValueError("component must be 'sin' or 'cos'.")

    return pd.Series(out, index=data.index, name=new_column)


def build_ratio_feature(
    data: pd.DataFrame,
    columns: List[str] | Tuple[str],
    new_column: str,
    default_value: float = np.nan,
) -> pd.Series:
    """
    Crea una feature de razón (ratio) con división segura.

    Args:
        data (pd.DataFrame): Dataframe de entrada.
        columns (List[str] | Tuple[str]): Lista o tupla con dos nombres de columna (numerador, denominador).
        new_column (str): Nombre de la nueva feature.
        default_value (float): Valor a usar cuando el denominador es 0
            (NaN por defecto, igual que antes).

    Returns:
        pd.Series: Feature de razón.
    """
    column_num, column_den = columns
    num = _to_float(_values(data, column_num))
    den = _to_float(_values(data, column_den))
    ratio = np.divide(num, den, out=np.full(num.shape, default_value), where=(den != 0))
    return pd.Series(ratio, index=data.index, name=new_column)

# ----------------------------
# Features de Encodings categóricos
# ----------------------------

def build_frequency_encoding(data: pd.DataFrame, column: str, new_column: str, relative: bool = True) -> pd.Series:
    """
    Frequency encoding para categóricas de alta cardinalidad.

    Implementado con pandas (`value_counts` + `map`): en columnas
    object/string, `numpy.unique` cae a comparaciones a nivel Python y es
    ~10-60x más lento que el `value_counts` de pandas (basado en hashtable).

    Args:
        data: Dataframe de entrada.
        column: Columna categórica.
        new_column: Nombre de la feature de salida.
        relative: Si es True, usa frecuencia relativa; si no, usa conteos.

    Returns:
        Serie numérica con frequency encoding.
    """
    counts = data[column].value_counts(dropna=False)
    if relative:
        counts = counts / len(data)
    return data[column].map(counts).astype(float).rename(new_column)


def build_rare_grouping(
    data: pd.DataFrame,
    column: str,
    new_column: str,
    min_frequency: float = 0.01,
    fallback_label: str = "__OTHER__",
) -> pd.Series:
    """
    Agrupa categorías raras en una única etiqueta.

    Implementado con pandas (`value_counts` + `isin`) por la misma razón
    que `build_frequency_encoding`: las operaciones de pandas basadas en
    hashtable superan a `numpy.unique` sobre arrays object/string sin
    importar la cardinalidad.

    Args:
        data: Dataframe de entrada.
        column: Columna categórica.
        new_column: Nombre de la feature de salida.
        min_frequency: Frecuencia relativa mínima para conservar una categoría.
        fallback_label: Etiqueta para las categorías agrupadas como raras.

    Returns:
        Serie categórica agrupada.
    """
    vc = data[column].value_counts(dropna=False, normalize=True)
    keep = set(vc[vc >= min_frequency].index)
    grouped = data[column].where(data[column].isin(keep), fallback_label)
    return grouped.astype("object").rename(new_column)


def build_target_encoding_with_cross_val(
    data: pd.DataFrame,
    column: str,
    target_column: str,
    new_column: str,
    num_folds: int = 5,
    shrinkage: float = 10.0,
    seed: int = 42,
) -> pd.Series:
    """
    Target encoding con validación cruzada (leakage-aware).
    Los folds se construyen con `numpy.random.Generator` (sin depender de
    `sklearn.model_selection.KFold`), pero la agregación por categoría en
    cada fold usa `groupby` de pandas (~4x más rápido que `numpy.unique` +
    `numpy.bincount` sobre categóricas object/string, misma razón que en
    `build_frequency_encoding`).
    Si tienes orden temporal, pasa los folds externamente en su lugar.

    Args:
        data: Dataframe de entrada.
        column: Columna categórica a codificar.
        target_column: Columna objetivo binaria (0/1).
        new_column: Nombre de la feature de salida.
        num_folds: Número de folds.
        shrinkage: Valores más altos -> más encogimiento hacia la media global.
        seed: Semilla.

    Returns:
        Serie con target encoding, alineada al índice del dataframe.
    """
    y = data[target_column].astype(float)
    x = data[column].astype("object")

    n = len(data)
    global_mean = y.mean()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, num_folds)

    out = pd.Series(index=data.index, dtype=float)

    for k in range(num_folds):
        va_idx = folds[k]
        tr_idx = np.concatenate([folds[j] for j in range(num_folds) if j != k])

        x_tr, y_tr = x.iloc[tr_idx], y.iloc[tr_idx]
        x_va = x.iloc[va_idx]

        stats = y_tr.groupby(x_tr).agg(["mean", "count"])
        smooth_mean = (stats["count"] * stats["mean"] + shrinkage * global_mean) / (stats["count"] + shrinkage)

        out.iloc[va_idx] = x_va.map(smooth_mean).fillna(global_mean).astype(float)

    return out.rename(new_column)


# ----------------------------
# Binning (listo para IV)
# ----------------------------

def build_quantile_bins_dropna(
    data: pd.DataFrame,
    column: str,
    num_bins: int,
    new_column: str,
    include_missing_bin: bool = True,
) -> pd.Series:
    """
    Bins por cuantiles que retornan ids de etiqueta en vez de rangos de intervalo.

    Implementado con `qcut` de pandas (~1.4x más rápido que el equivalente
    con bordes de cuantil de numpy + `digitize` sobre las columnas
    numéricas de este dataset).
    """
    x = pd.to_numeric(data[column], errors="coerce")
    valid = x.notna()

    out = pd.Series(pd.NA, index=data.index, dtype="object")

    if valid.sum() == 0:
        if include_missing_bin:
            return out.where(valid, "__MISSING__").rename(new_column)
        return out.rename(new_column)

    try:
        b = pd.qcut(x[valid], q=num_bins, labels=False, duplicates="drop")

        if b.nunique() < 2:
            if include_missing_bin:
                return out.where(valid, "__MISSING__").rename(new_column)
            return out.rename(new_column)

        out.loc[valid] = b.astype("Int64").astype(str)

    except ValueError:
        # fallback usando bins por percentil
        qs = np.unique(np.nanpercentile(x[valid], np.linspace(0, 100, num_bins + 1)))

        if qs.size < 3:
            if include_missing_bin:
                return out.where(valid, "__MISSING__").rename(new_column)
            return out.rename(new_column)

        qs[0], qs[-1] = -np.inf, np.inf
        b = pd.cut(x[valid], bins=qs, labels=False, include_lowest=True)
        out.loc[valid] = b.astype("Int64").astype(str)

    if include_missing_bin:
        out = out.where(valid, "__MISSING__")

    return out.rename(new_column)


# ----------------------------
# Features de PCA (opcional)
# ----------------------------

def build_pca_components(
    data: pd.DataFrame,
    columns: List[str] | Tuple[str, ...],
    num_components: int,
    new_column: str,
    scale: bool = False,
    fill_strategy: str = "median",
) -> pd.DataFrame:
    """
    Ajusta PCA sobre las columnas seleccionadas y retorna las features de componentes.
    Calculado vía `numpy.linalg.svd` en lugar de `sklearn.decomposition.PCA`
    (los componentes pueden tener el signo invertido respecto a los de
    sklearn, lo cual es una ambigüedad sin efecto real del PCA).

    Nota: En un pipeline real, ajusta el PCA solo con TRAIN para evitar leakage.

    Args:
        data: Dataframe de entrada.
        columns: Columnas a incluir (deben ser numéricas).
        num_components: Número de componentes.
        new_column: Prefijo de la feature, p.ej. 'pca_beh'.
        scale: Si es True, escala cada columna a varianza 1 antes del PCA
            (False por defecto, igual que antes: solo centra en la media).
            Útil cuando las columnas están en escalas muy distintas.
        fill_strategy: Cómo imputar nulos antes del PCA: "median"
            (default, igual que antes), "mean" o "zero".

    Returns:
        DataFrame con columnas {new_column}_1..{new_column}_n.
    """
    columns = list(columns)
    X = np.column_stack([_to_float(_values(data, c)) for c in columns])

    if fill_strategy == "median":
        fill = np.nanmedian(X, axis=0)
    elif fill_strategy == "mean":
        fill = np.nanmean(X, axis=0)
    elif fill_strategy == "zero":
        fill = np.zeros(X.shape[1])
    else:
        raise ValueError("fill_strategy must be 'median', 'mean' or 'zero'.")

    nan_rows, nan_cols = np.where(np.isnan(X))
    X[nan_rows, nan_cols] = np.take(fill, nan_cols)

    X_centered = X - X.mean(axis=0)
    if scale:
        std = X_centered.std(axis=0)
        std[std == 0] = 1.0
        X_centered = X_centered / std

    U, S, _ = np.linalg.svd(X_centered, full_matrices=False)
    Z = U[:, :num_components] * S[:num_components]

    out = pd.DataFrame(Z, index=data.index, columns=[f"{new_column}_{i + 1}" for i in range(num_components)])
    return out


class preprocessing:
    """
    Wrapper de feature engineering para transformaciones modulares (backend híbrido numpy/pandas).
    """

    def __init__(self, data: pd.DataFrame):
        """Guarda el DataFrame base sobre el que se calcularán las features."""
        self.data = data

    def missing_flag(self, column: str, new_column: str) -> pd.Series:
        """Indicador binario de valor faltante en `column`."""
        return build_missing_flag(self.data, column, new_column)

    def zero_flag(self, column: str, new_column: str, value: float = 0) -> pd.Series:
        """Indicador binario de que `column` es igual a `value`."""
        return build_zero_flag(self.data, column, new_column, value)

    def log_feature(self, column: str, new_column: str, offset: float = 0.0) -> pd.Series:
        """Transformación log1p de `column`, con un desplazamiento opcional previo."""
        return build_log_feature(self.data, column, new_column, offset)

    def ratio(self, columns: Tuple[str, str] | List[str], new_column: str, default_value: float = np.nan) -> pd.Series:
        """Razón entre dos columnas, con división seguro ante denominador cero."""
        return build_ratio_feature(self.data, columns, new_column, default_value)

    def top_percentile_flag(self, column: str, quantile: float, new_column: str, direction: str = "top") -> pd.Series:
        """Marca los valores de `column` por encima (o debajo) del percentil `quantile`."""
        return build_top_percentile_flag(self.data, column, quantile, new_column, direction)

    def hour(self, column: str, new_column: str) -> pd.Series:
        """Extrae la hora del día (0-23) de una columna fecha/hora."""
        return build_hour(self.data, column, new_column)

    def weekday(self, column: str, new_column: str) -> pd.Series:
        """Extrae el día de la semana (0=lunes .. 6=domingo) de una columna fecha/hora."""
        return build_weekday(self.data, column, new_column)

    def is_weekend_flag(self, column: str, new_column: str, off_days: Tuple[int, ...] = (5, 6)) -> pd.Series:
        """Indicador de fin de semana a partir de una columna fecha/hora."""
        return build_is_weekend_flag(self.data, column, new_column, off_days)

    def hour_cyclic_sin_cos(
        self, column: str, new_column: str, component: str = "sin", cycle_length: float = 24.0,
    ) -> pd.Series:
        """Codificación cíclica (seno o coseno) de la hora del día."""
        return build_hour_cyclic_sin_cos(self.data, column, new_column, component=component, cycle_length=cycle_length)

    def frequency_encoding(self, column: str, new_column: str, relative: bool = True) -> pd.Series:
        """Reemplaza cada categoría de `column` por su frecuencia (relativa o absoluta)."""
        return build_frequency_encoding(self.data, column, new_column, relative)

    def rare_grouping(self, column: str, new_column: str, min_frequency: float = 0.01, fallback_label: str = "__OTHER__") -> pd.Series:
        """Agrupa las categorías poco frecuentes de `column` en una única etiqueta."""
        return build_rare_grouping(self.data, column, new_column, min_frequency, fallback_label)

    def target_encoding_with_cross_val(self, column: str, target_column: str, new_column: str, num_folds: int = 5, shrinkage: float = 10.0, seed: int = 42) -> pd.Series:
        """Target encoding de `column` con suavizado y validación cruzada, evitando fuga de información."""
        return build_target_encoding_with_cross_val(self.data, column, target_column, new_column, num_folds, shrinkage, seed)

    def quantile_bins_dropna(self, column: str, num_bins: int, new_column: str, include_missing_bin: bool = True) -> pd.Series:
        """Bins por cuantiles de `column`, con etiquetas de texto y un bin dedicado para valores nulos."""
        return build_quantile_bins_dropna(self.data, column, num_bins, new_column, include_missing_bin)

    def pca_components(
        self,
        columns: List[str],
        num_components: int,
        new_column: str,
        scale: bool = False,
        fill_strategy: str = "median",
    ) -> pd.DataFrame:
        """Ajusta PCA sobre `columns` y devuelve los componentes principales resultantes."""
        return build_pca_components(self.data, columns, num_components, new_column, scale, fill_strategy)

    def apply_transformations(self, config_list: List[dict]) -> pd.DataFrame:
        """
        Aplica múltiples transformaciones de features según una lista de configuración.

        Args:
            config_list: Lista de diccionarios que definen las transformaciones.
                Cada diccionario usa las claves "process", "column",
                "new_column" y, opcionalmente, "params".

        Returns:
            DataFrame con todas las features generadas.
        """
        features = []

        for cfg in config_list:
            process = cfg["process"]
            column = cfg["column"]
            new_column = cfg["new_column"]
            params = cfg.get("params", {})

            try:
                method = getattr(self, process)
                result = method(column, new_column=new_column, **params)
                features.append(result)

            except Exception as e:
                print(f"[WARNING] Failed {process} on {column}: {e}")

        return pd.concat(features, axis=1)
