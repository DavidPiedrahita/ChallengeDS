import pandas as pd

WINDOW_ALIASES = {
    "1h": "1h",
    "1d": "1D",
    "3d": "3D",
    "7d": "7D",
    "10d": "10D",
    "15d": "15D",
}


def _validate_window(window: str) -> str:
    """Traduce un alias corto de ventana (p.ej. `"1d"`) al offset de pandas correspondiente (`"1D"`)."""
    if window not in WINDOW_ALIASES:
        raise ValueError(f"window must be one of {list(WINDOW_ALIASES)}.")
    return WINDOW_ALIASES[window]


def _rolling_window_aggregate(
    data: pd.DataFrame,
    column: str,
    new_column: str,
    group_column: str,
    date_column: str,
    window: str,
    agg_func: str,
) -> pd.Series:
    """
    Agregación causal por ventana de tiempo: para cada fila, agrega
    `column` sobre las transacciones previas (excluyendo la fila actual)
    de la misma entidad (`group_column`), dentro de la ventana
    `[t - window, t)`. No hay fuga de información hacia el futuro.
    """
    offset = _validate_window(window)

    sorted_data = data[[group_column, date_column, column]].sort_values([group_column, date_column])
    indexed = sorted_data.set_index(date_column)

    rolled = indexed.groupby(group_column)[column].rolling(offset, closed="left").agg(agg_func)
    rolled = rolled.reset_index(level=0, drop=True)
    rolled.index = sorted_data.index

    return rolled.reindex(data.index).rename(new_column)


def build_sum_window(
    data: pd.DataFrame,
    column: str,
    new_column: str,
    group_column: str = "j",
    date_column: str = "fecha",
    window: str = "1d",
) -> pd.Series:
    """
    Suma de `column` sobre las transacciones previas de la misma entidad,
    dentro de la ventana de tiempo dada (p.ej. "suma de monto en las
    últimas 24h para esta entidad").
    """
    return _rolling_window_aggregate(data, column, new_column, group_column, date_column, window, "sum")


def build_mean_window(
    data: pd.DataFrame,
    column: str,
    new_column: str,
    group_column: str = "j",
    date_column: str = "fecha",
    window: str = "1d",
) -> pd.Series:
    """Promedio de `column` sobre las transacciones previas de la misma entidad, en la ventana dada."""
    return _rolling_window_aggregate(data, column, new_column, group_column, date_column, window, "mean")


def build_median_window(
    data: pd.DataFrame,
    column: str,
    new_column: str,
    group_column: str = "j",
    date_column: str = "fecha",
    window: str = "1d",
) -> pd.Series:
    """Mediana de `column` sobre las transacciones previas de la misma entidad, en la ventana dada."""
    return _rolling_window_aggregate(data, column, new_column, group_column, date_column, window, "median")


def build_max_window(
    data: pd.DataFrame,
    column: str,
    new_column: str,
    group_column: str = "j",
    date_column: str = "fecha",
    window: str = "1d",
) -> pd.Series:
    """Máximo de `column` sobre las transacciones previas de la misma entidad, en la ventana dada."""
    return _rolling_window_aggregate(data, column, new_column, group_column, date_column, window, "max")


def build_min_window(
    data: pd.DataFrame,
    column: str,
    new_column: str,
    group_column: str = "j",
    date_column: str = "fecha",
    window: str = "1d",
) -> pd.Series:
    """Mínimo de `column` sobre las transacciones previas de la misma entidad, en la ventana dada."""
    return _rolling_window_aggregate(data, column, new_column, group_column, date_column, window, "min")


class AggregationFeatures:
    """
    Wrapper de features de agregación por ventana de tiempo: expone cada
    estadístico (`sum_window`, `mean_window`, etc.) como método de instancia
    y permite aplicar varias agregaciones en lote vía `apply_transformations`.
    """

    def __init__(self, data: pd.DataFrame, group_column: str = "j", date_column: str = "fecha"):
        """Guarda el DataFrame base y las columnas de entidad/fecha usadas por defecto en las agregaciones."""
        self.data = data
        self.group_column = group_column
        self.date_column = date_column

    def sum_window(self, column: str, new_column: str, window: str = "1d") -> pd.Series:
        """Suma causal de `column` en la ventana, agrupado por entidad."""
        return build_sum_window(self.data, column, new_column, self.group_column, self.date_column, window)

    def mean_window(self, column: str, new_column: str, window: str = "1d") -> pd.Series:
        """Promedio causal de `column` en la ventana, agrupado por entidad."""
        return build_mean_window(self.data, column, new_column, self.group_column, self.date_column, window)

    def median_window(self, column: str, new_column: str, window: str = "1d") -> pd.Series:
        """Mediana causal de `column` en la ventana, agrupado por entidad."""
        return build_median_window(self.data, column, new_column, self.group_column, self.date_column, window)

    def max_window(self, column: str, new_column: str, window: str = "1d") -> pd.Series:
        """Máximo causal de `column` en la ventana, agrupado por entidad."""
        return build_max_window(self.data, column, new_column, self.group_column, self.date_column, window)

    def min_window(self, column: str, new_column: str, window: str = "1d") -> pd.Series:
        """Mínimo causal de `column` en la ventana, agrupado por entidad."""
        return build_min_window(self.data, column, new_column, self.group_column, self.date_column, window)

    def apply_transformations(self, config_list: list[dict]) -> pd.DataFrame:
        """
        Aplica múltiples agregaciones según una lista de configuración.

        Args:
            config_list: Lista de diccionarios con "process", "column",
                "new_column" y, opcionalmente, "params" (p.ej. {"window": "7d"}).

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
