"""
Utilidades vectorizadas de bajo nivel (numpy/pandas) para los transformers de
producción de este paquete.

Son copias autocontenidas de la lógica equivalente usada en la fase de
exploración, no importes en tiempo de ejecución de ese código: el paquete de
exploración no tiene un contrato de estabilidad para producción (sin tests,
libre de cambiar su comportamiento por necesidades exploratorias), y un
pipeline productivo que dependiera de él en runtime quedaría acoplado a
cambios ajenos al servicio del modelo. Duplicar la lógica aquí evita esa
dependencia, al costo de que ambas implementaciones puedan divergir si una se
edita sin la otra -- aceptable, ya que este es el módulo bajo control de
cambios de producción.
"""
import numpy as np
import pandas as pd


def _values(data: pd.DataFrame, column: str) -> np.ndarray:
    """Array de numpy crudo que respalda una columna (sin el wrapping de dtype de pandas)."""
    return data[column].to_numpy()


def _is_missing(array: np.ndarray) -> np.ndarray:
    """Máscara vectorizada de valores faltantes, válida para arrays numéricos y de tipo object."""
    if array.dtype.kind in "fc":
        return np.isnan(array)
    if array.dtype.kind in "iub":
        return np.zeros(array.shape, dtype=bool)
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


_WINDOW_ALIASES = {
    "1h": "1h",
    "1d": "1D",
    "3d": "3D",
    "7d": "7D",
    "10d": "10D",
    "15d": "15D",
}


def _validate_window(window: str) -> str:
    """Traduce un alias corto de ventana (p.ej. `"1d"`) al offset de pandas correspondiente (`"1D"`)."""
    if window not in _WINDOW_ALIASES:
        raise ValueError(f"window must be one of {list(_WINDOW_ALIASES)}.")
    return _WINDOW_ALIASES[window]


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
