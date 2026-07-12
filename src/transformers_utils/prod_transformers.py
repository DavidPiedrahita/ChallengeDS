
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.transformers_utils._vectorized_ops import (
    _is_missing,
    _rolling_window_aggregate,
    _to_datetime64,
    _to_float,
    _values,
)


# ---------------------------------------------------------------------------
# Transformaciones
# ---------------------------------------------------------------------------

class LogFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Aplica log1p (con offset opcional). Requiere x > -1 tras el offset; si no
    se cumple, esa fila queda NaN.
    Output column name: log_{col}.
    """
    def __init__(self, offset: float = 0.0):
        """Guarda el offset a sumar antes de aplicar log1p."""
        self.offset = offset

    def fit(self, X, y=None):
        """No aprende parámetros; devuelve la instancia sin modificar su estado."""
        return self

    def transform(self, X):
        """Aplica log1p(x + offset) a la columna de entrada."""
        col = X.columns[0]
        x = _to_float(_values(X, col)) + self.offset
        x = np.where(x > -1, x, np.nan)
        result = np.log1p(x)
        name = self.get_feature_names_out(list(X.columns))[0]
        return pd.DataFrame({name: result}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        """Devuelve el nombre de la columna de salida (`log_{col}`)."""
        return np.array([f"log_{input_features[0]}"])


class MissingFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Indicador binario de valor faltante.
    Output column name: {col}_is_missing.
    """
    def fit(self, X, y=None):
        """No aprende parámetros; devuelve la instancia sin modificar su estado."""
        return self

    def transform(self, X):
        """Marca con 1 los valores faltantes de la columna de entrada y con 0 el resto."""
        col = X.columns[0]
        mask = _is_missing(_values(X, col))
        name = self.get_feature_names_out(list(X.columns))[0]
        return pd.DataFrame({name: mask.astype(int)}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        """Devuelve el nombre de la columna de salida (`{col}_is_missing`)."""
        return np.array([f"{input_features[0]}_is_missing"])


class ZeroFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Indicador binario de valor == `value`.
    Output column name: {col}_is_zero (o {col}_eq_{value}_flag si value != 0).
    """
    def __init__(self, value: float = 0):
        """Guarda el valor de referencia contra el que se compara la columna."""
        self.value = value

    def fit(self, X, y=None):
        """No aprende parámetros; devuelve la instancia sin modificar su estado."""
        return self

    def transform(self, X):
        """Marca con 1 las filas donde la columna de entrada es igual a `value`."""
        col = X.columns[0]
        flag = (_values(X, col) == self.value).astype(int)
        name = self.get_feature_names_out(list(X.columns))[0]
        return pd.DataFrame({name: flag}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        """Devuelve el nombre de la columna de salida (`{col}_is_zero` o `{col}_eq_{value}_flag`)."""
        col = input_features[0]
        suffix = "is_zero" if self.value == 0 else f"eq_{self.value}_flag"
        return np.array([f"{col}_{suffix}"])


class HourTransformer(BaseEstimator, TransformerMixin):
    """Hora del día (0..23) desde una columna fecha/hora. Output column name: hour_{col}."""
    def fit(self, X, y=None):
        """No aprende parámetros; devuelve la instancia sin modificar su estado."""
        return self

    def transform(self, X):
        """Extrae la hora del día (0-23) de la columna fecha/hora de entrada."""
        col = X.columns[0]
        dt = _to_datetime64(_values(X, col))
        valid = ~np.isnat(dt)
        hours = np.full(dt.shape, np.nan)
        hours[valid] = dt[valid].astype("datetime64[h]").astype(np.int64) % 24
        name = self.get_feature_names_out(list(X.columns))[0]
        return pd.DataFrame({name: hours}, index=X.index).astype({name: "Int64"})

    def get_feature_names_out(self, input_features=None):
        """Devuelve el nombre de la columna de salida (`hour_{col}`)."""
        return np.array([f"hour_{input_features[0]}"])


class WeekdayTransformer(BaseEstimator, TransformerMixin):
    """Día de la semana (Lun=0..Dom=6) desde una columna fecha/hora. Output column name: weekday_{col}."""
    def fit(self, X, y=None):
        """No aprende parámetros; devuelve la instancia sin modificar su estado."""
        return self

    def transform(self, X):
        """Extrae el día de la semana (0=lunes .. 6=domingo) de la columna fecha/hora de entrada."""
        col = X.columns[0]
        dt = _to_datetime64(_values(X, col))
        valid = ~np.isnat(dt)
        weekday = np.full(dt.shape, np.nan)
        days = dt[valid].astype("datetime64[D]").astype(np.int64)
        weekday[valid] = (days + 3) % 7  # el epoch (1970-01-01) fue jueves
        name = self.get_feature_names_out(list(X.columns))[0]
        return pd.DataFrame({name: weekday}, index=X.index).astype({name: "Int64"})

    def get_feature_names_out(self, input_features=None):
        """Devuelve el nombre de la columna de salida (`weekday_{col}`)."""
        return np.array([f"weekday_{input_features[0]}"])


class IsWeekendFlagTransformer(BaseEstimator, TransformerMixin):
    """Indicador de fin de semana desde una columna fecha/hora. Output column name: is_weekend_{col}."""
    def __init__(self, off_days: tuple = (5, 6)):
        """Guarda los índices de día de la semana (0=lunes) considerados fin de semana."""
        self.off_days = off_days

    def fit(self, X, y=None):
        """No aprende parámetros; devuelve la instancia sin modificar su estado."""
        return self

    def transform(self, X):
        """Marca con 1 las fechas cuyo día de la semana está en `off_days`."""
        col = X.columns[0]
        dt = _to_datetime64(_values(X, col))
        valid = ~np.isnat(dt)
        is_weekend = np.zeros(dt.shape, dtype=int)
        days = dt[valid].astype("datetime64[D]").astype(np.int64)
        weekday = (days + 3) % 7
        is_weekend[valid] = np.isin(weekday, self.off_days).astype(int)
        name = self.get_feature_names_out(list(X.columns))[0]
        return pd.DataFrame({name: is_weekend}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        """Devuelve el nombre de la columna de salida (`is_weekend_{col}`)."""
        return np.array([f"is_weekend_{input_features[0]}"])


class HourCyclicSinCosTransformer(BaseEstimator, TransformerMixin):
    """
    Codificación cíclica sin/cos de la hora.
    Output column name: hour_{component}_{col}.
    """
    def __init__(self, component: str = "sin", cycle_length: float = 24.0):
        """Guarda el componente (`sin`/`cos`) y la longitud del ciclo a codificar."""
        self.component = component
        self.cycle_length = cycle_length

    def fit(self, X, y=None):
        """No aprende parámetros; devuelve la instancia sin modificar su estado."""
        return self

    def transform(self, X):
        """Codifica la hora del día como seno o coseno de su ángulo en el ciclo configurado."""
        col = X.columns[0]
        dt = _to_datetime64(_values(X, col))
        valid = ~np.isnat(dt)
        hour = np.full(dt.shape, np.nan)
        hour[valid] = dt[valid].astype("datetime64[h]").astype(np.int64) % 24
        angle = 2.0 * np.pi * hour / self.cycle_length

        component = self.component.lower().strip()
        if component == "sin":
            out = np.sin(angle)
        elif component == "cos":
            out = np.cos(angle)
        else:
            raise ValueError("component must be 'sin' or 'cos'.")

        name = self.get_feature_names_out(list(X.columns))[0]
        return pd.DataFrame({name: out}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        """Devuelve el nombre de la columna de salida (`hour_{component}_{col}`)."""
        return np.array([f"hour_{self.component}_{input_features[0]}"])


class RatioTransformer(BaseEstimator, TransformerMixin):
    """
    Razón col1/col2 con división segura.
    Expects exactly two columns. Output column name: {col1}_per_{col2}.
    """
    def __init__(self, default_value: float = np.nan):
        """Guarda el valor a usar cuando el denominador es cero."""
        self.default_value = default_value

    def fit(self, X, y=None):
        """No aprende parámetros; devuelve la instancia sin modificar su estado."""
        return self

    def transform(self, X):
        """Calcula la razón entre las dos columnas de entrada, evitando división por cero."""
        col1, col2 = X.columns[0], X.columns[1]
        num = _to_float(_values(X, col1))
        den = _to_float(_values(X, col2))
        ratio = np.divide(num, den, out=np.full(num.shape, self.default_value), where=(den != 0))
        name = self.get_feature_names_out(list(X.columns))[0]
        return pd.DataFrame({name: ratio}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        """Devuelve el nombre de la columna de salida (`{col1}_per_{col2}`)."""
        return np.array([f"{input_features[0]}_per_{input_features[1]}"])


# ---------------------------------------------------------------------------
# Transformaciones que aprenden parámetros en fit (umbrales, bordes de bin,
# mapas de frecuencia) y los congelan para transform, evitando fuga de
# información entre train y test.
# ---------------------------------------------------------------------------

class TopPercentileFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Marca valores por encima (o debajo) de un percentil aprendido en train.
    Output column name: {col}_top{pct}_flag o {col}_bottom{pct}_flag.
    """
    def __init__(self, quantile: float = 0.99, direction: str = "top"):
        """Guarda el cuantil de referencia y la dirección (`top`/`bottom`) a marcar."""
        self.quantile = quantile
        self.direction = direction

    def fit(self, X, y=None):
        """Aprende el percentil `quantile` de la columna sobre los datos de entrenamiento."""
        col = X.columns[0]
        x = _to_float(_values(X, col))
        self.threshold_ = np.nanquantile(x, self.quantile)
        return self

    def transform(self, X):
        """Marca con 1 los valores que superan (o no alcanzan) el percentil aprendido en fit."""
        col = X.columns[0]
        x = _to_float(_values(X, col))
        direction = self.direction.lower().strip()
        if direction == "top":
            flag = (x >= self.threshold_).astype(int)
        elif direction == "bottom":
            flag = (x <= self.threshold_).astype(int)
        else:
            raise ValueError("direction must be 'top' or 'bottom'.")
        name = self.get_feature_names_out(list(X.columns))[0]
        return pd.DataFrame({name: flag}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        """Devuelve el nombre de la columna de salida (`{col}_top{pct}_flag` o `{col}_bottom{pct}_flag`)."""
        col = input_features[0]
        if self.direction == "top":
            pct_int = int(round((1 - self.quantile) * 100))
            return np.array([f"{col}_top{pct_int}_flag"])
        pct_int = int(round(self.quantile * 100))
        return np.array([f"{col}_bottom{pct_int}_flag"])


class QuantileBinsDropnaTransformer(BaseEstimator, TransformerMixin):
    """
    Bins por cuantiles con etiquetas de texto y bin dedicado para NaN,
    con bordes aprendidos en train.
    Output column name: {col}_qbin_dropna.
    """
    def __init__(self, num_bins: int = 5, include_missing_bin: bool = True):
        """Guarda la cantidad de bins y si se debe reservar una etiqueta explícita para nulos."""
        self.num_bins = num_bins
        self.include_missing_bin = include_missing_bin

    def fit(self, X, y=None):
        """Aprende los bordes de los bins por cuantiles sobre los datos de entrenamiento, ignorando los valores nulos."""
        col = X.columns[0]
        x = pd.to_numeric(X[col], errors="coerce")
        valid = x.notna()
        self.edges_ = None

        if valid.sum() == 0:
            return self

        try:
            codes, edges = pd.qcut(x[valid], q=self.num_bins, labels=False, duplicates="drop", retbins=True)
            if pd.Series(codes).nunique() < 2:
                return self
            edges = np.asarray(edges, dtype=float)
            edges[0], edges[-1] = -np.inf, np.inf
            self.edges_ = edges
        except ValueError:
            qs = np.unique(np.nanpercentile(x[valid], np.linspace(0, 100, self.num_bins + 1)))
            if qs.size >= 3:
                qs[0], qs[-1] = -np.inf, np.inf
                self.edges_ = qs
        return self

    def transform(self, X):
        """Asigna cada valor a su bin (como texto) según los bordes aprendidos en fit, o a un bin dedicado si es nulo."""
        col = X.columns[0]
        x = pd.to_numeric(X[col], errors="coerce")
        valid = x.notna()
        out = pd.Series(pd.NA, index=X.index, dtype="object")

        if self.edges_ is not None and valid.any():
            b = pd.cut(x[valid], bins=self.edges_, labels=False, include_lowest=True)
            out.loc[valid] = b.astype("Int64").astype(str)

        if self.include_missing_bin:
            out = out.where(valid, "__MISSING__")

        name = self.get_feature_names_out(list(X.columns))[0]
        return out.rename(name).to_frame()

    def get_feature_names_out(self, input_features=None):
        """Devuelve el nombre de la columna de salida (`{col}_qbin_dropna`)."""
        return np.array([f"{input_features[0]}_qbin_dropna"])


class FrequencyEncodingTransformer(BaseEstimator, TransformerMixin):
    """
    Frequency encoding aprendido en train vía `value_counts`. Categorías no
    vistas en transform reciben la frecuencia media entre las categorías
    conocidas.
    Output column name: {col}_freq_encoded.
    """
    def __init__(self, relative: bool = True):
        """Guarda si la frecuencia se expresa como proporción (True) o como conteo absoluto (False)."""
        self.relative = relative

    def fit(self, X, y=None):
        """Aprende la frecuencia de cada categoría sobre los datos de entrenamiento."""
        col = X.columns[0]
        counts = X[col].value_counts(dropna=False)
        if self.relative:
            counts = counts / len(X)
        self.encoding_map_ = counts.to_dict()
        self.mean_freq_ = counts.mean()
        return self

    def transform(self, X):
        """Reemplaza cada categoría por su frecuencia aprendida en fit; las categorías no vistas reciben la frecuencia media."""
        col = X.columns[0]
        mapped = X[col].map(self.encoding_map_).fillna(self.mean_freq_).astype(float)
        name = self.get_feature_names_out(list(X.columns))[0]
        return mapped.rename(name).to_frame()

    def get_feature_names_out(self, input_features=None):
        """Devuelve el nombre de la columna de salida (`{col}_freq_encoded`)."""
        return np.array([f"{input_features[0]}_freq_encoded"])


class RareGroupingTransformer(BaseEstimator, TransformerMixin):
    """
    Agrupa categorías raras (frecuencia < min_frequency en train) en una
    única etiqueta.
    Output column name: {col}_rare_grouped.
    """
    def __init__(self, min_frequency: float = 0.01, fallback_label: str = "__OTHER__"):
        """Guarda el umbral mínimo de frecuencia y la etiqueta de reemplazo para categorías raras."""
        self.min_frequency = min_frequency
        self.fallback_label = fallback_label

    def fit(self, X, y=None):
        """Aprende qué categorías superan `min_frequency` en los datos de entrenamiento."""
        col = X.columns[0]
        vc = X[col].value_counts(dropna=False, normalize=True)
        self.keep_ = set(vc[vc >= self.min_frequency].index)
        return self

    def transform(self, X):
        """Reemplaza las categorías no frecuentes por `fallback_label`, dejando el resto sin cambios."""
        col = X.columns[0]
        grouped = X[col].where(X[col].isin(self.keep_), self.fallback_label).astype("object")
        name = self.get_feature_names_out(list(X.columns))[0]
        return grouped.rename(name).to_frame()

    def get_feature_names_out(self, input_features=None):
        """Devuelve el nombre de la columna de salida (`{col}_rare_grouped`)."""
        return np.array([f"{input_features[0]}_rare_grouped"])


class TargetEncodingCVTransformer(BaseEstimator, TransformerMixin):
    """
    Target encoding con suavizado y validación cruzada (leakage-aware).

    - fit(X, y): calcula el encoding out-of-fold (K-fold) sobre los datos de
      train, usado en transform() cuando se llama sobre esos mismos datos
      (p.ej. dentro de `ColumnTransformer.fit_transform` para generar las
      meta-features de entrenamiento sin leakage). También aprende un mapa
      global (media suavizada por categoría, sin folds) para aplicar sobre
      datos genuinamente nuevos (test/producción).
    - transform(X): si X coincide con el índice usado en fit, retorna los
      valores out-of-fold ya calculados; si no, aplica el mapa global.

    Output column name: {col}_target_enc.
    """
    def __init__(self, num_folds: int = 5, shrinkage: float = 10.0, seed: int = 42):
        """Guarda el número de folds, el factor de suavizado hacia la media global y la semilla del particionado."""
        self.num_folds = num_folds
        self.shrinkage = shrinkage
        self.seed = seed

    def fit(self, X, y):
        """Calcula el encoding out-of-fold sobre train y un mapa global (media suavizada por categoría) para datos nuevos."""
        col = X.columns[0]
        x = X[col].astype("object")
        y = pd.Series(np.asarray(y), index=X.index).astype(float)

        n = len(X)
        global_mean = y.mean()

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        folds = np.array_split(perm, self.num_folds)

        oof = pd.Series(index=X.index, dtype=float)
        for k in range(self.num_folds):
            va_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(self.num_folds) if j != k])

            x_tr, y_tr = x.iloc[tr_idx], y.iloc[tr_idx]
            x_va = x.iloc[va_idx]

            # dropna=False: el nulo se trata como una categoría más (con su propia
            # tasa aprendida), en vez de perder esa señal al caer en fillna(global_mean).
            stats = y_tr.groupby(x_tr, dropna=False).agg(["mean", "count"])
            smooth_mean = (stats["count"] * stats["mean"] + self.shrinkage * global_mean) / (stats["count"] + self.shrinkage)
            oof.iloc[va_idx] = x_va.map(smooth_mean).fillna(global_mean).astype(float)

        self.oof_values_ = oof
        self.train_index_ = X.index

        stats_full = y.groupby(x, dropna=False).agg(["mean", "count"])
        self.global_map_ = (stats_full["count"] * stats_full["mean"] + self.shrinkage * global_mean) / (stats_full["count"] + self.shrinkage)
        self.global_mean_ = global_mean
        return self

    def transform(self, X):
        """Devuelve los valores out-of-fold si X es el mismo set usado en fit, o aplica el mapa global en caso contrario."""
        col = X.columns[0]
        name = self.get_feature_names_out(list(X.columns))[0]
        if X.index.equals(self.train_index_):
            result = self.oof_values_.reindex(X.index)
        else:
            result = X[col].map(self.global_map_).fillna(self.global_mean_).astype(float)
        return result.rename(name).to_frame()

    def get_feature_names_out(self, input_features=None):
        """Devuelve el nombre de la columna de salida (`{col}_target_enc`)."""
        return np.array([f"{input_features[0]}_target_enc"])


class PCATransformer(BaseEstimator, TransformerMixin):
    """
    PCA vía `numpy.linalg.svd` (sin sklearn.decomposition.PCA). Los valores
    de imputación, la media (y desviación si scale=True) y los componentes
    (`Vt`) se aprenden en fit sobre train, y se reutilizan en transform
    sobre cualquier dato nuevo.

    Output column names: {prefix}_1, {prefix}_2, ...
    """
    def __init__(self, num_components: int = 2, prefix: str = "pca", scale: bool = False, fill_strategy: str = "median"):
        """Guarda el número de componentes, el prefijo de salida, si se escala y la estrategia de imputación."""
        self.num_components = num_components
        self.prefix = prefix
        self.scale = scale
        self.fill_strategy = fill_strategy

    def fit(self, X, y=None):
        """Aprende la imputación, el centrado (y escalado si scale=True) y los componentes principales sobre los datos de entrenamiento."""
        self.columns_ = list(X.columns)
        arr = np.column_stack([_to_float(_values(X, c)) for c in self.columns_])

        if self.fill_strategy == "median":
            fill = np.nanmedian(arr, axis=0)
        elif self.fill_strategy == "mean":
            fill = np.nanmean(arr, axis=0)
        elif self.fill_strategy == "zero":
            fill = np.zeros(arr.shape[1])
        else:
            raise ValueError("fill_strategy must be 'median', 'mean' or 'zero'.")
        self.fill_ = fill

        nan_rows, nan_cols = np.where(np.isnan(arr))
        arr[nan_rows, nan_cols] = np.take(fill, nan_cols)

        self.mean_ = arr.mean(axis=0)
        centered = arr - self.mean_

        if self.scale:
            std = centered.std(axis=0)
            std[std == 0] = 1.0
            self.std_ = std
            centered = centered / std
        else:
            self.std_ = None

        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = Vt[: self.num_components]
        return self

    def transform(self, X):
        """Proyecta los datos de entrada sobre los componentes principales aprendidos en fit."""
        arr = np.column_stack([_to_float(_values(X, c)) for c in self.columns_])
        nan_rows, nan_cols = np.where(np.isnan(arr))
        arr[nan_rows, nan_cols] = np.take(self.fill_, nan_cols)

        centered = arr - self.mean_
        if self.scale:
            centered = centered / self.std_

        Z = centered @ self.components_.T
        return pd.DataFrame(Z, index=X.index, columns=self.get_feature_names_out())

    def get_feature_names_out(self, input_features=None):
        """Devuelve los nombres de las columnas de salida (`{prefix}_1`, `{prefix}_2`, ...)."""
        return np.array([f"{self.prefix}_{i + 1}" for i in range(self.num_components)])


# ---------------------------------------------------------------------------
# Agregaciones por ventana de tiempo.
# ---------------------------------------------------------------------------

class _BaseWindowTransformer(BaseEstimator, TransformerMixin):
    """
    Agregación causal por ventana de tiempo, con separación fit/transform:
    - fit(X): guarda X como histórico (contexto por entidad) de train.
    - transform(X): si X es el mismo fit (mismo índice), aplica la
      agregación causal directamente sobre esos datos -- sin fuga hacia el
      futuro dentro de la partición. Si X es una partición nueva
      (test/producción), concatena el histórico de train + X, recalcula la
      agregación sobre el conjunto combinado y devuelve solo las filas de
      X, de forma que las primeras transacciones de test puedan usar el
      historial real de train de la misma entidad en vez de arrancar con la
      ventana vacía.
    """
    agg_func: str = ""

    def __init__(self, value_column: str, group_column: str = "j", date_column: str = "fecha", window: str = "1d"):
        """Guarda las columnas de valor/entidad/fecha y el tamaño de la ventana a agregar."""
        self.value_column = value_column
        self.group_column = group_column
        self.date_column = date_column
        self.window = window

    def fit(self, X, y=None):
        """Guarda X como histórico de la entidad, para usarlo como contexto causal en transform."""
        cols = [self.group_column, self.date_column, self.value_column]
        self.history_ = X[cols].copy()
        return self

    def transform(self, X):
        """Calcula la agregación causal por ventana; si X es una partición nueva, la antecede con el histórico de fit."""
        cols = [self.group_column, self.date_column, self.value_column]

        if X.index.equals(self.history_.index):
            combined = self.history_.reset_index(drop=True)
            new_positions = combined.index
        else:
            # Se re-indexa con un RangeIndex propio antes de concatenar: si X viene
            # de una partición leída por separado del histórico de fit, su índice
            # puede colisionar con el de self.history_ (ambos arrancando en 0), lo
            # que rompe el reindex final más abajo.
            history_part = self.history_.reset_index(drop=True)
            new_part = X[cols].reset_index(drop=True)
            new_part.index = new_part.index + len(history_part)
            combined = pd.concat([history_part, new_part], axis=0)
            new_positions = new_part.index

        name = self.get_feature_names_out()[0]
        rolled = _rolling_window_aggregate(
            combined, self.value_column, name, self.group_column, self.date_column, self.window, self.agg_func,
        )
        result = rolled.reindex(new_positions)
        result.index = X.index
        return result.to_frame()

    def get_feature_names_out(self, input_features=None):
        """Devuelve el nombre de la columna de salida (`{value_column}_{agg_func}_{window}`)."""
        return np.array([f"{self.value_column}_{self.agg_func}_{self.window}"])


class SumWindowTransformer(_BaseWindowTransformer):
    """Suma causal de `value_column` en la ventana, por entidad. Ver `_BaseWindowTransformer`."""
    agg_func = "sum"


class MeanWindowTransformer(_BaseWindowTransformer):
    """Promedio causal de `value_column` en la ventana, por entidad. Ver `_BaseWindowTransformer`."""
    agg_func = "mean"


class MedianWindowTransformer(_BaseWindowTransformer):
    """Mediana causal de `value_column` en la ventana, por entidad. Ver `_BaseWindowTransformer`."""
    agg_func = "median"


class MaxWindowTransformer(_BaseWindowTransformer):
    """Máximo causal de `value_column` en la ventana, por entidad. Ver `_BaseWindowTransformer`."""
    agg_func = "max"


class MinWindowTransformer(_BaseWindowTransformer):
    """Mínimo causal de `value_column` en la ventana, por entidad. Ver `_BaseWindowTransformer`."""
    agg_func = "min"
