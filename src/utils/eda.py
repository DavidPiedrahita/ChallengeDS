"""Utilidades de EDA reutilizables para el dataset de fraude."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

try:
    from IPython import get_ipython
    _IN_NOTEBOOK = get_ipython() is not None
except ImportError:
    _IN_NOTEBOOK = False

if not _IN_NOTEBOOK:
    # En ejecución como script plano (sin kernel de notebook) se fuerza un
    # backend no interactivo para poder guardar figuras sin display.
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class EDA:
    """Encapsula las rutinas de análisis exploratorio del dataset de fraude.

    Parameters
    ----------
    path : str | Path
        Ruta al archivo CSV con el dataset.
    id_uniqueness_threshold : float
        Fracción mínima de valores únicos (sobre el total de filas) para
        considerar una columna como candidata a identificador único.
    """

    def __init__(self, path: str | Path, id_uniqueness_threshold: float = 0.9):
        """Guarda la ruta del dataset y el umbral de unicidad; el DataFrame se carga en `load()`."""
        self.path = Path(path)
        self.id_uniqueness_threshold = id_uniqueness_threshold
        self.df: pd.DataFrame | None = None
        self.date_columns: list[str] = []

    # ------------------------------------------------------------------
    # Carga de datos
    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Carga el CSV y detecta automáticamente columnas de fecha."""
        self.df = pd.read_csv(self.path)
        self.date_columns = self._detect_date_columns(self.df)
        for col in self.date_columns:
            self.df[col] = pd.to_datetime(self.df[col], format="mixed")
        return self.df

    @staticmethod
    def _detect_date_columns(df: pd.DataFrame, sample_size: int = 500) -> list[str]:
        """Detecta columnas de tipo texto que sean parseables como fecha.

        Se apoya en el nombre de la columna (heurística rápida) y, si eso
        falla, intenta parsear una muestra de valores no nulos.
        """
        detected = []
        for col in df.columns:
            if df[col].dtype != object and not pd.api.types.is_string_dtype(df[col]):
                continue
            name_hint = any(k in col.lower() for k in ["fecha", "date", "timestamp"])
            sample = df[col].dropna().head(sample_size)
            if sample.empty:
                continue
            try:
                parsed = pd.to_datetime(sample, errors="raise", format="mixed")
                # Evita falsos positivos: exige que se hayan podido parsear
                # todos los valores de la muestra.
                if parsed.notna().all():
                    detected.append(col)
                    continue
            except (ValueError, TypeError):
                pass
            if name_hint:
                detected.append(col)
        return detected

    # ------------------------------------------------------------------
    # 1. Estructura general
    # ------------------------------------------------------------------
    def general_structure(self) -> dict:
        """Resumen de shape, dtypes y clasificación numérica/categórica."""
        df = self._require_df()
        n_rows, n_cols = df.shape

        datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in datetime_cols]
        categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]

        # Marca binarias (0/1 o Y/N) como sub-tipo, útil para no confundirlas
        # con variables numéricas continuas.
        binary_cols = [c for c in df.columns if df[c].dropna().nunique() == 2]

        def tipo_general(col: str) -> str:
            """Clasifica una columna como fecha/datetime, numérica o categórica/texto."""
            if col in datetime_cols:
                return "fecha/datetime"
            if col in numeric_cols:
                return "numérica"
            return "categórica/texto"

        def n_ceros(col: str):
            """Cuenta los valores iguales a cero en una columna numérica (NaN para las demás)."""
            # Solo tiene sentido para columnas numéricas; en categóricas/fecha
            # se deja como NaN para no confundir con "0 ocurrencias".
            if col not in numeric_cols:
                return np.nan
            return int((df[col] == 0).sum())

        ceros = [n_ceros(c) for c in df.columns]

        dtypes_summary = pd.DataFrame({
            "columna": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "tipo_general": [tipo_general(c) for c in df.columns],
            "binaria": [c in binary_cols for c in df.columns],
            "n_unicos": [df[c].nunique(dropna=True) for c in df.columns],
            "pct_unicos": [round(df[c].nunique(dropna=True) / n_rows, 4) for c in df.columns],
            "n_nulos": [int(df[c].isna().sum()) for c in df.columns],
            "pct_nulos": [round(df[c].isna().mean(), 4) for c in df.columns],
            "n_ceros": ceros,
            "pct_ceros": [round(c / n_rows, 4) if not pd.isna(c) else np.nan for c in ceros],
        })

        return {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "binary_cols": binary_cols,
            "dtypes_summary": dtypes_summary,
        }

    # ------------------------------------------------------------------
    # 2. Rango de fechas
    # ------------------------------------------------------------------
    def date_range_summary(self) -> dict:
        """Rango de fechas por cada columna de fecha detectada."""
        df = self._require_df()
        summary = []
        for col in self.date_columns:
            serie = df[col].dropna()
            summary.append({
                "columna": col,
                "fecha_min": serie.min(),
                "fecha_max": serie.max(),
                "dias_cubiertos": (serie.max() - serie.min()).days,
                "n_registros": len(serie),
            })
        return {
            "date_columns": self.date_columns,
            "summary": pd.DataFrame(summary),
        }

    # ------------------------------------------------------------------
    # 3. Columnas candidatas a ID
    # ------------------------------------------------------------------
    def id_like_columns(self) -> pd.DataFrame:
        """Heurística para detectar columnas que probablemente sean IDs.

        Se marca una columna como candidata si:
        - Su nombre sugiere un identificador (id, key, code, cat_, etc.), o
        - Su ratio de valores únicos respecto al total de filas supera el
          umbral configurado (`id_uniqueness_threshold`).
        """
        df = self._require_df()
        n_rows = len(df)
        name_pattern = ("id", "key", "code", "uuid")

        rows = []
        for col in df.columns:
            if col in self.date_columns:
                # Las columnas de fecha son casi únicas por granularidad de
                # timestamp; se analizan aparte en `date_range_summary`.
                continue
            n_unique = df[col].nunique(dropna=True)
            uniqueness_ratio = round(n_unique / n_rows, 4)
            name_hint = any(p in col.lower() for p in name_pattern)

            # Heurística adicional: columnas de texto con prefijo repetido
            # tipo "cat_xxxxx" (identificador de categoría/entidad).
            prefix_hint = False
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                sample = df[col].dropna().astype(str).head(50)
                if len(sample) > 0 and sample.str.match(r"^[a-zA-Z]+_[0-9a-fA-F]+$").mean() > 0.8:
                    prefix_hint = True

            is_candidate = (
                uniqueness_ratio >= self.id_uniqueness_threshold
                or name_hint
                or prefix_hint
            )

            if is_candidate:
                if uniqueness_ratio >= self.id_uniqueness_threshold:
                    motivo = f"~{uniqueness_ratio:.0%} de valores únicos (posible identificador de fila)"
                elif prefix_hint:
                    motivo = "patrón tipo prefijo_hash (posible identificador de categoría/entidad)"
                else:
                    motivo = "nombre de columna sugiere identificador"
                rows.append({
                    "columna": col,
                    "n_unicos": n_unique,
                    "pct_unicos": uniqueness_ratio,
                    "motivo": motivo,
                })

        return pd.DataFrame(rows).sort_values("pct_unicos", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Distribución del target por periodo (día / semana / mes)
    # ------------------------------------------------------------------
    _FREQ_LABELS = {"D": "día", "W": "semana", "M": "mes"}

    def target_distribution_by_period(
        self,
        target_col: str,
        freq: str = "M",
        date_col: str | None = None,
    ) -> pd.DataFrame:
        """Cantidad y tasa de la clase positiva del target por periodo.

        Parameters
        ----------
        freq : str
            Granularidad temporal: "D" (día a día), "W" (semana a semana) o
            "M" (mes a mes).
        """
        if freq not in self._FREQ_LABELS:
            raise ValueError(f"freq debe ser una de {list(self._FREQ_LABELS)}, recibido: {freq!r}")

        df = self._require_df()
        date_col = date_col or (self.date_columns[0] if self.date_columns else None)
        if date_col is None:
            raise ValueError("No se detectó ninguna columna de fecha; especifica `date_col`.")

        tmp = df[[date_col, target_col]].dropna(subset=[date_col]).copy()
        tmp["periodo"] = tmp[date_col].dt.to_period(freq).astype(str)

        summary = (
            tmp.groupby("periodo")[target_col]
            .agg(n_total="count", n_positivos="sum")
            .reset_index()
        )
        summary["n_negativos"] = summary["n_total"] - summary["n_positivos"]
        summary["tasa_positivos"] = round(summary["n_positivos"] / summary["n_total"], 4)
        return summary

    def target_daily_distribution(self, target_col: str, date_col: str | None = None) -> pd.DataFrame:
        """Atajo de `target_distribution_by_period` con freq="D" (día a día)."""
        return self.target_distribution_by_period(target_col, freq="D", date_col=date_col)

    def target_weekly_distribution(self, target_col: str, date_col: str | None = None) -> pd.DataFrame:
        """Atajo de `target_distribution_by_period` con freq="W" (semana a semana).

        La columna `periodo` se reemplaza por una etiqueta secuencial
        ("semana_1", "semana_2", ...) en orden cronológico, para que los
        gráficos y tablas sean más legibles que el rango de fechas crudo
        (p. ej. "2020-03-02/2020-03-08"). Ese rango se conserva en la
        columna `rango_fechas` como referencia.
        """
        summary = self.target_distribution_by_period(target_col, freq="W", date_col=date_col)
        summary = summary.rename(columns={"periodo": "rango_fechas"})
        summary.insert(0, "periodo", [f"semana_{i}" for i in range(1, len(summary) + 1)])
        return summary

    def target_monthly_distribution(self, target_col: str, date_col: str | None = None) -> pd.DataFrame:
        """Atajo de `target_distribution_by_period` con freq="M" (mes a mes)."""
        return self.target_distribution_by_period(target_col, freq="M", date_col=date_col)

    def plot_target_by_period(
        self,
        period_df: pd.DataFrame,
        output_path: str | Path | None = None,
        target_name: str = "target",
        freq: str = "M",
        close: bool = True,
    ):
        """Genera (y opcionalmente guarda) el gráfico del target por periodo.

        Si `close=False`, la figura no se cierra al final, lo que permite
        que backends interactivos (p. ej. el inline de Jupyter) la muestren
        automáticamente al terminar la celda.
        """
        period_label = self._FREQ_LABELS.get(freq, freq)
        n_periods = len(period_df)

        needs_rotation = freq in ("D", "W")
        width_per_period = 0.35 if freq == "D" else 1.1 if freq == "W" else 0.9
        fig, ax1 = plt.subplots(figsize=(max(9, n_periods * width_per_period), 5))
        sns.barplot(x="periodo", y="n_total", data=period_df, ax=ax1, color="#cfd8dc", label="Total registros", errorbar=None)
        sns.barplot(x="periodo", y="n_positivos", data=period_df, ax=ax1, color="#e53935", label=f"{target_name} = 1", errorbar=None)
        ax1.set_ylabel("N° de registros")
        ax1.set_xlabel(period_label.capitalize())
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        if needs_rotation:
            plt.setp(ax1.get_xticklabels(), rotation=45 if freq == "D" else 45, ha="right", fontsize=8)

        ax2 = ax1.twinx()
        sns.lineplot(x="periodo", y="tasa_positivos", data=period_df, ax=ax2, color="#1e88e5", marker="o", label="Tasa positivos")
        ax2.set_ylabel("Tasa de positivos")
        ax2.legend(loc="upper right")
        ax2.grid(False)

        plt.title(f"Distribución de {target_name} por {period_label}")
        fig.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)

        if close:
            plt.close(fig)
            return output_path
        return fig

    # ------------------------------------------------------------------
    # 5. Concentración de variables categóricas
    # ------------------------------------------------------------------
    def categorical_concentration(self, cols: list[str] | None = None, top_n: int = 10) -> dict:
        """Analiza qué tan concentrada está cada variable categórica.

        Para cada columna retorna una tabla con las `top_n` categorías más
        frecuentes (el resto se agrupa en "otros") y un resumen con métricas
        de concentración: participación de la categoría top 1, top 3, y el
        índice de Herfindahl-Hirschman (HHI) sobre las proporciones
        (0 = totalmente disperso entre muchas categorías, 1 = una sola
        categoría concentra el 100% de los registros).

        Parameters
        ----------
        cols : list[str] | None
            Columnas a analizar. Si es None, se auto-detectan las columnas
            categóricas/texto (excluyendo las de fecha).
        top_n : int
            Número de categorías principales a mostrar por columna.
        """
        df = self._require_df()
        n_rows = len(df)

        if cols is None:
            cols = [
                c for c in df.columns
                if (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))
                and c not in self.date_columns
            ]

        tables: dict[str, pd.DataFrame] = {}
        summary_rows = []

        for col in cols:
            # `fillna` deja el nulo como una categoría explícita ("(nulo)")
            # y evita mezclar tipos (str/NaN) en el índice de value_counts.
            counts = df[col].fillna("(nulo)").astype(str).value_counts()
            table = pd.DataFrame({
                "categoria": counts.index,
                "n": counts.values,
            })
            table["pct"] = round(table["n"] / n_rows, 4)
            table["pct_acumulado"] = round(table["pct"].cumsum(), 4)

            top_table = table.head(top_n).copy()
            if len(table) > top_n:
                resto = table.iloc[top_n:]
                otros = pd.DataFrame([{
                    "categoria": f"otros ({len(resto)} categorías)",
                    "n": int(resto["n"].sum()),
                    "pct": round(resto["pct"].sum(), 4),
                    "pct_acumulado": 1.0,
                }])
                top_table = pd.concat([top_table, otros], ignore_index=True)

            tables[col] = top_table

            hhi = round(float((table["pct"] ** 2).sum()), 4)
            summary_rows.append({
                "columna": col,
                "n_categorias": df[col].nunique(dropna=False),
                "categoria_top1": table["categoria"].iloc[0],
                "top1_pct": table["pct"].iloc[0],
                "top3_pct": round(table["pct"].iloc[:3].sum(), 4),
                "hhi": hhi,
            })

        summary = pd.DataFrame(summary_rows).sort_values("top1_pct", ascending=False).reset_index(drop=True)
        return {"tables": tables, "summary": summary}

    def plot_categorical_concentration(
        self,
        col: str,
        table: pd.DataFrame,
        output_path: str | Path | None = None,
        close: bool = True,
    ):
        """Gráfico de barras con la participación (%) de cada categoría.

        `table` es la tabla top_n + "otros" devuelta en
        `categorical_concentration()["tables"][col]`.
        """
        fig, ax = plt.subplots(figsize=(8, 4.5))
        categorias = table["categoria"].astype(str)
        sns.barplot(x=categorias, y=table["pct"], hue=categorias, ax=ax, palette="tab10", legend=False)
        ax.set_ylabel("Proporción del total")
        ax.set_xlabel(col)
        ax.set_title(f"Concentración de categorías - {col}")
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)

        if close:
            plt.close(fig)
            return output_path
        return fig

    # ------------------------------------------------------------------
    # 7. Correlación y multicolinealidad
    # ------------------------------------------------------------------
    def correlation_matrix(self, cols: list[str] | None = None, method: str = "pearson") -> pd.DataFrame:
        """Matriz de correlación entre columnas numéricas."""
        df = self._require_df()
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return df[cols].corr(method=method)

    def correlation_with_target(
        self,
        target_col: str,
        cols: list[str] | None = None,
        method: str = "pearson",
    ) -> pd.DataFrame:
        """Correlación de cada variable numérica con el target, ordenada por |correlación|."""
        df = self._require_df()
        if cols is None:
            cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

        corr = df[cols + [target_col]].corr(method=method)[target_col].drop(target_col)
        result = corr.reset_index()
        result.columns = ["columna", "corr_con_target"]
        result["abs_corr"] = result["corr_con_target"].abs().round(4)
        result["corr_con_target"] = result["corr_con_target"].round(4)
        return result.sort_values("abs_corr", ascending=False).reset_index(drop=True)

    def high_correlation_pairs(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float = 0.7,
        exclude_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Pares de variables con |correlación| >= threshold (candidatos a multicolinealidad)."""
        exclude_cols = exclude_cols or []
        cols = [c for c in corr_matrix.columns if c not in exclude_cols]

        pairs = []
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                val = corr_matrix.loc[c1, c2]
                if pd.notna(val) and abs(val) >= threshold:
                    pairs.append({"variable_1": c1, "variable_2": c2, "corr": round(float(val), 4)})

        if not pairs:
            return pd.DataFrame(columns=["variable_1", "variable_2", "corr"])
        return (
            pd.DataFrame(pairs)
            .reindex(columns=["variable_1", "variable_2", "corr"])
            .assign(abs_corr=lambda d: d["corr"].abs())
            .sort_values("abs_corr", ascending=False)
            .drop(columns="abs_corr")
            .reset_index(drop=True)
        )

    def plot_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        output_path: str | Path | None = None,
        close: bool = True,
    ):
        """Heatmap anotado de una matriz de correlación."""
        n = len(corr_matrix.columns)
        fig, ax = plt.subplots(figsize=(max(7, 0.65 * n), max(6, 0.65 * n)))
        sns.heatmap(
            corr_matrix,
            ax=ax,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 7},
            cbar_kws={"shrink": 0.8, "label": "Correlación"},
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.set_title("Matriz de correlación")
        fig.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)

        if close:
            plt.close(fig)
            return output_path
        return fig

    def variance_inflation_factors(self, cols: list[str] | None = None) -> dict:
        """Calcula el VIF (Variance Inflation Factor) para detectar multicolinealidad.

        VIF > 10 indica multicolinealidad severa, 5-10 moderada, < 5 baja.
        El cálculo requiere filas completas (sin nulos) en las columnas
        analizadas; las filas con algún nulo se excluyen y se reporta cuántas.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant

        df = self._require_df()
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        sub = df[cols].dropna()
        n_dropped = len(df) - len(sub)

        X = add_constant(sub)
        rows = []
        for i, col in enumerate(X.columns):
            if col == "const":
                continue
            vif_val = variance_inflation_factor(X.values, i)
            rows.append({"columna": col, "vif": round(float(vif_val), 2)})

        def interpretacion(vif: float) -> str:
            """Traduce un VIF a una etiqueta cualitativa de severidad de multicolinealidad."""
            if vif > 10:
                return "severa"
            if vif > 5:
                return "moderada"
            return "baja"

        result = pd.DataFrame(rows)
        result["multicolinealidad"] = result["vif"].apply(interpretacion)
        result = result.sort_values("vif", ascending=False).reset_index(drop=True)

        return {
            "vif": result,
            "n_rows_used": len(sub),
            "n_rows_dropped": n_dropped,
        }

    # ------------------------------------------------------------------
    # 8. Sesgo (skewness) y outliers en variables numéricas
    # ------------------------------------------------------------------
    def numeric_distribution_summary(
        self,
        cols: list[str] | None = None,
        exclude_binary: bool = True,
    ) -> pd.DataFrame:
        """Sesgo, curtosis y outliers (regla IQR) por variable numérica.

        Clasifica el sesgo como "bajo" (|skew|<=0.5), "moderado" (<=1) o
        "alto" (>1); y el nivel de outliers (% fuera de rango IQR) como
        "bajo" (<=1%), "moderado" (<=5%) o "alto" (>5%).
        """
        df = self._require_df()
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if exclude_binary:
                cols = [c for c in cols if df[c].dropna().nunique() > 2]

        def skew_label(skew: float) -> str:
            """Traduce un coeficiente de asimetría a una etiqueta cualitativa (alto/moderado/bajo)."""
            if abs(skew) > 1:
                return "alto"
            if abs(skew) > 0.5:
                return "moderado"
            return "bajo"

        def outlier_label(pct: float) -> str:
            """Traduce el porcentaje de outliers de una columna a una etiqueta cualitativa (alto/moderado/bajo)."""
            if pct > 0.05:
                return "alto"
            if pct > 0.01:
                return "moderado"
            return "bajo"

        rows = []
        for col in cols:
            serie = df[col].dropna()
            q1, q3 = serie.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((serie < lower) | (serie > upper)).sum())
            pct_out = round(n_out / len(serie), 4) if len(serie) else 0.0
            skew = float(serie.skew())

            rows.append({
                "columna": col,
                "media": round(float(serie.mean()), 4),
                "mediana": round(float(serie.median()), 4),
                "std": round(float(serie.std()), 4),
                "min": round(float(serie.min()), 4),
                "max": round(float(serie.max()), 4),
                "skewness": round(skew, 4),
                "sesgo": skew_label(skew),
                "n_outliers_iqr": n_out,
                "pct_outliers_iqr": pct_out,
                "outliers": outlier_label(pct_out),
            })

        return pd.DataFrame(rows).sort_values("pct_outliers_iqr", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 9. Utilidad de variables categóricas para detectar el target
    # ------------------------------------------------------------------
    def categorical_target_association(
        self,
        target_col: str,
        cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fuerza de asociación (Cramér's V) entre cada categórica y el target.

        Usa un test chi-cuadrado de independencia sobre la tabla de
        contingencia (categoría x target). Cramér's V se interpreta como:
        <0.1 despreciable, 0.1-0.3 débil, 0.3-0.5 moderada, >0.5 fuerte.
        """
        from scipy.stats import chi2_contingency

        df = self._require_df()
        if cols is None:
            cols = [
                c for c in df.columns
                if (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))
                and c not in self.date_columns
            ]

        def interpretacion(v: float) -> str:
            """Traduce una V de Cramér a una etiqueta cualitativa de fuerza de asociación."""
            if v < 0.1:
                return "despreciable"
            if v < 0.3:
                return "débil"
            if v < 0.5:
                return "moderada"
            return "fuerte"

        rows = []
        for col in cols:
            tmp_col = df[col].fillna("(nulo)").astype(str)
            contingency = pd.crosstab(tmp_col, df[target_col])
            chi2, p_value, _, _ = chi2_contingency(contingency)
            n = contingency.values.sum()
            k = min(contingency.shape)
            cramers_v = float(np.sqrt((chi2 / n) / (k - 1))) if k > 1 else np.nan

            rows.append({
                "columna": col,
                "n_categorias": contingency.shape[0],
                "cramers_v": round(cramers_v, 4),
                "p_value": p_value,
                "asociacion_con_target": interpretacion(cramers_v),
            })

        return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)

    def category_fraud_rate(
        self,
        target_col: str,
        col: str,
        min_support: int = 30,
    ) -> pd.DataFrame:
        """Tasa del target y lift por categoría (solo categorías con soporte suficiente).

        `lift` = tasa de la categoría / tasa global. Un lift muy por encima o
        por debajo de 1 indica que esa categoría podría ser señal útil para
        distinguir la clase positiva del target.
        """
        df = self._require_df()
        # Se usa un nombre de columna temporal para la categoría (en vez de
        # `col` tal cual) porque `col` puede coincidir con "n" o "n_total"
        # (los nombres de las columnas agregadas), lo que rompe `reset_index`.
        tmp_col = df[col].fillna("(nulo)").astype(str)
        global_rate = df[target_col].mean()

        grouped = (
            pd.DataFrame({"categoria": tmp_col, target_col: df[target_col]})
            .groupby("categoria")[target_col]
            .agg(n_total="count", n_positivos="sum")
            .reset_index()
            .rename(columns={"categoria": col})
        )
        grouped["tasa"] = round(grouped["n_positivos"] / grouped["n_total"], 4)
        grouped["lift"] = round(grouped["tasa"] / global_rate, 2)
        grouped = grouped[grouped["n_total"] >= min_support]
        return grouped.sort_values("tasa", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _require_df(self) -> pd.DataFrame:
        """Devuelve el DataFrame cargado, o lanza un error si aún no se llamó a `load()`."""
        if self.df is None:
            raise RuntimeError("Debes llamar a `load()` antes de usar este método.")
        return self.df
