"""Funciones de graficación para el análisis exploratorio del dataset de fraude.

Complementa los métodos `plot_*` de la clase `EDA` con gráficos ad-hoc que no
dependen de esa clase: evolución del monto en fraude e impacto económico, y
distribuciones numéricas diferenciadas por el target.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save_and_close(fig, output_path: str | Path | None, close: bool, dpi: int = 150) -> None:
    """Guarda la figura en `output_path` (si se indica) y la cierra (si `close` es True)."""
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
    if close:
        plt.close(fig)


def plot_daily_amount_evolution(
    daily_df: pd.DataFrame,
    period_col: str = "periodo",
    no_fraud_col: str = "monto_no_fraude",
    fraud_col: str = "monto_fraude",
    pct_col: str = "pct_monto_fraude",
    output_path: str | Path | None = None,
    close: bool = True,
):
    """Evolución diaria del monto transado, apilando monto fraude/no fraude.

    Usa un eje secundario (`twinx`) para el `% del monto diario que es
    fraude`, lo que es correcto aquí porque las unidades son distintas
    (monto en $ vs. porcentaje), a diferencia de graficar dos series ya
    expresadas en $ en ejes separados (ver `plot_economic_impact`).
    """
    periodos = daily_df[period_col].astype(str)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    sns.barplot(x=periodos, y=daily_df[no_fraud_col], ax=ax1, color="#cfd8dc", label="Monto no fraude", errorbar=None)
    sns.barplot(
        x=periodos,
        y=daily_df[fraud_col],
        ax=ax1,
        bottom=daily_df[no_fraud_col],
        color="#e53935",
        label="Monto fraude",
        errorbar=None,
    )
    ax1.set_ylabel("Monto ($)")
    ax1.set_xlabel("Día")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    sns.lineplot(x=periodos, y=daily_df[pct_col], ax=ax2, color="#1e88e5", marker="o", label="% monto fraude")
    ax2.set_ylabel("% del monto diario que es fraude")
    ax2.legend(loc="upper right")
    ax2.grid(False)

    ax1.set_title("Evolución diaria del monto en fraude")
    fig.tight_layout()
    _save_and_close(fig, output_path, close)
    return fig


def plot_economic_impact(
    impact_df: pd.DataFrame,
    period_col: str = "periodo",
    gain_col: str = "ganancia_25pct_legit",
    loss_col: str = "perdida_100pct_fraude",
    net_col: str = "resultado_neto",
    output_path: str | Path | None = None,
    close: bool = True,
):
    """Ganancia (25% legítimo), pérdida (100% fraude) y resultado neto por día.

    Las tres series están en las mismas unidades ($), por lo que se grafican
    en un único eje: un eje secundario (`twinx`) aquí sería engañoso, ya que
    su autoescalado independiente desalinearía el "0" de la línea de
    resultado neto respecto al "0" de las barras, aunque los valores reales
    nunca sean negativos.
    """
    df = impact_df[impact_df[period_col].astype(str) != "TOTAL"]
    periodos = df[period_col].astype(str)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    sns.barplot(x=periodos, y=df[gain_col], ax=ax1, color="#43a047", label="Ganancia (25% legítimo)", errorbar=None)
    sns.barplot(x=periodos, y=-df[loss_col], ax=ax1, color="#e53935", label="Pérdida (100% fraude)", errorbar=None)
    sns.lineplot(x=periodos, y=df[net_col], ax=ax1, color="#1e88e5", marker="o", label="Resultado neto")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Monto ($)")
    ax1.set_xlabel("Día")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax1.set_title("Impacto económico diario: ganancia (25% legítimo) vs. pérdida (100% fraude)")
    fig.tight_layout()
    _save_and_close(fig, output_path, close)
    return fig


def plot_numeric_grid_by_target(
    data: pd.DataFrame,
    cols: list[str],
    target_col: str,
    kind: str = "hist",
    ncols: int = 4,
    bins: int = 40,
    class_labels: dict | None = None,
    output_path: str | Path | None = None,
    close: bool = True,
):
    """Grilla de histogramas o boxplots por columna, separados por clase del target.

    - `kind="hist"`: histogramas superpuestos semi-transparentes por clase,
      normalizados a densidad (`density=True`) para que sean comparables aun
      cuando las clases tengan tamaños muy distintos (p. ej. ~5% de fraude).
    - `kind="box"`: un boxplot por clase, lado a lado, dentro de cada subplot.
    """
    if kind not in ("hist", "box"):
        raise ValueError('kind debe ser "hist" o "box"')

    class_labels = class_labels or {0: "No fraude", 1: "Fraude"}
    classes = sorted(data[target_col].dropna().unique())
    class_colors = ["#43a047", "#e53935", "#1e88e5", "#fb8c00"]

    n = len(cols)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.3 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols):
        ax = axes[i]
        groups = [data.loc[data[target_col] == cls, col].dropna() for cls in classes]
        labels = [class_labels.get(cls, str(cls)) for cls in classes]

        if kind == "hist":
            for serie, label, color in zip(groups, labels, class_colors):
                sns.histplot(serie, bins=bins, stat="density", alpha=0.5, color=color, label=label, ax=ax)
        else:
            tmp = pd.DataFrame({"clase": np.repeat(labels, [len(g) for g in groups]), "valor": np.concatenate(groups)})
            sns.boxplot(x="clase", y="valor", data=tmp, hue="clase", palette=class_colors[: len(classes)], legend=False, ax=ax)
            ax.set_xlabel("")

        ax.set_title(col, fontsize=10)
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    if kind == "hist":
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc="upper right")

    fig.tight_layout()
    _save_and_close(fig, output_path, close)
    return fig
