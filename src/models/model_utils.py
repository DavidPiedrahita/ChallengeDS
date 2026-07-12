import numpy as np
import pandas as pd

def _align_split_to_day(fecha_sorted, idx):

    """
    Ajusta un índice de corte al límite de día calendario más cercano.

    Evita que un mismo día quede repartido entre ambos lados del corte
    (p. ej. la mitad de las filas del 9-abr en train y la otra mitad en
    test): busca, entre las posiciones donde realmente cambia el día en
    `fecha_sorted`, la más cercana al índice original `idx`.

    Args:
        fecha_sorted (np.ndarray): Fechas ya ordenadas cronológicamente.
        idx (int): Índice de corte propuesto (antes de alinear).

    Returns:
        int: Índice de corte alineado a un límite de día. Si `idx` está en
            un extremo (0 o len(fecha_sorted)) o todas las fechas caen en
            un único día, se retorna sin modificar.
    """
    n = len(fecha_sorted)
    if idx <= 0 or idx >= n:
        return idx

    day_values = fecha_sorted.astype('datetime64[D]')
    day_starts = np.flatnonzero(np.diff(day_values) != np.timedelta64(0, 'D')) + 1
    if len(day_starts) == 0:
        return idx  

    return int(day_starts[np.argmin(np.abs(day_starts - idx))])


def train_test_split_time(
    df,
    features,
    target,
    test_size = 0.2,
    val_size = None,
    strategy = 'split_last',
    align_to_day = True,
):
    """
    Realiza una partición de train / (val opcional) / test basada en tiempo.

    Es el equivalente funcional de `time_based_split`, pero el ordenamiento
    cronológico y el cálculo de los cortes de la partición se hacen con
    numpy (`np.argsort` para ordenar por fecha y `np.arange`/máscaras
    booleanas para definir los cortes) en lugar de `df.sort_values` y
    `.iloc` con slices de pandas.

    Args:
        df (pd.DataFrame): Debe contener una columna 'fecha' para ordenar.
        features (list): Nombres de las columnas de features para los conjuntos X.
        target (str): Nombre de la columna del target.
        test_size (float): Proporción de los datos reservada para el conjunto de test.
        val_size (float | None): Proporción para validación. Si es None, no se genera partición de validación.
        strategy (str): Estrategia de asignación cuando se define `val_size`.
            'split_last'  — val y test ocupan periodos consecutivos al final:
                            [------train------][---val---][---test---]
            'same_period' — val y test se toman del mismo periodo final
                            (cola total = test_size + val_size), intercalados
                            por índice de fila para que ambos cubran el mismo
                            rango de tiempo:
                            [------train------][val/test intercalados]
        align_to_day (bool): Si es True (default), cada corte se desplaza al
            límite de día calendario más cercano para que ningún día quede
            repartido entre dos particiones consecutivas (evita fuga de
            información cuando hay features agregadas por día/entidad o
            patrones de fraude concentrados en un mismo día). Las
            proporciones (`test_size`/`val_size`) quedan entonces como un
            objetivo aproximado, no exacto, porque el corte se redondea al
            día más cercano. Si es False, se corta exactamente en la
            proporción indicada, aunque eso parta un día entre dos lados.

    Returns:
        Sin val : (X_train, X_test,       y_train, y_test)
        Con val : (X_train, X_val, X_test, y_train, y_val, y_test)
    """

    fecha_values = df['fecha'].to_numpy()
    sorted_idx = np.argsort(fecha_values, kind='mergesort')
    df_sorted = df.iloc[sorted_idx]
    fecha_sorted = fecha_values[sorted_idx]
    n = len(df_sorted)
    row_idx = np.arange(n)

    # ── Sin validación ──────────────────────────────────────────────────────
    if val_size is None:
        split_idx = int(n * (1 - test_size))
        if align_to_day:
            split_idx = _align_split_to_day(fecha_sorted, split_idx)

        train_mask = row_idx < split_idx
        test_mask = ~train_mask

        train_df = df_sorted.iloc[train_mask]
        test_df = df_sorted.iloc[test_mask]

        X_train, y_train = train_df[features], train_df[target]
        X_test, y_test = test_df[features], test_df[target]

        print("Partición basada en tiempo (sin validación):")
        print(f"  Train : {X_train.shape}")
        print(f"  Test  : {X_test.shape}")
        return X_train, X_test, y_train, y_test

    # ── Con validación ────────────────────────────────────────────────────────
    if strategy == 'split_last':
        # Orden cronológico: [train][val][test]
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))
        if align_to_day:
            train_end = _align_split_to_day(fecha_sorted, train_end)
            val_end = _align_split_to_day(fecha_sorted, val_end)
            val_end = max(val_end, train_end)  # salvaguarda si ambos cortes alinean al mismo día

        train_mask = row_idx < train_end
        val_mask = (row_idx >= train_end) & (row_idx < val_end)
        test_mask = row_idx >= val_end

        train_df = df_sorted.iloc[train_mask]
        val_df = df_sorted.iloc[val_mask]
        test_df = df_sorted.iloc[test_mask]

    elif strategy == 'same_period':
        # Mismo periodo final para ambos; se intercalan por fila (par→val, impar→test)
        train_end = int(n * (1 - test_size - val_size))
        if align_to_day:
            train_end = _align_split_to_day(fecha_sorted, train_end)

        train_mask = row_idx < train_end
        held_mask = row_idx >= train_end

        train_df = df_sorted.iloc[train_mask]
        held_df = df_sorted.iloc[held_mask]

        held_idx = np.arange(len(held_df))
        val_df = held_df.iloc[held_idx % 2 == 0]   # filas pares
        test_df = held_df.iloc[held_idx % 2 == 1]  # filas impares

    else:
        raise ValueError(
            f"Estrategia desconocida '{strategy}'. Elige 'split_last' o 'same_period'."
        )

    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]

    print(f"Partición basada en tiempo (estrategia='{strategy}'):")
    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def time_based_split(
    df,
    features,
    target,
    test_size=0.2,
    val_size=None,
    strategy='split_last',
):
    """
    Performs a time-based train / (optional val) / test split.

    Args:
        df (pd.DataFrame): Must contain a 'fecha' column for sorting.
        features (list): Feature column names for X sets.
        target (str): Target column name.
        test_size (float): Proportion of data reserved for the test set.
        val_size (float | None): Proportion for validation. If None, no val split.
        strategy (str): Allocation strategy when val_size is provided.
            'split_last'  — val and test occupy consecutive tail periods:
                            [------train------][---val---][---test---]
            'same_period' — val and test are drawn from the same tail period
                            (total tail = test_size + val_size), interleaved
                            by row index so both cover the same time range:
                            [------train------][val/test interleaved]

    Returns:
        Without val : (X_train, X_test,       y_train, y_test)
        With val    : (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    df_sorted = df.sort_values('fecha')
    n = len(df_sorted)

    # ── No validation ────────────────────────────────────────────────────────
    if val_size is None:
        split_idx = int(n * (1 - test_size))
        train_df  = df_sorted.iloc[:split_idx]
        test_df   = df_sorted.iloc[split_idx:]

        X_train, y_train = train_df[features], train_df[target]
        X_test,  y_test  = test_df[features],  test_df[target]

        print("Time-based split (no validation):")
        print(f"  Train : {X_train.shape}")
        print(f"  Test  : {X_test.shape}")
        return X_train, X_test, y_train, y_test

    # ── With validation ───────────────────────────────────────────────────────
    if strategy == 'split_last':
        # Chronological order: [train][val][test]
        train_end = int(n * (1 - test_size - val_size))
        val_end   = int(n * (1 - test_size))
        train_df  = df_sorted.iloc[:train_end]
        val_df    = df_sorted.iloc[train_end:val_end]
        test_df   = df_sorted.iloc[val_end:]

    elif strategy == 'same_period':
        # Same tail period for both; interleave by row (even→val, odd→test)
        train_end = int(n * (1 - test_size - val_size))
        train_df  = df_sorted.iloc[:train_end]
        held_df   = df_sorted.iloc[train_end:]
        val_df    = held_df.iloc[::2]   # even rows
        test_df   = held_df.iloc[1::2]  # odd rows

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose 'split_last' or 'same_period'."
        )

    X_train, y_train = train_df[features], train_df[target]
    X_val,   y_val   = val_df[features],   val_df[target]
    X_test,  y_test  = test_df[features],  test_df[target]

    print(f"Time-based split (strategy='{strategy}'):")
    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test