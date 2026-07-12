
import json
from pathlib import Path

from sklearn.compose import ColumnTransformer

from src.transformers_utils.prod_transformers import (
    FrequencyEncodingTransformer,
    HourCyclicSinCosTransformer,
    HourTransformer,
    IsWeekendFlagTransformer,
    LogFeatureTransformer,
    MaxWindowTransformer,
    MeanWindowTransformer,
    MedianWindowTransformer,
    MinWindowTransformer,
    MissingFlagTransformer,
    PCATransformer,
    QuantileBinsDropnaTransformer,
    RareGroupingTransformer,
    RatioTransformer,
    SumWindowTransformer,
    TargetEncodingCVTransformer,
    TopPercentileFlagTransformer,
    WeekdayTransformer,
    ZeroFlagTransformer,
)

_STRING_TRANSFORMER_ = {
    "LogFeatureTransformer": LogFeatureTransformer,
    "MissingFlagTransformer":MissingFlagTransformer,
    "ZeroFlagTransformer": ZeroFlagTransformer,
    "HourTransformer": HourTransformer,
    "WeekdayTransformer": WeekdayTransformer,
    "IsWeekendFlagTransformer": IsWeekendFlagTransformer,
    "HourCyclicSinCosTransformer": HourCyclicSinCosTransformer,
    "RatioTransformer": RatioTransformer,
    "TopPercentileFlagTransformer": TopPercentileFlagTransformer,
    "QuantileBinsDropnaTransformer": QuantileBinsDropnaTransformer,
    "FrequencyEncodingTransformer": FrequencyEncodingTransformer,
    "RareGroupingTransformer": RareGroupingTransformer,
    "TargetEncodingCVTransformer": TargetEncodingCVTransformer,
    "PCATransformer": PCATransformer,
    "SumWindowTransformer": SumWindowTransformer,
    "MeanWindowTransformer": MeanWindowTransformer,
    "MedianWindowTransformer": MedianWindowTransformer,
    "MaxWindowTransformer": MaxWindowTransformer,
    "MinWindowTransformer": MinWindowTransformer,
}


def load_config(config_path) -> dict:
    """
    Reads and parses a JSON pipeline config from disk.

    Args:
        config_path (str | Path): Path to the JSON config file.

    Returns:
        dict: parsed config, e.g. {"steps": [{"name", "type", "cols", "params"}, ...]}
    """
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def build_preprocessor(config: dict) -> ColumnTransformer:
    """
    Builds an unfitted ColumnTransformer from an already-parsed config dict.

    Each entry in config["steps"] must have:
        name:   step name (string)
        type:   transformer class name from _REGISTRY, or "passthrough"
        cols:   list of column names passed to the transformer
        params: (optional) dict of constructor kwargs

    For the *WindowTransformer steps (Sum/Mean/Median/Max/MinWindow), `cols`
    must include the value column plus the group/date columns (e.g.
    `["monto", "j", "fecha"]`), and `params` must set `value_column`
    (`group_column`/`date_column` default to "j"/"fecha" if omitted).

    Args:
        config (dict): parsed pipeline config (see `load_config`).

    Returns:
        sklearn.compose.ColumnTransformer (unfitted)
    """
    steps = []

    for step in config["steps"]:
        name   = step["name"]
        type_  = step["type"]
        cols   = step["cols"]
        params = dict(step.get("params") or {})

        if type_ == "passthrough":
            steps.append((name, "passthrough", cols))
            continue

        cls = _STRING_TRANSFORMER_[type_]
        steps.append((name, cls(**params), cols))

    return ColumnTransformer(
        transformers = steps,
        remainder = "passthrough",
        verbose_feature_names_out = False,
    )


def build_preprocessor_from_config(config_path) -> ColumnTransformer:

    """
    Convenience wrapper: reads a JSON config from disk and builds the
    ColumnTransformer in one call. Equivalent to
    `build_preprocessor(load_config(config_path))`.

    Args:
        config_path (str | Path): Path to the JSON config file.

    Returns:
        sklearn.compose.ColumnTransformer (unfitted)
    """

    return build_preprocessor(load_config(config_path))


def filter_config_by_features(config: dict, target_features: list[str], passthrough_cols: list[str] | None = None) -> dict:
    """
    Reduces a parsed pipeline config to only the steps needed to produce
    `target_features`, plus raw passthrough for `passthrough_cols`.

    This exists so a single general config (e.g. `preprocessing_config.json`)
    can serve as the one source of truth, while individual training runs
    (e.g. a baseline model) select a curated subset of its features without
    duplicating the config file.

    How it decides which steps to keep:
        - For each non-passthrough step, the transformer is instantiated
          (via `_STRING_TRANSFORMER_`) and `get_feature_names_out(cols)` is
          called to get its *real* output name(s) -- which can differ from
          the step's `name` field (e.g. step "b_bin" outputs "b_qbin_dropna").
          The step is kept if any of those real output names is in
          `target_features`.
        - Existing "passthrough" steps are kept if their column is in
          `target_features`.
        - For each column in `passthrough_cols` that isn't already passed
          through (explicitly or as an automatic ColumnTransformer
          remainder, i.e. it's still consumed by a kept step), an explicit
          extra `{"type": "passthrough"}` step is added -- this is what
          lets e.g. `f` show up both raw and as `f_is_zero`.

    Args:
        config: parsed pipeline config (see `load_config`).
        target_features: output feature names to keep (as produced by
            `get_feature_names_out`, not the config's `name` field).
        passthrough_cols: raw input columns to also include as-is.

    Returns:
        dict: filtered config, usable directly with `build_preprocessor`.
    """
    target_features = set(target_features)
    kept_steps = []
    passthrough_covered = set()

    for step in config["steps"]:
        type_ = step["type"]
        cols = step["cols"]

        if type_ == "passthrough":
            if any(c in target_features for c in cols):
                kept_steps.append(step)
                passthrough_covered.update(cols)
            continue

        cls = _STRING_TRANSFORMER_[type_]
        params = dict(step.get("params") or {})
        output_names = list(cls(**params).get_feature_names_out(cols))

        if any(name in target_features for name in output_names):
            kept_steps.append(step)

    for col in passthrough_cols or []:
        if col in passthrough_covered:
            continue
        used_by_kept_step = any(
            s["type"] != "passthrough" and col in s["cols"] for s in kept_steps
        )
        if not used_by_kept_step:
            # No step in kept_steps references this column, so
            # ColumnTransformer's remainder="passthrough" already covers it.
            continue
        kept_steps.append({"name": f"{col}_raw", "type": "passthrough", "cols": [col]})

    return {"steps": kept_steps}
