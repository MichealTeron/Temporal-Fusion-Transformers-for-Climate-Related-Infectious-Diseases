from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer


@dataclass(frozen=True)
class DatasetSpec:
    csv_path: Path
    date_col: str = "Date"
    date_format: str = "%d/%m/%Y"
    target: str = "mal"
    group_col: str = "group_ids"
    time_idx_col: str = "time_idx"

    static_reals: Tuple[str, ...] = ("sin_week", "cos_week", "sin_month", "cos_month")
    known_reals: Tuple[str, ...] = ("MaxT", "MinT", "Max_Rain", "Min_Rain")
    unknown_reals: Tuple[str, ...] = (
        "temp",
        "PEVR",
        "PWV",
        "RH",
        "SMC",
        "rain",
        "Uwind",
        "Vwind",
        "Wspd",
        "dmi_1",
        "dmi_2",
        "dmi_3",
        "Nino",
    )

    # for plotting per group
    group_ids: Tuple[int, ...] = (1, 2)


def load_dataframe(spec: DatasetSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.csv_path)
    df[spec.date_col] = pd.to_datetime(df[spec.date_col], format=spec.date_format)

    if spec.time_idx_col not in df.columns:
        df[spec.time_idx_col] = range(len(df))

    return df


def make_datasets(
    df: pd.DataFrame,
    spec: DatasetSpec,
    min_encoder_length: int,
    max_encoder_length: int,
    max_prediction_length: int,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    training_cutoff = df[spec.time_idx_col].max() - max_prediction_length

    training = TimeSeriesDataSet(
        df[lambda x: x[spec.time_idx_col] <= training_cutoff],
        time_idx=spec.time_idx_col,
        target=spec.target,
        group_ids=[spec.group_col],
        target_normalizer=GroupNormalizer(groups=[spec.group_col], transformation="softplus"),
        allow_missing_timesteps=True,
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=list(spec.static_reals),
        time_varying_known_categoricals=[],
        time_varying_known_reals=list(spec.known_reals),
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=list(spec.unknown_reals),
    )

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    return training, validation

