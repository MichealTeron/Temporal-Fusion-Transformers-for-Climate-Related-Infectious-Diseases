from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def moving_average(series: pd.Series, window_size: int) -> pd.Series:
    return series.rolling(window=window_size).mean()


def evaluate_and_log(
    model,
    val_dataloader,
    max_prediction_length: int,
    out_dir: Path,
    model_label: str,
    smoothing_window: int = 6,
) -> Tuple[float, float, float, float]:
    """
    - Predict on validation loader
    - Smooth (moving average)
    - Compute MAE, MSE, RMSE, R2
    - Append predictions to Act_Prdt.csv
    """
    predictions = model.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
    y_actual = predictions.y[0].detach().cpu().numpy().ravel()
    y_pred = predictions.output.detach().cpu().numpy().ravel()

    df = pd.DataFrame({"Actual": y_actual, "Predicted": y_pred})
    df["Smoothed_Actual"] = moving_average(df["Actual"], smoothing_window)
    df["Smoothed_Predicted"] = moving_average(df["Predicted"], smoothing_window)
    df = df.dropna().reset_index(drop=True)

    mae = mean_absolute_error(df["Smoothed_Actual"], df["Smoothed_Predicted"])
    mse = mean_squared_error(df["Smoothed_Actual"], df["Smoothed_Predicted"])
    rmse = float(np.sqrt(mse))
    r2 = r2_score(df["Smoothed_Actual"], df["Smoothed_Predicted"])

    df["ID"] = max_prediction_length

    act_prdt_path = out_dir / "Act_Prdt.csv"
    write_header = not act_prdt_path.exists()
    df.to_csv(act_prdt_path, mode="a", index=False, header=write_header)

    return mae, mse, rmse, r2

