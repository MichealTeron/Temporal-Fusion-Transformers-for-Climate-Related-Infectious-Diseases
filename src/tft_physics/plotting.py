from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def plot_predictions_for_groups(
    model,
    training_dataset,
    group_ids: Iterable[int],
    out_dir: Path,
    model_idx: int,
):
    """
    For each group_id, makes a prediction and saves plot_prediction PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for gid in group_ids:
        ds = training_dataset.filter(lambda x: (x.group_ids == gid))
        raw_prediction = model.predict(ds, mode="raw", return_x=True)
        fig = model.plot_prediction(raw_prediction.x, raw_prediction.output, idx=0, add_loss_to_title=True)
        fig_path = out_dir / f"plot_prediction_model_{model_idx}_group_{gid}.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)


def plot_interpretation(
    model,
    val_dataloader,
    out_dir: Path,
    model_idx: int,
):
    """
    Plots TFT interpretation for a given model on validation data.
    """
    raw_predictions = model.predict(val_dataloader, mode="raw", return_x=True)
    interpretation = model.interpret_output(raw_predictions.output, reduction="sum")
    fig = model.plot_interpretation(interpretation)
    fig_path = out_dir / f"plot_interpretation_model_{model_idx}.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

