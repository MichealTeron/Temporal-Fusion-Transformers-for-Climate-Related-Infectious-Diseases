from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from .losses import CustomLoss, Thresholds


@dataclass(frozen=True)
class TrainSpec:
    out_dir: Path
    accelerator: str = "gpu"  # "cpu" or "gpu"
    max_epochs: int = 200
    batch_size: int = 128
    grad_clip: float = 0.0667
    seed: int = 42

    learning_rate: float = 0.0328
    hidden_size: int = 30
    attention_head_size: int = 2
    dropout: float = 0.2658
    hidden_continuous_size: int = 14
    output_size: int = 7

    physics_coeff: float = 0.1
    early_stop_patience: int = 50


class TFTWithExtrasLoss(TemporalFusionTransformer):
    """
    Wrap TFT so we can pass batch variables (temp/rain/etc) into CustomLoss.
    """

    def __init__(self, *args, custom_loss: Optional[CustomLoss] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss = custom_loss

    def _extras_from_batch(self, x: Dict) -> Dict[str, torch.Tensor]:
        """
        TODO: Map climate variables from batch to extras.

        You can inspect a batch to fill this in:

            x, y = next(iter(train_dataloader))
            print(x.keys())
            print(self.dataset.reals)
            print(x["decoder_cont"].shape)

        Then slice x["encoder_cont"] / x["decoder_cont"] by index.
        For now, this returns {} so physics terms are effectively off.
        """
        return {}

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        if self.custom_loss is None:
            return super().training_step(batch, batch_idx)

        target = y[0]
        extras = self._extras_from_batch(x)
        loss = self.custom_loss(out["prediction"], target, extras=extras)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        if self.custom_loss is None:
            return super().validation_step(batch, batch_idx)

        target = y[0]
        extras = self._extras_from_batch(x)
        loss = self.custom_loss(out["prediction"], target, extras=extras)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


def build_trainer(spec: TrainSpec, run_name: str) -> pl.Trainer:
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    logger = TensorBoardLogger(save_dir=str(spec.out_dir / "lightning_logs"), name=run_name)

    early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=spec.early_stop_patience, mode="min")
    lr_logger = LearningRateMonitor()

    pl.seed_everything(spec.seed)

    return pl.Trainer(
        max_epochs=spec.max_epochs,
        accelerator=spec.accelerator,
        gradient_clip_val=spec.grad_clip,
        callbacks=[lr_logger, early_stop],
        logger=logger,
        enable_model_summary=True,
    )


def build_model(training_dataset, spec: TrainSpec) -> TemporalFusionTransformer:
    base_loss = QuantileLoss()
    custom_loss = CustomLoss(physics_coeff=spec.physics_coeff, thresholds=Thresholds())

    model = TFTWithExtrasLoss.from_dataset(
        training_dataset,
        learning_rate=spec.learning_rate,
        hidden_size=spec.hidden_size,
        attention_head_size=spec.attention_head_size,
        dropout=spec.dropout,
        hidden_continuous_size=spec.hidden_continuous_size,
        output_size=spec.output_size,
        loss=base_loss,  # used internally by PF, but we override in training_step
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    model.custom_loss = custom_loss
    return model

