from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
from pytorch_forecasting.metrics import QuantileLoss


@dataclass(frozen=True)
class Thresholds:
    temp_low: float = 13.24
    temp_high: float = 24.91
    rain_low: float = 0.00
    rain_high: float = 26.04
    rh_low: float = 39.0
    rh_high: float = 86.0
    smc_low: float = 0.12
    smc_high: float = 0.31
    dmi1_low: float = -0.34
    dmi1_high: float = 0.34
    nino_low: float = 27.5
    nino_high: float = 29.0


def _safe_zero_like(y_pred: torch.Tensor) -> torch.Tensor:
    return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)


def temp_constraint_loss(y_pred, target, temp, th: Thresholds):
    if temp is None:
        return _safe_zero_like(y_pred)
    penalty = ((temp < th.temp_low) | (temp > th.temp_high)).to(y_pred.dtype)
    return (penalty * torch.abs(y_pred - target)).mean()


def rain_constraint_loss(y_pred, target, rain, th: Thresholds):
    if rain is None:
        return _safe_zero_like(y_pred)
    penalty = ((rain < th.rain_low) | (rain > th.rain_high)).to(y_pred.dtype)
    return (penalty * torch.abs(y_pred - target)).mean()


def humidity_constraint_loss(y_pred, target, rh, th: Thresholds):
    if rh is None:
        return _safe_zero_like(y_pred)
    low = torch.relu(rh - th.rh_low) * torch.relu(y_pred - target)
    high = torch.relu(th.rh_high - rh) * torch.relu(y_pred - target)
    return low.mean() + high.mean()


def smc_constraint_loss(y_pred, target, smc, th: Thresholds):
    if smc is None:
        return _safe_zero_like(y_pred)
    low = torch.relu(smc - th.smc_low) * torch.relu(y_pred - target)
    high = torch.relu(th.smc_high - smc) * torch.relu(y_pred - target)
    return low.mean() + high.mean()


def dmi1_constraint_loss(y_pred, target, dmi1, th: Thresholds):
    if dmi1 is None:
        return _safe_zero_like(y_pred)
    low = torch.relu(dmi1 - th.dmi1_low) * torch.relu(y_pred - target)
    high = torch.relu(th.dmi1_high - dmi1) * torch.relu(y_pred - target)
    return low.mean() + high.mean()


def nino_constraint_loss(y_pred, target, nino, th: Thresholds):
    if nino is None:
        return _safe_zero_like(y_pred)
    low = torch.relu(nino - th.nino_low) * torch.relu(y_pred - target)
    high = torch.relu(th.nino_high - nino) * torch.relu(y_pred - target)
    return low.mean() + high.mean()


def lagged_effects_constraint_loss(
    y_pred,
    target,
    temp,
    rain,
    temperature_lag: int = 2,
    rainfall_lag: int = 16,
):
    if temp is None or rain is None:
        return _safe_zero_like(y_pred)
    lagged_temp = torch.roll(temp, shifts=temperature_lag, dims=0)
    lagged_rain = torch.roll(rain, shifts=rainfall_lag, dims=0)
    temperature_effects = torch.relu(lagged_temp - 25.0) * torch.relu(y_pred - target)
    rainfall_effects = torch.relu(lagged_rain - 20.0) * torch.relu(y_pred - target)
    return (temperature_effects + rainfall_effects).mean()


class CustomLoss(QuantileLoss):
    """
    Quantile loss + physics penalties.

    NOTE:
    - PyTorch Forecasting calls loss(y_pred, target) by default.
    - To use the physics terms, we pass an `extras` dict from a custom LightningModule.
    """

    def __init__(self, physics_coeff: float = 0.1, thresholds: Optional[Thresholds] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.physics_coeff = physics_coeff
        self.thresholds = thresholds or Thresholds()

    def forward(self, y_pred, target, extras: Optional[Dict[str, torch.Tensor]] = None):
        base = super().forward(y_pred, target)
        if self.physics_coeff <= 0 or not extras:
            return base

        th = self.thresholds
        temp = extras.get("temp")
        rain = extras.get("rain")
        smc = extras.get("SMC")
        dmi1 = extras.get("dmi_1")
        nino = extras.get("Nino")
        rh = extras.get("RH")

        physics = (
            temp_constraint_loss(y_pred, target, temp, th)
            + rain_constraint_loss(y_pred, target, rain, th)
            + smc_constraint_loss(y_pred, target, smc, th)
            + dmi1_constraint_loss(y_pred, target, dmi1, th)
            + nino_constraint_loss(y_pred, target, nino, th)
            + humidity_constraint_loss(y_pred, target, rh, th)
            + lagged_effects_constraint_loss(y_pred, target, temp, rain)
        )
        return base + self.physics_coeff * physics

