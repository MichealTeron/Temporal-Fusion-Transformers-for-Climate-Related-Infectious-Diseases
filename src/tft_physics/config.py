from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml

from .data import DatasetSpec
from .train import TrainSpec


@dataclass(frozen=True)
class SearchSpec:
    i_start: int = 1
    i_end: int = 49
    encoder_base: int = 52
    pred_base: int = 10
    min_encoder_length: int = 1


@dataclass(frozen=True)
class Config:
    dataset: DatasetSpec
    train: TrainSpec
    search: SearchSpec

    @staticmethod
    def from_yaml(path: Path) -> "Config":
        d = yaml.safe_load(path.read_text())

        dataset_cfg = d["dataset"]
        dataset = DatasetSpec(
            csv_path=Path(dataset_cfg["csv_path"]),
            date_col=dataset_cfg.get("date_col", "Date"),
            date_format=dataset_cfg.get("date_format", "%d/%m/%Y"),
            target=dataset_cfg.get("target", "mal"),
            group_col=dataset_cfg.get("group_col", "group_ids"),
            group_ids=tuple(dataset_cfg.get("group_ids_list", [1, 2])),
        )

        train_cfg = d["train"]
        train = TrainSpec(
            out_dir=Path(train_cfg["out_dir"]),
            accelerator=train_cfg.get("accelerator", "gpu"),
            max_epochs=int(train_cfg.get("max_epochs", 200)),
            batch_size=int(train_cfg.get("batch_size", 128)),
            grad_clip=float(train_cfg.get("grad_clip", 0.0667)),
            seed=int(train_cfg.get("seed", 42)),
            learning_rate=float(train_cfg.get("learning_rate", 0.0328)),
            hidden_size=int(train_cfg.get("hidden_size", 30)),
            attention_head_size=int(train_cfg.get("attention_head_size", 2)),
            dropout=float(train_cfg.get("dropout", 0.2658)),
            hidden_continuous_size=int(train_cfg.get("hidden_continuous_size", 14)),
            output_size=int(train_cfg.get("output_size", 7)),
            physics_coeff=float(train_cfg.get("physics_coeff", 0.1)),
            early_stop_patience=int(train_cfg.get("early_stop_patience", 50)),
        )

        search = SearchSpec(**d.get("search", {}))
        return Config(dataset=dataset, train=train, search=search)

