from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .data import load_dataframe, make_datasets
from .train import build_model, build_trainer
from .eval import evaluate_and_log
from .plotting import plot_predictions_for_groups, plot_interpretation


def parse_args():
    p = argparse.ArgumentParser(prog="tft-physics")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train TFT models with physics-informed loss")
    t.add_argument("--config", type=Path, required=True, help="Path to YAML config")

    return p.parse_args()


def main():
    args = parse_args()

    if args.cmd == "train":
        run_train(args.config)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


def run_train(config_path: Path):
    cfg = Config.from_yaml(config_path)

    df = load_dataframe(cfg.dataset)
    results_dir = cfg.train.out_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = results_dir / "model_results.csv"
    if not metrics_file.exists():
        metrics_file.write_text("Model,MaxEncoder,MaxPrediction,MAE,MSE,RMSE,R2\n")

    for i in range(cfg.search.i_start, cfg.search.i_end + 1):
        max_encoder_length = cfg.search.encoder_base + i
        max_prediction_length = cfg.search.pred_base + i

        print(f"=== Training model_{i} (encoder={max_encoder_length}, pred={max_prediction_length}) ===")

        training, validation = make_datasets(
            df=df,
            spec=cfg.dataset,
            min_encoder_length=cfg.search.min_encoder_length,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
        )

        train_dl = training.to_dataloader(train=True, batch_size=cfg.train.batch_size, num_workers=0)
        val_dl = validation.to_dataloader(train=False, batch_size=cfg.train.batch_size * 10, num_workers=0)

        model = build_model(training, cfg.train)
        trainer = build_trainer(cfg.train, run_name=f"run_{i}")

        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        ckpt_dir = results_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / f"ptmodel_{i}.ckpt"
        trainer.save_checkpoint(str(ckpt_path))

        # Evaluation
        mae, mse, rmse, r2 = evaluate_and_log(
            model=model,
            val_dataloader=val_dl,
            max_prediction_length=max_prediction_length,
            out_dir=results_dir,
            model_label=f"model_{i}",
        )

        with metrics_file.open("a") as f:
            f.write(f"model_{i},{max_encoder_length},{max_prediction_length},{mae},{mse},{rmse},{r2}\n")

        # Plots
        plot_predictions_for_groups(
            model=model,
            training_dataset=training,
            group_ids=cfg.dataset.group_ids,
            out_dir=results_dir,
            model_idx=i,
        )
        plot_interpretation(
            model=model,
            val_dataloader=val_dl,
            out_dir=results_dir,
            model_idx=i,
        )

        print(f"model_{i}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    print(f"All done. Results in: {results_dir}")

