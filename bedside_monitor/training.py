from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from bedside_monitor.dataset_prep import prepare_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or fine-tune the bedside activity detector.")
    parser.add_argument("--data", default="config_refined.yaml", help="Dataset config YAML.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model or checkpoint.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="cpu", help="Training device.")
    parser.add_argument("--project", default="runs/detect", help="Output project directory.")
    parser.add_argument("--name", default="bedside_activity", help="Run name.")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--prepare-dataset", action="store_true", help="Prepare the refined dataset before training.")
    parser.add_argument("--source-data-config", default="config.yaml", help="Source dataset config used for preparation.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_path = Path(args.data)
    if args.prepare_dataset or not data_path.exists():
        prepare_dataset(
            source_config=args.source_data_config,
            output_root="data_refined",
            output_config=args.data,
            report_path="reports/dataset_prepare_report.json",
        )

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
    )
    return 0
