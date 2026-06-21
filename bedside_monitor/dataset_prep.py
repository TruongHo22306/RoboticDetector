from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(slots=True)
class SourceConfig:
    dataset_root: Path
    image_dirs: dict[str, Path]
    label_dirs: dict[str, Path]
    class_names: dict[int, str]


@dataclass(slots=True)
class LabelBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_xyxy(self) -> tuple[float, float, float, float]:
        x1 = self.x_center - self.width / 2
        y1 = self.y_center - self.height / 2
        x2 = self.x_center + self.width / 2
        y2 = self.y_center + self.height / 2
        return x1, y1, x2, y2

    def to_line(self) -> str:
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


@dataclass(slots=True)
class DatasetItem:
    key: str
    image_path: Path
    label_path: Path | None
    source_split: str
    boxes: list[LabelBox]


def load_source_config(path: str | Path) -> SourceConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid dataset config: {config_path}")

    dataset_root = Path(payload["path"])
    names_payload = payload["names"]
    if isinstance(names_payload, dict):
        class_names = {int(key): str(value) for key, value in names_payload.items()}
    else:
        class_names = {index: str(value) for index, value in enumerate(names_payload)}

    image_dirs: dict[str, Path] = {}
    label_dirs: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        relative = payload.get(split)
        if not relative:
            continue
        image_dir = dataset_root / Path(relative)
        image_dirs[split] = image_dir

        parts = list(Path(relative).parts)
        if not parts or parts[0] != "images":
            raise ValueError(f"Expected split path to start with 'images': {relative}")
        parts[0] = "labels"
        label_dirs[split] = dataset_root / Path(*parts)

    return SourceConfig(
        dataset_root=dataset_root,
        image_dirs=image_dirs,
        label_dirs=label_dirs,
        class_names=class_names,
    )


def parse_label_file(label_path: Path, class_names: dict[int, str]) -> list[LabelBox]:
    if not label_path.exists():
        return []

    boxes: list[LabelBox] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label line in {label_path}:{line_number}: {raw_line}")

        class_id = int(float(parts[0]))
        if class_id not in class_names:
            raise ValueError(f"Unknown class id {class_id} in {label_path}:{line_number}")

        x_center, y_center, width, height = (float(part) for part in parts[1:])
        for value in (x_center, y_center, width, height):
            if value < 0 or value > 1:
                raise ValueError(f"Out-of-range label value in {label_path}:{line_number}")
        if width <= 0 or height <= 0:
            raise ValueError(f"Non-positive label size in {label_path}:{line_number}")

        boxes.append(LabelBox(class_id, x_center, y_center, width, height))
    return boxes


def collect_items(config: SourceConfig) -> tuple[list[DatasetItem], list[dict[str, str]]]:
    items_by_key: dict[str, DatasetItem] = {}
    duplicates: list[dict[str, str]] = []

    for split, image_dir in config.image_dirs.items():
        label_dir = config.label_dirs[split]
        if not image_dir.exists():
            continue

        for image_path in sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
            key = image_path.stem
            label_path = label_dir / f"{key}.txt"
            boxes = parse_label_file(label_path, config.class_names)
            item = DatasetItem(
                key=key,
                image_path=image_path,
                label_path=label_path if label_path.exists() else None,
                source_split=split,
                boxes=boxes,
            )

            if key in items_by_key:
                duplicates.append(
                    {
                        "key": key,
                        "existing_split": items_by_key[key].source_split,
                        "duplicate_split": split,
                    }
                )
                continue
            items_by_key[key] = item

    return list(items_by_key.values()), duplicates


def validate_output_target(source: SourceConfig, output_root: Path, output_config: Path, source_config_path: Path) -> None:
    resolved_output_root = output_root.resolve()
    protected_paths = [source.dataset_root, *source.image_dirs.values(), *source.label_dirs.values()]
    for protected_path in protected_paths:
        if resolved_output_root == protected_path.resolve():
            raise ValueError(f"Refusing to overwrite source dataset path: {output_root}")

    if output_config.resolve() == source_config_path.resolve():
        raise ValueError("Output config path must be different from the source config path.")


def build_class_remap(class_names: dict[int, str], drop_class_names: set[str]) -> tuple[dict[int, int | None], dict[int, str]]:
    remap: dict[int, int | None] = {}
    kept_names: dict[int, str] = {}
    next_id = 0
    for class_id in sorted(class_names):
        class_name = class_names[class_id]
        if class_name in drop_class_names:
            remap[class_id] = None
            continue
        remap[class_id] = next_id
        kept_names[next_id] = class_name
        next_id += 1
    return remap, kept_names


def iou(box_a: LabelBox, box_b: LabelBox) -> float:
    ax1, ay1, ax2, ay2 = box_a.to_xyxy()
    bx1, by1, bx2, by2 = box_b.to_xyxy()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection_width = max(0.0, ix2 - ix1)
    intersection_height = max(0.0, iy2 - iy1)
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def count_patient_action_overlaps(items: Iterable[DatasetItem], class_names: dict[int, str]) -> int:
    overlap_count = 0
    patient_ids = {class_id for class_id, class_name in class_names.items() if class_name == "patient"}
    for item in items:
        patient_boxes = [box for box in item.boxes if box.class_id in patient_ids]
        action_boxes = [box for box in item.boxes if box.class_id not in patient_ids]
        for patient_box in patient_boxes:
            if any(iou(patient_box, action_box) >= 0.5 for action_box in action_boxes):
                overlap_count += 1
    return overlap_count


def remap_item_boxes(
    items: Iterable[DatasetItem],
    class_names: dict[int, str],
    remap: dict[int, int | None],
) -> tuple[list[DatasetItem], Counter[str]]:
    dropped_boxes = Counter()
    remapped_items: list[DatasetItem] = []

    for item in items:
        new_boxes: list[LabelBox] = []
        for box in item.boxes:
            new_class_id = remap[box.class_id]
            if new_class_id is None:
                dropped_boxes[class_names[box.class_id]] += 1
                continue
            new_boxes.append(
                LabelBox(
                    class_id=new_class_id,
                    x_center=box.x_center,
                    y_center=box.y_center,
                    width=box.width,
                    height=box.height,
                )
            )

        remapped_items.append(
            DatasetItem(
                key=item.key,
                image_path=item.image_path,
                label_path=item.label_path,
                source_split=item.source_split,
                boxes=new_boxes,
            )
        )

    return remapped_items, dropped_boxes


def compute_split_sizes(total_items: int, val_ratio: float, test_ratio: float) -> dict[str, int]:
    if total_items < 3:
        raise ValueError("Need at least 3 unique images to build train/val/test splits.")

    test_size = max(1, round(total_items * test_ratio))
    val_size = max(1, round(total_items * val_ratio))
    train_size = total_items - val_size - test_size

    if train_size < 1:
        train_size = 1
        if val_size > test_size:
            val_size -= 1
        else:
            test_size -= 1

    while train_size + val_size + test_size > total_items:
        if val_size >= test_size and val_size > 1:
            val_size -= 1
        elif test_size > 1:
            test_size -= 1
        else:
            train_size -= 1

    while train_size + val_size + test_size < total_items:
        train_size += 1

    return {"train": train_size, "val": val_size, "test": test_size}


def assign_splits(items: list[DatasetItem], val_ratio: float, test_ratio: float, seed: int) -> dict[str, str]:
    target_sizes = compute_split_sizes(len(items), val_ratio, test_ratio)
    splits = ("train", "val", "test")
    total_class_counts = Counter()
    for item in items:
        total_class_counts.update(box.class_id for box in item.boxes)

    total_boxes = sum(total_class_counts.values()) or 1
    class_targets: dict[str, dict[int, int]] = {split: {} for split in splits}
    for class_id, count in total_class_counts.items():
        ratios = {
            "train": target_sizes["train"] / len(items),
            "val": target_sizes["val"] / len(items),
            "test": target_sizes["test"] / len(items),
        }
        assigned = 0
        for split in splits:
            target = int(round(count * ratios[split]))
            class_targets[split][class_id] = target
            assigned += target
        while assigned > count:
            largest_split = max(splits, key=lambda split: class_targets[split].get(class_id, 0))
            class_targets[largest_split][class_id] -= 1
            assigned -= 1
        while assigned < count:
            best_split = max(splits, key=lambda split: target_sizes[split])
            class_targets[best_split][class_id] = class_targets[best_split].get(class_id, 0) + 1
            assigned += 1

    rng = random.Random(seed)
    ordered_items = list(items)
    rng.shuffle(ordered_items)
    ordered_items.sort(
        key=lambda item: (
            -sum(1.0 / max(total_class_counts[box.class_id], 1) for box in item.boxes),
            -len({box.class_id for box in item.boxes}),
            -len(item.boxes),
            item.key,
        )
    )

    split_boxes = {split: Counter() for split in splits}
    split_items = Counter()
    assignment: dict[str, str] = {}

    for item in ordered_items:
        item_classes = Counter(box.class_id for box in item.boxes)
        candidates = []
        for split in splits:
            if split_items[split] >= target_sizes[split]:
                continue
            deficit_score = sum(
                max(class_targets[split].get(class_id, 0) - split_boxes[split].get(class_id, 0), 0) * count
                for class_id, count in item_classes.items()
            )
            space_score = target_sizes[split] - split_items[split]
            current_box_density = sum(split_boxes[split].values()) / max(split_items[split], 1)
            candidates.append((deficit_score, space_score, -current_box_density, split))

        if not candidates:
            split = min(splits, key=lambda name: split_items[name])
        else:
            split = max(candidates)[3]

        assignment[item.key] = split
        split_items[split] += 1
        split_boxes[split].update(item_classes)

    return assignment


def write_prepared_dataset(
    items: Iterable[DatasetItem],
    assignment: dict[str, str],
    output_root: Path,
) -> dict[str, Counter | defaultdict[str, list[str]]]:
    if output_root.exists():
        shutil.rmtree(output_root)

    stats = {
        "images_per_split": Counter(),
        "boxes_per_split": Counter(),
        "empty_labels_per_split": Counter(),
        "split_class_box_counts": defaultdict(Counter),
        "empty_label_items": defaultdict(list),
    }

    for item in items:
        split = assignment[item.key]
        image_output_dir = output_root / "images" / split
        label_output_dir = output_root / "labels" / split
        image_output_dir.mkdir(parents=True, exist_ok=True)
        label_output_dir.mkdir(parents=True, exist_ok=True)

        image_output_path = image_output_dir / item.image_path.name
        label_output_path = label_output_dir / f"{item.image_path.stem}.txt"
        shutil.copy2(item.image_path, image_output_path)
        label_output_path.write_text("\n".join(box.to_line() for box in item.boxes), encoding="utf-8")

        stats["images_per_split"][split] += 1
        stats["boxes_per_split"][split] += len(item.boxes)
        stats["split_class_box_counts"][split].update(box.class_id for box in item.boxes)
        if not item.boxes:
            stats["empty_labels_per_split"][split] += 1
            stats["empty_label_items"][split].append(item.image_path.name)

    return stats


def write_config(output_config: Path, output_root: Path, class_names: dict[int, str]) -> None:
    payload = {
        "path": output_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {class_id: class_name for class_id, class_name in sorted(class_names.items())},
    }
    output_config.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def build_report(
    source_config_path: Path,
    source_items: list[DatasetItem],
    duplicates: list[dict[str, str]],
    cleaned_items: list[DatasetItem],
    class_names: dict[int, str],
    dropped_boxes: Counter[str],
    prepared_stats: dict[str, Counter],
    output_root: Path,
    output_config: Path,
    overlap_count: int,
) -> dict[str, object]:
    class_counts = Counter()
    for item in cleaned_items:
        class_counts.update(box.class_id for box in item.boxes)

    warnings: list[str] = []
    if duplicates:
        warnings.append("Source dataset had duplicated entries across train/val splits.")
    if overlap_count > 0:
        warnings.append("Patient boxes overlapped action boxes and were removed in the refined dataset.")

    low_sample_classes = [class_names[class_id] for class_id, count in sorted(class_counts.items()) if count < 10]
    if low_sample_classes:
        warnings.append("These classes still have very few labeled boxes: " + ", ".join(low_sample_classes))

    for split in ("val", "test"):
        split_counts = prepared_stats["split_class_box_counts"].get(split, Counter())
        missing_classes = [class_names[class_id] for class_id in sorted(class_names) if split_counts.get(class_id, 0) == 0]
        if missing_classes:
            warnings.append(f"Split '{split}' is missing these classes: {', '.join(missing_classes)}")

    return {
        "source_config": str(source_config_path),
        "output_root": str(output_root),
        "output_config": str(output_config),
        "source_unique_images": len(source_items),
        "duplicate_entries_removed": len(duplicates),
        "patient_action_overlaps_detected": overlap_count,
        "dropped_boxes_by_class": dict(sorted(dropped_boxes.items())),
        "prepared_class_names": {class_id: class_name for class_id, class_name in sorted(class_names.items())},
        "prepared_class_box_counts": {class_names[class_id]: count for class_id, count in sorted(class_counts.items())},
        "images_per_split": dict(prepared_stats["images_per_split"]),
        "boxes_per_split": dict(prepared_stats["boxes_per_split"]),
        "empty_labels_per_split": dict(prepared_stats["empty_labels_per_split"]),
        "split_class_box_counts": {
            split: {
                class_names[class_id]: count
                for class_id, count in sorted(counter.items())
            }
            for split, counter in prepared_stats["split_class_box_counts"].items()
        },
        "empty_label_items": {split: names for split, names in prepared_stats["empty_label_items"].items()},
        "warnings": warnings,
        "duplicates": duplicates,
    }


def write_report(report_path: Path, report: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_dataset(
    source_config: str | Path = "config.yaml",
    output_root: str | Path = "data_refined",
    output_config: str | Path = "config_refined.yaml",
    report_path: str | Path = "reports/dataset_prepare_report.json",
    drop_class_names: Iterable[str] = ("patient",),
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, object]:
    source_config_path = Path(source_config)
    output_root_path = Path(output_root)
    output_config_path = Path(output_config)
    report_path_obj = Path(report_path)

    config = load_source_config(source_config_path)
    validate_output_target(config, output_root_path, output_config_path, source_config_path)
    source_items, duplicates = collect_items(config)
    overlap_count = count_patient_action_overlaps(source_items, config.class_names)
    remap, prepared_class_names = build_class_remap(config.class_names, set(drop_class_names))
    cleaned_items, dropped_boxes = remap_item_boxes(source_items, config.class_names, remap)
    assignment = assign_splits(cleaned_items, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    prepared_stats = write_prepared_dataset(cleaned_items, assignment, output_root_path)
    write_config(output_config_path, output_root_path, prepared_class_names)
    report = build_report(
        source_config_path=source_config_path,
        source_items=source_items,
        duplicates=duplicates,
        cleaned_items=cleaned_items,
        class_names=prepared_class_names,
        dropped_boxes=dropped_boxes,
        prepared_stats=prepared_stats,
        output_root=output_root_path,
        output_config=output_config_path,
        overlap_count=overlap_count,
    )
    write_report(report_path_obj, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a cleaner YOLO dataset for bedside activity training.")
    parser.add_argument("--source-config", default="config.yaml", help="Source dataset YAML.")
    parser.add_argument("--output-root", default="data_refined", help="Output dataset root.")
    parser.add_argument("--output-config", default="config_refined.yaml", help="Output dataset YAML.")
    parser.add_argument("--report-path", default="reports/dataset_prepare_report.json", help="Output report JSON.")
    parser.add_argument("--drop-classes", default="patient", help="Comma-separated class names to remove.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    drop_class_names = [name.strip() for name in args.drop_classes.split(",") if name.strip()]
    report = prepare_dataset(
        source_config=args.source_config,
        output_root=args.output_root,
        output_config=args.output_config,
        report_path=args.report_path,
        drop_class_names=drop_class_names,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0
