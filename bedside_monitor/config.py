from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class MonitorConfig:
    model_path: str = "runs/detect/train-4/weights/best.pt"
    source: int | str = 0
    imgsz: int = 416
    conf: float = 0.35
    iou: float = 0.45
    device: str = "cpu"
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 20
    display: bool = True
    frame_skip: int = 1
    smoothing_window: int = 8
    critical_confirmation_frames: int = 3
    bed_exit_confirmation_frames: int = 4
    event_cooldown_seconds: float = 5.0
    bed_zone: tuple[float, float, float, float] | None = (0.18, 0.16, 0.82, 0.95)
    record_output: str | None = None
    event_log: str = "logs/activity_events.csv"
    alert_frame_dir: str = "alerts"
    save_alert_frames: bool = True
    fallback_posture_inference: bool = True


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def parse_source(value: Any) -> int | str:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def parse_bed_zone(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 4:
            raise ValueError("Bed zone must have 4 comma-separated values.")
        numbers = tuple(float(part) for part in parts)
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        numbers = tuple(float(part) for part in value)
    else:
        raise ValueError("Bed zone must be a list, tuple, or comma-separated string.")

    if any(number < 0 or number > 1 for number in numbers):
        raise ValueError("Bed zone values must be normalized between 0 and 1.")
    x1, y1, x2, y2 = numbers
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bed zone values must satisfy x2 > x1 and y2 > y1.")
    return numbers


def load_monitor_config(path: str | Path) -> MonitorConfig:
    config_path = Path(path)
    if not config_path.exists():
        return MonitorConfig()

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = MonitorConfig()
    for key, value in payload.items():
        if not hasattr(config, key):
            continue
        setattr(config, key, value)

    config.source = parse_source(config.source)
    config.display = parse_bool(config.display)
    config.save_alert_frames = parse_bool(config.save_alert_frames)
    config.fallback_posture_inference = parse_bool(config.fallback_posture_inference)
    config.bed_zone = parse_bed_zone(config.bed_zone)
    config.frame_skip = max(int(config.frame_skip), 1)
    config.smoothing_window = max(int(config.smoothing_window), 1)
    config.critical_confirmation_frames = max(int(config.critical_confirmation_frames), 1)
    config.bed_exit_confirmation_frames = max(int(config.bed_exit_confirmation_frames), 1)
    config.imgsz = max(int(config.imgsz), 160)
    config.camera_width = max(int(config.camera_width), 160)
    config.camera_height = max(int(config.camera_height), 120)
    config.camera_fps = max(int(config.camera_fps), 1)
    config.conf = float(config.conf)
    config.iou = float(config.iou)
    config.event_cooldown_seconds = float(config.event_cooldown_seconds)
    config.model_path = str(config.model_path)
    config.event_log = str(config.event_log)
    config.alert_frame_dir = str(config.alert_frame_dir)
    config.record_output = None if config.record_output in {None, ""} else str(config.record_output)
    return config
