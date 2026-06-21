from __future__ import annotations

import argparse
import csv
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
from ultralytics import YOLO

from bedside_monitor.config import MonitorConfig, load_monitor_config, parse_bed_zone, parse_bool, parse_source

RAW_TO_CANONICAL = {
    "sleep": "lying_down",
    "lying": "lying_down",
    "sit": "sitting",
    "stand_up": "standing",
    "eat": "eating",
    "shake": "convulsing",
    "fall": "falling",
    "vomit": "vomiting",
}

ACTION_PRIORITY = {
    "fall": 7,
    "shake": 6,
    "vomit": 5,
    "eat": 4,
    "stand_up": 3,
    "sit": 2,
    "sleep": 1,
    "lying": 1,
}

EVENT_LABELS = {"falling", "convulsing", "getting_out_of_bed", "vomiting"}
CRITICAL_LABELS = {"falling", "convulsing", "getting_out_of_bed"}
ACTION_COLORS = {
    "standing": (255, 110, 0),
    "sitting": (0, 170, 255),
    "lying_down": (0, 220, 120),
    "getting_out_of_bed": (0, 0, 255),
    "eating": (180, 60, 255),
    "convulsing": (0, 0, 255),
    "falling": (0, 0, 255),
    "vomiting": (0, 60, 255),
    "no_patient": (160, 160, 160),
}


@dataclass(slots=True)
class Detection:
    raw_label: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(slots=True)
class ActivityDecision:
    label: str
    confidence: float
    box: tuple[int, int, int, int] | None
    raw_label: str | None
    inside_bed: bool


@dataclass(slots=True)
class EventRecord:
    timestamp: str
    frame_index: int
    event: str
    activity: str
    confidence: float
    inside_bed: bool
    source_label: str


class ThreadedVideoStream:
    def __init__(self, source: int | str, width: int, height: int, fps: int) -> None:
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.lock = threading.Lock()
        self.stopped = False
        self.grabbed, self.frame = self.capture.read()
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self) -> "ThreadedVideoStream":
        self.thread.start()
        return self

    def _reader(self) -> None:
        while not self.stopped:
            grabbed, frame = self.capture.read()
            with self.lock:
                self.grabbed = grabbed
                if grabbed:
                    self.frame = frame
            if not grabbed:
                self.stopped = True
                break

    def read(self) -> tuple[bool, object]:
        with self.lock:
            if not self.grabbed:
                return False, None
            return True, self.frame.copy()

    def stop(self) -> None:
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1)
        self.capture.release()


class PatientActivityRecognizer:
    def __init__(self, model: YOLO, config: MonitorConfig) -> None:
        self.model = model
        self.config = config
        if isinstance(model.names, dict):
            self.names = {int(key): value for key, value in model.names.items()}
        else:
            self.names = {index: value for index, value in enumerate(model.names)}
        self.history: deque[tuple[str, float]] = deque(maxlen=config.smoothing_window)
        self.last_state: str | None = None
        self.last_event_at: dict[str, float] = {}
        self.persistence: Counter[str] = Counter()
        self.exit_armed = False
        self.outside_bed_standing_frames = 0

    def predict(self, frame, frame_index: int) -> tuple[object, ActivityDecision, list[EventRecord]]:
        results = self.model.predict(
            source=frame,
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            device=self.config.device,
            verbose=False,
        )
        detections = self._extract_detections(results[0])
        decision = self._select_activity(detections, frame.shape)
        smoothed = self._smooth(decision)
        events = self._build_events(smoothed, frame_index)
        if any(event.event == "getting_out_of_bed" for event in events):
            smoothed = ActivityDecision(
                label="getting_out_of_bed",
                confidence=smoothed.confidence,
                box=smoothed.box,
                raw_label=smoothed.raw_label,
                inside_bed=smoothed.inside_bed,
            )
        annotated = self._annotate_frame(frame, detections, smoothed, events)
        return annotated, smoothed, events

    def _extract_detections(self, result) -> list[Detection]:
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            cls_index = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    raw_label=self.names.get(cls_index, str(cls_index)),
                    confidence=confidence,
                    box=(x1, y1, x2, y2),
                )
            )
        return detections

    def _select_activity(self, detections: Iterable[Detection], frame_shape: tuple[int, ...]) -> ActivityDecision:
        detections = list(detections)
        specific = [detection for detection in detections if detection.raw_label in RAW_TO_CANONICAL]
        if specific:
            chosen = max(
                specific,
                key=lambda detection: (ACTION_PRIORITY.get(detection.raw_label, 0), detection.confidence),
            )
            label = RAW_TO_CANONICAL[chosen.raw_label]
            inside_bed = self._inside_bed(chosen.box, frame_shape)
            return ActivityDecision(label, chosen.confidence, chosen.box, chosen.raw_label, inside_bed)

        patient_boxes = [detection for detection in detections if detection.raw_label == "patient"]
        if patient_boxes:
            chosen = max(patient_boxes, key=lambda detection: detection.confidence)
            label = self._infer_posture(chosen.box, frame_shape)
            inside_bed = self._inside_bed(chosen.box, frame_shape)
            return ActivityDecision(label, chosen.confidence * 0.6, chosen.box, chosen.raw_label, inside_bed)

        return ActivityDecision("no_patient", 0.0, None, None, False)

    def _infer_posture(self, box: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> str:
        if not self.config.fallback_posture_inference:
            return "standing"

        x1, y1, x2, y2 = box
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        aspect_ratio = width / height
        inside_bed = self._inside_bed(box, frame_shape)

        if inside_bed and aspect_ratio >= 1.05:
            return "lying_down"
        if inside_bed:
            return "sitting"
        return "standing"

    def _inside_bed(self, box: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> bool:
        if self.config.bed_zone is None:
            return False

        frame_height, frame_width = frame_shape[:2]
        zone = self._bed_zone_pixels(frame_width, frame_height)
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return zone[0] <= center_x <= zone[2] and zone[1] <= center_y <= zone[3]

    def _bed_zone_pixels(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        assert self.config.bed_zone is not None
        x1, y1, x2, y2 = self.config.bed_zone
        return (
            int(x1 * frame_width),
            int(y1 * frame_height),
            int(x2 * frame_width),
            int(y2 * frame_height),
        )

    def _smooth(self, decision: ActivityDecision) -> ActivityDecision:
        self.history.append((decision.label, decision.confidence))
        scores: Counter[str] = Counter()
        for label, confidence in self.history:
            scores[label] += confidence if confidence > 0 else 0.01

        smoothed_label, smoothed_score = scores.most_common(1)[0]
        if smoothed_label == decision.label:
            return decision
        return ActivityDecision(smoothed_label, smoothed_score / len(self.history), decision.box, decision.raw_label, decision.inside_bed)

    def _build_events(self, decision: ActivityDecision, frame_index: int) -> list[EventRecord]:
        now = time.time()
        events: list[EventRecord] = []
        timestamp = datetime.now().isoformat(timespec="seconds")

        if decision.label != self.last_state:
            events.append(
                EventRecord(
                    timestamp=timestamp,
                    frame_index=frame_index,
                    event="activity_changed",
                    activity=decision.label,
                    confidence=decision.confidence,
                    inside_bed=decision.inside_bed,
                    source_label=decision.raw_label or "none",
                )
            )
            self.last_state = decision.label

        for label in list(self.persistence):
            if label != decision.label:
                self.persistence[label] = 0

        if decision.label in {"falling", "convulsing", "vomiting"}:
            self.persistence[decision.label] += 1
            if self.persistence[decision.label] >= self.config.critical_confirmation_frames:
                event = self._maybe_create_event(
                    timestamp=timestamp,
                    frame_index=frame_index,
                    event=decision.label,
                    decision=decision,
                    now=now,
                )
                if event is not None:
                    events.append(event)

        if decision.label in {"lying_down", "sitting"} and decision.inside_bed:
            self.exit_armed = True
            self.outside_bed_standing_frames = 0
        elif self.exit_armed and decision.label == "standing":
            if decision.inside_bed:
                self.outside_bed_standing_frames = 0
            else:
                self.outside_bed_standing_frames += 1
                if self.outside_bed_standing_frames >= self.config.bed_exit_confirmation_frames:
                    event = self._maybe_create_event(
                        timestamp=timestamp,
                        frame_index=frame_index,
                        event="getting_out_of_bed",
                        decision=decision,
                        now=now,
                    )
                    if event is not None:
                        events.append(event)
                    self.exit_armed = False
                    self.outside_bed_standing_frames = 0

        if decision.label in {"no_patient", "falling"}:
            self.outside_bed_standing_frames = 0
        return events

    def _maybe_create_event(
        self,
        timestamp: str,
        frame_index: int,
        event: str,
        decision: ActivityDecision,
        now: float,
    ) -> EventRecord | None:
        last_seen = self.last_event_at.get(event, 0.0)
        if now - last_seen < self.config.event_cooldown_seconds:
            return None

        self.last_event_at[event] = now
        return EventRecord(
            timestamp=timestamp,
            frame_index=frame_index,
            event=event,
            activity=decision.label,
            confidence=decision.confidence,
            inside_bed=decision.inside_bed,
            source_label=decision.raw_label or "none",
        )

    def _annotate_frame(
        self,
        frame,
        detections: Iterable[Detection],
        decision: ActivityDecision,
        events: Iterable[EventRecord],
    ):
        annotated = frame.copy()

        if self.config.bed_zone is not None:
            frame_height, frame_width = annotated.shape[:2]
            x1, y1, x2, y2 = self._bed_zone_pixels(frame_width, frame_height)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                annotated,
                "bed zone",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        for detection in detections:
            mapped_label = RAW_TO_CANONICAL.get(detection.raw_label)
            color = ACTION_COLORS.get(mapped_label, (255, 255, 255))
            x1, y1, x2, y2 = detection.box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{detection.raw_label} {detection.confidence:.2f}"
            cv2.putText(annotated, text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        banner_color = ACTION_COLORS.get(decision.label, (255, 255, 255))
        cv2.rectangle(annotated, (10, 10), (420, 110), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            f"activity: {decision.label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            banner_color,
            2,
        )
        cv2.putText(
            annotated,
            f"confidence: {decision.confidence:.2f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"in bed: {'yes' if decision.inside_bed else 'no'}",
            (20, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        active_events = [event.event for event in events if event.event in EVENT_LABELS]
        if active_events:
            cv2.putText(
                annotated,
                "alert: " + ", ".join(active_events),
                (20, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return annotated


def ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def open_event_writer(path: str | Path):
    csv_path = ensure_parent(path)
    file_exists = csv_path.exists()
    handle = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    if not file_exists:
        writer.writerow(["timestamp", "frame_index", "event", "activity", "confidence", "inside_bed", "source_label"])
        handle.flush()
    return handle, writer


def save_alert_frame(frame, decision: ActivityDecision, events: Iterable[EventRecord], directory: str | Path) -> None:
    critical_events = [event.event for event in events if event.event in CRITICAL_LABELS]
    if not critical_events:
        return

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"{timestamp}_{decision.label}.jpg"
    cv2.imwrite(str(filename), frame)


def coerce_overrides(config: MonitorConfig, args: argparse.Namespace) -> MonitorConfig:
    if args.model_path is not None:
        config.model_path = args.model_path
    if args.source is not None:
        config.source = parse_source(args.source)
    if args.imgsz is not None:
        config.imgsz = args.imgsz
    if args.conf is not None:
        config.conf = args.conf
    if args.iou is not None:
        config.iou = args.iou
    if args.device is not None:
        config.device = args.device
    if args.display is not None:
        config.display = parse_bool(args.display)
    if args.frame_skip is not None:
        config.frame_skip = max(args.frame_skip, 1)
    if args.bed_zone is not None:
        config.bed_zone = parse_bed_zone(args.bed_zone)
    if args.event_log is not None:
        config.event_log = args.event_log
    if args.record_output is not None:
        config.record_output = args.record_output
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time bedside patient activity recognition.")
    parser.add_argument("--config", default="monitor_config.yaml", help="Path to monitor configuration YAML.")
    parser.add_argument("--model-path", help="YOLO model path.")
    parser.add_argument("--source", help="Webcam index or video source path.")
    parser.add_argument("--imgsz", type=int, help="Inference image size.")
    parser.add_argument("--conf", type=float, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, help="NMS IoU threshold.")
    parser.add_argument("--device", help="Inference device, for example cpu.")
    parser.add_argument("--display", help="Set true to show the OpenCV window, false for headless mode.")
    parser.add_argument("--frame-skip", type=int, help="Process every Nth frame.")
    parser.add_argument("--bed-zone", help="Normalized bed zone: x1,y1,x2,y2")
    parser.add_argument("--event-log", help="CSV path for activity and alert events.")
    parser.add_argument("--record-output", help="Optional output video path.")
    return parser


def run_monitor(config: MonitorConfig) -> int:
    model_path = Path(config.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    recognizer = PatientActivityRecognizer(model, config)
    stream = ThreadedVideoStream(config.source, config.camera_width, config.camera_height, config.camera_fps).start()
    event_handle, event_writer = open_event_writer(config.event_log)
    writer = None
    last_processed = None
    frame_index = 0
    start_time = time.time()

    try:
        while True:
            grabbed, frame = stream.read()
            if not grabbed or frame is None:
                break

            frame_index += 1
            if frame_index % config.frame_skip == 0:
                annotated, decision, events = recognizer.predict(frame, frame_index)
                fps = frame_index / max(time.time() - start_time, 0.001)
                cv2.putText(
                    annotated,
                    f"fps: {fps:.1f}",
                    (annotated.shape[1] - 120, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                last_processed = annotated

                for event in events:
                    event_writer.writerow(
                        [
                            event.timestamp,
                            event.frame_index,
                            event.event,
                            event.activity,
                            f"{event.confidence:.4f}",
                            event.inside_bed,
                            event.source_label,
                        ]
                    )
                event_handle.flush()

                if config.save_alert_frames:
                    save_alert_frame(annotated, decision, events, config.alert_frame_dir)

                if config.record_output:
                    if writer is None:
                        output_path = ensure_parent(config.record_output)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            str(output_path),
                            fourcc,
                            config.camera_fps,
                            (annotated.shape[1], annotated.shape[0]),
                        )
                    writer.write(annotated)

                if config.display:
                    cv2.imshow("Bedside Activity Monitor", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            elif config.display and last_processed is not None:
                cv2.imshow("Bedside Activity Monitor", last_processed)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        event_handle.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    args = build_parser().parse_args()
    config = load_monitor_config(args.config)
    config = coerce_overrides(config, args)
    return run_monitor(config)
