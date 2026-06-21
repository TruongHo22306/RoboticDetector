# Bedside Patient Activity Recognition

This project turns the trained YOLOv8 model in this folder into a real-time bedside activity monitor that can run from a USB webcam on a Raspberry Pi 4 or a normal PC.

## What the system does

- Detects a patient in each frame with YOLOv8.
- Maps trained labels into bedside activities:
  - `sleep`, `lying` -> `lying_down`
  - `sit` -> `sitting`
  - `stand_up` -> `standing`
  - `eat` -> `eating`
  - `shake` -> `convulsing`
  - `fall` -> `falling`
- Uses temporal smoothing so the displayed activity is more stable than raw frame-by-frame detections.
- Uses a configurable bed zone to infer `getting_out_of_bed` when the patient changes from `lying_down` or `sitting` in bed to `standing` outside the bed area.
- Logs activity changes and alert events to CSV.
- Can save alert snapshots for critical events such as convulsing, falling, and bed exit.

## Current model limitation

The current dataset already supports most of the requested actions, but `getting_out_of_bed` is not a direct trained class in the model. In this codebase it is inferred from a sequence of detections and the patient position relative to the bed zone. For a thesis-grade deployment, you should still collect more real bedside video and retrain with more samples for:

- `getting_out_of_bed`
- `convulsing`
- `falling`
- multi-angle standing and sitting transitions

The existing checkpoint in [runs/detect/train-4/weights/best.pt](/D:/Thesis/runs/detect/train-4/weights/best.pt) is not accurate enough yet for real deployment. Its final validation metrics in [runs/detect/train-4/results.csv](/D:/Thesis/runs/detect/train-4/results.csv) are approximately:

- `mAP50 = 0.01198`
- `mAP50-95 = 0.00344`

That means the software pipeline is now ready, but the model should be retrained with a better bedside dataset before you treat the output as reliable.

## Install

```bash
pip install -r requirements.txt
```

On Raspberry Pi OS, `opencv-python` can be replaced with the system package if needed:

```bash
sudo apt install python3-opencv
```

## Run the real-time monitor

```bash
python main.py
```

Useful options:

```bash
python main.py --source 0 --imgsz 416 --conf 0.35 --device cpu
python main.py --display false
python main.py --bed-zone 0.18,0.16,0.82,0.95
python main.py --record-output recordings/session.mp4
```

The monitor reads defaults from [monitor_config.yaml](/D:/Thesis/monitor_config.yaml).

## Prepare the dataset before retraining

The current raw dataset has two structural problems:

- `train` and `val` were duplicated.
- `patient` boxes overlap the action labels on the same person.

This project now includes a preparation step that creates a cleaner action-only dataset in [data_refined](/D:/Thesis/data_refined), writes a new config to [config_refined.yaml](/D:/Thesis/config_refined.yaml), and saves an audit report to [reports/dataset_prepare_report.json](/D:/Thesis/reports/dataset_prepare_report.json).

Run:

```bash
python prepare_dataset.py
```

## Train or fine-tune again

```bash
python train.py --prepare-dataset --epochs 100 --imgsz 640
```

By default, the training script now uses [config_refined.yaml](/D:/Thesis/config_refined.yaml) and starts from `yolov8n.pt`. If you want to continue the current local checkpoint instead:

```bash
python train.py --prepare-dataset --model runs/detect/train-4/weights/best.pt
```

The recommended workflow for the upgraded project is:

1. Add more raw images and labels into the source dataset.
2. Run `python prepare_dataset.py`.
3. Inspect [reports/dataset_prepare_report.json](/D:/Thesis/reports/dataset_prepare_report.json).
4. Retrain with `python train.py --prepare-dataset`.
5. Test the new weights in the real-time monitor.

## Raspberry Pi 4 recommendations

- Use `imgsz=320` or `416`.
- Keep the camera at `640x480`.
- Run on `device=cpu`.
- Disable the preview window with `--display false` for better throughput.
- Export a smaller model later if you need more speed.

## Output files

- Activity log: [logs/activity_events.csv](/D:/Thesis/logs/activity_events.csv)
- Alert frames: [alerts](/D:/Thesis/alerts)

## Thesis note

This code now satisfies the software structure for a Python-based real-time webcam monitoring system, but the final scientific quality still depends on better bedside data coverage and validation on real patient-like scenarios.
>>>>>>> 6f909c2 (Initial commit)
