# Assignment 3: UAV Drone Detection and Tracking

## Overview

End-to-end pipeline for detecting and tracking drones in video using a fine-tuned YOLOv8 object detector and a Kalman filter tracker.

---

## Dataset

**Source:** [`pathikg/drone-detection-dataset`](https://huggingface.co/datasets/pathikg/drone-detection-dataset) (HuggingFace)

- 3,000 training images, 500 validation images (subset of 54K total)
- Single class: `drone`
- Bounding box format: COCO (converted to YOLO format for training)
- License: MIT

The dataset contains images of drones captured from the ground — the drone itself is the detection target, not objects viewed from a drone.

---

## Detector Configuration

**Model:** YOLOv8n (nano) — pretrained on COCO, fine-tuned on drone dataset

**Fine-tuning settings:**

| Parameter | Value |
|---|---|
| Epochs | 30 |
| Image size | 640x640 |
| Batch size | 16 |
| Optimizer | AdamW (auto) |
| Learning rate | 0.002 |
| Early stopping patience | 10 |
| Device | CUDA (Tesla T4) |

**Final validation performance (best.pt):**

| Metric | Value |
|---|---|
| Precision | 0.870 |
| Recall | 0.714 |
| mAP50 | 0.816 |
| mAP50-95 | 0.392 |

**Inference settings:**

| Parameter | Value |
|---|---|
| Confidence threshold | 0.1 |
| IOU threshold (NMS) | 0.5 |
| Target class | `drone` |

---

## Kalman Filter State Design

The tracker uses a **constant velocity motion model** with a 4-dimensional state vector:

```
State: [x, y, vx, vy]
  x, y   — center of bounding box (pixels)
  vx, vy — velocity (pixels per frame)

Measurement: [x, y]
  Position only — bounding box center from detector
```

**State transition matrix (F):**

```
F = [[1, 0, dt, 0 ],
     [0, 1, 0,  dt],
     [0, 0, 1,  0 ],
     [0, 0, 0,  1 ]]
```

Encodes: new position = old position + velocity * dt

**Measurement matrix (H):**

```
H = [[1, 0, 0, 0],
     [0, 1, 0, 0]]
```

Only position is observed; velocity is inferred.

**Noise parameters:**

| Matrix | Parameter | Value | Reasoning |
|---|---|---|---|
| Q (process noise) | Position | 1.0 | Low — drone motion is relatively smooth |
| Q (process noise) | Velocity | 0.1 | Very low — velocity changes slowly |
| R (measurement noise) | Position | 10.0 | Moderate — detector bounding box center has some jitter |
| P (initial uncertainty) | Position | 100.0 | High — position uncertain at initialization |
| P (initial uncertainty) | Velocity | 10.0 | Moderate — velocity unknown at start |

---

## Failure Cases and Missed Detection Handling

**How missed detections are handled:**

When the detector fails to find a drone in a frame, the tracker calls `update_without_detection()`:
1. The Kalman filter runs the **predict step only** — the state estimate propagates forward using the motion model
2. A counter `frames_since_detection` is incremented
3. The predicted position is used to draw an orange bounding box on the output video (vs green for detected frames)
4. If `frames_since_detection > MAX_FRAMES_TO_SKIP` (10 frames), the tracker is marked inactive and tracking stops

**Known failure cases:**

| Case | Description | Observed In |
|---|---|---|
| Small/distant drone | Drone occupies very few pixels — below detection threshold | Video 2 (28% detection rate) |
| Motion blur | Fast-moving drone blurs across frames | Both videos |
| Background clutter | Sky gradients or birds triggering false negatives | Video 2 |
| Occlusion | Drone passes behind objects | Not observed |

**Detection rates:**

| Video | Frames | Detected | Rate |
|---|---|---|---|
| drone_video_1 | 828 | 684 | 82.6% |
| drone_video_2 | 2580 | 722 | 28.0% |

Video 2's low rate is primarily due to the drone being very small relative to frame size throughout most of the footage.

---

## Output Videos

[![drone_video_1_tracked](https://img.youtube.com/vi/RNi-ososllk/0.jpg)](https://youtu.be/RNi-ososllk)

[![drone_video_2_tracked](https://img.youtube.com/vi/YdHYPkkGWlg/0.jpg)](https://youtu.be/YdHYPkkGWlg)

---

## HuggingFace Dataset

Detection results packaged as Parquet: https://huggingface.co/datasets/nazzzz5265/drone_tracking

| Column | Type | Description |
|---|---|---|
| `video_name` | str | Source video identifier |
| `frame_number` | int | Frame index in the video |
| `image` | bytes | Raw detection frame image |
| `num_detections` | int | Number of drones detected in frame |
| `bbox_x1/y1/x2/y2` | int | Bounding box coordinates (pixels) |
| `center_x/y` | float | Bounding box center coordinates |
| `confidence` | float | Model detection confidence (0-1) |
| `class_name` | str | Detected class (`drone`) |

---

## Repository Structure

```
assignments-3/
├── drone_tracking.ipynb          # Main notebook
├── videos/                       # Downloaded test videos
├── frames/                       # Extracted frames (5 fps)
├── detections/                   # Per-frame detection images
├── output/                       # Tracked output videos
├── models/
│   ├── yolov8n.pt                # Base pretrained model
│   ├── drone_finetune_data/      # Training dataset (YOLO format)
│   └── drone_finetune/
│       └── weights/
│           └── best.pt           # Fine-tuned weights
└── data/
    └── drone_detections.parquet  # HuggingFace dataset
```
