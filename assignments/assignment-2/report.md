# Image-Based Semantic Search Over Video
**Assignment 2 — Car Parts Detection and Retrieval**

---

## Introduction

This report describes a system that performs image-based semantic search over a video. Given a query image of a car exterior component, the system identifies which component(s) appear in the image, searches a pre-built video detection index, and returns the timestamps of every contiguous segment in the video where that component is visible.

The pipeline operates in four stages: (1) **detect** — run an object detector on sampled video frames to produce class-labeled bounding boxes; (2) **index** — store all detections in a Parquet file keyed by timestamp; (3) **match** — run the same detector on a query image to obtain its component class labels, then find matching entries in the index; (4) **retrieve** — merge nearby matching timestamps into contiguous intervals and return them with verification URLs. The system operates entirely through detected semantic structure — no manual labeling, hardcoded timestamps, or query-specific heuristics are used.

---

## Section 1: Detector Choice and Configuration

### Model Selection

The detector used in this system is **YOLOv8n-seg** (Ultralytics), a nano-scale instance segmentation model from the YOLOv8 family. The base COCO-pretrained model was fine-tuned on the **Ultralytics Car Parts Segmentation dataset** (`carparts-seg.yaml`), which provides 23 labeled classes of automobile exterior components:

> `back_bumper`, `front_bumper`, `hood`, `wheel`, `front_left_door`, `front_right_door`, `back_left_door`, `back_right_door`, `front_glass`, `back_glass`, `front_left_light`, `front_right_light`, `back_left_light`, `back_right_light`, `left_mirror`, `right_mirror`, `trunk`, `tailgate`, and others.

The generic COCO-pretrained YOLOv8 model was rejected because its class vocabulary (person, car, truck, etc.) is too coarse for this task — it cannot distinguish between a wheel, a bumper, and a hood. Fine-tuning on the car parts dataset was necessary to produce semantically meaningful detections aligned with the query images.

The nano variant (`yolov8n`) was chosen over larger variants (small, medium, large) because it runs on CPU without crashing, 69% mAP50 is sufficient for class-label-based retrieval, and lower inference time matters when processing every frame of a video and every image in a query dataset.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `yolov8n-seg.pt` (COCO pretrained) |
| Dataset | `carparts-seg.yaml` (3,156 train / 401 val images) |
| Epochs | 50 |
| Image size | 640 × 640 |
| Hardware | Google Colab T4 GPU (16GB VRAM) |
| Training time | ~1.13 hours |
| Best mAP50 (box) | 69.3% |
| Best mAP50 (mask) | 71.0% |

### Confidence Threshold

A confidence threshold of **0.5** was applied at inference time. Detections below 50% confidence are discarded to reduce false positives from partial occlusions and motion blur in video frames.

### Preprocessing

No custom preprocessing was applied beyond what YOLOv8 performs internally. The model resizes input images to 640×640 and normalizes pixel values before inference. Video frames were fed as JPEG images directly from disk.

### Sample Detection

A sample detection from the system shows multiple car parts — including `wheel`, `front_bumper`, and `front_glass` — detected simultaneously in a single frame with bounding boxes. This confirms the fine-tuned model correctly distinguishes between distinct exterior components rather than labeling the entire vehicle as a single class.

---

## Section 2: Video Sampling Strategy

### Clip and Frame Rate Selection

The input video is a Toyota RAV4 2026 exterior review ([YouTube: YcvECxtXoxQ](https://www.youtube.com/watch?v=YcvECxtXoxQ)). Rather than processing the full video (~10 minutes, 4.4GB), a **45-second clip from 2:00 to 2:45** (timestamps 120–164 seconds) was selected. This segment contains a systematic exterior walkaround showing the front, sides, rear, and roof of the vehicle — maximizing component variety within a manageable processing window.

Frames were extracted at **1 frame per second** using ffmpeg:

```bash
ffmpeg -ss 00:02:00 -i input_video.mp4 -t 45 -c copy clip_120_165.mp4
ffmpeg -i clip_120_165.mp4 -vf "fps=1" frames/frame_%04d.jpg
```

This produced **45 frames**. A 1fps rate balances coverage against redundancy — consecutive video frames are nearly identical, so sub-second sampling would produce duplicate detections without meaningfully increasing recall.

### Detection Summary

Running the trained model over all 45 frames produced **92 detections** across 20 unique car part classes. Each frame was assigned a `timestamp_sec` value of `120 + (frame_index − 1)`, mapping each detection to its absolute position in the original video. No frames were filtered post-detection; confidence filtering (threshold 0.5) was applied during inference.

### Parquet Schema

All detections are stored in `video_detections.parquet`. Each row corresponds to a single detection in a single frame:

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | string | YouTube video ID (`YcvECxtXoxQ`) |
| `frame_idx` | int | Frame number within the extracted clip |
| `timestamp_sec` | int | Absolute time position in seconds (120–164) |
| `class_label` | string | Detected car part class name |
| `confidence_score` | float | Detector confidence (0–1) |
| `x_min` | float | Bounding box left edge (pixels) |
| `y_min` | float | Bounding box top edge (pixels) |
| `x_max` | float | Bounding box right edge (pixels) |
| `y_max` | float | Bounding box bottom edge (pixels) |

---

## Section 3: Image-to-Video Matching Logic

### Query Image Processing

For each query image from the `aegean-ai/rav4-exterior-images` HuggingFace dataset, the same `best.pt` detector is run with a confidence threshold of 0.5. The set of unique detected class labels becomes the **query class set**:

```python
def get_query_classes(image_path, confidence_threshold=0.5):
    results = model(image_path, verbose=False)[0]
    return list(set([
        model.names[int(box.cls)]
        for box in results.boxes
        if float(box.conf) >= confidence_threshold
    ]))
```

Using the same model for both video indexing and query detection ensures the class label vocabulary is consistent — a `wheel` detected in the video and a `wheel` detected in a query image use identical class name strings.

### Label Matching

The detection index is filtered to rows where `class_label` is in the query class set:

```python
matches = df[df["class_label"].isin(query_classes)]
```

This is a simple string equality match. No embedding similarity or visual comparison is performed — matching is purely semantic via class label overlap.

### Interval Construction

Matching timestamps are sorted and merged into contiguous `(start, end)` intervals using a gap threshold of **10 seconds**. Timestamps within 10 seconds of the previous one are merged into the same interval; otherwise a new interval begins. A 10-second gap threshold was chosen because component detections can be intermittent — a wheel may not be detected in every frame even when consistently visible, due to motion blur or partial occlusion. Gaps under 10 seconds are treated as detection misses rather than genuine absence.

### Concrete Example

A query image from `aegean-ai/rav4-exterior-images` (timestamp `00:00`) shows the front-right side of a RAV4. The detector identifies: `front_right_door`, `wheel`, `front_glass`, `front_left_door`, `right_mirror`. Matching these against the video index produces two intervals:

| Field | Segment 1 | Segment 2 |
|-------|-----------|-----------|
| `start_timestamp` | 120 | 131 |
| `end_timestamp` | 120 | 165 |
| `number_of_supporting_detections` | 2 | 21 |
| `verify_url` | `...?start=120&end=120` | `...?start=131&end=165` |

Opening `https://www.youtube.com/embed/YcvECxtXoxQ?start=131&end=165` plays the 34-second matched segment, confirming the front-right side of the RAV4 is visible throughout. The complete retrieval run over all query images produced **72 retrieval results**.

---

## Section 4: Failure Cases and Limitations

### Missed Detections and Fragmented Clips

At 1fps, a component appearing briefly or during a fast camera pan may be detected in only 1 frame. This produces single-timestamp intervals (e.g., `start=120, end=120`) with very few supporting detections. These are technically valid but carry low confidence. The `number_of_supporting_detections` field can be used to filter such weak matches.

### Label Mismatch Across Viewpoints

If a query image shows a component at a very different scale or angle than what appears in the video, the detector may assign different class labels to each. For example, a close-up of a headlight unit might be labeled `front_left_light` in the query but undetected in a wide-angle video frame where only `front_bumper` is confidently returned. This causes false negatives in retrieval.

### Coarse Sampling Window

The system indexes only a single 45-second clip. Components visible outside this window — such as close-up angles or the beginning/end of the walkaround — are never indexed, and queries for those angles return no results. Extending the indexed clip to the full video duration would significantly improve recall.

### Class Ambiguity

Some car parts share visual and semantic overlap. `front_left_door` and `front_right_door` are symmetric and can be confused under camera mirroring. `tailgate` and `trunk` cover similar areas depending on vehicle type. These ambiguities produce noisy retrieval where incorrect component labels are matched.

### No Visual Similarity

The matching logic is class-label-only. Two images match even if their visual appearance differs substantially — a scratched bumper query matches any frame containing any front bumper detection, regardless of damage, lighting, or color. A more robust system would compute visual embeddings (e.g., CLIP) and score matches by cosine similarity in addition to class overlap.

---

## Conclusion

This assignment produced a working image-to-video semantic retrieval system built entirely on detected structure. A YOLOv8n-seg model fine-tuned on 23 car part classes was used to index 45 video frames and process a HuggingFace query dataset, producing 72 retrieval results with timestamped, verifiable video segments. The system is reproducible — all detections are stored in a publicly available Parquet file on HuggingFace ([nazzzz5265/rav4-video-detection](https://huggingface.co/datasets/nazzzz5265/rav4-video-detection)), and the retrieval logic operates solely through that index.

With more time, the main improvements would be: (1) denser frame sampling (2–5fps) to reduce missed detections, (2) extending the video index beyond a single 45-second clip to cover the full walkaround, and (3) incorporating embedding-based visual similarity scoring alongside class label matching to handle viewpoint variation more robustly.
