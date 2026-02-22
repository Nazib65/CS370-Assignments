# RAV4 Video Detections

Object detection index and image-to-video retrieval results for the Toyota RAV4 exterior video ([YcvECxtXoxQ](https://www.youtube.com/watch?v=YcvECxtXoxQ)).

## Files

| File | Description |
|------|-------------|
| `video_detections.parquet` | All car-part detections from the video corpus (92 rows) |
| `retrieval_results.parquet` | Retrieval results matching query images to video segments (72 rows) |

## Schema: `video_detections.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | string | YouTube video ID (`YcvECxtXoxQ`) |
| `frame_idx` | int | Sampled frame number |
| `timestamp_sec` | int | Time position in seconds (120–164, clip from 2:00–2:45) |
| `class_label` | string | Detected car part class (e.g. `wheel`, `hood`, `front_bumper`) |
| `confidence_score` | float | Detector confidence (0–1) |
| `x_min` | float | Bounding box left edge (pixels) |
| `y_min` | float | Bounding box top edge (pixels) |
| `x_max` | float | Bounding box right edge (pixels) |
| `y_max` | float | Bounding box bottom edge (pixels) |

## Schema: `retrieval_results.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `query_index` | int | Index of the query image in `aegean-ai/rav4-exterior-images` |
| `query_timestamp` | string | Timestamp of the query image (e.g. `00:05`) |
| `start_timestamp` | int | Start of matched video segment (seconds) |
| `end_timestamp` | int | End of matched video segment (seconds) |
| `class_label` | string | Car part classes shared between query and video |
| `number_of_supporting_detections` | int | Number of video frame detections supporting the match |
| `verify_url` | string | YouTube embed URL to verify the matched segment |

## Detector

- **Model**: YOLOv8n-seg (Ultralytics), fine-tuned on `carparts-seg` dataset
- **Training**: 50 epochs, 640px, Google Colab T4 GPU — mAP50: 69.3% (box), 71.0% (mask)
- **Car part classes**: back_bumper, front_bumper, hood, wheel, front_door, back_door, front_glass, back_glass, front_light, back_light, mirrors, trunk, tailgate, and others (23 total)
- **Confidence threshold**: 0.5
- **Video sampling**: 1 frame/second from a 45-second clip (2:00–2:45)

## Query Dataset

Query images sourced from [`aegean-ai/rav4-exterior-images`](https://huggingface.co/datasets/aegean-ai/rav4-exterior-images).
