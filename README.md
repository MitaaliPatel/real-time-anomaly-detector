# Real-Time Anomaly Detection using Object Detection & USB Camera

A real-time workplace safety monitoring system built on **NVIDIA Jetson** using the Jetson Inference library. The system detects unsafe or unusual situations in live camera feeds by applying rule-based anomaly logic on top of object detection — flagging violations, logging events with timestamps, and saving anomalous frames for review.

---

## Overview

Each frame from a USB camera is passed through **DetectNet** (SSD-MobileNet-v2, trained on COCO) to identify objects in the scene. The detections are then evaluated against a set of configurable safety rules. When a rule is violated, the system draws a red border around the offending object, logs the event to a CSV file, and saves the frame.

---

## Anomaly Rules

A scene is considered **normal** when:
- At most 2 people are visible
- No cell phone is present in the frame
- A helmet is detected on any person present

The following anomaly types are implemented:

| Type | Trigger |
|---|---|
| **Count-based** | More than 2 people detected in a single frame |
| **Forbidden object** | A cell phone is detected anywhere in the frame |
| **Missing equipment** | A person is present but no helmet is detected |

Each violation is logged with a timestamp and anomaly type to `anomaly_log.csv`, and the frame is saved to `anomaly_images/`.

---

## Output Artifacts

| File / Folder | Contents |
|---|---|
| `anomaly_log.csv` | Timestamped log of all anomaly events and their types |
| `anomaly_images/` | Saved frames captured at the moment of each violation |

**Example log entries:**
```csv
timestamp,anomaly_type,detail
2025-10-27 02:14:32,forbidden_present,cell phone
2025-10-27 02:15:10,count_exceeded,person:4
2025-10-27 02:17:45,missing_equipment,helmet not found
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Hardware | NVIDIA Jetson Nano / Xavier / Orin |
| Camera | USB camera via `/dev/video0` |
| Inference | Jetson Inference — DetectNet |
| Model | SSD-MobileNet-v2 (COCO) |
| Streaming | GStreamer |
| Language | Python 3 |

---

## How to Run

Make sure your Jetson is connected to a USB camera, then run:

```bash
python3 /jetson-inference/anomaly_detector.py
```

**What you'll see:**
- Live camera window with bounding boxes and red borders on violations
- Terminal output when an anomaly is detected
- Events logged automatically to `anomaly_log.csv`

---

## Observations

- Object count detection and forbidden object flagging work reliably under normal lighting
- Occasional false positives occur in low-light or heavily crowded scenes

---

## Future Work

- [ ] Confidence threshold tuning to reduce false positives in low-light conditions
- [ ] Cooldown logic to prevent duplicate entries for the same ongoing violation
- [ ] Replace generic COCO helmet class with a fine-tuned helmet detection model for higher accuracy
- [ ] Add a lightweight dashboard to visualize anomaly frequency over time

---

## Requirements

- NVIDIA Jetson device (Nano / Xavier / Orin)
- JetPack SDK with:
  - `jetson-inference`
  - `jetson-utils`
- Python 3
- USB camera
