cat > /jetson-inference/anomaly_detector_rules.py << 'PY'
#!/usr/bin/env python3
import os
import time
import csv
from datetime import datetime

import jetson.inference
import jetson.utils

# Configuration
MODEL_NAME = "ssd-mobilenet-v2"
THRESHOLD = 0.5
CAMERA_URI = "v4l2:///dev/video0"   # use /dev/video1 if needed
CAMERA_ARGS = ["--input-width=640", "--input-height=480", "--input-rate=10"]
DISPLAY_URI = "display://0"         # "display://0" for connected screen, "none" for headless

LOG_CSV = "/jetson-inference/data/anomaly_log.csv"
IMG_DIR = "/jetson-inference/data/anomaly_images"
os.makedirs(IMG_DIR, exist_ok=True)

# Anomaly rules (modify these to match your chosen "normal" scene)
MAX_PERSONS = 3               # anomaly if person_count > MAX_PERSONS
REQUIRED_OBJECTS = []         # e.g. ["helmet"] if helmet should be present when person is present
FORBIDDEN_OBJECTS = {"cell phone", "cellphone"}  # any of these triggers anomaly

# Cooldown protects from duplicate logging of the same continuous anomaly
COOLDOWN_SECONDS = 3.0
last_anomaly_time = 0.0
anomaly_count = 0

# Initialize model, camera, display
print("Loading model:", MODEL_NAME)
net = jetson.inference.detectNet(MODEL_NAME, threshold=THRESHOLD)

print("Opening camera:", CAMERA_URI)
camera = jetson.utils.videoSource(CAMERA_URI, argv=CAMERA_ARGS)
display = jetson.utils.videoOutput(DISPLAY_URI)

# Initialize CSV log if not present
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "anomaly_reason", "detected_labels", "counts", "image_path"])

print("Starting anomaly detection loop")
try:
    while display.IsStreaming():
        img = camera.Capture()
        if img is None:
            # no frame captured; small pause to avoid busy-loop
            time.sleep(0.05)
            continue

        # Run detection (overlay draws boxes, labels, confidence)
        detections = net.Detect(img, overlay="box,labels,conf")

        # Collect labels and counts
        labels = []
        counts = {}
        for d in detections:
            label = net.GetClassDesc(d.ClassID).lower()
            labels.append(label)
            counts[label] = counts.get(label, 0) + 1

        # Evaluate anomaly rules
        anomaly_reasons = []

        # Count-based rule: too many people
        person_count = counts.get("person", 0)
        if person_count > MAX_PERSONS:
            anomaly_reasons.append(f"person_count_exceeded:{person_count}>{MAX_PERSONS}")

        # Missing required objects (if person present)
        if person_count > 0:
            for req in REQUIRED_OBJECTS:
                if counts.get(req, 0) == 0:
                    anomaly_reasons.append(f"missing_required:{req}")

        # Forbidden object presence
        for forb in FORBIDDEN_OBJECTS:
            if forb in labels:
                anomaly_reasons.append(f"forbidden_present:{forb}")

        # If any anomaly reason found and cooldown passed => log + save image
        now = time.time()
        if anomaly_reasons and (now - last_anomaly_time) >= COOLDOWN_SECONDS:
            last_anomaly_time = now
            anomaly_count += 1
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            reason_str = ";".join(anomaly_reasons)
            counts_str = ",".join([f"{k}:{v}" for k, v in counts.items()])

            # Save annotated image
            img_name = f"anomaly_{anomaly_count}_{int(now)}.jpg"
            img_path = os.path.join(IMG_DIR, img_name)
            try:
                jetson.utils.saveImage(img_path, img)
            except Exception:
                img_path = ""

            # Append to CSV
            with open(LOG_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ts, reason_str, "|".join(labels), counts_str, img_path])

            print(f"[{ts}] Anomaly #{anomaly_count}: {reason_str}  counts={counts_str}  saved={img_path}")

        # Render frame and status
        display.Render(img)
        display.SetStatus(f"Anomalies: {anomaly_count}  Last: {datetime.fromtimestamp(last_anomaly_time).strftime('%Y-%m-%d %H:%M:%S') if last_anomaly_time>0 else 'None'}")

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    try:
        camera.Close()
    except:
        pass
    try:
        display.Close()
    except:
        pass
    print("Exited cleanly")
PY
