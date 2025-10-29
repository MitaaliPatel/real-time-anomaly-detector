Project: Real-Time Anomaly Detection using Object Detector and USB Camera

Overview:

This project implements a real-time anomaly detection system using NVIDIA Jetson and the jetson-inference library. The goal is to detect unusual or unsafe situations in live camera feeds using object detection. The system flags anomalies based on object presence, count, or absence.

Objective:

This project uses NVIDIA Jetson Inference’s detectNet model to build a real-time anomaly detection system. The system identifies anomalies based on object presence, count, or absence in the live camera feed.

Normal Scene Definition:

For this implementation, a normal scene is defined as:
At most 2 people are visible.

No cell phone visible in the frame.

No missing safety equipment (example: person should have helmet).

Anomaly Rules Implemented:

The following rules were implemented in the code:
Count-based anomaly:
 If more than 2 people are detected in a frame, it flags an anomaly.

Forbidden object anomaly:
 If a “cell phone” is detected in the frame, it flags an anomaly.

Missing object anomaly:
 If a person is detected but a “helmet” is not, it flags an anomaly.

Each anomaly is logged with a timestamp and type in a CSV file.

Real-Time Detection:

The system runs using:
python3 /jetson-inference/anomaly_detector.py

Each camera frame is passed through detectNet to detect objects. The detections are then checked against the anomaly rules. When a rule is violated:
A red border appears around the detected object.

The event is logged in anomaly_log.csv.

The anomalous frame is saved in anomaly_images/.

Output Artifacts:

anomaly_log.csv — contains timestamps and reasons for anomalies.

anomaly_images/ — contains frames saved when anomalies occurred.

Example anomaly_log.csv content:
2025-10-27 02:14:32,forbidden_present,cell phone
2025-10-27 02:15:10,count_exceeded,person:4

Observations and Improvements:

The system accurately detects object counts and forbidden objects.

Occasional false positives occur in low-light or crowded scenes.

Could be improved with:

Confidence threshold tuning

Adding cooldown logic to prevent duplicate anomaly logs

Using a trained helmet detection model for better accuracy

Tools and Frameworks Used:

Hardware: NVIDIA Jetson board with USB camera

Software: Jetson Inference (DetectNet), GStreamer, Python

Model: SSD-Mobilenet-v2 trained on COCO dataset

How to Run:

Start Docker Jetson Inference container:

cd ~/jetson-inference
sudo ./docker/run.sh
Run the anomaly detection script:

 python3 /jetson-inference/anomaly_detector.py
View real-time detection and anomaly logs in /jetson-inference/data/.



