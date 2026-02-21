# Smart Vision: LiDAR + IMU + Camera Fusion

A standalone system that fuses **TF-Mini Plus** LiDAR, **BNO085** IMU, and MacBook camera with YOLOv8 object detection to determine what you are looking at. Outputs results via display overlay and TTS (text-to-speech) audio.

---

## Overview

- **LiDAR** — Measures distance to objects (0–12 m)
- **IMU** — Provides orientation (yaw, pitch, roll) for direction of look
- **Camera** — Object detection (YOLOv8) to identify what’s in view
- **Fusion** — Combines sensors to announce: *"Looking at person at 2 meters"*

---

## Hardware

| Component | Description |
|-----------|-------------|
| Arduino Mega 2560 | Bridge between sensors and laptop |
| TF-Mini Plus | LiDAR distance sensor |
| BNO085 | 9-axis IMU (orientation) |
| MacBook | Built-in FaceTime camera |

---

## Wiring

| Component | Pin | Arduino Mega Pin |
|-----------|-----|------------------|
| TF-Mini Plus | Red (5V) | 5V |
| TF-Mini Plus | Black (GND) | GND |
| TF-Mini Plus | White (TX) | RX1 (pin 19) |
| TF-Mini Plus | Green/Blue (RX) | TX1 (pin 18) |
| BNO085 | VIN (3.3V) | 3.3V |
| BNO085 | GND | GND |
| BNO085 | SDA | Pin 20 |
| BNO085 | SCL | Pin 21 |

LiDAR and IMU connect to the Arduino; the Arduino is connected to the laptop via USB.

---

## Prerequisites

- **Arduino IDE** — [arduino.cc/software](https://www.arduino.cc/en/software)
- **Python 3.8+**
- **macOS** (for built-in camera)

---

## Setup

### 1. Arduino

1. Install libraries via **Sketch → Include Library → Manage Libraries**:
   - **TFMPlus** (by Bud Ryerson)
   - **Adafruit BNO08x**

2. Open `arduino/smart_vision_sensors/smart_vision_sensors.ino` in Arduino IDE.

3. Connect the Arduino Mega via USB.

4. Select **Tools → Board → Arduino Mega 2560**.

5. Select **Tools → Port** → your Arduino port (e.g. `/dev/cu.usbmodem...`).

6. Click **Upload** (→).

### 2. Python (virtual environment recommended)

```bash
cd python
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

### Test connections

Verify camera, Arduino, LiDAR, and IMU before running the main app:

```bash
cd python
source venv/bin/activate
python test_connections.py
```

### Run Smart Vision

```bash
cd python
source venv/bin/activate
python smart_vision.py
```

- Press **q** to quit.
- Display: camera feed, bounding boxes, distance, orientation, and fused result.
- TTS announces “Looking at [object] at [distance]” when you look at a detected object (with throttling).

---

## Project Structure

```
Smart Vision/
├── arduino/
│   └── smart_vision_sensors/
│       └── smart_vision_sensors.ino
├── python/
│   ├── smart_vision.py       # Main script
│   ├── test_connections.py   # Connection test
│   ├── sensors.py            # Serial reader (LiDAR + IMU)
│   ├── detector.py           # Camera + YOLOv8 object detection
│   ├── fusion.py             # Fusion logic
│   └── requirements.txt
└── README.md
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Arduino port not found | Check USB cable and port in Arduino IDE |
| LiDAR/IMU test fails | Confirm wiring and that firmware is uploaded |
| Camera not opening | Grant camera permissions to Terminal/IDE |
| YOLOv8 slow | First run downloads the model; later runs are faster |
