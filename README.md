# Smart Vision: LiDAR + IMU + Camera Fusion

A standalone system that fuses **TF-Mini Plus** LiDAR, **BNO085** IMU, and MacBook camera with YOLOv8 object detection to determine what you are looking at. Outputs results via display overlay and TTS (text-to-speech) audio.

---

## Overview

- **LiDAR** — Measures distance to objects (0–12 m)
- **IMU** — Provides orientation (yaw, pitch, roll) for direction of look
- **Camera** — Object detection (YOLOv8) to identify what's in view
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
| TF-Mini Plus | Green/Blue (RX) | RX1 (pin 19) |
| TF-Mini Plus | White (TX) | TX1 (pin 18) |
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
   - **SparkFun BNO080** — search `SparkFun BNO080`
2. Open `arduino/smart_vision_sensors/smart_vision_sensors.ino` in Arduino IDE.
3. Connect the Arduino Mega via USB.
4. Select **Tools → Board → Arduino Mega 2560**.
5. Select **Tools → Port** → your Arduino port (e.g. `/dev/cu.usbmodem...`).
6. Click **Upload** (→).

### 2. IMU Direction Calibration

The BNO085 needs to know which direction is North, South, East, West, Up, and Down in your environment. This is done once and saved to EEPROM — it survives power off.

1. Open **Serial Monitor** at baud rate `115200`
2. Set line ending to **Newline** (dropdown at bottom of Serial Monitor)
3. On first boot, the calibration wizard starts automatically:

```
=== CALIBRATION MODE ===
Point device NORTH
  >> Press Enter when ready...
```

4. For each direction:
   - Physically point the device that way
   - Hold it steady
   - Press **Enter** in Serial Monitor
   - The saved value prints to confirm

5. You will be prompted for all 6 directions: North, East, South, West, Up, Down
6. Send `c` in Serial Monitor anytime to recalibrate

### 3. Python (virtual environment recommended)

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

- Press **q** to quit
- Display shows: camera feed, bounding boxes, distance, orientation, and fused result
- TTS announces *"Looking at [object] at [distance]"* when you look at a detected object (with throttling)

### Demo mode (without Arduino)
If Arduino is not connected, the app uses `python/data/sample_sensor_data.csv` for simulated sensor data. Camera and object detection still run normally.

### Serial Monitor Output (IMU debug)
While the Arduino is running you can monitor raw IMU output:
```
Yaw: 45.23° Pitch: 2.10° Roll: -1.05°  -->  North
Yaw: 134.87° Pitch: 1.50° Roll: 0.30°  -->  East
Yaw: 45.00° Pitch: 88.20° Roll: 0.10°  -->  Up
```

---

## Understanding Yaw, Pitch and Roll

These are the three axes of rotation used to describe orientation in 3D space. The IMU uses all three to determine which direction the device is pointing.

```
                        YAW (Z axis)
                        Rotate left/right
                             |
                             |
                    +--------+--------+
                   /         |        \
                  /     PITCH (Y axis) \
                 /     Tilt forward     \
                /       /backward        \
               +-------+                  +
               |      /                   |
               |     /  ROLL (X axis)     |
               |    /   Tilt left/right   |
               +---+---------------------+


         +-----------------+
         |                 |
         |    [DEVICE]     |   <-- top view
         |                 |
         +-----------------+

    YAW        PITCH         ROLL
   (Z axis)   (Y axis)     (X axis)

  Rotating    Nodding      Tilting
  like a      like         like a
  compass     saying       see-saw
              "yes"


              PITCH
              ^ (nose up)
              |
              |
   ROLL <-----+-----> ROLL
   (left)     |       (right)
              |
              v (nose down)

         Viewed from behind device:

              up
              |
    YAW left--+--YAW right
              |
             down

```

### In Plain Terms

| Rotation | Axis | Real World Example |
|----------|------|--------------------|
| **Yaw**  | Z (vertical) | Turning your head left or right, like a compass needle rotating |
| **Pitch**| Y (side to side) | Nodding your head up and down, like saying "yes" |
| **Roll** | X (front to back) | Tilting your head to your shoulder, like saying "maybe" |

### Ranges

| Value | Range | Zero Point |
|-------|-------|------------|
| Yaw   | 0° to 360° | Depends on calibrated North reference |
| Pitch | -180° to 180° | Device level/horizontal |
| Roll  | -90° to 90° | Device level/horizontal |

### Gimbal Lock
Euler angles (yaw/pitch/roll) have a limitation called **gimbal lock** — when pitch reaches ±90°, yaw and roll become indistinguishable. The BNO085 uses quaternions internally to avoid this, but converted yaw/pitch/roll values can behave oddly at extreme tilt angles. For this reason, Up/Down detection uses pitch thresholds rather than reading yaw at those angles.

---

## How Direction Matching Works

During calibration, the raw yaw angle is saved for each horizontal direction and raw pitch for Up/Down. At runtime, the current reading is compared against all saved references and the closest match wins.

Up/Down are checked first (using pitch), then horizontal direction (using yaw). The matching uses shortest angular distance to correctly handle the 0°/360° wraparound boundary.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Arduino port not found | Check USB cable and port in Arduino IDE |
| LiDAR/IMU test fails | Confirm wiring and that firmware is uploaded |
| Camera not opening | Grant camera permissions to Terminal/IDE |
| YOLOv8 slow | First run downloads the model; later runs are faster |
| IMU always says same direction | See below — EEPROM may have stale data |
| Calibration prompt not appearing | Set Serial Monitor line ending to **Newline** |

**IMU always says West (or same direction)**
EEPROM may contain random data from a fresh board. Add this temporarily at the start of `setup()` to force recalibration:
```cpp
refs.valid = false;
EEPROM.put(0, refs);
```
Remove these lines after successful calibration.

---

## Project Structure

```
smart-vision/
├── arduino/
│   └── smart_vision_sensors/
│       └── smart_vision_sensors.ino   — Arduino firmware (LiDAR + IMU)
├── python/
│   ├── data/
│   │   └── sample_sensor_data.csv     — Sample sensor data (demo mode when Arduino not connected)
│   ├── models/
│   │   └── yolov8n.pt                 — YOLOv8-nano weights (auto-downloaded by ultralytics if missing)
│   ├── src/
│   │   ├── sensors.py                 — Serial reader (LiDAR + IMU)
│   │   ├── detector.py                — Camera + YOLOv8 object detection
│   │   └── fusion.py                  — Fusion logic
│   ├── smart_vision.py                — Main script
│   ├── test_connections.py            — Connection test
│   └── requirements.txt
└── README.md
```
