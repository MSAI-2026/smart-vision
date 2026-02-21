# Smart Vision Arduino Firmware

Arduino Mega firmware for reading **TF-Mini Plus** LiDAR (distance) and **BNO085** IMU (orientation). Streams `distance_cm,roll,pitch,yaw` over USB Serial at 115200 baud for the Python app.

---

## Hardware

- **Arduino Mega 2560**
- **TF-Mini Plus** LiDAR
- **BNO085** IMU breakout

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

---

## Setup

1. Install Arduino IDE from [arduino.cc](https://www.arduino.cc/en/software).

2. Install libraries via **Sketch → Include Library → Manage Libraries**:
   - **TFMPlus** (by Bud Ryerson)
   - **Adafruit BNO08x**

3. Connect the Arduino Mega via USB.

4. Open `smart_vision_sensors/smart_vision_sensors.ino`.

5. Select **Tools → Board → Arduino Mega 2560**.

6. Select **Tools → Port** → your Arduino (e.g. `/dev/cu.usbmodem...`).

7. Click **Upload** (→).

---

## Output Format

One CSV line per sample over USB Serial (115200 baud):

```
distance_cm,roll,pitch,yaw
```

Example: `125,2.3,-15.1,45.2`
