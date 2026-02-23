#!/usr/bin/env python3
"""
Connection test for Smart Vision.
Verifies that camera, Arduino/Serial, LiDAR, and IMU are working.
Run before the main app to verify wiring and setup.
"""

import re
import sys
import time
from typing import Tuple

# Match human-readable IMU output: "Yaw: 12.34° Pitch: -5.67° Roll: 0.00°" (with optional " --> North")
_IMU_TEXT_RE = re.compile(
    r"Yaw:\s*([-\d.]+)\s*°?\s*Pitch:\s*([-\d.]+)\s*°?\s*Roll:\s*([-\d.]+)",
    re.IGNORECASE,
)

try:
    import cv2
    import serial
    import serial.tools.list_ports
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install pyserial opencv-python")
    sys.exit(1)


def find_arduino_port():
    """Auto-detect Arduino Mega USB serial port (macOS, Linux, Windows)."""
    candidates = []
    for port in serial.tools.list_ports.comports():
        dev = (port.device or "").lower()
        desc = (port.description or "").lower()
        hwid = (port.hwid or "").lower()
        # Prefer explicit matches first
        if "usbmodem" in dev or "usbserial" in dev:
            return port.device
        if "arduino" in desc or "mega" in desc or "ch340" in desc or "cp210" in desc:
            return port.device
        # Linux: ttyACM0, ttyUSB0
        if "ttyacm" in dev or "ttyusb" in dev:
            candidates.append(port.device)
        # Windows: COM3 etc.; often Arduino shows as "Arduino" in description or USB VID in hwid
        if dev.startswith("com") and ("2341" in hwid or "1a86" in hwid or "arduino" in desc):
            return port.device
    return candidates[0] if candidates else None


def list_serial_ports():
    """Return a short list of available serial ports for error messages."""
    try:
        ports = serial.tools.list_ports.comports()
        return [f"{p.device} ({p.description or 'unknown'})" for p in ports]
    except Exception:
        return []


def test_camera() -> Tuple[bool, str]:
    """Test MacBook camera. Returns (pass, message)."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False, "Could not open camera (index 0)"
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False, "Could not read frame from camera"
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return False, "Invalid frame dimensions"
    return True, f"Camera OK ({w}x{h})"


def _port_not_found_msg():
    msg = "Arduino port not found (connect Arduino via USB)"
    available = list_serial_ports()
    if available:
        msg += ". Available: " + ", ".join(available[:5])
    return msg


def test_serial() -> Tuple[bool, str]:
    """Test Arduino serial connection. Returns (pass, message)."""
    port = find_arduino_port()
    if not port:
        return False, _port_not_found_msg()
    try:
        ser = serial.Serial(port, 115200, timeout=0.5)
    except serial.SerialException as e:
        return False, f"Serial open failed: {e}"
    try:
        time.sleep(2)  # Allow Arduino reset
        lines = []
        deadline = time.time() + 3
        while time.time() < deadline and len(lines) < 5:
            line = ser.readline()
            if line:
                lines.append(line)
    except Exception as e:
        ser.close()
        return False, f"Serial read error: {e}"
    ser.close()
    if len(lines) < 2:
        return False, f"Too few serial lines received ({len(lines)})"
    return True, f"Serial OK (received {len(lines)} lines)"


def test_lidar() -> Tuple[bool, str]:
    """Test LiDAR (distance from Arduino CSV). Returns (pass, message)."""
    port = find_arduino_port()
    if not port:
        return False, _port_not_found_msg()
    try:
        ser = serial.Serial(port, 115200, timeout=0.5)
    except serial.SerialException:
        return False, "Serial open failed"
    try:
        time.sleep(2)
        valid_count = 0
        last_dist = None
        deadline = time.time() + 3
        while time.time() < deadline and valid_count < 5:
            line = ser.readline()
            if not line:
                continue
            try:
                text = line.decode("utf-8").strip()
                parts = text.split(",")
                if len(parts) >= 1:
                    dist = int(parts[0])
                    if 0 <= dist <= 1200:  # TF-Mini Plus range 0-1200 cm
                        valid_count += 1
                        last_dist = dist
            except (ValueError, UnicodeDecodeError):
                pass
        ser.close()
        if valid_count < 2:
            return False, f"LiDAR: insufficient valid distance samples ({valid_count})"
        return True, f"LiDAR OK (distance: {last_dist} cm)"
    except Exception as e:
        ser.close()
        return False, f"LiDAR test error: {e}"


def test_imu() -> Tuple[bool, str]:
    """Test IMU (roll, pitch, yaw from Arduino). Accepts CSV or human-readable format."""
    port = find_arduino_port()
    if not port:
        return False, _port_not_found_msg()
    try:
        ser = serial.Serial(port, 115200, timeout=0.5)
    except serial.SerialException as e:
        return False, f"Serial open failed: {e}"
    try:
        time.sleep(2)  # Allow Arduino reset after open
        valid_count = 0
        last_rpy = None
        saw_imu_working = False
        saw_imu_not_detected = False
        deadline = time.time() + 5  # Longer window after reset
        while time.time() < deadline and valid_count < 5:
            line = ser.readline()
            if not line:
                continue
            try:
                text = line.decode("utf-8").strip()
                if "IMU not detected" in text:
                    saw_imu_not_detected = True
                    continue
                if "IMU working" in text:
                    saw_imu_working = True
                    continue
                # CSV format: something,roll,pitch,yaw
                parts = text.split(",")
                if len(parts) >= 4:
                    roll = float(parts[1])
                    pitch = float(parts[2])
                    yaw = float(parts[3])
                    if -180 <= roll <= 180 and -180 <= pitch <= 180 and -180 <= yaw <= 180:
                        valid_count += 1
                        last_rpy = (roll, pitch, yaw)
                    continue
                # Human-readable: "Yaw: x° Pitch: y° Roll: z°" or "Yaw: x°  Pitch: y°  Roll: z°"
                match = _IMU_TEXT_RE.search(text)
                if match:
                    yaw = float(match.group(1))
                    pitch = float(match.group(2))
                    roll = float(match.group(3))
                    if -180 <= roll <= 180 and -180 <= pitch <= 180 and -180 <= yaw <= 180:
                        valid_count += 1
                        last_rpy = (roll, pitch, yaw)
            except (ValueError, UnicodeDecodeError):
                pass
        ser.close()
        if saw_imu_not_detected:
            return False, "IMU: Arduino reports 'IMU not detected' (check I2C wiring and BNO085 power)"
        if valid_count >= 2:
            r, p, y = last_rpy or (0, 0, 0)
            return True, f"IMU OK (roll={r:.1f} pitch={p:.1f} yaw={y:.1f})"
        if saw_imu_working:
            return True, "IMU OK (Arduino reports IMU working; no orientation lines yet—move the device or wait)"
        return False, f"IMU: insufficient valid orientation samples ({valid_count}). Ensure firmware sends Yaw/Pitch/Roll or CSV."
    except Exception as e:
        ser.close()
        return False, f"IMU test error: {e}"


def main():
    print("=" * 50)
    print("Smart Vision - Connection Test")
    print("=" * 50)

    results = []

    # Camera (independent - no Arduino needed)
    print("\n[1/4] Testing camera...", end=" ")
    ok, msg = test_camera()
    results.append(("Camera", ok, msg))
    print("PASS" if ok else "FAIL")
    print(f"      {msg}")

    # Serial
    print("\n[2/4] Testing Arduino/Serial...", end=" ")
    ok, msg = test_serial()
    results.append(("Arduino/Serial", ok, msg))
    print("PASS" if ok else "FAIL")
    print(f"      {msg}")

    # LiDAR (requires Arduino with firmware)
    print("\n[3/4] Testing LiDAR...", end=" ")
    ok, msg = test_lidar()
    results.append(("LiDAR", ok, msg))
    print("PASS" if ok else "FAIL")
    print(f"      {msg}")

    # IMU (requires Arduino with firmware)
    print("\n[4/4] Testing IMU...", end=" ")
    ok, msg = test_imu()
    results.append(("IMU", ok, msg))
    print("PASS" if ok else "FAIL")
    print(f"      {msg}")

    # Summary
    print("\n" + "=" * 50)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    if passed == total:
        print("All connections OK.")
    else:
        failed = [name for name, ok, _ in results if not ok]
        print(f"Failed: {', '.join(failed)}")
        print("Ensure Arduino is connected, firmware is flashed, and wiring is correct.")
    print("=" * 50)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
