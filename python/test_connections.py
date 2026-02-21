#!/usr/bin/env python3
"""
Connection test for Smart Vision.
Verifies that camera, Arduino/Serial, LiDAR, and IMU are working.
Run before the main app to verify wiring and setup.
"""

import sys
import time
from typing import Tuple

try:
    import cv2
    import serial
    import serial.tools.list_ports
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install pyserial opencv-python")
    sys.exit(1)


def find_arduino_port():
    """Auto-detect Arduino Mega USB serial port."""
    for port in serial.tools.list_ports.comports():
        if "usbmodem" in port.device.lower() or "usbserial" in port.device.lower():
            return port.device
        desc = (port.description or "").lower()
        if "arduino" in desc or "mega" in desc or "ch340" in desc or "cp210" in desc:
            return port.device
    return None


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


def test_serial() -> Tuple[bool, str]:
    """Test Arduino serial connection. Returns (pass, message)."""
    port = find_arduino_port()
    if not port:
        return False, "Arduino port not found (connect Arduino via USB)"
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
        return False, "Arduino port not found"
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
    """Test IMU (roll, pitch, yaw from Arduino CSV). Returns (pass, message)."""
    port = find_arduino_port()
    if not port:
        return False, "Arduino port not found"
    try:
        ser = serial.Serial(port, 115200, timeout=0.5)
    except serial.SerialException:
        return False, "Serial open failed"
    try:
        time.sleep(2)
        valid_count = 0
        last_rpy = None
        deadline = time.time() + 3
        while time.time() < deadline and valid_count < 5:
            line = ser.readline()
            if not line:
                continue
            try:
                text = line.decode("utf-8").strip()
                parts = text.split(",")
                if len(parts) >= 4:
                    roll = float(parts[1])
                    pitch = float(parts[2])
                    yaw = float(parts[3])
                    if -180 <= roll <= 180 and -180 <= pitch <= 180 and -180 <= yaw <= 180:
                        valid_count += 1
                        last_rpy = (roll, pitch, yaw)
            except (ValueError, UnicodeDecodeError):
                pass
        ser.close()
        if valid_count < 2:
            return False, f"IMU: insufficient valid orientation samples ({valid_count})"
        r, p, y = last_rpy or (0, 0, 0)
        return True, f"IMU OK (roll={r:.1f} pitch={p:.1f} yaw={y:.1f})"
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
