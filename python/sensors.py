"""
Serial sensor reader for Smart Vision.
Reads distance (LiDAR) and orientation (roll, pitch, yaw) from Arduino over USB serial.
Runs in a background thread and exposes latest values via thread-safe access.
"""

import serial
import serial.tools.list_ports
import threading
import time
import sys


def find_arduino_port():
    """Auto-detect Arduino Mega USB serial port."""
    for port in serial.tools.list_ports.comports():
        # Arduino Mega typically shows as usbmodem (macOS) or USB Serial (Linux)
        if "usbmodem" in port.device.lower() or "usbserial" in port.device.lower():
            return port.device
        # Windows: COM ports, check for Arduino in description
        desc = (port.description or "").lower()
        if "arduino" in desc or "mega" in desc or "ch340" in desc or "cp210" in desc:
            return port.device
    return None


class SensorReader:
    """Reads distance and orientation from Arduino, provides thread-safe access."""

    def __init__(self, port=None, baud=115200):
        self.port = port or find_arduino_port()
        self.baud = baud
        self._serial = None
        self._lock = threading.Lock()
        self._distance = 0
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._running = False
        self._thread = None

    def connect(self):
        """Open serial connection to Arduino."""
        if not self.port:
            raise RuntimeError("Arduino port not found. Connect Arduino and try again.")
        self._serial = serial.Serial(self.port, self.baud, timeout=0.1)
        time.sleep(2)  # Allow Arduino to reset
        return True

    def disconnect(self):
        """Close serial connection."""
        self.stop()
        if self._serial and self._serial.is_open:
            self._serial.close()
            self._serial = None

    def start(self):
        """Start background thread reading serial data."""
        if self._running:
            return
        if not self._serial or not self._serial.is_open:
            self.connect()
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _read_loop(self):
        """Background loop: read lines and parse CSV."""
        while self._running and self._serial and self._serial.is_open:
            try:
                line = self._serial.readline()
                if line:
                    self._parse_line(line)
            except (serial.SerialException, OSError) as e:
                sys.stderr.write(f"Serial error: {e}\n")
                break
            except Exception as e:
                sys.stderr.write(f"Parse error: {e}\n")
        self._running = False

    def _parse_line(self, line: bytes):
        """Parse CSV line: distance_cm,roll,pitch,yaw"""
        try:
            text = line.decode("utf-8").strip()
            if not text:
                return
            parts = text.split(",")
            if len(parts) >= 4:
                dist = int(parts[0])
                roll = float(parts[1])
                pitch = float(parts[2])
                yaw = float(parts[3])
                with self._lock:
                    self._distance = dist
                    self._roll = roll
                    self._pitch = pitch
                    self._yaw = yaw
        except (ValueError, UnicodeDecodeError):
            pass

    def get(self):
        """Get latest sensor values (thread-safe). Returns dict with distance, roll, pitch, yaw."""
        with self._lock:
            return {
                "distance": self._distance,
                "roll": self._roll,
                "pitch": self._pitch,
                "yaw": self._yaw,
            }

    def is_connected(self):
        """Check if serial port is open."""
        return self._serial is not None and self._serial.is_open


if __name__ == "__main__":
    reader = SensorReader()
    try:
        reader.connect()
        reader.start()
        for _ in range(20):
            d = reader.get()
            print(f"Distance: {d['distance']} cm | Roll: {d['roll']:.1f} Pitch: {d['pitch']:.1f} Yaw: {d['yaw']:.1f}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        reader.disconnect()
