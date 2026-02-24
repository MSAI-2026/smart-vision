"""
Serial sensor reader for Smart Vision.
Reads distance (LiDAR) and orientation (roll, pitch, yaw) from Arduino over USB serial.
When Arduino is not connected, can replay from a sample CSV file (same format).
Runs in a background thread and exposes latest values via thread-safe access.
"""

import os
import serial
import serial.tools.list_ports
import threading
import time
import sys


def find_arduino_port():
    """Auto-detect Arduino Mega USB serial port (macOS, Linux, Windows)."""
    candidates = []
    for port in serial.tools.list_ports.comports():
        dev = (port.device or "").lower()
        desc = (port.description or "").lower()
        hwid = (port.hwid or "").lower()
        if "usbmodem" in dev or "usbserial" in dev:
            return port.device
        if "arduino" in desc or "mega" in desc or "ch340" in desc or "cp210" in desc:
            return port.device
        if "ttyacm" in dev or "ttyusb" in dev:
            candidates.append(port.device)
        if dev.startswith("com") and ("2341" in hwid or "1a86" in hwid or "arduino" in desc):
            return port.device
    return candidates[0] if candidates else None


class SensorReader:
    """Reads distance and orientation from Arduino (or sample file), provides thread-safe access."""

    def __init__(self, port=None, sample_file=None, baud=115200):
        self.baud = baud
        self._serial = None
        self._lock = threading.Lock()
        self._distance = 0
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._running = False
        self._thread = None
        if sample_file is not None:
            self.port = None
            self.sample_file = sample_file
            self._file_mode = True
        else:
            self.port = port or find_arduino_port()
            self.sample_file = None
            self._file_mode = False
        self._sample_file_path = None

    def connect(self):
        """Open serial connection to Arduino, or validate sample file."""
        if self._file_mode:
            path = self.sample_file
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(__file__), path)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Sample file not found: {path}")
            self._sample_file_path = path
            return True
        if not self.port:
            raise RuntimeError("Arduino port not found. Connect Arduino and try again.")
        self._serial = serial.Serial(self.port, self.baud, timeout=0.1)
        time.sleep(2)  # Allow Arduino to reset
        return True

    def disconnect(self):
        """Close serial connection or stop file replay."""
        self.stop()
        if self._serial and self._serial.is_open:
            self._serial.close()
            self._serial = None
        self._sample_file_path = None

    def start(self):
        """Start background thread reading serial or sample file."""
        if self._running:
            return
        if self._file_mode:
            if self._sample_file_path is None:
                self.connect()
            self._running = True
            self._thread = threading.Thread(target=self._read_loop_file, daemon=True)
            self._thread.start()
        else:
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
        """Background loop: read lines from serial and parse CSV."""
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

    def _read_loop_file(self):
        """Background loop: read lines from sample file, parse CSV, loop on EOF."""
        while self._running and self._sample_file_path:
            try:
                with open(self._sample_file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not self._running:
                            break
                        line = line.strip()
                        if line:
                            self._parse_line(line.encode("utf-8"))
                        time.sleep(0.03)  # ~33 Hz replay
            except (OSError, IOError) as e:
                sys.stderr.write(f"Sample file error: {e}\n")
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
        """Check if serial port is open or sample file is replaying."""
        if self._file_mode:
            return self._running
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
