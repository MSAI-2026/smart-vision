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
    """Reads distance and orientation from Arduino (or sample file), provides thread-safe access.

    CSV format from Arduino (extended BNO085 output), one sample per line:
        distance_cm,
        roll_deg,pitch_deg,yaw_deg,
        quat_w,quat_x,quat_y,quat_z,
        lin_accel_x,lin_accel_y,lin_accel_z,
        gyro_x,gyro_y,gyro_z,
        mag_x,mag_y,mag_z,
        accel_x,accel_y,accel_z,
        gravity_x,gravity_y,gravity_z

    Older firmware may only send: distance_cm,roll_deg,pitch_deg,yaw_deg
    This reader remains backward compatible and treats missing fields as 0.0.
    """

    def __init__(self, port=None, sample_file=None, baud=115200):
        self.baud = baud
        self._serial = None
        self._lock = threading.Lock()
        self._distance = 0
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        # Extended IMU fields (BNO085)
        self._quat_w = 0.0
        self._quat_x = 0.0
        self._quat_y = 0.0
        self._quat_z = 0.0
        self._lin_ax = 0.0
        self._lin_ay = 0.0
        self._lin_az = 0.0
        self._gyro_x = 0.0
        self._gyro_y = 0.0
        self._gyro_z = 0.0
        self._mag_x = 0.0
        self._mag_y = 0.0
        self._mag_z = 0.0
        self._accel_x = 0.0
        self._accel_y = 0.0
        self._accel_z = 0.0
        self._grav_x = 0.0
        self._grav_y = 0.0
        self._grav_z = 0.0
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
        """Parse CSV line from Arduino.

        Expected extended format (see class docstring) but remains compatible
        with older 4-field lines: distance_cm,roll,pitch,yaw.
        """
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

                # Optional extended fields; keep previous values on parse error
                quat_w = self._quat_w
                quat_x = self._quat_x
                quat_y = self._quat_y
                quat_z = self._quat_z
                lin_ax = self._lin_ax
                lin_ay = self._lin_ay
                lin_az = self._lin_az
                gyro_x = self._gyro_x
                gyro_y = self._gyro_y
                gyro_z = self._gyro_z
                mag_x = self._mag_x
                mag_y = self._mag_y
                mag_z = self._mag_z
                accel_x = self._accel_x
                accel_y = self._accel_y
                accel_z = self._accel_z
                grav_x = self._grav_x
                grav_y = self._grav_y
                grav_z = self._grav_z

                # Quaternion (indices 4-7)
                if len(parts) >= 8:
                    try:
                        quat_w = float(parts[4])
                        quat_x = float(parts[5])
                        quat_y = float(parts[6])
                        quat_z = float(parts[7])
                    except ValueError:
                        pass

                # Linear acceleration (indices 8-10)
                if len(parts) >= 11:
                    try:
                        lin_ax = float(parts[8])
                        lin_ay = float(parts[9])
                        lin_az = float(parts[10])
                    except ValueError:
                        pass

                # Gyro (indices 11-13)
                if len(parts) >= 14:
                    try:
                        gyro_x = float(parts[11])
                        gyro_y = float(parts[12])
                        gyro_z = float(parts[13])
                    except ValueError:
                        pass

                # Magnetometer (indices 14-16)
                if len(parts) >= 17:
                    try:
                        mag_x = float(parts[14])
                        mag_y = float(parts[15])
                        mag_z = float(parts[16])
                    except ValueError:
                        pass

                # Raw accelerometer (indices 17-19)
                if len(parts) >= 20:
                    try:
                        accel_x = float(parts[17])
                        accel_y = float(parts[18])
                        accel_z = float(parts[19])
                    except ValueError:
                        pass

                # Gravity vector (indices 20-22)
                if len(parts) >= 23:
                    try:
                        grav_x = float(parts[20])
                        grav_y = float(parts[21])
                        grav_z = float(parts[22])
                    except ValueError:
                        pass

                with self._lock:
                    self._distance = dist
                    self._roll = roll
                    self._pitch = pitch
                    self._yaw = yaw
                    self._quat_w = quat_w
                    self._quat_x = quat_x
                    self._quat_y = quat_y
                    self._quat_z = quat_z
                    self._lin_ax = lin_ax
                    self._lin_ay = lin_ay
                    self._lin_az = lin_az
                    self._gyro_x = gyro_x
                    self._gyro_y = gyro_y
                    self._gyro_z = gyro_z
                    self._mag_x = mag_x
                    self._mag_y = mag_y
                    self._mag_z = mag_z
                    self._accel_x = accel_x
                    self._accel_y = accel_y
                    self._accel_z = accel_z
                    self._grav_x = grav_x
                    self._grav_y = grav_y
                    self._grav_z = grav_z
        except (ValueError, UnicodeDecodeError):
            pass

    def get(self):
        """Get latest sensor values (thread-safe).

        Returns dict with:
            distance, roll, pitch, yaw,
            quaternion, linear_accel, gyro, mag, accel, gravity.
        """
        with self._lock:
            return {
                "distance": self._distance,
                "roll": self._roll,
                "pitch": self._pitch,
                "yaw": self._yaw,
                "quaternion": {
                    "w": self._quat_w,
                    "x": self._quat_x,
                    "y": self._quat_y,
                    "z": self._quat_z,
                },
                "linear_accel": {
                    "x": self._lin_ax,
                    "y": self._lin_ay,
                    "z": self._lin_az,
                },
                "gyro": {
                    "x": self._gyro_x,
                    "y": self._gyro_y,
                    "z": self._gyro_z,
                },
                "mag": {
                    "x": self._mag_x,
                    "y": self._mag_y,
                    "z": self._mag_z,
                },
                "accel": {
                    "x": self._accel_x,
                    "y": self._accel_y,
                    "z": self._accel_z,
                },
                "gravity": {
                    "x": self._grav_x,
                    "y": self._grav_y,
                    "z": self._grav_z,
                },
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
