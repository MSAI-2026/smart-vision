#!/usr/bin/env python3
"""
Smart Vision - Main standalone script.
Reads LiDAR + IMU from Arduino, runs camera object detection,
fuses to determine what the user is looking at, and outputs via display + TTS.
When Arduino is not connected, uses sample_sensor_data.csv for demo mode.
"""

import os
import sys
import time
import threading

import cv2

from sensors import SensorReader, find_arduino_port
from detector import ObjectDetector
from fusion import fuse, FusedResult


# TTS setup (optional - graceful fallback if unavailable)
_tts_engine = None


def _init_tts():
    global _tts_engine
    if _tts_engine is not None:
        return
    try:
        import pyttsx3
        _tts_engine = pyttsx3.init()
    except Exception as e:
        print(f"TTS unavailable: {e}")


def speak(text: str, throttle_sec: float = 4.0):
    """Speak text via TTS, with throttling to avoid excessive announcements."""
    global _tts_engine
    _init_tts()
    if _tts_engine is None:
        return
    try:
        _tts_engine.say(text)
        _tts_engine.runAndWait()
    except Exception as e:
        print(f"TTS error: {e}")


def format_distance(distance_cm: int) -> str:
    """Convert cm to human-readable (e.g., '2 meters' or '50 cm')."""
    if distance_cm >= 100:
        meters = distance_cm / 100
        if meters == int(meters):
            return f"{int(meters)} meter{'s' if meters != 1 else ''}"
        return f"{meters:.1f} meters"
    return f"{distance_cm} cm"


def main():
    print("Smart Vision - Starting...")
    print("Press 'q' to quit.")

    # Sensor reader (Arduino or sample file)
    port = find_arduino_port()
    if port:
        reader = SensorReader(port=port)
        reader.connect()
        reader.start()
    else:
        sample_path = os.path.join(os.path.dirname(__file__), "sample_sensor_data.csv")
        try:
            reader = SensorReader(sample_file=sample_path)
            reader.connect()
            reader.start()
            print("Arduino not found; using sample_sensor_data.csv")
        except FileNotFoundError as e:
            print(f"{e}")
            print("Connect Arduino with LiDAR and IMU, or ensure sample_sensor_data.csv exists.")
            sys.exit(1)

    # Object detector (camera)
    detector = ObjectDetector(camera_index=0)
    if not detector.open_camera():
        print("Failed to open MacBook camera")
        reader.disconnect()
        sys.exit(1)

    last_spoken = ""
    last_speak_time = 0
    speak_cooldown = 4.0  # seconds

    try:
        while True:
            frame = detector.read_frame()
            if frame is None:
                break

            h, w = frame.shape[:2]
            sensor_data = reader.get()
            objects = detector.detect(frame)

            fused = fuse(
                objects,
                sensor_data["distance"],
                w,
                h,
                center_region_ratio=0.4,
            )

            # Draw detections
            frame = detector.draw_detections(frame, objects)

            # Overlay: sensor data
            overlay_text = [
                f"Distance: {sensor_data['distance']} cm",
                f"Yaw: {sensor_data['yaw']:.1f} Pitch: {sensor_data['pitch']:.1f} Roll: {sensor_data['roll']:.1f}",
            ]
            y_offset = 30
            for line in overlay_text:
                cv2.putText(
                    frame,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                )
                y_offset += 25

            # Overlay: fused result
            if fused:
                looking_at = f"Looking at: {fused.object_class} at {format_distance(fused.distance_cm)}"
                cv2.putText(
                    frame,
                    looking_at,
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                # TTS: announce when target changes, with cooldown
                msg = f"Looking at {fused.object_class} at {format_distance(fused.distance_cm)}"
                now = time.time()
                if msg != last_spoken and (now - last_speak_time) >= speak_cooldown:
                    last_spoken = msg
                    last_speak_time = now
                    threading.Thread(target=speak, args=(msg,), daemon=True).start()
            else:
                cv2.putText(
                    frame,
                    "Looking at: (nothing in center)",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (128, 128, 128),
                    2,
                )
                last_spoken = ""

            cv2.imshow("Smart Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        reader.disconnect()
        detector.release_camera()
        cv2.destroyAllWindows()
        print("Smart Vision - Stopped.")


if __name__ == "__main__":
    main()
