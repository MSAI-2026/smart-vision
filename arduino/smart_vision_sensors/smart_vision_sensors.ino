/*
 * Smart Vision Sensors - Arduino Mega
 * Reads TF-Mini Plus LiDAR (raw Serial1) and BNO085 IMU (SparkFun BNO080)
 * Streams: distance_cm,roll,pitch,yaw over USB Serial at 115200 baud
 *
 * Wiring:
 *   TF-Mini Plus: TX->RX1(19), RX->TX1(18), 5V, GND
 *   BNO085: SDA->20, SCL->21, 3.3V, GND
 */

#include <Wire.h>
#include "SparkFun_BNO080_Arduino_Library.h"

BNO080 imu;

// LiDAR - last reading (cm)
uint16_t dist = 0;

// IMU - last reading (degrees)
float roll = 0, pitch = 0, yaw = 0;

void setup() {
  Serial.begin(115200);   // USB Serial Monitor / Python
  Serial1.begin(115200);  // TF-Mini Plus

  Wire.begin();

  if (!imu.begin()) {
    Serial.println("IMU not detected");
    while (1);
  }
  Serial.println("IMU working");
  imu.enableRotationVector(50);
}

void loop() {
  // Read TF-Mini Plus (9-byte frame) on Serial1
  if (Serial1.available() >= 9) {
    if (Serial1.read() == 0x59) {
      if (Serial1.read() == 0x59) {
        dist = Serial1.read() | (Serial1.read() << 8);
        Serial1.read();  // strength low
        Serial1.read();  // strength high
        Serial1.read();
        Serial1.read();
        Serial1.read();  // reserved + checksum
      }
    }
  }

  // Read IMU and output CSV: distance_cm,roll,pitch,yaw
  if (imu.dataAvailable()) {
    roll = imu.getRoll() * 180.0f / PI;
    pitch = imu.getPitch() * 180.0f / PI;
    yaw = imu.getYaw() * 180.0f / PI;

    Serial.print(dist);
    Serial.print(",");
    Serial.print(roll, 2);
    Serial.print(",");
    Serial.print(pitch, 2);
    Serial.print(",");
    Serial.println(yaw, 2);
  }
}
