/*
 * Smart Vision Sensors - Arduino Mega
 * Reads TF-Mini Plus LiDAR (raw Serial1) and BNO085 IMU (SparkFun BNO080)
 *
 * Streams extended IMU CSV over USB Serial at 115200 baud:
 *   distance_cm,
 *   roll_deg,pitch_deg,yaw_deg,
 *   quat_w,quat_x,quat_y,quat_z,
 *   lin_accel_x,lin_accel_y,lin_accel_z,
 *   gyro_x,gyro_y,gyro_z,
 *   mag_x,mag_y,mag_z,
 *   accel_x,accel_y,accel_z,
 *   gravity_x,gravity_y,gravity_z
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

// IMU - last reading (degrees and extended vectors)
float roll = 0, pitch = 0, yaw = 0;
float quatW = 0, quatX = 0, quatY = 0, quatZ = 0;
float linAx = 0, linAy = 0, linAz = 0;
float gyroX = 0, gyroY = 0, gyroZ = 0;
float magX = 0, magY = 0, magZ = 0;
float accelX = 0, accelY = 0, accelZ = 0;
float gravX = 0, gravY = 0, gravZ = 0;

void setup() {
  Serial.begin(115200);   // USB Serial Monitor / Python
  Serial1.begin(115200);  // TF-Mini Plus

  Wire.begin();

  if (!imu.begin()) {
    Serial.println("IMU not detected");
    while (1);
  }
  Serial.println("IMU working");
  // Enable all relevant sensors at 50 Hz
  imu.enableRotationVector(50);
  imu.enableGyro(50);
  imu.enableAccelerometer(50);
  imu.enableMagnetometer(50);
  imu.enableLinearAccelerometer(50);
  imu.enableGravity(50);
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

  // Read IMU and output extended CSV (see header comment)
  if (imu.dataAvailable()) {
    roll = imu.getRoll() * 180.0f / PI;
    pitch = imu.getPitch() * 180.0f / PI;
    yaw = imu.getYaw() * 180.0f / PI;

    // Quaternion
    quatW = imu.getQuatReal();
    quatX = imu.getQuatI();
    quatY = imu.getQuatJ();
    quatZ = imu.getQuatK();

    // Linear acceleration (m/s^2)
    linAx = imu.getLinAccelX();
    linAy = imu.getLinAccelY();
    linAz = imu.getLinAccelZ();

    // Gyro (rad/s)
    gyroX = imu.getGyroX();
    gyroY = imu.getGyroY();
    gyroZ = imu.getGyroZ();

    // Magnetometer (uT)
    magX = imu.getMagX();
    magY = imu.getMagY();
    magZ = imu.getMagZ();

    // Raw accelerometer (m/s^2)
    accelX = imu.getAccelX();
    accelY = imu.getAccelY();
    accelZ = imu.getAccelZ();

    // Gravity vector (m/s^2)
    gravX = imu.getGravityX();
    gravY = imu.getGravityY();
    gravZ = imu.getGravityZ();

    Serial.print(dist);
    Serial.print(",");
    Serial.print(roll, 2);
    Serial.print(",");
    Serial.print(pitch, 2);
    Serial.print(",");
    Serial.print(yaw, 2);

    // Quaternion
    Serial.print(",");
    Serial.print(quatW, 6);
    Serial.print(",");
    Serial.print(quatX, 6);
    Serial.print(",");
    Serial.print(quatY, 6);
    Serial.print(",");
    Serial.print(quatZ, 6);

    // Linear acceleration
    Serial.print(",");
    Serial.print(linAx, 3);
    Serial.print(",");
    Serial.print(linAy, 3);
    Serial.print(",");
    Serial.print(linAz, 3);

    // Gyro
    Serial.print(",");
    Serial.print(gyroX, 3);
    Serial.print(",");
    Serial.print(gyroY, 3);
    Serial.print(",");
    Serial.print(gyroZ, 3);

    // Magnetometer
    Serial.print(",");
    Serial.print(magX, 3);
    Serial.print(",");
    Serial.print(magY, 3);
    Serial.print(",");
    Serial.print(magZ, 3);

    // Raw accelerometer
    Serial.print(",");
    Serial.print(accelX, 3);
    Serial.print(",");
    Serial.print(accelY, 3);
    Serial.print(",");
    Serial.print(accelZ, 3);

    // Gravity vector
    Serial.print(",");
    Serial.print(gravX, 3);
    Serial.print(",");
    Serial.print(gravY, 3);
    Serial.print(",");
    Serial.println(gravZ, 3);
  }
}
