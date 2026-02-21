/*
 * Smart Vision Sensors - Arduino Mega
 * Reads TF-Mini Plus LiDAR (distance) and BNO085 IMU (orientation)
 * Streams: distance_cm,roll,pitch,yaw over USB Serial at 115200 baud
 *
 * Wiring:
 *   TF-Mini Plus: TX->RX1(19), RX->TX1(18), 5V, GND
 *   BNO085: SDA->20, SCL->21, 3.3V, GND
 */

#include <TFMPlus.h>
#include <Adafruit_BNO08x.h>

TFMPlus tfmP;
Adafruit_BNO08x bno08x(-1);  // -1 = no reset pin
sh2_SensorValue_t sensorValue;

// LiDAR
int16_t dist = 0;
int16_t flux = 0;
int16_t temp = 0;

// IMU quaternion and Euler
float qw, qx, qy, qz;
float roll = 0, pitch = 0, yaw = 0;

void quaternionToEuler(float w, float x, float y, float z, float& r, float& p, float& y) {
  // Convert quaternion (w,x,y,z) to Euler angles (roll, pitch, yaw) in degrees
  float sinr_cosp = 2.0f * (w * x + y * z);
  float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
  r = atan2(sinr_cosp, cosr_cosp) * 180.0f / PI;

  float sinp = 2.0f * (w * y - z * x);
  if (fabs(sinp) >= 1.0f) {
    p = copysign(90.0f, sinp);
  } else {
    p = asin(sinp) * 180.0f / PI;
  }

  float siny_cosp = 2.0f * (w * z + x * y);
  float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
  y = atan2(siny_cosp, cosy_cosp) * 180.0f / PI;
}

void setReports(void) {
  if (!bno08x.enableReport(SH2_GAME_ROTATION_VECTOR)) {
    Serial.println("Could not enable rotation vector");
  }
}

void setup() {
  Serial.begin(115200);
  Serial1.begin(115200);

  delay(500);

  // Init TF-Mini Plus on Serial1
  tfmP.begin(&Serial1);
  delay(100);

  // Init BNO085 on I2C
  if (!bno08x.begin_I2C()) {
    Serial.println("BNO085 not found");
  } else {
    setReports();
  }

  delay(200);
}

void loop() {
  // Read LiDAR
  if (tfmP.getData(dist, flux, temp)) {
    // dist in cm
  } else {
    dist = 0;
  }

  // Read IMU
  if (bno08x.wasReset()) {
    setReports();
  }

  if (bno08x.getSensorEvent(&sensorValue)) {
    if (sensorValue.sensorId == SH2_GAME_ROTATION_VECTOR) {
      qw = sensorValue.un.gameRotationVector.real;
      qx = sensorValue.un.gameRotationVector.i;
      qy = sensorValue.un.gameRotationVector.j;
      qz = sensorValue.un.gameRotationVector.k;
      quaternionToEuler(qw, qx, qy, qz, roll, pitch, yaw);
    }
  }

  // Output CSV: distance_cm,roll,pitch,yaw
  Serial.print(dist);
  Serial.print(",");
  Serial.print(roll, 2);
  Serial.print(",");
  Serial.print(pitch, 2);
  Serial.print(",");
  Serial.println(yaw, 2);
}
