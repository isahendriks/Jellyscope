/***************************************************************************
  This is a software for Isa and the underwater camera humidity, temperature & pressure sensor
  These sensors use I2C, SPI or OneWire to communicate
  Sensors used are Bar3XT, BME280 (or BMP280 fallback), DS18B20 & leak detector
  Written by Johan B, Lund University Last update 2026-07-22 15:19
 ***************************************************************************/

#include <Wire.h>
#include <OneWire.h>
#include <SPI.h>
#include <Adafruit_BME280.h>
#include <Adafruit_BMP280.h>
#include <DallasTemperature.h>
#include "KellerLD.h"

#define BME_SCK  13
#define BME_MISO 11  //SDI
#define BME_MOSI 12  //SDO
#define BME_CS   10
#define ONE_WIRE_BUS 9

KellerLD sensor;

// Primary sensor is a BME280 (adds humidity). If the board still has the old
// BMP280 installed, we fall back to that instead of halting - same pins,
// just no humidity reading available.
Adafruit_BME280 bme(BME_CS, BME_MOSI, BME_MISO,  BME_SCK);
Adafruit_BMP280 bmp(BME_CS, BME_MOSI, BME_MISO,  BME_SCK);
bool usingBME280 = true;
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

bool debug = false;
int leakPin = 8;    // Leak Signal Pin
int leak = 0;       // 0 = Dry , 1 = Leak
char ctrl = 'C';

float airPressure() {
  return usingBME280 ? bme.readPressure() : bmp.readPressure();
}

float airTemperature() {
  return usingBME280 ? bme.readTemperature() : bmp.readTemperature();
}

float airAltitude(float seaLevelhPa) {
  return usingBME280 ? bme.readAltitude(seaLevelhPa) : bmp.readAltitude(seaLevelhPa);
}

// NAN when running on the BMP280 fallback, which has no humidity sensor.
float airHumidity() {
  return usingBME280 ? bme.readHumidity() : NAN;
}
void setup() {
  Serial.begin(115200);
  while ( !Serial ) delay(100);   // wait for native usb
  Serial.println(F("Isa submarine"));
  Wire.begin();
  // DS18B20 setup
  sensors.begin();
  sensors.setResolution(11);
  // Setup air pressure sensor
 
  if (bme.begin()) {
    usingBME280 = true;
    Serial.println(F("BME280 detected (pressure, temperature & humidity)."));
    /* Settings from datasheet. */
    bme.setSampling(Adafruit_BME280::MODE_NORMAL,     // Operating Mode.
                    Adafruit_BME280::SAMPLING_X2,     // Temp. oversampling
                    Adafruit_BME280::SAMPLING_X16,    // Pressure oversampling
                    Adafruit_BME280::SAMPLING_X1,     // Humidity oversampling
                    Adafruit_BME280::FILTER_X16,      // Filtering.
                    Adafruit_BME280::STANDBY_MS_500); // Standby time.
  } else {
    uint8_t chipID = bme.sensorID();
    Serial.print("No BME280 found, SensorID was: 0x"); Serial.println(chipID,16);
    if (bmp.begin()) {
      usingBME280 = false;
      Serial.println(F("Falling back to BMP280 (pressure & temperature only, no humidity) - "
                        "check that the BME280 swap was actually done if humidity is expected."));
      bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,     // Operating Mode.
                      Adafruit_BMP280::SAMPLING_X2,     // Temp. oversampling
                      Adafruit_BMP280::SAMPLING_X16,    // Pressure oversampling
                      Adafruit_BMP280::FILTER_X16,      // Filtering.
                      Adafruit_BMP280::STANDBY_MS_500); // Standby time.
    } else {
      Serial.print("Halting: no valid BME280 or BMP280 found. SensorID was: 0x");
      Serial.println(chipID,16);
      Serial.print("   ID of 0x60 represents a BME280, 0x56-0x58 a BMP280.\n");
      while (1) delay(12);
    }
  }

  // Setup water pressure sensor Bar3XT
  sensor.init();
  sensor.setFluidDensity(997); // kg/m^3 (freshwater, 1029 for seawater)

  if (sensor.isInitialized()) {
    Serial.println("Sensor connected.");
  } else {
    Serial.println("Sensor not connected.");
  }
  pinMode(leakPin, INPUT);
  Serial.println("p=BME280/BMP280 pressure, h=BME280 humidity (NaN if running on BMP280 fallback), P=Bar3XT pressure, T=Bar3XT temperature, d=Bar3XT depth, l=leak (1->leak detected, 0->no leak), t=DS18B20 temperature");
}

void loop() {
  if (debug) {
    // BME280/BMP280 data
    Serial.print(F("Air temperature = "));
    Serial.print(airTemperature());
    Serial.println(" *C");
    Serial.print(F("Air pressure = "));
    Serial.print(airPressure());
    Serial.println(" Pa");
    Serial.print(F("Air humidity = "));
    Serial.print(airHumidity());
    Serial.println(" %");
    Serial.print(F("Air approx altitude = "));
    Serial.print(airAltitude(1013.25)); // Adjusted to local forecast!
    Serial.println(" m");
    Serial.println();
    delay(1000);

    // Bar3XT data
    sensor.read();
    Serial.print("Bar3XT pressure: ");
    Serial.print(sensor.pressure());
    Serial.println(" mbar");
 
    Serial.print("Bar3XT temperature: ");
    Serial.print(sensor.temperature());
    Serial.println(" deg C");
 
    Serial.print("Bar3XT depth: ");
    Serial.print(sensor.depth());
    Serial.println(" m");
 
    Serial.print("Bar3XT altitude: ");
    Serial.print(sensor.altitude());
    Serial.println(" m above mean sea level");
    delay(500);
    sensors.requestTemperatures(); // Send the command to get temperatures
    Serial.print("DS18B20 temperature: ");
    Serial.println(sensors.getTempCByIndex(0));
    delay(500);
    // Detect leaks
    leak = digitalRead(leakPin);   // Read the Leak Sensor Pin
    if (leak == 1) {
      Serial.println("Leak Detected!");
    }

    delay(2000);
  } else { 
    if (Serial.available() > 0) {
      ctrl = Serial.read();
    }
    if (ctrl == 'P' ) {
      sensor.read();
      Serial.print("P:");
      Serial.println(sensor.pressure());
    } else if (ctrl == 'C' ) {
      Serial.print(F("p:"));
      Serial.print(airPressure());
      Serial.print(",h:");
      Serial.print(airHumidity());
      sensor.read();
      Serial.print(",P:");
      Serial.print(sensor.pressure());
      Serial.print(",T:");
      Serial.print(sensor.temperature());
      Serial.print(",d:");
      Serial.print(sensor.depth());
      leak = digitalRead(leakPin);
      Serial.print(",l:");
      Serial.print(leak);
      sensors.requestTemperatures(); // Send the command to get temperatures
      Serial.print(",t:");
      Serial.println(sensors.getTempCByIndex(0));
    }
  }
  // ca 85 reading/sec when NOT using the DS18B20
}