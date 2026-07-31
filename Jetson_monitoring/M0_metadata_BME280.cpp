#include <BME280.h>
#include <BME280I2C.h>
#include <BME280I2C_BRZO.h>
#include <BME280Spi.h>
#include <BME280SpiSw.h>
#include <EnvironmentCalculations.h>

/***************************************************************************
  This is a software for Isa and the underwater camera humidity, temperature & pressure sensor
  These sensors use I2C, SPI or OneWire to communicate
  Sensors used are Bar3XT, BME280, DS18B20 & leak detector
  Written by Johan B, Lund University Last update 2026-07-22 15:19
 ***************************************************************************/

#include <Wire.h>
#include <OneWire.h>
#include <SPI.h>
#include <Adafruit_BME280.h>
#include <DallasTemperature.h>
#include "KellerLD.h"

#define BMP_SCK  13
#define BMP_MISO 11  //SDI
#define BMP_MOSI 12  //SDO
#define BMP_CS   10
#define ONE_WIRE_BUS 9

KellerLD sensor;

Adafruit_BME280 bme(BMP_CS, BMP_MOSI, BMP_MISO,  BMP_SCK);
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

bool debug = false;
int leakPin = 8;    // Leak Signal Pin
int leak = 0;       // 0 = Dry , 1 = Leak
char ctrl = 'C';
void setup() {
  Serial.begin(115200);
  while ( !Serial ) delay(100);   // wait for native usb
  Serial.println(F("Isa submarine"));
  Wire.begin();
  // DS18B20 setup
  sensors.begin();
  sensors.setResolution(11);
  // Setup air pressure/humidity sensor
 
  unsigned status;
  //status = bme.begin(BME280_CHIPID);
  status = bme.begin();
  if (!status) {
    Serial.println(F("Could not find a valid BME280 sensor, check wiring or "
                      "try a different address!"));
    Serial.print("SensorID was: 0x"); Serial.println(bme.sensorID(),16);
    Serial.print("Halting on missing BME280 pressure sensor\n");
    Serial.print("   ID of 0x56-0x58 represents a BMP 280,\n");
    Serial.print("   ID of 0x60 represents a BME 280.\n");
    while (1) delay(12);
  }
 
  /* Settings from datasheet. */
 
  bme.setSampling(Adafruit_BME280::MODE_NORMAL,     // Operating Mode.
                  Adafruit_BME280::SAMPLING_X2,     // Temp. oversampling
                  Adafruit_BME280::SAMPLING_X16,    // Pressure oversampling
                  Adafruit_BME280::SAMPLING_X1,     // Humidity oversampling
                  Adafruit_BME280::FILTER_X16,      // Filtering.
                  Adafruit_BME280::STANDBY_MS_500); // Standby time.
                 
  // Setup water pressure sensor Bar3XT
  sensor.init();
  sensor.setFluidDensity(997); // kg/m^3 (freshwater, 1029 for seawater)

  if (sensor.isInitialized()) {
    Serial.println("Sensor connected.");
  } else {
    Serial.println("Sensor not connected.");
  }
  pinMode(leakPin, INPUT);
  Serial.println("p=BME280 pressure, h=BME280 humidity, b=BME280 temperature, P=Bar3XT pressure, d=Bar3XT depth, T=Bar3XT temperature, l=leak (1->leak detected, 0->no leak), t=DS18B20 temperature");
}

void loop() {
  if (debug) {
    // BME280 data
    Serial.print(F("BME280 temperature = "));
    Serial.print(bme.readTemperature());
    Serial.println(" *C");
    Serial.print(F("BME280 pressure = "));
    Serial.print(bme.readPressure());
    Serial.println(" Pa");
    Serial.print(F("BME280 humidity = "));
    Serial.print(bme.readHumidity());
    Serial.println(" %");
    Serial.print(F("BME280 approx altitude = "));
    Serial.print(bme.readAltitude(1013.25)); // Adjusted to local forecast!
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
      Serial.print(bme.readPressure());
      Serial.print(",h:");
      Serial.print(bme.readHumidity());
      Serial.print(",b:");
      Serial.print(bme.readTemperature());
      sensor.read();
      Serial.print(",P:");
      Serial.print(sensor.pressure());
      Serial.print(",d:");
      Serial.print(sensor.depth());
      Serial.print(",T:");
      Serial.print(sensor.temperature());
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
