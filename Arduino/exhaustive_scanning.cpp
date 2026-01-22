#include <WiFi.h>
#include <AccelStepper.h>

// WiFi Configuration
// -----------------------------------
const char *ssid = "2_4G_SSID";
const char *password = "WiFiPassword";

// Stepper Configuration
// -----------------------------------
#define DIR_PIN 15
#define STEP_PIN 2
#define STEPS_PER_DEGREE 5
#define STEP_MODE AccelStepper::DRIVER
AccelStepper stepper(STEP_MODE, STEP_PIN, DIR_PIN);

const int SETTLE_TIME_MS = 500; // antenna settle time after rotation

// Scan Configuration
// -----------------------------------
const int ANGLE_STEP_COARSE = 10; // coarse scan step in degrees
const int ANGLE_STEP_FINE = 2;    // fine scan step in degrees
const int SAMPLES_PER_POINT = 20; // RSSI averaging per angle
const int HYSTERESIS_COUNT = 3;   // consecutive low RSSI measurements
#define BUTTON_PIN 0              // manual reset button
#define SCAN_COOLDOWN 1800000     // 30 min cooldown in ms
int RSSI_THRESHOLD = -72;         // initial threshold (adaptive)

// Global Variables
// -----------------------------------
unsigned long lastScanTime = 0;
int currentAngle = 0;
int lowRssiCounter = 0;

struct ScanPoint
{
    int angle;
    int rssi;
    unsigned long timestamp;
};

#define MAX_POINTS 72
ScanPoint scanData[MAX_POINTS];
int scanIndex = 0;

// -------------------------- WiFi Connection --------------------------
void connectWiFi()
{
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);

    Serial.print("Connecting to WiFi...");
    unsigned long start = millis();
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
        if (millis() - start > 15000)
        { // 15s timeout
            Serial.println("\nWiFi connection failed!");
            return;
        }
    }
    Serial.println("\nConnected to WiFi!");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
}

// -------------------------- RSSI Measurement --------------------------
int measureRSSI(int samples = SAMPLES_PER_POINT)
{
    long sum = 0;
    for (int i = 0; i < samples; i++)
    {
        sum += WiFi.RSSI();
        delay(20);
    }
    return sum / samples;
}

// -------------------------- Motor Rotation --------------------------
void rotateTo(int targetAngle)
{
    int deltaAngle = targetAngle - currentAngle;
    int steps = deltaAngle * STEPS_PER_DEGREE;

    stepper.move(steps);
    unsigned long start = millis();
    while (stepper.distanceToGo() != 0)
    {
        stepper.run();
        if (millis() - start > 5000)
            break;
    }
    currentAngle = targetAngle;
    delay(SETTLE_TIME_MS);
}

// -------------------------- Exhaustive Scan --------------------------
int performExhaustiveScan(int stepSize = ANGLE_STEP_COARSE)
{
    int bestAngle = 0;
    int bestRSSI = -1000;
    scanIndex = 0;

    for (int angle = 0; angle < 180; angle += stepSize)
    {
        rotateTo(angle);
        int rssi = measureRSSI();
        scanData[scanIndex++] = {angle, rssi, millis()};

        Serial.print("Angle: ");
        Serial.print(angle);
        Serial.print(" | RSSI: ");
        Serial.println(rssi);

        if (rssi > bestRSSI)
        {
            bestRSSI = rssi;
            bestAngle = angle;
        }
    }

    Serial.print("Best angle: ");
    Serial.print(bestAngle);
    Serial.print(" | RSSI: ");
    Serial.println(bestRSSI);

    rotateTo(bestAngle);
    return bestAngle;
}

// -------------------------- Adaptive Threshold --------------------------
void updateThreshold()
{
    int maxRssi = -1000;
    for (int i = 0; i < scanIndex; i++)
    {
        if (scanData[i].rssi > maxRssi)
            maxRssi = scanData[i].rssi;
    }
    RSSI_THRESHOLD = maxRssi - 10; // adaptive: 10dBm below max
    Serial.print("Adaptive RSSI threshold set to: ");
    Serial.println(RSSI_THRESHOLD);
}

// -------------------------- Setup --------------------------
void setup()
{
    Serial.begin(115200);
    delay(2000);
    Serial.println("BOOT OK!");

    stepper.setMaxSpeed(1000);
    stepper.setAcceleration(500);

    pinMode(BUTTON_PIN, INPUT_PULLUP);

    connectWiFi();

    // Perform initial exhaustive scan
    int optimalAngle = performExhaustiveScan();
    lastScanTime = millis();

    updateThreshold();
    Serial.print("Antenna aligned at angle: ");
    Serial.println(optimalAngle);
}

// -------------------------- Main Loop --------------------------
void loop()
{
    unsigned long now = millis();

    // Manual button (active LOW)
    bool buttonPressed = (digitalRead(BUTTON_PIN) == LOW);

    // Measure RSSI
    int avgRssi = measureRSSI();
    if (avgRssi < RSSI_THRESHOLD)
        lowRssiCounter++;
    else
        lowRssiCounter = 0;

    bool rssiTrigger = (lowRssiCounter >= HYSTERESIS_COUNT);
    bool cooldownPassed = (now - lastScanTime >= SCAN_COOLDOWN);

    if ((rssiTrigger && cooldownPassed) || buttonPressed)
    {
        Serial.println("Triggering exhaustive scan...");

        rotateTo(0); // reset antenna
        int optimalAngle = performExhaustiveScan(ANGLE_STEP_COARSE);

        // fine scan around best coarse angle
        int fineStart = max(0, optimalAngle - ANGLE_STEP_COARSE);
        int fineEnd = min(359, optimalAngle + ANGLE_STEP_COARSE);
        scanIndex = 0; // reset for fine scan
        int bestFineAngle = optimalAngle;
        int bestFineRSSI = -1000;

        for (int angle = fineStart; angle <= fineEnd; angle += ANGLE_STEP_FINE)
        {
            rotateTo(angle);
            int rssi = measureRSSI();
            scanData[scanIndex++] = {angle, rssi, millis()};
            if (rssi > bestFineRSSI)
            {
                bestFineRSSI = rssi;
                bestFineAngle = angle;
            }
        }
        rotateTo(bestFineAngle);

        lastScanTime = millis();
        updateThreshold();

        Serial.print("Scan complete. Optimal angle: ");
        Serial.print(bestFineAngle);
        Serial.print(" | RSSI: ");
        Serial.println(bestFineRSSI);
    }

    // Debug output
    Serial.print("Current RSSI: ");
    Serial.println(avgRssi);
    Serial.print("Time since last scan (min): ");
    Serial.println((now - lastScanTime) / 60000);

    delay(1000); // loop stability
}
