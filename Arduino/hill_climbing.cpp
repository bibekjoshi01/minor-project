
#include <WiFi.h>
#include <AccelStepper.h>

// ------------------- WiFi -------------------
const char *ssid = "2_4G_SSID";
const char *password = "WiFiPassword";

// ------------------- Stepper -------------------
#define DIR_PIN 15
#define STEP_PIN 2
#define STEP_MODE AccelStepper::DRIVER
#define STEPS_PER_DEGREE 5

AccelStepper stepper(STEP_MODE, STEP_PIN, DIR_PIN);

// ------------------- Parameters -------------------
#define BUTTON_PIN 0

const int STEP_ANGLE = 5;         // hill-climb step size (degrees)
const int SAMPLES_PER_POINT = 20; // RSSI averaging
const int SETTLE_TIME_MS = 500;

#define HYSTERESIS_COUNT 3
#define SCAN_COOLDOWN 1800000UL // 30 minutes

int RSSI_THRESHOLD = -72; // adaptive later

// ------------------- State -------------------
int currentAngle = 0;
unsigned long lastScanTime = 0;
int lowRssiCounter = 0;

// ------------------- WiFi -------------------
void connectWiFi()
{
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);

    Serial.print("Connecting WiFi");
    unsigned long start = millis();
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
        if (millis() - start > 15000)
        {
            Serial.println("\nWiFi failed");
            return;
        }
    }
    Serial.println("\nWiFi connected");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
}

// ------------------- RSSI -------------------
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

// ------------------- Motor -------------------
void rotateTo(int targetAngle)
{
    targetAngle = (targetAngle + 360) % 360;

    int delta = targetAngle - currentAngle;
    int steps = delta * STEPS_PER_DEGREE;

    stepper.move(steps);
    unsigned long start = millis();
    while (stepper.distanceToGo() != 0)
    {
        stepper.run();
        if (millis() - start > 5000)
            break; // safety
    }

    currentAngle = targetAngle;
    delay(SETTLE_TIME_MS);
}

// ------------------- Hill Climbing -------------------
int hillClimb()
{
    Serial.println("Starting hill-climbing...");

    int currentRSSI = measureRSSI();
    bool improved = true;
    int iterations = 0;

    while (improved && iterations < 100)
    {
        iterations++;
        improved = false;

        int leftAngle = currentAngle - STEP_ANGLE;
        int rightAngle = currentAngle + STEP_ANGLE;

        rotateTo(leftAngle);
        int leftRSSI = measureRSSI();

        rotateTo(rightAngle);
        int rightRSSI = measureRSSI();

        Serial.print("Angle ");
        Serial.print(currentAngle);
        Serial.print(" | RSSI ");
        Serial.println(currentRSSI);

        if (leftRSSI > currentRSSI && leftRSSI >= rightRSSI)
        {
            rotateTo(leftAngle);
            currentRSSI = leftRSSI;
            improved = true;
        }
        else if (rightRSSI > currentRSSI)
        {
            rotateTo(rightAngle);
            currentRSSI = rightRSSI;
            improved = true;
        }
    }

    Serial.print("Hill-climb converged at angle ");
    Serial.print(currentAngle);
    Serial.print(" | RSSI ");
    Serial.println(currentRSSI);

    RSSI_THRESHOLD = currentRSSI - 10; // adaptive threshold
    Serial.print("New RSSI threshold: ");
    Serial.println(RSSI_THRESHOLD);

    return currentAngle;
}

// ------------------- Setup -------------------
void setup()
{
    Serial.begin(115200);
    delay(2000);
    Serial.println("BOOT OK");

    pinMode(BUTTON_PIN, INPUT_PULLUP);

    stepper.setMaxSpeed(1000);
    stepper.setAcceleration(500);

    connectWiFi();

    rotateTo(0);
    hillClimb();
    lastScanTime = millis();
}

// ------------------- Loop -------------------
void loop()
{
    unsigned long now = millis();

    bool buttonPressed = (digitalRead(BUTTON_PIN) == LOW);
    int avgRSSI = measureRSSI();

    if (avgRSSI < RSSI_THRESHOLD)
        lowRssiCounter++;
    else
        lowRssiCounter = 0;

    bool rssiTrigger = (lowRssiCounter >= HYSTERESIS_COUNT);
    bool cooldownPassed = (now - lastScanTime >= SCAN_COOLDOWN);

    if ((rssiTrigger && cooldownPassed) || buttonPressed)
    {
        Serial.println("Re-triggering hill-climb...");
        hillClimb();
        lastScanTime = millis();
        lowRssiCounter = 0;
    }

    Serial.print("Current RSSI: ");
    Serial.println(avgRSSI);
    Serial.print("Minutes since last climb: ");
    Serial.println((now - lastScanTime) / 60000);

    delay(1000);
}
