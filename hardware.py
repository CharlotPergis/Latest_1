import requests
import time
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD

# =====================
# CONFIG
# =====================
FLASK_URL = "http://127.0.0.1:5000/api/simulate"

GREEN_LED = 17
RED_LED = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(GREEN_LED, GPIO.OUT)
GPIO.setup(RED_LED, GPIO.OUT)

# =====================
# LCD (20x4 I2C)
# =====================
lcd = CharLCD(
    i2c_expander='PCF8574',
    address=0x27,      # change if needed (0x3F common alternative)
    cols=20,
    rows=4
)

lcd.clear()

# =====================
# BLINK CONTROL
# =====================
blink_state = False

# =====================
# UPDATE OUTPUTS
# =====================
def update_outputs(data, blink_state):
    temp = data.get("temperature", 0)
    current = data.get("current", 0)
    state = data.get("breakerState", "Unknown")
    mode = data.get("simulation_mode", "")

    # =====================
    # LED LOGIC
    # =====================
    if state == "Overheating":
        GPIO.output(GREEN_LED, 0)
        GPIO.output(RED_LED, blink_state)  # BLINKING RED

    elif state == "Overload":
        GPIO.output(GREEN_LED, 0)
        GPIO.output(RED_LED, 1)  # SOLID RED

    else:
        GPIO.output(GREEN_LED, 1)
        GPIO.output(RED_LED, 0)

    # =====================
    # LCD 20x4 DISPLAY
    # =====================
    lcd.clear()

    lcd.write_string(f"STATE: {state}\n")
    lcd.write_string(f"TEMP : {temp:.1f} C\n")
    lcd.write_string(f"CURR : {current:.1f} A\n")
    lcd.write_string(f"MODE : {mode}")


# =====================
# MAIN LOOP
# =====================
def run():
    global blink_state

    print("🚀 Hardware running (20x4 LCD + Blink RED mode)")

    while True:
        try:
            response = requests.get(FLASK_URL, timeout=5)
            data = response.json()

            # toggle blink only for overheating
            if data.get("breakerState") == "Overheating":
                blink_state = not blink_state
            else:
                blink_state = False

            update_outputs(data, blink_state)

            print(
                f"[{data['simulation_mode']}] "
                f"{data['temperature']}°C | {data['current']}A | {data['breakerState']}"
            )

        except Exception as e:
            print("❌ Connection error:", e)

            GPIO.output(GREEN_LED, 0)
            GPIO.output(RED_LED, 0)

            lcd.clear()
            lcd.write_string("NO SIGNAL\nCHECK FLASK")

        time.sleep(0.7)  # faster refresh for smoother blinking


# =====================
# CLEAN EXIT
# =====================
if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        GPIO.cleanup()
        lcd.clear()