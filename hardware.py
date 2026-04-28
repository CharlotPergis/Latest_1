import requests
import time
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD

# ======================
# GPIO SETUP
# ======================
GREEN_LED = 17
RED_LED = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(GREEN_LED, GPIO.OUT)
GPIO.setup(RED_LED, GPIO.OUT)

# ======================
# LCD SETUP (4x20 I2C)
# ======================
lcd = CharLCD('PCF8574', 0x27, cols=20, rows=4)
lcd.clear()

FLASK_URL = "http://127.0.0.1:5000/api/simulate"


# ======================
# LED CONTROL
# ======================
def set_leds(state, blink=False):
    if state == "Overheating" and blink:
        GPIO.output(GREEN_LED, 0)
        GPIO.output(RED_LED, 1)
        time.sleep(0.3)
        GPIO.output(RED_LED, 0)
        time.sleep(0.3)
    elif state == "Overheating":
        GPIO.output(GREEN_LED, 0)
        GPIO.output(RED_LED, 1)
    elif state == "Overload":
        GPIO.output(GREEN_LED, 0)
        GPIO.output(RED_LED, 1)
    else:
        GPIO.output(GREEN_LED, 1)
        GPIO.output(RED_LED, 0)


# ======================
# LCD DISPLAY
# ======================
def update_lcd(temp, current, state):
    lcd.clear()
    lcd.write_string(f"Temp: {temp:.1f}C")
    lcd.cursor_pos = (1, 0)
    lcd.write_string(f"Current: {current:.1f}A")
    lcd.cursor_pos = (2, 0)
    lcd.write_string(f"State: {state}")


# ======================
# MAIN LOOP
# ======================
def run():
    print("System running...")

    while True:
        try:
            res = requests.get(FLASK_URL, timeout=5)
            data = res.json()

            temp = data["temperature"]
            current = data["current"]
            state = data["breakerState"]

            # LED behavior
            if state == "Overheating":
                set_leds(state, blink=True)
            else:
                set_leds(state)

            # LCD update
            update_lcd(temp, current, state)

            print(f"{state} | {temp}°C | {current}A")

        except Exception as e:
            print("Error:", e)
            GPIO.output(GREEN_LED, 0)
            GPIO.output(RED_LED, 1)
            lcd.clear()
            lcd.write_string("SYSTEM ERROR")

        time.sleep(1)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        GPIO.cleanup()
        lcd.clear()
