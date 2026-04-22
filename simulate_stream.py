import requests
import time
import random

FLASK_URL = "http://127.0.0.1:5000/api/check-alert"

def generate_sensor_data(i):
    """
    Raw sensor simulation (like ESP32 / Raspberry Pi input)
    """
    temperature = 60 + i * 0.15 + random.uniform(-2, 2)
    current = 15 + random.uniform(-5, 10)
    ambient = 35

    return {
        "temperature": round(temperature, 2),
        "current": round(current, 2),
        "ambient_temp_c": ambient
    }


def run():
    print("🚀 Starting RAW sensor simulation (Design A)...")

    for i in range(1000):
        payload = generate_sensor_data(i)

        try:
            response = requests.post(FLASK_URL, json=payload, timeout=5)
            print(f"[{i}] Sent:", payload)
            print("Response:", response.json())
            print("-" * 50)

        except requests.exceptions.ConnectionError:
            print("❌ Flask server not running")
            break

        time.sleep(1)


if __name__ == "__main__":
    run()