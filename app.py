from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
import joblib
from dataclasses import dataclass
import pandas as pd
from flask_mail import Mail, Message
from flask_cors import CORS
import json
import time
import os
from datetime import datetime
from collections import deque
import random

latest_data_store = {}

# ----------------------
# DEFINE MODELS
# ----------------------
hotspot_model = None
overload_model = None

# ----------------------
# Setup Flask
# ----------------------
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)

# Email config
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'breaker.monitor.system@gmail.com'
app.config['MAIL_PASSWORD'] = 'kzng lhzr elww gyyu'
app.config['MAIL_DEFAULT_SENDER'] = 'breaker.monitor.system@gmail.com'
app.config['MAIL_DEBUG'] = True

try:
    mail = Mail(app)
    print("✓ Email service initialized")
except Exception as e:
    print(f"✗ Email initialization error: {e}")
    mail = None

# ----------------------
# Load Models
# ----------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    hotspot_path = os.path.join(BASE_DIR, "ml", "hotspot_model.pkl")
    overload_path = os.path.join(BASE_DIR, "ml", "overload_model.pkl")

    if os.path.exists(hotspot_path):
        hotspot_model = joblib.load(hotspot_path)
        print("✓ Hotspot model loaded")
        if hasattr(hotspot_model, 'feature_names_in_'):
            print(f"  Hotspot expects {len(hotspot_model.feature_names_in_)} features")
    else:
        print("✗ hotspot_model.pkl NOT FOUND")

    if os.path.exists(overload_path):
        overload_model = joblib.load(overload_path)
        print("✓ Overload model loaded")
        if hasattr(overload_model, 'feature_names_in_'):
            print(f"  Overload expects {len(overload_model.feature_names_in_)} features")
    else:
        print("✗ overload_model.pkl NOT FOUND")

except Exception as e:
    print(f"✗ MODEL LOAD ERROR: {e}")

# ----------------------
# STREAMING SLOPE
# ----------------------
temp_buffer = deque(maxlen=10)
current_buffer = deque(maxlen=10)

def compute_slope(temp, current):
    temp_buffer.append(temp)
    current_buffer.append(current)
    if len(temp_buffer) < 2:
        return 0.0, 0.0
    dt = (len(temp_buffer) - 1)
    if dt == 0:
        return 0.0, 0.0
    thermal_slope = (temp_buffer[-1] - temp_buffer[0]) / dt * 5
    current_slope = (current_buffer[-1] - current_buffer[0]) / dt * 5
    return thermal_slope, current_slope

# ----------------------
# Time to Trip Calculation
# ----------------------
def calculate_time_to_trip(current_a, temperature_c, overload_prob, breaker_state):
    if breaker_state != "Overload":
        return None
    if overload_prob < 0.5:
        return None
    
    rated_current = 20.0
    overload_ratio = current_a / rated_current
    temp_factor = max(1.0, temperature_c / 65.0)
    
    if overload_ratio > 2.0:
        time_seconds = 3 / (overload_ratio * temp_factor)
        time_seconds = max(1, min(5, time_seconds))
        urgency = "CRITICAL"
    elif overload_ratio > 1.5:
        time_seconds = 20 / (overload_ratio * temp_factor)
        time_seconds = max(5, min(30, time_seconds))
        urgency = "HIGH"
    elif overload_ratio > 1.2:
        time_seconds = 60 / (overload_ratio * temp_factor)
        time_seconds = max(30, min(120, time_seconds))
        urgency = "MEDIUM"
    else:
        time_seconds = 180 / (overload_ratio * temp_factor)
        time_seconds = max(120, min(300, time_seconds))
        urgency = "LOW"
    
    if time_seconds < 60:
        time_str = f"{int(time_seconds)} seconds"
    elif time_seconds < 3600:
        minutes = time_seconds / 60
        time_str = f"{int(minutes)} minute{'s' if minutes >= 2 else ''}"
    else:
        hours = time_seconds / 3600
        time_str = f"{hours:.1f} hours"
    
    return {
        "seconds": time_seconds,
        "formatted": time_str,
        "urgency": urgency,
        "overload_ratio": overload_ratio
    }

# ----------------------
# Sensor Reading Dataclass
# ----------------------
@dataclass
class SensorReading:
    ambient_temp_c: float
    temperature_c: float
    temperature_rise_c: float
    current_a: float
    thermal_slope_c_per_5s: float
    current_slope_a_per_5s: float

temp_history = deque(maxlen=20)
current_history = deque(maxlen=20)

# ----------------------
# Prediction Function
# ----------------------
def predict_risk(reading: SensorReading) -> dict:
    global hotspot_model, overload_model, temp_history, current_history

    temp_history.append(reading.temperature_c)
    current_history.append(reading.current_a)

    def get_lag(data, lag):
        if len(data) > lag:
            return list(data)[-lag-1]
        return 0.0

    features = {
        "ambient_temp_c": reading.ambient_temp_c,
        "temperature_c": reading.temperature_c,
        "temperature_rise_c": reading.temperature_rise_c,
        "current_a": reading.current_a,
        "thermal_slope_c_per_5s": reading.thermal_slope_c_per_5s,
        "current_slope_a_per_5s": reading.current_slope_a_per_5s,
        
        "current_lag_1": get_lag(current_history, 1),
        "current_lag_2": get_lag(current_history, 2),
        "current_lag_3": get_lag(current_history, 3),
        "current_lag_4": get_lag(current_history, 4),
        "current_lag_5": get_lag(current_history, 5),
        "current_lag_6": get_lag(current_history, 6),
        "current_lag_7": get_lag(current_history, 7),
        "current_lag_8": get_lag(current_history, 8),
        "current_lag_9": get_lag(current_history, 9),
        "current_lag_10": get_lag(current_history, 10),
        
        "current_avg_3": sum(list(current_history)[-3:]) / min(len(current_history), 3) if len(current_history) > 0 else 0,
        "current_avg_5": sum(list(current_history)[-5:]) / min(len(current_history), 5) if len(current_history) > 0 else 0,
        "current_avg_10": sum(list(current_history)[-10:]) / min(len(current_history), 10) if len(current_history) > 0 else 0,
        
        "temp_lag_1": get_lag(temp_history, 1),
        "temp_lag_2": get_lag(temp_history, 2),
        "temp_lag_3": get_lag(temp_history, 3),
        "temp_lag_4": get_lag(temp_history, 4),
        "temp_lag_5": get_lag(temp_history, 5),
        
        "temp_avg_3": sum(list(temp_history)[-3:]) / min(len(temp_history), 3) if len(temp_history) > 0 else 0,
        "temp_avg_5": sum(list(temp_history)[-5:]) / min(len(temp_history), 5) if len(temp_history) > 0 else 0,
    }

    if hotspot_model is None or overload_model is None:
        if reading.temperature_c >= 72:
            hotspot_prob = min(0.95, (reading.temperature_c - 70) / 30)
            overload_prob = min(0.95, reading.current_a / 35)
        elif reading.current_a >= 19:
            hotspot_prob = min(0.7, reading.temperature_c / 100)
            overload_prob = min(0.85, (reading.current_a - 15) / 25)
        else:
            hotspot_prob = reading.temperature_c / 100 * 0.3
            overload_prob = reading.current_a / 40 * 0.3
        
        return {
            "hotspot_prob": hotspot_prob,
            "overload_prob": overload_prob,
            "hotspot_flag": int(hotspot_prob >= 0.75),
            "overload_flag": int(overload_prob >= 0.5),
            "composite_risk": 0.5 * hotspot_prob + 0.5 * overload_prob,
        }

    try:
        x_new = pd.DataFrame([features])
        
        if hasattr(hotspot_model, 'feature_names_in_'):
            hotspot_features = hotspot_model.feature_names_in_
            for feature in hotspot_features:
                if feature not in x_new.columns:
                    x_new[feature] = 0
            x_new_hotspot = x_new[hotspot_features]
            hotspot_prob = float(hotspot_model.predict_proba(x_new_hotspot)[0, 1])
        else:
            hotspot_prob = float(hotspot_model.predict_proba(x_new)[0, 1])
        
        if hasattr(overload_model, 'feature_names_in_'):
            overload_features = overload_model.feature_names_in_
            for feature in overload_features:
                if feature not in x_new.columns:
                    x_new[feature] = 0
            x_new_overload = x_new[overload_features]
            overload_prob = float(overload_model.predict_proba(x_new_overload)[0, 1])
        else:
            overload_prob = float(overload_model.predict_proba(x_new)[0, 1])
        
    except Exception as e:
        print(f"⚠️ ML Prediction error: {e}")
        if reading.temperature_c >= 72:
            hotspot_prob = 0.85
            overload_prob = 0.75
        elif reading.current_a >= 19:
            hotspot_prob = 0.6
            overload_prob = 0.7
        else:
            hotspot_prob = 0.1
            overload_prob = 0.1

    return {
        "hotspot_prob": hotspot_prob,
        "overload_prob": overload_prob,
        "hotspot_flag": int(hotspot_prob >= 0.75),
        "overload_flag": int(overload_prob >= 0.5),
        "composite_risk": 0.5 * hotspot_prob + 0.5 * overload_prob,
    }

# ----------------------
# Email Alert Function
# ----------------------
def send_breaker_alert(reading, risk, alert_type, time_to_trip=None):
    if mail is None:
        return False, "Email service not configured"
    
    recipients = ['gwenlykapergis@gmail.com']
    
    time_to_trip_text = ""
    if time_to_trip and alert_type in ["overheating", "prevention"]:
        time_to_trip_text = f"\nEstimated Time to Trip: {time_to_trip['formatted']}\nUrgency: {time_to_trip['urgency']}"
    
    if alert_type == "overheating":
        subject = "🔥 CRITICAL: Breaker Overheating Alert!"
        body = f"""IMMEDIATE ACTION REQUIRED

BREAKER OVERHEATING DETECTED!

Temperature: {reading.temperature_c:.1f}°C
Current: {reading.current_a:.1f}A
Hotspot Probability: {risk['hotspot_prob']*100:.1f}%
Overload Probability: {risk['overload_prob']*100:.1f}%
{time_to_trip_text}

Action: Isolate circuit immediately!

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
    elif alert_type == "prevention":
        subject = "⚠️ PREVENTION: Potential Overload Detected!"
        body = f"""PREVENTIVE ACTION RECOMMENDED

POTENTIAL OVERLOAD DEVELOPING!

Temperature: {reading.temperature_c:.1f}°C
Current: {reading.current_a:.1f}A
Hotspot Probability: {risk['hotspot_prob']*100:.1f}%
Overload Probability: {risk['overload_prob']*100:.1f}%
{time_to_trip_text}

Action: Reduce load by 15-20%

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
    else:
        return False, "Unknown alert type"
    
    try:
        msg = Message(subject=subject, sender=app.config['MAIL_USERNAME'], recipients=recipients)
        msg.body = body
        mail.send(msg)
        print(f"✓ Email sent: {subject}")
        return True, "Alert sent"
    except Exception as e:
        print(f"✗ Email error: {e}")
        return False, str(e)

# ----------------------
# Alert Tracking
# ----------------------
last_alert_time = {}
ALERT_COOLDOWN_SECONDS = 300

def should_send_alert(alert_type):
    current_time = time.time()
    if alert_type in last_alert_time:
        if current_time - last_alert_time[alert_type] < ALERT_COOLDOWN_SECONDS:
            return False
    last_alert_time[alert_type] = current_time
    return True

# ----------------------
# AUTO-CYCLING SIMULATION
# ----------------------
sim_temp = 35
sim_current = 12
simulation_mode = "normal"
simulation_step = 0
last_mode_change = time.time()

STATE_RANGES = {
    "normal": {"temp_range": (30, 45), "current_range": (8, 15)},
    "overload": {"temp_range": (55, 68), "current_range": (21, 28)},
    "overheating": {"temp_range": (75, 95), "current_range": (30, 38)}
}

def generate_values_for_state(state, step):
    ranges = STATE_RANGES[state]
    if state == "normal":
        temp = random.uniform(*ranges["temp_range"])
        current = random.uniform(*ranges["current_range"])
    elif state == "overload":
        progress = min(1.0, step / 30)
        temp = ranges["temp_range"][0] + (ranges["temp_range"][1] - ranges["temp_range"][0]) * progress
        current = ranges["current_range"][0] + (ranges["current_range"][1] - ranges["current_range"][0]) * progress
        temp += random.uniform(-2, 2)
        current += random.uniform(-1, 1)
    else:
        progress = min(1.0, step / 40)
        temp = ranges["temp_range"][0] + (ranges["temp_range"][1] - ranges["temp_range"][0]) * progress
        current = ranges["current_range"][0] + (ranges["current_range"][1] - ranges["current_range"][0]) * progress
        temp += random.uniform(-3, 3)
        current += random.uniform(-1.5, 1.5)
    
    return round(max(20, min(100, temp)), 1), round(max(3, min(45, current)), 1)

def get_simulation_scenario():
    global simulation_mode, simulation_step, sim_temp, sim_current, last_mode_change
    
    current_time = time.time()
    if current_time - last_mode_change > 30:
        states = ["normal", "overload", "overheating"]
        current_index = states.index(simulation_mode) if simulation_mode in states else 0
        next_index = (current_index + 1) % 3
        simulation_mode = states[next_index]
        simulation_step = 0
        last_mode_change = current_time
        print(f"\n🔄 Auto-cycling to: {simulation_mode.upper()}")
    
    simulation_step += 1
    if simulation_step > 50:
        simulation_step = 50
    
    sim_temp, sim_current = generate_values_for_state(simulation_mode, simulation_step)
    return simulation_mode

# ----------------------
# ROUTES - ALL NECESSARY ROUTES
# ----------------------

# Main dashboard route
@app.route('/')
def index():
    return render_template('index.html')

# Full history page route
@app.route('/full_history')
@app.route('/full_history.html')
def full_history():
    return render_template('full_history.html')

# Redirect for index.html
@app.route('/index.html')
def index_html():
    return redirect(url_for('index'))

# Static files route (CSS, JS)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# API Routes
@app.route("/api/simulate", methods=["GET"])
def simulate():
    global sim_temp, sim_current, latest_data_store

    current_mode = get_simulation_scenario()
    thermal_slope, current_slope = compute_slope(sim_temp, sim_current)

    reading = SensorReading(
        ambient_temp_c=25.0,
        temperature_c=sim_temp,
        temperature_rise_c=sim_temp - 25.0,
        current_a=sim_current,
        thermal_slope_c_per_5s=thermal_slope,
        current_slope_a_per_5s=current_slope
    )

    try:
        risk = predict_risk(reading)
        hotspot_prob = risk['hotspot_prob']
        overload_prob = risk['overload_prob']
        
        if sim_temp >= 75 or hotspot_prob > 0.7:
            breaker_state = "Overheating"
        elif sim_current >= 21 or overload_prob > 0.6:
            breaker_state = "Overload"
        else:
            breaker_state = "Normal"
        
        time_to_trip = None
        if breaker_state == "Overload":
            time_to_trip = calculate_time_to_trip(sim_current, sim_temp, overload_prob, breaker_state)
        
        if breaker_state == "Overheating":
            clean_status = "🔥 CRITICAL - Immediate action required!"
        elif breaker_state == "Overload":
            clean_status = "⚠️ WARNING - Reduce load to prevent damage!"
        else:
            clean_status = "✓ System operating normally"
        
    except Exception as e:
        print(f"Error: {e}")
        breaker_state = "Normal"
        time_to_trip = None
        clean_status = "✓ System operating normally"
        risk = {"hotspot_prob": 0, "overload_prob": 0, "composite_risk": 0}
    
    latest_data_store = {
        "temperature": sim_temp,
        "current": sim_current,
        "breakerState": breaker_state,
        "systemStatus": clean_status,
        "time": datetime.now().strftime('%H:%M:%S'),
        "date": datetime.now().strftime('%Y-%m-%d'),
        "time_to_trip": time_to_trip,
        "simulation_mode": current_mode,
        "ml_predictions": {
            "hotspot_prob": round(risk.get('hotspot_prob', 0), 3),
            "overload_prob": round(risk.get('overload_prob', 0), 3),
            "composite_risk": round(risk.get('composite_risk', 0), 3)
        }
    }
    
    print(f"[{current_mode.upper():12}] Temp: {sim_temp:5.1f}°C | Current: {sim_current:5.1f}A | State: {breaker_state}")
    return jsonify(latest_data_store)

@app.route("/api/check-alert", methods=["POST"])
def check_alert():
    try:
        data = request.json
        temperature = data.get('temperature', 0)
        current = data.get('current', 0)
        
        reading = SensorReading(
            ambient_temp_c=25.0,
            temperature_c=temperature,
            temperature_rise_c=temperature - 25.0,
            current_a=current,
            thermal_slope_c_per_5s=0,
            current_slope_a_per_5s=0
        )
        
        risk = predict_risk(reading)
        
        if temperature >= 75 or risk['hotspot_prob'] > 0.7:
            alert_type = "overheating"
        elif current >= 21 or risk['overload_prob'] > 0.6:
            alert_type = "prevention"
        else:
            return jsonify({"success": True, "alert_sent": False, "messages": ["No alert needed"]})
        
        if should_send_alert(alert_type):
            time_to_trip = calculate_time_to_trip(current, temperature, risk['overload_prob'], "Overload")
            success, msg = send_breaker_alert(reading, risk, alert_type, time_to_trip)
            return jsonify({"success": True, "alert_sent": success, "messages": [msg], "alert_type": alert_type})
        else:
            return jsonify({"success": True, "alert_sent": False, "messages": ["Alert on cooldown"]})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/latest-data", methods=['GET'])
def get_latest_data():
    if not latest_data_store:
        return jsonify({"temperature": 0, "current": 0, "breakerState": "Unknown", "systemStatus": "Waiting..."})
    return jsonify(latest_data_store)

@app.route("/test-overheating-alert")
def test_overheating_alert():
    test_reading = SensorReading(ambient_temp_c=25.0, temperature_c=85.0, temperature_rise_c=60.0, current_a=32.0, thermal_slope_c_per_5s=25.0, current_slope_a_per_5s=5.0)
    risk = predict_risk(test_reading)
    success, msg = send_breaker_alert(test_reading, risk, "overheating", None)
    return f"Test alert: {msg}"

@app.route("/test-overload-alert")
def test_overload_alert():
    test_reading = SensorReading(ambient_temp_c=25.0, temperature_c=62.0, temperature_rise_c=37.0, current_a=25.0, thermal_slope_c_per_5s=10.0, current_slope_a_per_5s=3.0)
    risk = predict_risk(test_reading)
    time_to_trip = calculate_time_to_trip(25.0, 62.0, risk['overload_prob'], "Overload")
    success, msg = send_breaker_alert(test_reading, risk, "prevention", time_to_trip)
    return f"Test alert: {msg}"

@app.route("/api/health", methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "models_loaded": hotspot_model is not None, "email_configured": mail is not None})

# 404 error handler - redirects to dashboard
@app.errorhandler(404)
def page_not_found(e):
    print(f"404 error: {e}")
    return redirect(url_for('index'))

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🔌 BREAKER MONITORING SYSTEM - AUTO-CYCLING MODE")
    print("="*50)
    print("Server: http://127.0.0.1:5000")
    print("Routes:")
    print("  - Dashboard: http://127.0.0.1:5000/")
    print("  - Full History: http://127.0.0.1:5000/full_history")
    print("  - API Health: http://127.0.0.1:5000/api/health")
    print("Auto-cycles every 30 seconds: Normal → Overload → Overheating")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)