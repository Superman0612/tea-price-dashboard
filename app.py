import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_tea_model.pkl')
model = None

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("⚠️  Model file not found. Using mock predictions.")
except Exception as e:
    print(f"⚠️  Error loading model: {e}. Using mock predictions.")


MONTH_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

RAINFALL_MAP = {'Low': 0, 'Medium': 1, 'High': 2}


def build_features(usd_kes, rainfall_code, month_num):
    """
    Build the feature vector expected by the model.
    Lag features are approximated using placeholder averages.
    """
    # Placeholder lag values (representative Kenyan tea market averages)
    price_lag1 = 2.85
    price_lag2 = 2.80
    price_lag3 = 2.75
    price_lag4 = 2.70
    volume_lag1 = 12500.0
    fx_lag1     = usd_kes * 0.998   # slight prior period FX

    # fx_return: percentage change in exchange rate
    fx_return = (usd_kes - fx_lag1) / (fx_lag1 + 1e-9)
    features = np.array([[
        price_lag1,
        price_lag2,
        price_lag3,
        price_lag4,
        volume_lag1,
        usd_kes,
        fx_lag1,
        fx_return,
        month_num,
        rainfall_code
    ]])
    return features
    


def mock_predict(usd_kes, rainfall_code, month_num):
    """Deterministic mock prediction when model is unavailable."""
    base = 2.75
    fx_effect = (usd_kes - 130) * 0.005
    rain_effect = (rainfall_code - 1) * 0.08
    season_effect = np.sin((month_num - 3) * np.pi / 6) * 0.12
    price = round(base + fx_effect + rain_effect + season_effect, 4)
    return max(1.5, min(6.0, price))


def get_market_signal(price, usd_kes, rainfall_code):
    baseline = 2.80
    if price > baseline * 1.05:
        return "Bullish"
    elif price < baseline * 0.95:
        return "Bearish"
    return "Neutral"


def get_risk_level(price, usd_kes, rainfall_code):
    volatility_score = 0
    if usd_kes > 145 or usd_kes < 115:
        volatility_score += 2
    if rainfall_code == 0:
        volatility_score += 2
    elif rainfall_code == 2:
        volatility_score += 1
    if price > 3.5 or price < 2.0:
        volatility_score += 1

    if volatility_score >= 4:
        return "High"
    elif volatility_score >= 2:
        return "Medium"
    return "Low"


def get_insight(signal, risk, rainfall_label, month_label):
    insights = {
        ("Bullish", "Low"):    f"Strong buying conditions in {month_label}. {rainfall_label} rainfall supports healthy crop yields, and exchange rates favour exporters.",
        ("Bullish", "Medium"): f"Positive price momentum in {month_label}. Monitor FX volatility closely — {rainfall_label} rainfall offers partial support.",
        ("Bullish", "High"):   f"Prices trending up but caution advised. High uncertainty from FX or {rainfall_label.lower()} rainfall conditions in {month_label}.",
        ("Neutral", "Low"):    f"Stable market in {month_label}. {rainfall_label} rainfall and steady FX suggest predictable conditions — good for forward contracts.",
        ("Neutral", "Medium"): f"Mixed signals in {month_label}. Hold positions and watch for FX shifts. {rainfall_label} rainfall keeps supply outlook uncertain.",
        ("Neutral", "High"):   f"Volatile conditions in {month_label} despite neutral pricing. Consider hedging strategies given current {rainfall_label.lower()} rainfall.",
        ("Bearish", "Low"):    f"Prices under pressure in {month_label}, but low risk means the dip may be short-lived. Watch auction volumes closely.",
        ("Bearish", "Medium"): f"Downward price pressure in {month_label}. {rainfall_label} rainfall and FX headwinds are compounding bearish sentiment.",
        ("Bearish", "High"):   f"Challenging outlook for {month_label}. {rainfall_label} rainfall combined with unfavourable FX is creating significant downside risk.",
    }
    return insights.get((signal, risk), f"Market conditions in {month_label} require careful monitoring of both FX and rainfall trends.")

def create_plot(predicted_price):
    # dummy past data (smooth look ke liye)
    past_prices = [2.5, 2.6, 2.7, 2.8]
    predicted = past_prices + [predicted_price]

    plt.figure(figsize=(5,3))
    plt.plot(past_prices, label='Past')
    plt.plot(range(len(predicted)), predicted, label='Predicted')
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def index():
    months = list(MONTH_MAP.keys())
    rainfall_options = list(RAINFALL_MAP.keys())
    return render_template('index.html', months=months, rainfall_options=rainfall_options)


@app.route('/predict', methods=['POST'])
def predict():
    errors = {}

    # --- Validate inputs ---
    try:
        usd_kes = float(request.form['usd_kes'])
        if not (80 <= usd_kes <= 200):
            errors['usd_kes'] = "USD/KES rate must be between 80 and 200."
    except (ValueError, KeyError):
        errors['usd_kes'] = "Please enter a valid exchange rate."
        usd_kes = None

    rainfall_label = request.form.get('rainfall', '')
    if rainfall_label not in RAINFALL_MAP:
        errors['rainfall'] = "Please select a valid rainfall level."
        rainfall_code = None
    else:
        rainfall_code = RAINFALL_MAP[rainfall_label]

    month_label = request.form.get('month', '')
    if month_label not in MONTH_MAP:
        errors['month'] = "Please select a valid month."
        month_num = None
    else:
        month_num = MONTH_MAP[month_label]

    if errors:
        months = list(MONTH_MAP.keys())
        rainfall_options = list(RAINFALL_MAP.keys())
        return render_template('index.html',
                               months=months,
                               rainfall_options=rainfall_options,
                               errors=errors,
                               form_data=request.form)

    # --- Predict ---
    try:
        if model is not None:
            features = build_features(usd_kes, rainfall_code, month_num)
            predicted_price = float(model.predict(features)[0])
        else:
            predicted_price = mock_predict(usd_kes, rainfall_code, month_num)
    except Exception as e:
        predicted_price = mock_predict(usd_kes, rainfall_code, month_num)
        print(f"Prediction error: {e}")

    signal   = get_market_signal(predicted_price, usd_kes, rainfall_code)
    risk     = get_risk_level(predicted_price, usd_kes, rainfall_code)
    insight  = get_insight(signal, risk, rainfall_label, month_label)
    graph = create_plot(predicted_price)

    result = {
        'predicted_price': f"{predicted_price:.4f}",
        'signal': signal,
        'risk': risk,
        'insight': insight,
        'usd_kes': usd_kes,
        'rainfall': rainfall_label,
        'month': month_label,
    }

    months = list(MONTH_MAP.keys())
    rainfall_options = list(RAINFALL_MAP.keys())
    return render_template('index.html',
                           months=months,
                           rainfall_options=rainfall_options,
                           graph=graph,
                           result=result,
                           form_data=request.form)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

