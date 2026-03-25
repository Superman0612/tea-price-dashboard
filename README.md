# 🍃 Kenyan Tea Market Intelligence Dashboard

An end-to-end Machine Learning project that predicts Kenyan tea prices based on macroeconomic and environmental factors.

---

## 📌 Project Overview

This project models tea price dynamics using:

* Historical price trends
* USD/KES exchange rate (FX exposure)
* Rainfall conditions (supply-side impact)

It provides:

* 📈 Price Prediction
* 📊 Market Signal (Bullish / Bearish / Neutral)
* ⚠️ Risk Level
* 💡 Market Insights

---

## ⚙️ Tech Stack

* Python (Pandas, NumPy, Scikit-learn, XGBoost)
* Flask (Backend)
* HTML, CSS (Frontend)
* Matplotlib (Visualization)

---

## 🧠 Machine Learning Approach

* Feature Engineering:

  * Lag features
  * Moving averages
  * Volatility
  * FX returns
  * Trend indicators

* Models Used:

  * Linear Regression (Baseline)
  * Random Forest
  * XGBoost (Final Model)

---

## 🚀 How to Run Locally

```bash
git clone <your-repo-link>
cd project-folder

pip install -r requirements.txt
python app.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

---

## Live Link:

```
https://tea-price.onrender.com
```

## 📊 Key Insights

* Tea prices are influenced by both macroeconomic and environmental factors.
* Exchange rate fluctuations impact export competitiveness.
* Tree-based models perform well for short-term forecasting.

---

## ⚠️ Note

This model provides short-term predictions and may not fully capture long-term trends.

---

## 👤 Author

**Aaran Sachin Patel**
Class 12, Commerce
Aga Khan Academy, Nairobi

---

## 🌟 Future Improvements

* Add export demand data
* Improve real-time feature generation
* Use advanced time-series models (ARIMA, LSTM)
