from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load trained ML model
model = joblib.load("best_power_models.pkl")    # Make sure file name matches

# Load dataset for ARIMA forecasting
df = pd.read_csv("household_power_consumption.csv", sep=',')
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(inplace=True)

# Monthly aggregated consumption
monthly_data = df.resample('M', on='Date')['Global_active_power'].sum()

# Train ARIMA model
arima_model = ARIMA(monthly_data, order=(2, 1, 2)).fit()

# French electricity price
UNIT_PRICE = 0.22     # €/kWh
USAGE_FACTOR = 0.40   # 40% realistic usage


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_power', methods=['POST'])
def predict_power():
    data = request.form

    # Input features from HTML form
    features = [
        float(data['Global_reactive_power']),
        float(data['Voltage']),
        float(data['Global_intensity']),
        float(data['Sub_metering_1']),
        float(data['Sub_metering_2']),
        float(data['Sub_metering_3']),
        int(data['Hour']),
        int(data['Day']),
        int(data['Month']),
        int(data['Weekday'])
    ]

    arr = np.array([features])
    power_kw = model.predict(arr)[0]

    # Daily & Monthly consumption (realistic usage)
    daily_kwh = round(power_kw * 24 * USAGE_FACTOR, 2)
    monthly_kwh = round(power_kw * 720 * USAGE_FACTOR, 2)
    monthly_bill = round(monthly_kwh * UNIT_PRICE, 2)

    # --------------------------
    # Sub-Metering Recommendations
    # --------------------------

    sm1 = float(data['Sub_metering_1'])   # Kitchen
    sm2 = float(data['Sub_metering_2'])   # Laundry
    sm3 = float(data['Sub_metering_3'])   # Heating / AC

    rec = ""

    if sm1 > sm2 and sm1 > sm3:
        rec = "High kitchen appliance usage — reduce microwave/oven time."
    elif sm2 > sm1 and sm2 > sm3:
        rec = "Washing machine / dishwasher consuming more — try full loads only."
    elif sm3 > sm1 and sm3 > sm2:
        rec = "Heating/AC usage is high — optimize thermostat settings."
    else:
        rec = "Energy usage is balanced — maintain efficient habits."

    result_text = f"""
        Predicted Power: {power_kw:.3f} kW<br>
        Estimated Daily Usage: {daily_kwh} kWh<br>
        Estimated Monthly Usage: {monthly_kwh} kWh<br>
        Estimated Monthly Bill: €{monthly_bill}<br><br>
        Recommendation: {rec}
    """

    return render_template('index.html', result=result_text)


@app.route('/future', methods=['POST'])
def future_prediction():
    month_input = int(request.form['future_month'])

    forecast = arima_model.forecast(steps=month_input)
    future_consumption = round(forecast.iloc[-1], 2)
    future_bill = round(future_consumption * UNIT_PRICE, 2)

    future_result = f"""
        Predicted Consumption for Month {month_input}: {future_consumption} kWh<br>
        Estimated Bill: €{future_bill}
    """

    return render_template('index.html', future=future_result)


if __name__ == "__main__":
    app.run(debug=True)
