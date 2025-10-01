# Install dependencies
!pip install xgboost shap streamlit --quiet

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Seed for reproducibility
np.random.seed(42)

# Create synthetic dataset
num_projects = 100
project_types = ['Substation', 'Overhead Cable', 'Underground Cable']
vendors = ['VendorA', 'VendorB', 'VendorC', 'VendorD']

data = pd.DataFrame({
    'project_id': range(1, num_projects + 1),
    'project_type': np.random.choice(project_types, num_projects),
    'vendor_name': np.random.choice(vendors, num_projects),
    'manpower_count': np.random.randint(20, 200, num_projects),
    'planned_cost': np.random.uniform(1e6, 10e6, num_projects).round(2),
    'planned_timeline_days': np.random.randint(30, 365, num_projects),
    'average_temperature': np.random.uniform(15, 45, num_projects).round(1),
    'rainfall_mm': np.random.uniform(0, 300, num_projects).round(1),
    'commodity_price_index': np.random.uniform(80, 120, num_projects).round(1)
})

base_cost_deviation = np.random.normal(1.0, 0.1, num_projects)
base_timeline_deviation = np.random.normal(1.0, 0.15, num_projects)

weather_effect_cost = 1 + 0.002 * (data['average_temperature'] - 25) + 0.001 * (data['rainfall_mm'] - 100)
commodity_effect_cost = data['commodity_price_index'] / 100
weather_effect_time = 1 + 0.003 * (data['average_temperature'] - 25) + 0.002 * (data['rainfall_mm'] - 100)

data['actual_cost'] = (data['planned_cost'] * base_cost_deviation * weather_effect_cost * commodity_effect_cost).round(2)
data['actual_timeline_days'] = (data['planned_timeline_days'] * base_timeline_deviation * weather_effect_time).round().astype(int)

data['cost_overrun'] = (data['actual_cost'] - data['planned_cost']).round(2)
data['timeline_delay_days'] = data['actual_timeline_days'] - data['planned_timeline_days']

data_encoded = pd.get_dummies(data, columns=['project_type', 'vendor_name'], drop_first=True)

features = ['manpower_count', 'planned_cost', 'planned_timeline_days',
            'average_temperature', 'rainfall_mm', 'commodity_price_index'] + \
           [col for col in data_encoded.columns if 'project_type_' in col or 'vendor_name_' in col]

X = data_encoded[features]

# Train cost overrun model
y_cost = data_encoded['cost_overrun']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cost, test_size=0.2, random_state=42)
model_cost = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_cost.fit(X_train_c, y_train_c)

# Train timeline delay model
y_time = data_encoded['timeline_delay_days']
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X, y_time, test_size=0.2, random_state=42)
model_time = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_time.fit(X_train_t, y_train_t)

# SHAP explainers
explainer_cost = shap.Explainer(model_cost)
explainer_time = shap.Explainer(model_time)

def predict_and_explain(input_df):
    input_encoded = pd.get_dummies(input_df, columns=['project_type', 'vendor_name'], drop_first=True)
    for col in features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[features]
    cost_pred = model_cost.predict(input_encoded)[0]
    time_pred = model_time.predict(input_encoded)[0]
    shap_vals_cost = explainer_cost(input_encoded)
    shap_vals_time = explainer_time(input_encoded)
    return cost_pred, time_pred, shap_vals_cost, shap_vals_time

# Save Streamlit app with traffic light visualization
streamlit_code = '''
import streamlit as st
import pandas as pd

def risk_color(value, thresholds):
    if value <= thresholds[0]:
        return 'green'
    elif value <= thresholds[1]:
        return 'yellow'
    else:
        return 'red'

def predict_and_explain(input_df):
    # Insert actual model prediction logic in deployment
    cost = input_df['planned_cost'].values[0] * 0.1
    time = input_df['planned_timeline_days'].values[0] * 0.15
    return cost, time, None, None

st.title("POWERGRID Project Cost & Timeline Risk Predictor")

st.sidebar.header("Input Parameters")
project_type = st.sidebar.selectbox("Project Type", ["Substation", "Overhead Cable", "Underground Cable"])
vendor_name = st.sidebar.selectbox("Vendor", ["VendorA", "VendorB", "VendorC", "VendorD"])
manpower_count = st.sidebar.slider("Manpower Count", 20, 200, 50)
planned_cost = st.sidebar.number_input("Planned Cost (INR)", 1_000_000, 10_000_000, 5_000_000, 100_000)
planned_timeline = st.sidebar.slider("Planned Timeline (days)", 30, 365, 180)
avg_temp = st.sidebar.slider("Average Temperature (°C)", 15.0, 45.0, 25.0)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 100.0)
commodity_price = st.sidebar.slider("Commodity Price Index", 80.0, 120.0, 100.0)

input_data = pd.DataFrame({
    "project_type": [project_type],
    "vendor_name": [vendor_name],
    "manpower_count": [manpower_count],
    "planned_cost": [planned_cost],
    "planned_timeline_days": [planned_timeline],
    "average_temperature": [avg_temp],
    "rainfall_mm": [rainfall],
    "commodity_price_index": [commodity_price]
})

if st.button("Predict Risk"):
    cost_pred, time_pred, _, _ = predict_and_explain(input_data)
    st.write(f"### Predicted Cost Overrun: INR {cost_pred:,.2f}")
    st.write(f"### Predicted Timeline Delay: {time_pred:.1f} days")

    cost_thresholds = [0, planned_cost * 0.1]
    time_thresholds = [0, planned_timeline * 0.1]

    cost_color = risk_color(cost_pred, cost_thresholds)
    time_color = risk_color(time_pred, time_thresholds)

    st.markdown(f"<h3>Cost Overrun Risk: <span style='color:{cost_color}'>●</span></h3>", unsafe_allow_html=True)
    st.markdown(f"<h3>Timeline Delay Risk: <span style='color:{time_color}'>●</span></h3>", unsafe_allow_html=True)
'''

with open('streamlit_app.py', 'w') as f:
    f.write(streamlit_code)

print("Streamlit app with traffic light saved as 'streamlit_app.py'. Run with: !streamlit run streamlit_app.py")
