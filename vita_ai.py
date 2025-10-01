import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

def create_synthetic_data(num_projects=100):
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

    return data

@st.cache_resource
def train_models():
    data = create_synthetic_data()

    data_encoded = pd.get_dummies(data, columns=['project_type', 'vendor_name'], drop_first=True)

    features = ['manpower_count', 'planned_cost', 'planned_timeline_days',
                'average_temperature', 'rainfall_mm', 'commodity_price_index'] + \
               [col for col in data_encoded.columns if 'project_type_' in col or 'vendor_name_' in col]

    X = data_encoded[features]
    y_cost = data_encoded['cost_overrun']
    y_time = data_encoded['timeline_delay_days']

    model_cost = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model_cost.fit(X, y_cost)

    model_time = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model_time.fit(X, y_time)

    explainer_cost = shap.Explainer(model_cost)
    explainer_time = shap.Explainer(model_time)

    return model_cost, model_time, explainer_cost, explainer_time, features

model_cost, model_time, explainer_cost, explainer_time, features = train_models()

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

def risk_color(value, thresholds):
    if value <= thresholds[0]:
        return 'green'
    elif value <= thresholds[1]:
        return 'yellow'
    else:
        return 'red'

st.sidebar.header("Input Parameters")

terrain_type = st.sidebar.selectbox("Type of Terrain", ["Hilly", "Plain"])
expected_timeline = st.sidebar.number_input("Expected Timeline (days)", min_value=1, max_value=1000, value=180)

def format_budget_input():
    val = st.sidebar.text_input("Total Budget")
    try:
        num = float(val.replace(',', '').replace(' ', ''))
        if num >= 1e7:
            formatted = f"{num/1e7:.2f} Crore"
        elif num >= 1e5:
            formatted = f"{num/1e5:.2f} Lakh"
        elif num >= 1e3:
            formatted = f"{num/1e3:.2f} Thousand"
        else:
            formatted = f"{num}"
        st.sidebar.write(f"Entered Budget: {formatted}")
        return num
    except:
        st.sidebar.write("Enter budget as number")
        return 0

total_budget = format_budget_input()
labour_efficiency = st.sidebar.slider("Labour Efficiency (1-10)", 1, 10, 5)
manpower_count = st.sidebar.number_input("Manpower Count", min_value=1, max_value=10000, value=50)

st.sidebar.subheader("Actual Project Data (optional for MAE)")
actual_timeline = st.sidebar.number_input("Actual Timeline (days)", min_value=0, max_value=2000, value=0)
actual_cost = st.sidebar.number_input("Actual Cost (INR)", min_value=0, value=0)

input_data = pd.DataFrame({
    "project_type": [terrain_type],  # substitute terrain for now
    "vendor_name": ["VendorA"],
    "manpower_count": [manpower_count],
    "planned_cost": [total_budget],
    "planned_timeline_days": [expected_timeline],
    "average_temperature": [25],
    "rainfall_mm": [100],
    "commodity_price_index": [100]
})

if st.button("Predict Risk"):
    cost_overrun_pred, timeline_delay_pred, _, _ = predict_and_explain(input_data)
    predicted_total_cost = total_budget + cost_overrun_pred
    predicted_total_timeline = expected_timeline + timeline_delay_pred

    st.write(f"### Predicted Cost Overrun: INR {cost_overrun_pred:,.2f}")
    st.write(f"### Predicted Timeline Delay: {timeline_delay_pred:.1f} days")
    st.write(f"### Predicted Total Cost: INR {predicted_total_cost:,.2f}")
    st.write(f"### Predicted Total Timeline: {predicted_total_timeline:.1f} days")

    if actual_timeline > 0 and actual_cost > 0:
        mae_timeline = mean_absolute_error([actual_timeline], [predicted_total_timeline])
        mae_cost = mean_absolute_error([actual_cost], [predicted_total_cost])
        st.write(f"### Mean Absolute Error - Timeline: {mae_timeline:.2f} days")
        st.write(f"### Mean Absolute Error - Cost: INR {mae_cost:,.2f}")
    else:
        st.write("Actual Timeline and Cost not provided - skipping MAE calculation.")

    cost_thresholds = [0, total_budget * 0.1]
    time_thresholds = [0, expected_timeline * 0.1]

    cost_color = risk_color(cost_overrun_pred, cost_thresholds)
    time_color = risk_color(timeline_delay_pred, time_thresholds)

    st.markdown(f"<h3>Cost Overrun Risk: <span style='color:{cost_color}'>●</span></h3>", unsafe_allow_html=True)
    st.markdown(f"<h3>Timeline Delay Risk: <span style='color:{time_color}'>●</span></h3>", unsafe_allow_html=True)

