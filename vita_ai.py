import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import shap
import plotly.express as px

# Dummy project data
data = pd.DataFrame({
    'project_type_substation':        [1, 0, 0, 1, 0, 0, 1, 0],
    'project_type_overhead_line':     [0, 1, 1, 0, 1, 1, 0, 0],
    'project_type_underground_cable':[0, 0, 0, 0, 0, 0, 0, 1],
    'terrain_plain':                  [1, 1, 0, 1, 0, 0, 1, 1],
    'terrain_hilly':                  [0, 0, 1, 0, 1, 1, 0, 0],
    'weather_rainy_days':             [5, 2, 10, 1, 8, 7, 0, 4],
    'vendor_performance_score':       [8.5, 7.1, 6.2, 9.0, 7.5, 6.5, 8.0, 7.9],
    'regulatory_delays_days':         [2, 3, 5, 1, 4, 6, 0, 2],
    'material_cost':                  [1200000, 1300000, 1250000, 1150000, 1400000, 1350000, 1100000, 1200000],
    'labour_cost':                   [800000, 850000, 750000, 700000, 900000, 880000, 600000, 800000],
    'cost':                          [2100000, 2200000, 2100000, 1800000, 2300000, 2250000, 1700000, 2000000],
    'timeline':                      [30, 35, 40, 25, 45, 50, 20, 35]
})

features = data.drop(['cost', 'timeline'], axis=1)
target_cost = data['cost']
target_timeline = data['timeline']

X_train, X_test, y_cost_train, y_cost_test = train_test_split(features, target_cost, test_size=0.2, random_state=42)
_, _, y_time_train, y_time_test = train_test_split(features, target_timeline, test_size=0.2, random_state=42)

rf_cost = RandomForestRegressor(n_estimators=100, random_state=42)
rf_cost.fit(X_train, y_cost_train)

xgb_time = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
xgb_time.fit(X_train, y_time_train)

def format_cost(cost):
    if cost >= 1e7:
        return f"{cost / 1e7:.2f} Cr"
    elif cost >= 1e5:
        return f"{cost / 1e5:.2f} L"
    elif cost >= 1e3:
        return f"{cost / 1e3:.2f} K"
    else:
        return f"{cost:.2f}"

def format_timeline(days):
    return f"{days:.1f} days"

def predict_project(new_data_dict):
    df = pd.DataFrame([new_data_dict])
    df = df.reindex(columns=features.columns, fill_value=0)
    cost_pred = rf_cost.predict(df)[0]
    time_pred = xgb_time.predict(df)[0]
    return cost_pred, time_pred

st.title("POWERGRID Project Prediction - Enhanced UI")

# Sidebar input section
st.sidebar.header("Project Configuration")

project_type = st.sidebar.selectbox("Project Type", ["Substation", "Overhead Line", "Underground Cable"])
terrain = st.sidebar.selectbox("Terrain Type", ["Plain", "Hilly"])
rainy_days = st.sidebar.number_input("Rainy Days", 0)
vendor_score = st.sidebar.number_input("Vendor Performance Score (0-10)", 0.0,10.0, 5.0)
regulatory_delays = st.sidebar.number_input("Regulatory Delays (Days)", 0)

with st.sidebar.expander("Cost Parameters"):
    material_cost = st.number_input("Material Cost (INR)", 0)
    labour_cost = st.number_input("Labour Cost (INR)", 0)

# Encode categorical selections to one-hot
project_map = {"Substation": [1, 0, 0], "Overhead Line": [0, 1, 0], "Underground Cable": [0, 0, 1]}
terrain_map = {"Plain": [1, 0], "Hilly": [0, 1]}

if st.sidebar.button("Predict"):
    input_data = {
        'project_type_substation': project_map[project_type][0],
        'project_type_overhead_line': project_map[project_type][1],
        'project_type_underground_cable': project_map[project_type][2],
        'terrain_plain': terrain_map[terrain][0],
        'terrain_hilly': terrain_map[terrain][1],
        'weather_rainy_days': rainy_days,
        'vendor_performance_score': vendor_score,
        'regulatory_delays_days': regulatory_delays,
        'material_cost': material_cost,
        'labour_cost': labour_cost
    }

    cost_pred, timeline_pred = predict_project(input_data)

    st.success(f"Predicted Cost: {format_cost(cost_pred)}")
    st.success(f"Predicted Timeline: {format_timeline(timeline_pred)}")

    explainer = shap.Explainer(xgb_time)
    shap_values = explainer(X_test)

    importance_vals = np.abs(shap_values.values).mean(0)
    feature_importance = pd.DataFrame({
        'Feature': features.columns,
        'Importance': importance_vals
    }).sort_values('Importance', ascending=False)

    feature_importance['Relative Importance'] = feature_importance['Importance'] / feature_importance['Importance'].sum()
    timeline_mae = mean_absolute_error(y_time_test, xgb_time.predict(X_test))
    feature_importance['Estimated Delay (Days)'] = feature_importance['Relative Importance'] * timeline_mae

    st.subheader("Top Hotspot Factors Impacting Timeline")

    fig = px.bar(
        feature_importance,
        x='Feature',
        y='Importance',
        hover_data=['Estimated Delay (Days)'],
        title='Feature Importance for Timeline Prediction',
        labels={'Importance': 'Importance', 'Feature': 'Feature'}
    )
    st.plotly_chart(fig, use_container_width=True)

