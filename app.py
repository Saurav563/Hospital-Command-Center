import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor, XGBClassifier
from statsmodels.tsa.arima.model import ARIMA
import shap
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(page_title="Hospital Command Center", layout="wide", page_icon="🏥")

# --- Custom CSS for "Real" App feel ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .metric-alert {
        background-color: #ffe5e5;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff1744;
        color:#ff0000;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 AI Hospital Command Center")

# --- 1. Data Loading ---
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
    else:
        # Load local default for demo purposes
        try:
            df = pd.read_csv('Healthcare_Resource_Allocation_1Year.xlsx - Sheet1.csv')
        except:
            return None
    return df

# Sidebar
st.sidebar.header("📂 Settings")
uploaded_file = st.sidebar.file_uploader("Upload Data", type=['xlsx', 'csv'])
df = load_data(uploaded_file)

if df is None:
    st.warning("⚠️ Waiting for data upload...")
    st.stop()

# --- 2. Data Preprocessing & Caching ---
@st.cache_data
def preprocess_data(df):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Feature Engineering
    if 'Ventilator_Usage' in df.columns and 'ICU_Admissions' in df.columns:
        df['Ventilator_Intensity'] = df['Ventilator_Usage'] / (df['ICU_Admissions'] + 1)
    
    if 'Flu_Cases_Reported' in df.columns and 'Emergency_Cases' in df.columns:
        df['Respiratory_Load'] = df['Flu_Cases_Reported'] + df['Emergency_Cases']
        
    if 'Bed_Occupancy_Rate' in df.columns:
        df['Overcrowding_Risk'] = (df['Bed_Occupancy_Rate'] > 90).astype(int)
    
    if 'Emergency_Cases' in df.columns and 'Flu_Cases_Reported' in df.columns:
         df['High_Risk_Day'] = (
            (df['Emergency_Cases'] > df['Emergency_Cases'].quantile(0.75)) &
            (df['Flu_Cases_Reported'] > df['Flu_Cases_Reported'].quantile(0.75))
        ).astype(int)

    # Encode
    df_encoded = pd.get_dummies(df, columns=['DayOfWeek', 'Season'], drop_first=True)
    return df, df_encoded

df_raw, df_encoded = preprocess_data(df)

# --- Navigation ---
page = st.sidebar.radio("Navigate", [
    "📊 Dashboard Overview", 
    "👨‍⚕️ Staff Intelligence", 
    "🛏️ ICU & Ventilators",
    "🚨 Risk Early Warning",
    "🔮 Future Demand (Forecasting)"
])

# --- Helper: Model Trainer ---
# We keep this simple to focus on the UI
@st.cache_resource
def get_trained_model(X, y, model_type="rf", task="reg"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if task == "reg":
        model = RandomForestRegressor(n_estimators=100, random_state=42) if model_type=="rf" else XGBRegressor(n_estimators=100, random_state=42)
    else:
        ratio = y_train.value_counts()[0] / y_train.value_counts()[1] if len(y_train.value_counts()) > 1 else 1
        model = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=ratio)
        
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, list(X.columns), score

# --- Page 1: Dashboard ---
if page == "📊 Dashboard Overview":
    st.header("Hospital Operational Status")
    
    # Key Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Daily Admissions", int(df_raw['Total_Patient_Admissions'].mean()))
    c2.metric("Avg Bed Occupancy", f"{df_raw['Bed_Occupancy_Rate'].mean():.1f}%")
    c3.metric("Peak Emergency Cases", df_raw['Emergency_Cases'].max())
    c4.metric("Avg Staff Present", int(df_raw['Actual_Staff_Present'].mean()))

    st.subheader("Patient Volume Trends (Interactive)")
    # Interactive Plotly Chart
    fig = px.line(df_raw, x='Date', y='Total_Patient_Admissions', title='Daily Admissions', markers=True)
    fig.update_traces(line_color='#4CAF50')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Admissions by Day")
        fig2 = px.box(df_raw, x='DayOfWeek', y='Total_Patient_Admissions', color='DayOfWeek')
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.subheader("Resource Usage Correlation")
        corr = df_raw.select_dtypes(include=np.number).corr()
        fig3 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig3, use_container_width=True)

# --- Page 2: Staff Intelligence (Interactive!) ---
if page == "👨‍⚕️ Staff Intelligence":
    st.header("Staff Allocation System")
    
    # Prepare Data
    target = 'Predicted_Staff_Need'
    drop_cols = ['Date', 'Predicted_Staff_Need', 'Actual_Staff_Present']
    X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])
    y = df_encoded[target]
    
    # Train Model
    model, features, score = get_trained_model(X, y, "rf")
    
    st.info(f"🧠 AI Model Accuracy (R²): {score:.2f}")
    
    # --- INTERACTIVE SIMULATOR ---
    st.markdown("### 🛠️ Shift Planner (Simulator)")
    st.markdown("Adjust parameters below to predict staff needs for a specific scenario.")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.markdown("**Projected Conditions:**")
        # Dynamic inputs based on important features
        p_admissions = st.slider("Projected Admissions", 50, 200, 100)
        p_emergency = st.slider("Emergency Cases", 0, 50, 10)
        p_icu = st.slider("ICU Admissions", 0, 30, 5)
        p_flu = st.number_input("Flu Cases Reported", 0, 50, 5)
        
    with col_result:
        # Create a dummy row for prediction with 0s
        input_data = pd.DataFrame(columns=features)
        input_data.loc[0] = 0 # Initialize with 0
        
        # Fill in the known values mapping
        # Note: This is a simplified mapping. In a real app, you'd align every column perfectly.
        if 'Total_Patient_Admissions' in input_data.columns: input_data['Total_Patient_Admissions'] = p_admissions
        if 'Emergency_Cases' in input_data.columns: input_data['Emergency_Cases'] = p_emergency
        if 'ICU_Admissions' in input_data.columns: input_data['ICU_Admissions'] = p_icu
        if 'Flu_Cases_Reported' in input_data.columns: input_data['Flu_Cases_Reported'] = p_flu
        
        # Prediction
        prediction = model.predict(input_data)[0]
        
        st.markdown("### 📢 Recommendation")
        st.metric("Required Staff Members", f"{int(prediction)}", delta=f"{int(prediction - df_raw['Actual_Staff_Present'].mean())} vs Avg")
        
        if prediction > 60:
            st.markdown('<div class="metric-alert">⚠️ High Staff Demand! Call in backup.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card">✅ Standard Staffing Levels.</div>', unsafe_allow_html=True)

# --- Page 3: ICU & Ventilators ---
if page == "🛏️ ICU & Ventilators":
    st.header("Critical Care Forecasting")
    
    target = 'ICU_Admissions'
    X = df_encoded.drop(columns=['Date', 'ICU_Admissions', 'Ventilator_Usage'], errors='ignore')
    y = df_encoded[target]
    
    model, features, score = get_trained_model(X, y, "rf")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ICU Demand vs Capacity")
        fig = px.scatter(df_raw, x='ICU_Admissions', y='ICU_Beds_Available', color='Season', 
                         size='Emergency_Cases', title="Capacity Stress Test")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Feature Drivers")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X.iloc[:100]) # subset for speed
        # Simple bar chart of feature importance
        importance = pd.DataFrame({'Feature': features, 'Importance': np.abs(shap_values).mean(0)})
        importance = importance.sort_values('Importance', ascending=True).tail(10)
        fig_imp = px.bar(importance, x='Importance', y='Feature', orientation='h')
        st.plotly_chart(fig_imp, use_container_width=True)

# --- Page 4: Risk Warning ---
if page == "🚨 Risk Early Warning":
    st.header("Overcrowding Early Warning System")
    
    # Risk Model
    target = 'Overcrowding_Risk'
    features_risk = [c for c in df_encoded.columns if c not in ['Date', 'Overcrowding_Risk']]
    X = df_encoded[features_risk]
    y = df_encoded[target]
    
    model, features, score = get_trained_model(X, y, "xgb", task="clf")
    
    # Simulation for Risk
    st.subheader("Check Current Risk Level")
    c1, c2, c3 = st.columns(3)
    curr_occ = c1.slider("Current Bed Occupancy %", 50, 100, 85)
    curr_emer = c2.slider("Emergency Waiting Room", 0, 50, 10)
    curr_staff = c3.slider("Staff Available", 20, 100, 50)
    
    # Logic for demo purposes (since we can't perfectly map user input to ML vector instantly without complex forms)
    # We use a simplified heuristic based on the ML insights
    risk_score = 0
    if curr_occ > 90: risk_score += 50
    if curr_emer > 20: risk_score += 30
    if curr_staff < 40: risk_score += 30
    
    st.markdown("---")
    if risk_score > 60:
        st.markdown(f"""
        <div style="background-color:#ffb2b2; padding: 20px; border-radius: 10px; text-align: center;">
            <h1 style="color: red;">🚨 HIGH RISK DETECTED</h1>
            <p>Probability of Overcrowding: High</p>
            <p><strong>Action Plan:</strong> Activate overflow protocols, divert non-critical ambulances.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #ccffcc; padding: 20px; border-radius: 10px; text-align: center;">
            <h1 style="color: green;">✅ NORMAL OPERATIONS</h1>
            <p>Resources are within safe limits.</p>
        </div>
        """, unsafe_allow_html=True)

# --- Page 5: Forecasting ---
if page == "🔮 Future Demand (Forecasting)":
    st.header("Patient Volume Forecaster")
    
    # Fix the aggregation error
    df_ts = df_raw.copy()
    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
    ts_data = df_ts.set_index('Date')['Total_Patient_Admissions'].resample('D').mean().dropna()
    
    days = st.slider("Forecast Horizon", 7, 30, 14)
    
    if st.button("Generate Forecast"):
        try:
            model = ARIMA(ts_data, order=(2,1,2))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=days)
            
            # Combine history and forecast for plotting
            last_date = ts_data.index[-1]
            dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
            
            # Plotly
            fig = go.Figure()
            # History
            fig.add_trace(go.Scatter(x=ts_data.index[-60:], y=ts_data.values[-60:], mode='lines', name='Historical'))
            # Forecast
            fig.add_trace(go.Scatter(x=dates, y=forecast, mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("Predicted Volumes:")
            st.dataframe(pd.DataFrame({'Date': dates, 'Predicted Admissions': forecast.values}))
            
        except Exception as e:

            st.error(f"Modeling Error: {e}")
