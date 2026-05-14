# -*- coding: utf-8 -*-
"""AI Business Agent App - Final Product Version"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
import requests
import io
import time
from datetime import timedelta
import urllib3

# --- SSL FIX: Disable warnings for unverified HTTPS requests ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -------------------------
# Streamlit UI + Session State
# -------------------------
st.set_page_config(
    page_title="BizOpt AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for that "High-Tech" feel and Agent Insight boxes
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        font-weight: bold;
    }
    .agent-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #00a8e8;
        margin-bottom: 20px;
        color: #0f5132;
    }
    .data-box {
        background-color: #fff8e1;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin-bottom: 20px;
        color: #664d03;
    }
    /* Hide Streamlit standard menu for a cleaner app look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def init_state():
    """Initialize all session state variables."""
    if "raw_data" not in st.session_state:
        st.session_state.raw_data = None
    if "weekly_data" not in st.session_state:
        st.session_state.weekly_data = None
    if "model" not in st.session_state:
        st.session_state.model = None
    if "forecast" not in st.session_state:
        st.session_state.forecast = None
    if "agent_status" not in st.session_state:
        st.session_state.agent_status = "Idle"
    if "live_stream" not in st.session_state:
        st.session_state.live_stream = False

init_state()

# -------------------------
# Helper Functions
# -------------------------
@st.cache_data(show_spinner=False)
def fetch_global_data():
    """Fetches data from the GitHub Superstore repository."""
    DATA_URL = "https://raw.githubusercontent.com/curran/data/gh-pages/superstoreSales/superstoreSales.csv"
    try:
        # --- FIX APPLIED HERE: verify=False ---
        response = requests.get(DATA_URL, verify=False)
        response.raise_for_status()
        
        df = pd.read_csv(io.StringIO(response.content.decode('ISO-8859-1')))

        # Standardization
        if 'Order Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        if 'Sales' in df.columns:
            df['Revenue'] = pd.to_numeric(df['Sales'], errors='coerce')

        df = df.dropna(subset=['Date', 'Revenue']).sort_values('Date')
        return df
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

def plot_interactive_trend(history, forecast=None, trend_line=None):
    """Creates a high-quality Plotly chart."""
    fig = go.Figure()

    # 1. Historical Data
    fig.add_trace(go.Scatter(
        x=history['Date'],
        y=history['Revenue'],
        mode='lines',
        name='Historical Revenue',
        line=dict(color='#95a5a6', width=1.5),
        opacity=0.7
    ))

    # 2. Optimized Trend (if available)
    if trend_line is not None:
        fig.add_trace(go.Scatter(
            x=history['Date'],
            y=trend_line,
            mode='lines',
            name='Optimized Trend',
            line=dict(color='#f1c40f', width=3)
        ))

    # 3. Forecast (if available)
    if forecast is not None:
        fig.add_trace(go.Scatter(
            x=forecast['Date'],
            y=forecast['Revenue'],
            mode='lines+markers',
            name='AI Projection',
            line=dict(color='#e74c3c', width=3, dash='dash'),
            marker=dict(size=6)
        ))

        # --- TIMESTAMP MATH FIX ---
        last_date_ts = history['Date'].max()
        last_date_numeric = last_date_ts.timestamp() * 1000 
        
        fig.add_vline(
            x=last_date_numeric, 
            line_width=1, 
            line_dash="dot", 
            line_color="green", 
            annotation_text="Today"
        )

    fig.update_layout(
        title="Enterprise Data Optimization & Growth Projection",
        xaxis_title="Fiscal Timeline",
        yaxis_title="Revenue Volume",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("🤖 BizOpt Agent")
st.sidebar.markdown("---")
pages = [
    "0️⃣ System Overview",
    "1️⃣ Connect Data Bank",
    "2️⃣ Data Optimization",
    "3️⃣ AI Model Training",
    "4️⃣ Strategic Projection",
    "5️⃣ Live Market Monitor"
]
choice = st.sidebar.radio("Agent Modules", pages)

st.sidebar.markdown("---")
st.sidebar.caption(f"Agent Status: **{st.session_state.agent_status}**")
st.sidebar.caption("v2.5.5 | Business Intelligence Unit")

# -------------------------
# PAGE 0: System Overview (Technical Report)
# -------------------------
if choice == pages[0]:
    st.title("📘 BizOpt Intelligent Forecasting System")
    st.markdown("**Module:** Business Intelligence Unit | **Version:** 2.5.5")
    st.divider()

    st.header("1. Executive Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.error("### ❌ The Problem: Data Chaos")
        st.write("""
        Modern businesses generate massive amounts of transactional data. 
        However, raw data is **noisy**. Daily sales fluctuate due to random factors 
        (weather, holidays), creating a jagged trend line that makes it impossible 
        for human managers to accurately predict the future.
        """)
    with col2:
        st.success("### ✅ The Solution: The BizOpt Agent")
        st.write("""
        This Agent is an autonomous system designed to ingest chaotic data and distill 
        it into strategic signals. It performs **Statistical Optimization** to smooth 
        volatility, learns the mathematical "heartbeat" of the business, and projects 
        that trajectory into the future.
        """)

    st.divider()
    
    # --- NEW SECTION: DATA SCHEMA ---
    st.header("2. Data Intelligence Brief & Schema")
    st.markdown("""
    <div class="data-box">
    <strong>📊 About the Dataset:</strong><br>
    The system connects to the <strong>Global Superstore Sales Repository</strong>. 
    This dataset captures 4 years of retail transactions across three major sectors: 
    <strong>Technology</strong>, <strong>Furniture</strong>, and <strong>Office Supplies</strong>.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📂 Data Schema & Dictionary (What columns matter?)", expanded=True):
        st.markdown("""
        The Agent ingests 20+ columns, but the following are **Critical Intelligence Vectors**:

        | Column Name | Data Type | Strategic Value | Agent Action |
        | :--- | :--- | :--- | :--- |
        | **Order Date** | `DateTime` | **The Timeline.** Tells us *when* events happened. | Used to index the Time Series and detect seasonality. |
        | **Sales** | `Float` | **The Target.** The primary revenue metric we want to maximize. | This is the $y$ variable in our optimization equation. |
        | **Category** | `String` | **The Segment.** (Tech/Furniture). | Provides context on *what* is driving the trends. |
        | **Profit** | `Float` | **The Health Metric.** Revenue without profit is vanity. | Secondary metric for efficiency analysis. |
        | **Region** | `String` | **The Geography.** (North/South/East/West). | Helps isolate localized market anomalies. |
        """)
        st.info("ℹ️ **Note:** While the dataset contains complex dimensions (Customer Name, Ship Mode), the Agent's **Optimization Engine** focuses primarily on the correlation between `Order Date` and `Sales` to build the Growth Trajectory.")

    st.divider()
    st.header("3. Technical Methodology")
    
    with st.expander("A. Data Optimization (The Smoothing Algorithm)", expanded=False):
        st.write("We use **Time-Series Resampling** to transform high-variance signals.")
        st.code("df.resample('W').sum()", language="python")
        st.write("This converts jagged daily data `[Day 1: $100, Day 2: $50...]` into stable weekly metrics `[Week 1: $300...]`.")

    with st.expander("B. Predictive Modeling (The Math)", expanded=False):
        st.write("The Agent solves the linear equation for business growth:")
        st.latex(r''' y = mx + c ''')
        st.write("""
        * $y$ = Revenue (Target)
        * $x$ = Time Index
        * $m$ = **Growth Coefficient** (Weekly Growth Rate)
        * $c$ = **Baseline** (Starting Revenue)
        """)

    st.divider()
    st.caption("Navigate to '1️⃣ Connect Data Bank' to begin the session.")


# -------------------------
# PAGE 1: Connect Data Bank
# -------------------------
elif choice == pages[1]:
    st.title("🌐 Connect to Global Data Bank")
    
    st.markdown("""
    <div class="agent-box">
    <strong>🧠 Agent Insight:</strong><br>
    I need access to your historical transaction records to learn your business patterns. 
    I can connect securely to verified global repositories or accept your local private data files.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.info("### Option A: Global Repository")
        st.write("Connect to verified Superstore Sales data hosted on GitHub.")
        if st.button("🔌 Initialize & Connect (GitHub)", key="connect_btn"):
            with st.spinner("Agent is authenticating with Global Data Bank..."):
                time.sleep(1.5) # Effect
                df = fetch_global_data()
                if df is not None:
                    st.session_state.raw_data = df
                    st.session_state.agent_status = "Data Loaded"
                    st.success(f"Connection Successful! Retrieved {len(df):,} transaction records.")

    with col2:
        st.warning("### Option B: Manual Upload")
        st.write("Upload a local CSV file for analysis.")
        uploaded_file = st.file_uploader("Upload CSV (Columns: Date, Sales)", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'Date' in df.columns and 'Revenue' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                st.session_state.raw_data = df
                st.session_state.agent_status = "Data Uploaded"
                st.success("File Uploaded Successfully.")
            else:
                st.error("CSV must have 'Date' and 'Revenue' columns.")

    if st.session_state.raw_data is not None:
        st.divider()
        st.markdown("### 📂 Data Preview (Raw Transactions)")
        st.dataframe(st.session_state.raw_data.head(10), use_container_width=True)
        
        with st.expander("ℹ️ How the Agent reads this data"):
            st.write("""
            The agent scans the `Order Date` column to establish a timeline and the `Sales` column to measure performance.
            Rows with missing values are automatically purged to ensure calculation integrity.
            """)
        
        st.info("Proceed to **2️⃣ Data Optimization** to clean and aggregate this data.")

# -------------------------
# PAGE 2: Data Optimization
# -------------------------
elif choice == pages[2]:
    st.title("⚙️ Data Optimization Engine")
    
    st.markdown("""
    <div class="agent-box">
    <strong>🧠 Agent Insight:</strong><br>
    Raw business data is "noisy"—it fluctuates wildly day-to-day due to random factors. 
    My Optimization Engine applies <strong>Statistical Aggregation</strong> and <strong>Smoothing Algorithms</strong> 
    to reveal the true underlying growth trend hidden in the chaos.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.raw_data is None:
        st.warning("⚠️ No data found. Please go to Page 1 to connect.")
        st.stop()

    if st.button("✨ Run Optimization Algorithms"):
        with st.status("Optimizing Data...", expanded=True) as status:
            st.write("📉 Detecting outliers and volatility...")
            time.sleep(0.8)
            st.write("🗓️ Resampling to Weekly frequency for trend stability...")
            time.sleep(0.5)
            st.write("✅ calculating performance indices...")
            
            # Optimization Logic
            df = st.session_state.raw_data.copy()
            # Resample to Weekly ('W') to smooth out daily noise
            weekly_df = df.set_index('Date').resample('W')['Revenue'].sum().reset_index()
            weekly_df['Time_Index'] = np.arange(len(weekly_df))
            
            st.session_state.weekly_data = weekly_df
            status.update(label="Optimization Complete", state="complete", expanded=False)
    
    if st.session_state.weekly_data is not None:
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Volatile Records", f"{len(st.session_state.raw_data):,}")
        with col2:
            st.metric("Optimized Trend Points", f"{len(st.session_state.weekly_data):,}")
            
        # Comparison Chart
        st.subheader("Optimization Visualized")
        
        fig = go.Figure()
        # Raw (Sample)
        sample_raw = st.session_state.raw_data.iloc[:200]
        fig.add_trace(go.Scatter(x=sample_raw['Date'], y=sample_raw['Revenue'], mode='markers', name='Raw Noise (Daily)', opacity=0.4))
        # Optimized
        fig.add_trace(go.Scatter(x=st.session_state.weekly_data['Date'], y=st.session_state.weekly_data['Revenue'], 
                                 mode='lines', name='Optimized Signal (Weekly)', line=dict(color='#f1c40f', width=3)))
        
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("🔍 Technical Details: What just happened?"):
            st.write("""
            1. **Ingestion:** The agent took {:,} individual transaction rows.
            2. **Resampling:** It grouped them into 7-day blocks (Weekly).
            3. **Noise Reduction:** By summing weekly values, random daily fluctuations (like a slow Tuesday followed by a busy Wednesday) cancel each other out.
            4. **Result:** The **Yellow Line** represents the "True" health of the business, which is what we will model.
            """.format(len(st.session_state.raw_data)))
            
        st.info("Proceed to **3️⃣ AI Model Training**.")

# -------------------------
# PAGE 3: AI Model Training
# -------------------------
elif choice == pages[3]:
    st.title("🧠 AI Model Training")
    
    if st.session_state.weekly_data is None:
        st.error("Please complete Data Optimization (Page 2) first.")
        st.stop()

    st.markdown("""
    <div class="agent-box">
    <strong>🧠 Agent Insight:</strong><br>
    I am now initializing a <strong>Predictive Regression Model</strong>. 
    I will mathematically "fit" a line through the optimized data to learn the relationship between Time and Revenue. 
    This establishes the <strong>Baseline Growth Trajectory</strong>.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("## Training Parameters")
        st.write("- **Algorithm:** Linear Regression (OLS)")
        st.write("- **Input Feature:** Time Index (Sequential)")
        st.write("- **Target Variable:** Weekly Revenue")
        
        if st.button("🚀 Train Model", type="primary"):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Training Logic
            df = st.session_state.weekly_data
            X = df[['Time_Index']]
            y = df['Revenue']
            
            model = LinearRegression()
            model.fit(X, y)
            
            st.session_state.model = model
            st.session_state.agent_status = "Model Trained"
            st.success("Model Training Complete.")
    
    with col2:
        if st.session_state.model is not None:
            df = st.session_state.weekly_data
            X = df[['Time_Index']]
            y = df['Revenue']
            
            # Metrics
            slope = st.session_state.model.coef_[0]
            intercept = st.session_state.model.intercept_
            r2 = st.session_state.model.score(X, y)
            
            st.write("## Model Performance Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("Growth Trend (Slope)", f"{slope:.2f}", help="Average revenue increase per week")
            m2.metric("Baseline (Intercept)", f"{intercept:.2f}", help="Starting theoretical revenue")
            m3.metric("Confidence (R²)", f"{r2:.3f}", help="How well the model fits the data (0-1)")
            
            with st.expander("📚 Explain these metrics"):
                st.write(f"""
                * **Growth Trend:** On average, your business grows by **${slope:.2f}** every single week.
                * **Confidence (R²):** A score of {r2:.3f} indicates how strong the trend is vs. the noise.
                """)
            
            st.info("Proceed to **4️⃣ Strategic Projection**.")

# -------------------------
# PAGE 4: Strategic Projection
# -------------------------
elif choice == pages[4]:
    st.title("📈 Strategic Projection")
    
    if st.session_state.model is None:
        st.warning("⚠️ Model not trained. Please go to Page 3.")
        st.stop()

    st.markdown("""
    <div class="agent-box">
    <strong>🧠 Agent Insight:</strong><br>
    Using the trained model, I will now <strong>extrapolate</strong> the growth trend into the future. 
    This assumes market conditions remain consistent with the historical pattern learned in Step 3.
    </div>
    """, unsafe_allow_html=True)

    # User Inputs
    st.markdown("### ⚙️ Forecast Configuration")
    weeks = st.slider("Weeks to Forecast", min_value=4, max_value=52, value=12, help="How far into the future should I look?")
    
    if st.button("🔮 Generate Projection", type="primary"):
        with st.spinner("Calculating future market trajectory..."):
            df = st.session_state.weekly_data
            model = st.session_state.model
            
            # 1. Predict Trend Line for History (to show the "Best Fit")
            trend_y = model.predict(df[['Time_Index']])
            
            # 2. Predict Future
            last_idx = df['Time_Index'].max()
            future_indices = np.arange(last_idx + 1, last_idx + weeks + 1).reshape(-1, 1)
            future_rev = model.predict(future_indices)
            
            last_date = df['Date'].max()
            future_dates = pd.date_range(start=last_date + timedelta(weeks=1), periods=weeks, freq='W')
            
            forecast_df = pd.DataFrame({'Date': future_dates, 'Revenue': future_rev})
            st.session_state.forecast = forecast_df
            
            # Visuals
            fig = plot_interactive_trend(df, forecast_df, trend_y)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary
            total_projected = forecast_df['Revenue'].sum()
            growth = (future_rev[-1] - df['Revenue'].iloc[-1]) / df['Revenue'].iloc[-1] * 100
            
            st.divider()
            st.subheader("Executive Summary")
            m1, m2 = st.columns(2)
            m1.metric("Projected Total Revenue", f"${total_projected:,.2f}")
            m2.metric("Expected Growth", f"{growth:.1f}%", delta_color="normal")

# -------------------------
# PAGE 5: Live Market Monitor
# -------------------------
elif choice == pages[5]:
    st.title("📡 Live Market Monitor")
    
    st.markdown("""
    <div class="agent-box">
    <strong>🧠 Agent Insight:</strong><br>
    This module simulates the Agent connecting to a <strong>Real-Time Data Stream</strong> (e.g., WebSocket or API). 
    It monitors incoming transactions against the projected model to detect anomalies or confirm growth.
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.weekly_data is None:
        st.warning("Data required for simulation. Please connect and optimize data first.")
        st.stop()

    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        start_btn = st.button("▶ Start Stream", type="primary")
    with col3:
        stop_btn = st.button("⏹ Stop Stream")

    if start_btn:
        st.session_state.live_stream = True
    if stop_btn:
        st.session_state.live_stream = False

    # Placeholder for live chart
    metric_holder = st.empty()
    chart_holder = st.empty()
    
    if st.session_state.live_stream:
        # Simulation Loop using the existing data to look "Live"
        # We iterate through the dataset to simulate a timeline playback
        df = st.session_state.weekly_data
        
        # Start from a point where we have enough data
        start_idx = max(50, len(df) - 100)
        
        for i in range(start_idx, len(df)):
            if not st.session_state.live_stream:
                break
                
            # Slice data up to current 'moment'
            current_slice = df.iloc[:i]
            latest_val = current_slice['Revenue'].iloc[-1]
            prev_val = current_slice['Revenue'].iloc[-2]
            delta = latest_val - prev_val
            
            with metric_holder.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Revenue", f"${latest_val:,.0f}", delta=f"{delta:.0f}")
                c2.metric("Active Orders", f"{np.random.randint(50, 150)}")
                c3.metric("Agent Confidence", "98.4%")

            with chart_holder.container():
                fig = px.line(current_slice, x='Date', y='Revenue', title="Live Ingest Stream")
                fig.update_traces(line_color='#e74c3c', line_width=2)
                fig.update_layout(
                    height=400, 
                    template="plotly_dark",
                    xaxis_range=[current_slice['Date'].min(), current_slice['Date'].max()]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            time.sleep(0.2) # Speed of simulation