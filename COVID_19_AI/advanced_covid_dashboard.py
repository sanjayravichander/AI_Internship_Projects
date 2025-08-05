import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
load_dotenv()

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# AI/LLM Integration
import os
from langchain_groq.chat_models import ChatGroq
import json

# Set page config
st.set_page_config(
    page_title="🦠 AI-Powered COVID-19 Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedCOVIDDashboard:
    def __init__(self):
        self.df = None
        self.processed_df = None
        self.groq_client = None
        self.setup_groq()
        
    def setup_groq(self):
        """Initialize Groq client for AI insights"""
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("Please set your GROQ_API_KEY in the .env file")

            # ✅ Specify model name explicitly here
            self.groq_client = ChatGroq(
                groq_api_key=api_key,
                model="llama-3.3-70b-versatile"  # ✅ or try "llama3-70b-8192" if needed
            )

        except Exception as e:
            st.warning(f"⚠️ Could not initialize Groq client: {str(e)}")


    def load_and_process_data(self):
        """Load and process the COVID-19 data"""
        try:
            self.df = pd.read_csv("c:/Users/DELL/AI_Internship_Projects/COVID_19/StatewiseTestingDetails.csv")
            
            # Data preprocessing
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df = self.df.sort_values(['State', 'Date'])
            
            # Handle missing values intelligently
            self.df['Positive'] = pd.to_numeric(self.df['Positive'], errors='coerce')
            self.df['Negative'] = pd.to_numeric(self.df['Negative'], errors='coerce')
            self.df['TotalSamples'] = pd.to_numeric(self.df['TotalSamples'], errors='coerce')
            
            # Forward fill missing values by state
            for state in self.df['State'].unique():
                mask = self.df['State'] == state
                self.df.loc[mask, 'Positive'] = self.df.loc[mask, 'Positive'].fillna(method='ffill').fillna(0)
                self.df.loc[mask, 'Negative'] = self.df.loc[mask, 'Negative'].fillna(method='ffill').fillna(0)
                self.df.loc[mask, 'TotalSamples'] = self.df.loc[mask, 'TotalSamples'].fillna(method='ffill').fillna(0)
            
            # Calculate additional metrics
            self.df['Daily_Positive'] = self.df.groupby('State')['Positive'].diff().fillna(0).clip(lower=0)
            self.df['Daily_Tests'] = self.df.groupby('State')['TotalSamples'].diff().fillna(0).clip(lower=0)
            self.df['Positivity_Rate'] = np.where(self.df['Daily_Tests'] > 0, 
                                                 (self.df['Daily_Positive'] / self.df['Daily_Tests']) * 100, 0)
            self.df['Week'] = self.df['Date'].dt.isocalendar().week
            self.df['Month'] = self.df['Date'].dt.month
            self.df['Year'] = self.df['Date'].dt.year
            
            # Calculate 7-day rolling averages
            self.df['Rolling_Avg_Positive'] = self.df.groupby('State')['Daily_Positive'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
            self.df['Rolling_Avg_Tests'] = self.df.groupby('State')['Daily_Tests'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def get_ai_insights(self, data_summary, chart_type="general"):
        """Generate AI-powered insights using Groq via LangChain"""
        if not self.groq_client:
            return "AI insights unavailable - Groq API key not configured."

        try:
            prompt = f"""
            As a data scientist analyzing COVID-19 data, provide 3-4 key insights based on this data summary:

            Data Summary:
            {data_summary}

            Chart Type: {chart_type}

            Please provide:
            1. Key trends and patterns
            2. Notable observations
            3. Potential implications
            4. Actionable recommendations

            Keep insights concise, data-driven, and actionable. Format as bullet points.
            """

            # ✅ Call the Groq model using LangChain's .invoke()
            response = self.groq_client.invoke(prompt)

            return response.content  # ✅ .content contains the LLM's response
        except Exception as e:
            return f"AI insight generation failed: {str(e)}"

    def detect_anomalies(self, state_data):
        """Detect anomalies in COVID-19 data using Isolation Forest"""
        try:
            # Prepare features for anomaly detection
            features = ['Daily_Positive', 'Daily_Tests', 'Positivity_Rate']
            data_for_anomaly = state_data[features].fillna(0)
            
            if len(data_for_anomaly) < 10:
                return []
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(data_for_anomaly)
            
            # Get anomaly dates
            anomaly_dates = state_data[anomalies == -1]['Date'].tolist()
            return anomaly_dates
        except Exception as e:
            st.warning(f"Anomaly detection failed: {str(e)}")
            return []
    
    def predict_future_cases(self, state_data, days_ahead=14):
        """Predict future COVID-19 cases using Random Forest"""
        try:
            # Prepare features
            state_data = state_data.copy()
            state_data['Day_Number'] = (state_data['Date'] - state_data['Date'].min()).dt.days
            
            # Create lag features
            for lag in [1, 3, 7]:
                state_data[f'Positive_Lag_{lag}'] = state_data['Positive'].shift(lag)
                state_data[f'Daily_Positive_Lag_{lag}'] = state_data['Daily_Positive'].shift(lag)
            
            # Drop rows with NaN values
            state_data = state_data.dropna()
            
            if len(state_data) < 30:
                return None, None, "Insufficient data for prediction"
            
            # Features and target
            feature_cols = ['Day_Number', 'Positive_Lag_1', 'Positive_Lag_3', 'Positive_Lag_7',
                           'Daily_Positive_Lag_1', 'Daily_Positive_Lag_3', 'Daily_Positive_Lag_7']
            X = state_data[feature_cols]
            y = state_data['Daily_Positive']
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions for test set
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Predict future
            last_row = state_data.iloc[-1]
            future_predictions = []
            
            for i in range(days_ahead):
                next_day = last_row['Day_Number'] + i + 1
                
                # Use last known values for lag features
                next_features = [
                    next_day,
                    last_row['Positive'],  # Lag 1
                    last_row['Positive'],  # Lag 3 (simplified)
                    last_row['Positive'],  # Lag 7 (simplified)
                    last_row['Daily_Positive'],  # Daily lag 1
                    last_row['Daily_Positive'],  # Daily lag 3
                    last_row['Daily_Positive']   # Daily lag 7
                ]
                
                pred = model.predict([next_features])[0]
                future_predictions.append(max(0, pred))  # Ensure non-negative
            
            return future_predictions, {'MAE': mae, 'R2': r2}, None
            
        except Exception as e:
            return None, None, f"Prediction failed: {str(e)}"
    
    def create_interactive_timeline(self, selected_states):
        """Create interactive timeline visualization"""
        filtered_df = self.df[self.df['State'].isin(selected_states)]
        
        fig = px.line(filtered_df, x='Date', y='Rolling_Avg_Positive', 
                     color='State', title='COVID-19 Daily Cases Timeline (7-day Rolling Average)',
                     labels={'Rolling_Avg_Positive': 'Daily Cases (7-day avg)', 'Date': 'Date'})
        
        fig.update_layout(
            hovermode='x unified',
            height=500,
            showlegend=True,
            xaxis_title="Date",
            yaxis_title="Daily Cases (7-day Rolling Average)"
        )
        
        return fig
    
    def create_heatmap(self):
        """Create state-wise heatmap"""
        # Calculate monthly totals by state
        monthly_data = self.df.groupby(['State', 'Year', 'Month'])['Daily_Positive'].sum().reset_index()
        monthly_data['Year_Month'] = monthly_data['Year'].astype(str) + '-' + monthly_data['Month'].astype(str).str.zfill(2)
        
        # Pivot for heatmap
        heatmap_data = monthly_data.pivot(index='State', columns='Year_Month', values='Daily_Positive').fillna(0)
        
        # Select top 15 states by total cases
        top_states = self.df.groupby('State')['Daily_Positive'].sum().nlargest(15).index
        heatmap_data = heatmap_data.loc[top_states]
        
        fig = px.imshow(heatmap_data, 
                       title='Monthly COVID-19 Cases Heatmap (Top 15 States)',
                       labels=dict(x="Month", y="State", color="Cases"),
                       aspect="auto")
        
        fig.update_layout(height=600)
        return fig
    
    def create_prediction_chart(self, state, predictions, metrics):
        """Create prediction visualization"""
        state_data = self.df[self.df['State'] == state].copy()
        
        # Historical data
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=state_data['Date'], 
            y=state_data['Daily_Positive'],
            mode='lines',
            name='Historical Cases',
            line=dict(color='blue')
        ))
        
        # Add predictions
        last_date = state_data['Date'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(len(predictions))]
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Predicted Cases',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'COVID-19 Cases Prediction for {state}',
            xaxis_title='Date',
            yaxis_title='Daily Cases',
            height=400
        )
        
        return fig
    
    def run_dashboard(self):
        """Main dashboard function"""
        # Header
        st.markdown('<h1 class="main-header">🦠 AI-Powered COVID-19 Dashboard</h1>', unsafe_allow_html=True)
        
        # Load data
        if not self.load_and_process_data():
            st.error("Failed to load data. Please check the file path.")
            return
        
        # Sidebar
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # State selection
        available_states = sorted(self.df['State'].unique())
        selected_states = st.sidebar.multiselect(
            "Select States for Analysis",
            available_states,
            default=['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu']
        )
        
        if not selected_states:
            st.warning("Please select at least one state.")
            return
        
        # Date range
        from datetime import date

        # Ensure 'Date' column is in datetime format
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')

        # Filter out any rows with NaT in Date
        self.df = self.df[self.df['Date'].notna()]

        # Compute min and max date from clean data
        min_date = self.df['Date'].min()
        max_date = self.df['Date'].max()

        # Safety fallback if min/max are missing
        if pd.isna(min_date) or pd.isna(max_date):
            min_date = date(2020, 1, 1)
            max_date = date.today()

        # Show caption of available data range
        st.sidebar.caption(f"📅 Available data: {min_date.date()} → {max_date.date()}")

        # Streamlit date selector
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Filter data
        if len(date_range) == 2:
            mask = (self.df['Date'] >= pd.to_datetime(date_range[0])) & (self.df['Date'] <= pd.to_datetime(date_range[1]))
            filtered_df = self.df[mask & self.df['State'].isin(selected_states)]
        else:
            filtered_df = self.df[self.df['State'].isin(selected_states)]
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Trends", "🔮 Predictions", "🚨 Anomalies", "🤖 AI Insights"])
        
        with tab1:
            st.subheader("📊 Key Metrics Overview")
            
            # Calculate key metrics
            total_cases = filtered_df['Positive'].max() if not filtered_df.empty else 0
            total_tests = filtered_df['TotalSamples'].max() if not filtered_df.empty else 0
            avg_positivity = filtered_df['Positivity_Rate'].mean() if not filtered_df.empty else 0
            peak_daily = filtered_df['Daily_Positive'].max() if not filtered_df.empty else 0
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Cases", f"{total_cases:,.0f}")
            with col2:
                st.metric("Total Tests", f"{total_tests:,.0f}")
            with col3:
                st.metric("Avg Positivity Rate", f"{avg_positivity:.2f}%")
            with col4:
                st.metric("Peak Daily Cases", f"{peak_daily:,.0f}")
            
            # Interactive timeline
            st.subheader("📈 Interactive Timeline")
            timeline_fig = self.create_interactive_timeline(selected_states)
            st.plotly_chart(timeline_fig, use_container_width=True)
            
            # State comparison
            st.subheader("🏛️ State Comparison")
            comparison_data = filtered_df.groupby('State').agg({
                'Positive': 'max',
                'Daily_Positive': 'sum',
                'TotalSamples': 'max',
                'Positivity_Rate': 'mean'
            }).round(2)
            
            st.dataframe(comparison_data, use_container_width=True)
        
        with tab2:
            st.subheader("📈 Trend Analysis")
            
            # Heatmap
            st.subheader("🗺️ Monthly Cases Heatmap")
            heatmap_fig = self.create_heatmap()
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Positivity rate trends
            st.subheader("📊 Positivity Rate Trends")
            positivity_fig = px.line(filtered_df, x='Date', y='Positivity_Rate', 
                                   color='State', title='COVID-19 Positivity Rate Over Time')
            st.plotly_chart(positivity_fig, use_container_width=True)
            
            # Weekly patterns
            st.subheader("📅 Weekly Patterns")
            weekly_data = filtered_df.groupby(['State', 'Week'])['Daily_Positive'].sum().reset_index()
            weekly_fig = px.bar(weekly_data, x='Week', y='Daily_Positive', 
                              color='State', title='Weekly COVID-19 Cases Distribution')
            st.plotly_chart(weekly_fig, use_container_width=True)
        
        with tab3:
            st.subheader("🔮 AI-Powered Predictions")
            
            # Single state selection for prediction
            prediction_state = st.selectbox("Select State for Prediction", selected_states)
            days_ahead = st.slider("Days to Predict", 7, 30, 14)
            
            if st.button("Generate Predictions"):
                state_data = self.df[self.df['State'] == prediction_state]
                predictions, metrics, error = self.predict_future_cases(state_data, days_ahead)
                
                if error:
                    st.error(error)
                elif predictions:
                    # Display prediction chart
                    pred_fig = self.create_prediction_chart(prediction_state, predictions, metrics)
                    st.plotly_chart(pred_fig, use_container_width=True)
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model Accuracy (R²)", f"{metrics['R2']:.3f}")
                    with col2:
                        st.metric("Mean Absolute Error", f"{metrics['MAE']:.1f}")
                    
                    # Prediction summary
                    st.subheader("📋 Prediction Summary")
                    pred_df = pd.DataFrame({
                        'Day': range(1, len(predictions) + 1),
                        'Predicted Cases': [int(p) for p in predictions]
                    })
                    st.dataframe(pred_df, use_container_width=True)
        
        with tab4:
            st.subheader("🚨 Anomaly Detection")
            
            anomaly_state = st.selectbox("Select State for Anomaly Detection", selected_states, key="anomaly")
            
            if st.button("Detect Anomalies"):
                state_data = self.df[self.df['State'] == anomaly_state]
                anomaly_dates = self.detect_anomalies(state_data)
                
                if anomaly_dates:
                    st.success(f"Found {len(anomaly_dates)} anomalies")
                    
                    # Visualize anomalies
                    fig = px.line(state_data, x='Date', y='Daily_Positive', 
                                title=f'COVID-19 Cases with Anomalies - {anomaly_state}')
                    
                    # Add anomaly markers
                    anomaly_data = state_data[state_data['Date'].isin(anomaly_dates)]
                    fig.add_scatter(x=anomaly_data['Date'], y=anomaly_data['Daily_Positive'],
                                  mode='markers', marker=dict(color='red', size=10),
                                  name='Anomalies')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # List anomaly dates
                    st.subheader("📅 Anomaly Dates")
                    for date in anomaly_dates:
                        st.write(f"• {date.strftime('%Y-%m-%d')}")
                else:
                    st.info("No significant anomalies detected.")
        
        with tab5:
            st.subheader("🤖 AI-Powered Insights")
            
            # Generate data summary for AI
            summary_data = {
                'total_states': len(selected_states),
                'date_range': f"{date_range[0]} to {date_range[1]}" if len(date_range) == 2 else "Full range",
                'total_cases': int(filtered_df['Positive'].max()) if not filtered_df.empty else 0,
                'peak_daily': int(filtered_df['Daily_Positive'].max()) if not filtered_df.empty else 0,
                'avg_positivity': round(filtered_df['Positivity_Rate'].mean(), 2) if not filtered_df.empty else 0,
                'states': selected_states
            }
            
            if st.button("Generate AI Insights"):
                with st.spinner("🤖 AI is analyzing your data..."):
                    insights = self.get_ai_insights(str(summary_data), "dashboard_overview")
                    
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.markdown("### 🧠 AI Analysis Results")
                    st.markdown(insights)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional AI features
            st.subheader("💡 Smart Recommendations")
            
            if st.button("Get Smart Recommendations"):
                recommendations = [
                    "📊 **Data Quality**: Consider investigating periods with missing negative test data",
                    "📈 **Trend Analysis**: Focus on states showing unusual positivity rate patterns",
                    "🎯 **Resource Allocation**: Prioritize testing in states with rising case trends",
                    "⚠️ **Early Warning**: Monitor states approaching historical peak levels",
                    "🔍 **Deep Dive**: Analyze weekly patterns for better resource planning"
                ]
                
                for rec in recommendations:
                    st.markdown(rec)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            🦠 Advanced COVID-19 Dashboard | Powered by AI & Machine Learning<br>
            Built with Streamlit, Plotly, Scikit-learn & Groq AI
        </div>
        """, unsafe_allow_html=True)

# Run the dashboard
if __name__ == "__main__":
    dashboard = AdvancedCOVIDDashboard()
    dashboard.run_dashboard()