import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from io import BytesIO

# Configuration
st.set_page_config(layout="wide", page_title="Advanced Wind Energy Analytics", page_icon="🌬️")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .metric-card {border-radius: 10px; padding: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .turbine-card {border-left: 5px solid #4CAF50;}
    .wind-card {border-left: 5px solid #2196F3;}
    .stSelectbox, .stSlider {background-color: white;}
</style>
""", unsafe_allow_html=True)

# API Functions
@st.cache_data(ttl=3600)
def get_coordinates(location):
    url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json"
    headers = {"User-Agent": "WindEnergyApp/1.0"}
    response = requests.get(url, headers=headers)
    data = response.json()
    if data:
        return float(data[0]['lat']), float(data[0]['lon'])
    return None, None

@st.cache_data(ttl=3600)
def get_weather_data(lat, lon, days=2):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m,surface_pressure&forecast_days={days}"
    response = requests.get(url)
    return response.json()

# Turbine Models
class WindTurbine:
    def __init__(self, name, cut_in, rated, cut_out, max_power, rotor_diam):
        self.name = name
        self.cut_in = cut_in
        self.rated = rated
        self.cut_out = cut_out
        self.max_power = max_power
        self.rotor_diam = rotor_diam
    
    def power_output(self, wind_speed):
        wind_speed = np.array(wind_speed)
        power = np.zeros_like(wind_speed)
        mask = (wind_speed >= self.cut_in) & (wind_speed <= self.rated)
        power[mask] = self.max_power * ((wind_speed[mask] - self.cut_in)/(self.rated - self.cut_in))**3
        power[wind_speed > self.rated] = self.max_power
        power[wind_speed > self.cut_out] = 0
        return power

# Turbine Database
TURBINES = {
    "Vestas V80-2.0MW": WindTurbine("Vestas V80-2.0MW", 4, 15, 25, 2000, 80),
    "GE 1.5sle": WindTurbine("GE 1.5sle", 3.5, 14, 25, 1500, 77),
    "Suzlon S88-2.1MW": WindTurbine("Suzlon S88-2.1MW", 3, 12, 25, 2100, 88),
    "Enercon E-53/800": WindTurbine("Enercon E-53/800", 2.5, 13, 25, 800, 53),
    "Custom Turbine": None
}

# Air Density Calculation
def calculate_air_density(temperature, humidity, pressure):
    R_d = 287.05  # Gas constant for dry air (J/kg·K)
    R_v = 461.495  # Gas constant for water vapor (J/kg·K)
    
    # Saturation vapor pressure (Buck equation)
    e_s = 0.61121 * np.exp((18.678 - temperature/234.5) * (temperature/(257.14 + temperature)))
    
    # Actual vapor pressure
    e = (humidity / 100) * e_s
    
    # Air density (kg/m³)
    rho = (pressure * 100) / (R_d * (temperature + 273.15)) * (1 - (0.378 * e) / (pressure * 100))
    return rho

# Weibull Distribution
def weibull(x, k, A):
    return (k/A) * ((x/A)**(k-1)) * np.exp(-(x/A)**k)

# UI Components
def main():
    st.title("🌬️ Advanced Wind Energy Analytics Dashboard")
    st.markdown("""
    **A comprehensive tool for wind energy assessment, turbine performance prediction, and energy generation forecasting**
    """)
    
    with st.expander("⚙️ Configuration Panel", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            location = st.text_input("📍 Location", "Chennai, India")
            days = st.slider("📅 Forecast Days", 1, 7, 3)
        
        with col2:
            turbine_model = st.selectbox("🌀 Turbine Model", list(TURBINES.keys()), index=0)
            if turbine_model == "Custom Turbine":
                st.number_input("Cut-in Speed (m/s)", min_value=1.0, max_value=10.0, value=3.0)
                st.number_input("Rated Speed (m/s)", min_value=5.0, max_value=20.0, value=12.0)
                st.number_input("Cut-out Speed (m/s)", min_value=15.0, max_value=30.0, value=25.0)
                st.number_input("Rated Power (kW)", min_value=100, max_value=10000, value=2000)
        
        with col3:
            analysis_type = st.selectbox("📊 Analysis Type", 
                                      ["Basic Forecast", "Technical Analysis", "Financial Evaluation"])
            st.checkbox("🔍 Show Raw Data", False)
    
    if st.button("🚀 Analyze Wind Data"):
        with st.spinner("Fetching wind data and performing analysis..."):
            # Data Acquisition
            lat, lon = get_coordinates(location)
            if not lat or not lon:
                st.error("Location not found. Please try a different location name.")
                return
            
            data = get_weather_data(lat, lon, days)
            if 'error' in data:
                st.error(f"API Error: {data['error']}")
                return
            
            # Data Processing
            hours = days * 24
            times = [datetime.now() + timedelta(hours=i) for i in range(hours)]
            df = pd.DataFrame({
                "Time": times,
                "Wind Speed (m/s)": data['hourly']['wind_speed_10m'][:hours],
                "Wind Direction": data['hourly']['wind_direction_10m'][:hours],
                "Temperature (°C)": data['hourly']['temperature_2m'][:hours],
                "Humidity (%)": data['hourly']['relative_humidity_2m'][:hours],
                "Pressure (hPa)": data['hourly']['surface_pressure'][:hours]
            })
            
            # Air Density Calculation
            df['Air Density (kg/m³)'] = calculate_air_density(
                df['Temperature (°C)'], 
                df['Humidity (%)'], 
                df['Pressure (hPa)']
            )
            
            # Power Calculation
            turbine = TURBINES[turbine_model]
            df['Power Output (kW)'] = turbine.power_output(df['Wind Speed (m/s)'])
            df['Energy Output (kWh)'] = df['Power Output (kW)']  # Assuming 1 hour intervals
            
            # Weibull Distribution Fit
            wind_speeds = df['Wind Speed (m/s)']
            hist, bin_edges = np.histogram(wind_speeds, bins=20, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            try:
                params, _ = curve_fit(weibull, bin_centers, hist, p0=[2, 6])
                k, A = params
            except:
                k, A = 2, 6  # Default values if fit fails
            
            # Dashboard Layout
            st.success(f"Analysis completed for {location} (Lat: {lat:.4f}, Lon: {lon:.4f})")
            
            # Key Metrics
            st.subheader("📊 Key Performance Indicators")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🌡️ Average Wind Speed", f"{df['Wind Speed (m/s)'].mean():.2f} m/s")
            col2.metric("💨 Max Wind Speed", f"{df['Wind Speed (m/s)'].max():.2f} m/s")
            col3.metric("⚡ Total Energy Output", f"{df['Energy Output (kWh)'].sum()/1000:.2f} MWh")
            col4.metric("🌀 Predominant Direction", f"{df['Wind Direction'].mode()[0]}°")
            
            # Main Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Wind Analysis", "Turbine Performance", "Energy Forecast", "Technical Reports"])
            
            with tab1:
                st.subheader("🌪️ Wind Characteristics Analysis")
                
                # Row 1
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.line(df, x="Time", y="Wind Speed (m/s)", 
                                title="<b>1. Wind Speed Time Series</b><br>Hourly variation of wind speed",
                                template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=df['Wind Speed (m/s)'],
                        theta=df['Wind Direction'],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=df['Wind Speed (m/s)'],
                            colorscale='Viridis',
                            showscale=True
                        )
                    ))
                    fig.update_layout(
                        title="<b>2. Wind Direction vs. Speed</b><br>Polar plot showing wind patterns",
                        polar=dict(
                            radialaxis=dict(visible=True),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Row 2
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(df, x="Wind Speed (m/s)", nbins=20,
                                     title="<b>3. Wind Speed Distribution</b><br>Frequency of different wind speeds",
                                     marginal="rug")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    x = np.linspace(0, df['Wind Speed (m/s)'].max()*1.2, 100)
                    y = weibull(x, k, A)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=y, name="Weibull Fit"))
                    fig.add_trace(go.Histogram(x=df['Wind Speed (m/s)'], histnorm='probability density', 
                                            name="Actual Data", opacity=0.5))
                    fig.update_layout(
                        title=f"<b>4. Weibull Distribution Fit</b><br>Shape (k)={k:.2f}, Scale (A)={A:.2f}",
                        xaxis_title="Wind Speed (m/s)",
                        yaxis_title="Probability Density"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Row 3
                st.subheader("Advanced Wind Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.scatter(df, x="Temperature (°C)", y="Wind Speed (m/s)", 
                                   color="Humidity (%)",
                                   title="<b>5. Wind Speed vs. Temperature</b><br>Colored by Humidity",
                                   trendline="lowess")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.density_heatmap(df, x="Time", y="Wind Speed (m/s)", 
                                           title="<b>6. Wind Speed Heatmap</b><br>Time vs. Speed density",
                                           nbinsx=24*days, nbinsy=20)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("🌀 Turbine Performance Analysis")
                
                # Row 1
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.area(df, x="Time", y="Power Output (kW)", 
                                 title=f"<b>7. {turbine_model} Power Output</b><br>Hourly generation forecast",
                                 template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    wind_range = np.linspace(0, turbine.cut_out*1.2, 100)
                    power_curve = turbine.power_output(wind_range)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=wind_range, y=power_curve, name="Power Curve"))
                    fig.add_vline(x=turbine.cut_in, line_dash="dash", annotation_text=f"Cut-in: {turbine.cut_in}m/s")
                    fig.add_vline(x=turbine.rated, line_dash="dash", annotation_text=f"Rated: {turbine.rated}m/s")
                    fig.add_vline(x=turbine.cut_out, line_dash="dash", annotation_text=f"Cut-out: {turbine.cut_out}m/s")
                    fig.update_layout(
                        title=f"<b>8. {turbine_model} Power Curve</b>",
                        xaxis_title="Wind Speed (m/s)",
                        yaxis_title="Power Output (kW)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Row 2
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.scatter(df, x="Wind Speed (m/s)", y="Power Output (kW)", 
                                    color="Air Density (kg/m³)",
                                    title="<b>9. Power Output vs. Wind Speed</b><br>Colored by Air Density",
                                    trendline="lowess")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    df['Hour'] = df['Time'].dt.hour
                    hourly_avg = df.groupby('Hour').agg({
                        'Wind Speed (m/s)': 'mean',
                        'Power Output (kW)': 'mean'
                    }).reset_index()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=hourly_avg['Hour'], y=hourly_avg['Power Output (kW)'], name="Power Output"))
                    fig.add_trace(go.Scatter(x=hourly_avg['Hour'], y=hourly_avg['Wind Speed (m/s)'], 
                                           name="Wind Speed", yaxis="y2"))
                    fig.update_layout(
                        title="<b>10. Diurnal Power Pattern</b><br>Average by hour of day",
                        xaxis_title="Hour of Day",
                        yaxis_title="Power Output (kW)",
                        yaxis2=dict(title="Wind Speed (m/s)", overlaying="y", side="right"),
                        barmode="group"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("⚡ Energy Production Forecast")
                
                # Row 1
                col1, col2 = st.columns(2)
                with col1:
                    df['Cumulative Energy (kWh)'] = df['Energy Output (kWh)'].cumsum()
                    fig = px.area(df, x="Time", y="Cumulative Energy (kWh)", 
                                 title="<b>11. Cumulative Energy Production</b>",
                                 template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.box(df, x=df['Time'].dt.day_name(), y="Energy Output (kWh)", 
                               title="<b>12. Daily Energy Distribution</b>",
                               color=df['Time'].dt.day_name())
                    st.plotly_chart(fig, use_container_width=True)
                
                # Row 2
                st.subheader("Energy Potential Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.scatter(df, x="Wind Speed (m/s)", y="Energy Output (kWh)", 
                                    trendline="ols",
                                    title="<b>13. Energy vs. Wind Speed Correlation</b>",
                                    trendline_color_override="red")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    capacity_factor = (df['Energy Output (kWh)'].sum() / (turbine.max_power * hours)) * 100
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=capacity_factor,
                        title="<b>14. Capacity Factor</b>",
                        gauge={'axis': {'range': [0, 100]}},
                        domain={'x': [0, 1], 'y': [0, 1]}
                    ))
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("📝 Technical Analysis Reports")
                
                # Wind Rose
                st.markdown("### 15. Wind Rose Diagram")
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='windrose'))
                ax.bar(df['Wind Direction'], df['Wind Speed (m/s)'], normed=True, opening=0.8, edgecolor='white')
                ax.set_legend(title="Wind Speed (m/s)")
                st.pyplot(fig)
                
                # Statistical Analysis
                st.subheader("Statistical Reports")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Wind Speed Statistics")
                    st.dataframe(df['Wind Speed (m/s)'].describe().to_frame().style.format("{:.2f}"))
                
                with col2:
                    st.markdown("#### Energy Production Stats")
                    st.dataframe(df['Energy Output (kWh)'].describe().to_frame().style.format("{:.2f}"))
                
                # Download Reports
                st.markdown("---")
                st.download_button(
                    label="📥 Download Full Report (PDF)",
                    data=generate_report(df, turbine, location),
                    file_name=f"Wind_Energy_Report_{location.replace(' ','_')}.pdf",
                    mime="application/pdf"
                )

def generate_report(df, turbine, location):
    # This would be replaced with actual PDF generation code
    # For demo purposes, returning a dummy PDF
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Wind Energy Report for {location}", ln=1, align="C")
    pdf.output("report.pdf")
    with open("report.pdf", "rb") as f:
        return f.read()

if __name__ == "__main__":
    main()
