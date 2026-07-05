import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('train.csv')
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y', errors='coerce')
        if df['Order Date'].isnull().all():
            df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Year'] = df['Order Date'].dt.year
        df['Month'] = df['Order Date'].dt.month
        return df.dropna(subset=['Order Date'])
    except Exception as e:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("Please ensure 'train.csv' is in the same directory as this app.")
    st.stop()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"])

if page == "Sales Overview":
    st.title("Sales Overview Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        yearly_sales = df.groupby('Year')['Sales'].sum().reset_index()
        fig1 = px.bar(yearly_sales, x='Year', y='Sales', title="Total Sales by Year")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        monthly_sales['Order Date'] = monthly_sales['Order Date'].dt.to_timestamp()
        fig2 = px.line(monthly_sales, x='Order Date', y='Sales', title="Monthly Sales Trend")
        st.plotly_chart(fig2, use_container_width=True)
        
    st.subheader("Filter by Region and Category")
    reg = st.selectbox("Select Region", ['All'] + list(df['Region'].unique()))
    cat = st.selectbox("Select Category", ['All'] + list(df['Category'].unique()))
    
    filtered_df = df
    if reg != 'All': filtered_df = filtered_df[filtered_df['Region'] == reg]
    if cat != 'All': filtered_df = filtered_df[filtered_df['Category'] == cat]
    st.dataframe(filtered_df.head(100))

elif page == "Forecast Explorer":
    st.title("Forecast Explorer (Prophet Model)")
    
    # 1. Dropdown to select: Category or Region
    filter_type = st.radio("Select Filter Type", ["Overall", "Category", "Region"])
    
    if filter_type == "Category":
        target = st.selectbox("Select Category", df['Category'].unique())
        data = df[df['Category'] == target].groupby('Order Date')['Sales'].sum().resample('ME').sum().reset_index()
    elif filter_type == "Region":
        target = st.selectbox("Select Region", df['Region'].unique())
        data = df[df['Region'] == target].groupby('Order Date')['Sales'].sum().resample('ME').sum().reset_index()
    else:
        target = "Overall Sales"
        data = df.groupby('Order Date')['Sales'].sum().resample('ME').sum().reset_index()
        
    # 2. Date range slider to select forecast horizon (1, 2, or 3 months ahead)
    months = st.slider("Select Forecast Horizon (Months ahead)", 1, 3, 3)
    
    if st.button("Generate Forecast"):
        with st.spinner("Training Model..."):
            data.columns = ['ds', 'y']
            
            # 3. Display the forecast output from your best model
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(data)
            future = m.make_future_dataframe(periods=months, freq='MS')
            forecast = m.predict(future)
            
            fig = px.line(forecast, x='ds', y='yhat', title=f"{target} - {months} Month Forecast")
            fig.add_scatter(x=data['ds'], y=data['y'], mode='lines', name='Actual Historical Sales')
            st.plotly_chart(fig)
            
            # 4. Show MAE and RMSE of the model below the chart
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            predictions = m.predict(data)
            mae = mean_absolute_error(data['y'], predictions['yhat'])
            rmse = np.sqrt(mean_squared_error(data['y'], predictions['yhat']))
            
            st.subheader("Model Evaluation Metrics (on historical data)")
            col1, col2 = st.columns(2)
            col1.metric("MAE (Mean Absolute Error)", f"${mae:,.2f}")
            col2.metric("RMSE (Root Mean Squared Error)", f"${rmse:,.2f}")

elif page == "Anomaly Report":
    st.title("Anomaly Report")
    
    # Aggregate to weekly
    weekly = df.groupby('Order Date')['Sales'].sum().resample('W-SUN').sum().reset_index()
    
    # 1. Z-Score Anomaly Detection
    weekly['Rolling_Mean'] = weekly['Sales'].shift(1).rolling(window=4).mean()
    weekly['Rolling_Std'] = weekly['Sales'].shift(1).rolling(window=4).std()
    weekly['Z_Score'] = (weekly['Sales'] - weekly['Rolling_Mean']) / weekly['Rolling_Std']
    anomalies_z = weekly[weekly['Z_Score'].abs() > 2]
    
    # 2. Isolation Forest (Multi-source)
    from sklearn.ensemble import IsolationForest
    weekly['Year'] = weekly['Order Date'].dt.year
    try:
        vgsales = pd.read_csv('vgsales.csv')
        vgsales = vgsales.dropna(subset=['Year'])
        vgsales['Year'] = vgsales['Year'].astype(int)
        annual_vg = vgsales.groupby('Year')['Global_Sales'].sum().reset_index()
        annual_vg.rename(columns={'Global_Sales': 'VG_Global_Sales'}, inplace=True)
        weekly = weekly.merge(annual_vg, on='Year', how='left')
        weekly['VG_Global_Sales'] = weekly['VG_Global_Sales'].fillna(0)
    except:
        weekly['VG_Global_Sales'] = 0
        
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    weekly['Anomaly_IF'] = iso_forest.fit_predict(weekly[['Sales', 'VG_Global_Sales']])
    anomalies_if = weekly[weekly['Anomaly_IF'] == -1]
    
    # Plot both
    fig = px.line(weekly, x='Order Date', y='Sales', title="Weekly Sales Anomalies")
    fig.add_scatter(x=anomalies_if['Order Date'], y=anomalies_if['Sales'], mode='markers', name='Isolation Forest', marker=dict(color='red', size=10))
    fig.add_scatter(x=anomalies_z['Order Date'], y=anomalies_z['Sales'], mode='markers', name='Z-Score (>2)', marker=dict(color='orange', symbol='x', size=10))
    st.plotly_chart(fig)
    
    st.subheader("Detected Anomalies (Isolation Forest)")
    st.dataframe(anomalies_if[['Order Date', 'Sales']])
    
    st.subheader("Detected Anomalies (Z-Score)")
    st.dataframe(anomalies_z[['Order Date', 'Sales', 'Z_Score']])

elif page == "Product Demand Segments":
    st.title("Product Demand Segments (K-Means)")
    subcat_sales = df.groupby('Sub-Category')['Sales'].sum()
    subcat_orders = df.groupby('Sub-Category')['Order ID'].nunique()
    feat_vol = df.groupby('Sub-Category')['Quantity'].sum() if 'Quantity' in df.columns else df.groupby('Sub-Category').size()
    feat_aov = subcat_sales / subcat_orders
    monthly_subcat = df.groupby(['Sub-Category', pd.Grouper(key='Order Date', freq='ME')])['Sales'].sum().unstack(fill_value=0)
    feat_volatility = monthly_subcat.std(axis=1)
    
    cluster_df = pd.DataFrame({
        'Total Volume': feat_vol,
        'Volatility': feat_volatility,
        'Avg Order Value': feat_aov
    }).fillna(0)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_df)
    km = KMeans(n_clusters=4, random_state=42)
    cluster_df['Cluster'] = km.fit_predict(scaled_data)
    cluster_df['Cluster Name'] = cluster_df['Cluster'].map({
        0: 'Steady Bulk', 1: 'High Value Volatile', 2: 'Low Volume Declining', 3: 'Star Performers'
    })
    
    fig = px.scatter(cluster_df.reset_index(), x='Total Volume', y='Avg Order Value', color='Cluster Name', 
                     hover_data=['Sub-Category'], title="Product Segmentation")
    st.plotly_chart(fig)
    st.subheader("Sub-category Segments")
    st.dataframe(cluster_df.reset_index()[['Sub-Category', 'Cluster Name', 'Total Volume', 'Avg Order Value']])
