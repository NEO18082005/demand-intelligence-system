# 📊 Retail Sales Forecasting & Demand Intelligence System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url-here.streamlit.app/)
*(Note: Replace the URL above with your actual live Streamlit link once deployed!)*

## 📝 Project Overview
This project is an end-to-end AI-driven Sales Forecasting and Demand Intelligence system designed for a retail/e-commerce company. By leveraging 4 years of historical transaction data, this system predicts future product demand with high accuracy, automatically flags anomalous sales spikes or drops, and segments the product catalog based on distinct purchasing behaviors. 

This tool enables supply chain and finance teams to optimize inventory levels, minimize overstock costs, and ensure high-demand items are consistently available, driving both operational efficiency and revenue growth.

## ✨ Features
1. **Sales Overview Dashboard**: Interactive filters to view historical sales trends by region and product category.
2. **Forecast Explorer**: An interactive 3-month forecasting tool powered by **Facebook Prophet**, allowing users to generate localized predictions for specific categories or regions with evaluated MAE/RMSE metrics.
3. **Anomaly Detection**: Uses **Isolation Forest** and **Z-Score** statistical methods to flag historically unusual sales weeks (e.g., massive holiday spikes or supply chain disruptions).
4. **Product Demand Segmentation**: Applies **K-Means Clustering** and PCA to segment product sub-categories into actionable demand profiles (e.g., "Steady Bulk", "High Value Volatile").

## 🛠️ Technology Stack
*   **Language:** Python 3.x
*   **Data Manipulation:** Pandas, NumPy
*   **Time Series Forecasting:** Statsmodels (SARIMA), Facebook Prophet, XGBoost
*   **Machine Learning (Clustering & Anomalies):** Scikit-Learn
*   **Visualization:** Matplotlib, Seaborn, Plotly
*   **Web Application / Deployment:** Streamlit

## 📈 Key Findings & Recommendations

### 1. Forecasting & Growth
*   **Revenue Drivers:** The *Technology* category remains the highest revenue generator, consistently outperforming Furniture and Office Supplies.
*   **Regional Consistency:** The *West* region has demonstrated the most consistent and stable sales growth year-over-year.
*   **Seasonality:** There is a massive, predictable sales spike occurring every November and December, aligning with holiday shopping.
*   **Recommendation:** Given the massive predictable spikes in Q4, Supply Chain should automate holiday procurement and lock in vendor contracts by late Q3 to avoid premium last-minute logistics costs.

### 2. Anomaly Detection
*   **Recurring Late November Spikes:** Sales volume deviated by over +3 standard deviations due to Black Friday and Cyber Monday.
*   **Mid-January Drops:** Sharp declines below expected post-holiday levels, likely caused by post-holiday buying fatigue and inventory restocking delays.
*   **Recommendation:** Compare Z-Score (sensitive to sudden local momentum shifts) against Isolation Forest (global dataset distribution) to separate isolated marketing wins from global supply chain failures.

### 3. Product Demand Segmentation (Stocking Strategies)
Using K-Means clustering, the catalog was segmented into four distinct profiles:
*   🟢 **Steady Bulk (High Vol, Stable):** Maintain high safety stock levels and utilize automated reordering.
*   🟡 **High Value Volatile:** Implement Just-In-Time (JIT) inventory management or vendor drop-shipping to minimize warehousing costs.
*   🔴 **Low Volume Declining:** Halt reorders, phase out inventory, and apply aggressive discounts.
*   ⭐ **Star Performers (High Growth):** Prioritize warehouse allocation and increase buffer stock to capture upside.

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/NEO18082005/retail-sales-forecasting-dashboard.git
   cd retail-sales-forecasting-dashboard
