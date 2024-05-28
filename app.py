import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, 
    HistGradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
import streamlit_shadcn_ui as ui

# Load the dataset
df = pd.read_csv('Dataset_modified_pom.csv')
df = df.drop(['Product Availibility index', 'City'], axis=1)

# Process date column
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Features and target variables
X = df[['Year', 'Month', 'Day']]
y_mrp = df['MRP']
y_sp = df['SP']
y_msp = df['MSP']

# Split the data
X_train, X_test, y_mrp_train, y_mrp_test, y_sp_train, y_sp_test, y_msp_train, y_msp_test = train_test_split(
    X, y_mrp, y_sp, y_msp, test_size=0.2, random_state=42
)

# Header and caption
st.header("Seller Name: Ashok")
st.caption("Optimise your sales with Real-time online retail market analytics")

st.subheader("Dashboard")

# Create tabs
selected_tab = st.radio("Select Tab", ['Today', 'History/Forecast'])

if selected_tab == 'Today':
    selected_date = st.date_input("Select Date")
    selected_date_str = selected_date.strftime('%d-%m-%Y')

    filtered_df = df[df['Date'].dt.strftime('%d-%m-%Y') == selected_date_str]

    if not filtered_df.empty:
        st.write(f"Selected Date: {selected_date_str}")
        st.info(f"MSP: {float(filtered_df['MSP'].values)}")
        st.info(f"MRP: {float(filtered_df['MRP'].values)}")
        opt = float(filtered_df['Base Price'].values)
        st.success(f"Optimised Price: {opt}")

        min_price, max_price = st.slider(
            "Price Range", 
            float(filtered_df['MRP'].min()), 
            float(filtered_df['MSP'].max()), 
            (float(filtered_df['MRP'].min()), float(filtered_df['MSP'].max()))
        )

        value = st.text_area("Enter the value between the above range", value="2.0")
        value = float(value)

        filtered_price_range_df = filtered_df[(filtered_df['MRP'] >= min_price) & (filtered_df['MSP'] <= max_price)]

        if not filtered_price_range_df.empty:
            selected_row_index = st.selectbox("Select Row", range(len(filtered_price_range_df)), format_func=lambda i: f"Row {i}")
            selected_row = filtered_price_range_df.iloc[selected_row_index]

            # Scatter Plot and Trendline
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.scatter(selected_row['MRP'], selected_row['DR1'], color='red', label='MRP Data Points')
            ax.scatter(selected_row['SP'], selected_row['DR2'], color='green', label='SP Data Points')
            ax.scatter(selected_row['MSP'], selected_row['DR3'], color='blue', label='MSP Data Points')

            prices = list(selected_row[['MRP', 'SP', 'MSP']])
            dr_values = list(selected_row[['DR1', 'DR2', 'DR3']])
            z = np.polyfit(prices, dr_values, 2)
            p = np.poly1d(z)
            x_trendline = np.linspace(min(prices), max(prices), 100)
            ax.plot(x_trendline, p(x_trendline), linestyle='dashed', color='black', label='Trendline')
            ax.set_xlabel('Price (MRP, SP, MSP)')
            ax.set_ylabel('DRs (DR1, DR2, DR3)')
            ax.set_title('Scatter Plot of MRP, SP, MSP vs. DR1, DR2, DR3')
            ax.legend()

            # Display Trendline Function
            st.info(f"Trendline function: {np.poly1d(z)}")
            # Calculate DRs at Base Price
            base_price = float(selected_row['Base Price'])
            dr_at_base_price = p(base_price)
            st.info(f"Demand Ratio (DR) at Base Price ({base_price}): {dr_at_base_price}")

            with st.container():
                st.info("Selected Row Dataset Values:")
                st.write(f"- **SP:** {selected_row['SP']}")
                st.write(f"- **Base Price:** {selected_row['Base Price']}")

            st.info("Demand Ratios:")
            cols = st.columns(4)
            with cols[0]:
                ui.metric_card(title="Demand Ratio(MRP)", content=f"{selected_row['DR1']:.2f}", key="card1")
            with cols[1]:
                ui.metric_card(title="Demand Ratio(SP)", content=f"{selected_row['DR2']:.2f}", key="card2")
            with cols[2]:
                ui.metric_card(title="Demand Ratio(MSP)", content=f"{selected_row['DR3']:.2f}", key="card3")
            with cols[3]:
                ui.metric_card(title="Order Probability", content=f"{selected_row['Probable Index for Bp']:.2f}", key="card4")

            area_under_curve = abs(np.trapz(p(x_trendline), x=x_trendline)) * 100
            turn_75 = abs(0.75 * area_under_curve)
            turn_90 = abs(0.90 * area_under_curve)
            x2 = selected_row['MRP']
            x1 = selected_row['MSP']
            avg_new = (area_under_curve) / (x2 - x1)
            result = abs(p(value) * value)

            with st.container():
                st.error("Demand Values")
                st.info(f"- **Area Under the Curve:** {area_under_curve}")
                st.success(f"- **75% of Turnover:** {turn_75}")
                st.info(f"- **90% of Turnover:** {turn_90}")
                st.success(f"- **Average Turnover:** {avg_new}")
                st.info(f"- **Total Turnover at Given Price:** {result}")

            st.info("Price Statistics:")
            cols = st.columns(3)
            if not filtered_price_range_df.empty:
                selected_row_mean_price = filtered_price_range_df['MRP'].mean()
                selected_row_median_price = filtered_price_range_df['MRP'].median()
                selected_row_mode_price = filtered_price_range_df['MRP'].mode().iloc[0]

                with cols[0]:
                    ui.metric_card(title="Mean Price", content=f"₹{selected_row_mean_price:.2f}", key="mean_card")
                with cols[1]:
                    ui.metric_card(title="Median Price", content=f"₹{selected_row_median_price:.2f}", key="median_card")
                with cols[2]:
                    ui.metric_card(title="Mode Price", content=f"₹{selected_row_mode_price:.2f}", key="mode_card")

            st.pyplot(fig)

        else:
            st.warning("No data available for the selected price range.")
    else:
        st.warning("No data available for the selected date.")

elif selected_tab == 'History/Forecast':
    algorithm = st.selectbox("Select Algorithm", ["Linear Regression", "Extra Trees", "Decision Trees", "Gradient Boosting", "HistGBM", "k-Nearest Neighbour", "XgBoost", "Random Forest Regressor"])

    model_dict = {
        "Linear Regression": LinearRegression,
        "Extra Trees": ExtraTreesRegressor,
        "Decision Trees": DecisionTreeRegressor,
        "Gradient Boosting": GradientBoostingRegressor,
        "HistGBM": HistGradientBoostingRegressor,
        "k-Nearest Neighbour": KNeighborsRegressor,
        "XgBoost": XGBRegressor,
        "Random Forest Regressor": RandomForestRegressor,
    }

    model_class = model_dict[algorithm]

    model_mrp = model_class()
    model_sp = model_class()
    model_msp = model_class()

    model_mrp.fit(X_train, y_mrp_train)
    model_sp.fit(X_train, y_sp_train)
    model_msp.fit(X_train, y_msp_train)

    df['MRP_Pred'] = model_mrp.predict(X)
    df['SP_Pred'] = model_sp.predict(X)
    df['MSP_Pred'] = model_msp.predict(X)

    if st.button("Show Result"):
        df_resampled = df.resample('M', on='Date').mean()
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(df_resampled['MRP'], label='MRP (Actual)', color='red')
        ax.plot(df_resampled['SP'], label='SP (Actual)', color='green')
        ax.plot(df_resampled['MSP'], label='MSP (Actual)', color='blue')

        ax.plot(df_resampled['MRP_Pred'], label='MRP (Predicted)', color='red', linestyle='dashed')
        ax.plot(df_resampled['SP_Pred'], label='SP (Predicted)', color='green', linestyle='dashed')
        ax.plot(df_resampled['MSP_Pred'], label='MSP (Predicted)', color='blue', linestyle='dashed')

        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'Price predictions for {algorithm}')
        ax.legend()
        st.pyplot(fig)
