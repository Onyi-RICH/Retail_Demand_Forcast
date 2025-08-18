import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from app import config
from model.model_utils import load_model
from data.data_utils import load_data, load_features

# -------------------
# Trim top spacing
# -------------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.25rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------
# Title
# -------------------
st.markdown(
    f"<h1 style='text-align:center; color:#1E88E5;'>{config.APP_TITLE}</h1>",
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style="display:flex; align-items:center; width:100%;">
        <hr style="flex:1; border:none; border-top:1px solid #bbb; margin:0 10px;">
        <p style="margin:0; font-size:16px; font-weight:bold; text-align:center;">
            📊 Forecasting demand in <b>Guayas region</b>
        </p>
        <hr style="flex:1; border:none; border-top:1px solid #bbb; margin:0 10px;">
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------
# Load model, data, features
# -------------------
model = load_model()
data = load_data()
feature_columns = load_features()  # list of feature names used to train the model
data['date'] = pd.to_datetime(data['date'])

# -------------------
# Layout
# -------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Settings")

    # Select store and item
    store_nbr = st.selectbox('Select Store ID', data['store_nbr'].unique())
    data_store = data[data['store_nbr'] == store_nbr]

    item_nbr = st.selectbox('Select Item ID', data_store['item_nbr'].unique())
    series_df = data_store[data_store['item_nbr'] == item_nbr].copy()

    # Date range selector (default to March 2014)
    min_date = series_df['date'].min()
    max_date = series_df['date'].max()
    default_start = pd.to_datetime("2014-03-01")
    default_end   = pd.to_datetime("2014-03-31")

    date_range = st.date_input(
        "Select forecast period",
        value=[default_start, default_end],
        min_value=min_date,
        max_value=max_date,
        key="forecast_date_selector"
    )

# ---- Build predictions on FULL history for this store-item ----
# (So lag/rolling features remain valid. We'll slice to the chosen window after predicting.)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date, end_date = default_start, default_end

# Keep only features that actually exist in the data
available_features = [c for c in feature_columns if c in series_df.columns]
missing = sorted(set(feature_columns) - set(available_features))
if missing:
    st.info(f"Using {len(available_features)} features. Missing in data: {missing}")

X_full = series_df[available_features].copy()
# Ensure we don't pass target/date by accident
for col in ['unit_sales', 'date']:
    if col in X_full.columns:
        X_full.drop(columns=[col], inplace=True)

# Predict across the full series, then cut to the selected dates
if not X_full.empty:
    pred_full = model.predict(X_full)
else:
    pred_full = pd.Series([0] * len(series_df), index=series_df.index)

result_full = pd.DataFrame({
    'date': series_df['date'],
    'actual_sales': series_df['unit_sales'],
    'predicted_sales': pred_full
})

# Slice to chosen window for display/metrics
mask = (result_full['date'] >= start_date) & (result_full['date'] <= end_date)
result_df = result_full.loc[mask].reset_index(drop=True)

# ----- Column 1: Metrics -----
with col1:
    st.markdown("---")
    st.subheader("📊 Sales Summary")

    if result_df.empty:
        st.warning("No data in the selected window. Try expanding the date range.")
    else:
        avg_actual = round(result_df['actual_sales'].mean(), 2)
        avg_pred   = round(result_df['predicted_sales'].mean(), 2)
        st.markdown(
            f"<p style='font-size:14px'><b>Average Sales</b><br>"
            f"Actual: <b>{avg_actual:.2f}</b><br>"
            f"Predicted: <b>{avg_pred:.2f}</b></p>",
            unsafe_allow_html=True
        )

        max_actual = round(result_df['actual_sales'].max(), 2)
        max_pred   = round(result_df['predicted_sales'].max(), 2)
        st.markdown(
            f"<p style='font-size:14px'><b>Max Sales</b><br>"
            f"Actual: <b>{max_actual:.2f}</b><br>"
            f"Predicted: <b>{max_pred:.2f}</b></p>",
            unsafe_allow_html=True
        )

        min_actual = round(result_df['actual_sales'].min(), 2)
        min_pred   = round(result_df['predicted_sales'].min(), 2)
        st.markdown(
            f"<p style='font-size:14px'><b>Min Sales</b><br>"
            f"Actual: <b>{min_actual:.2f}</b><br>"
            f"Predicted: <b>{min_pred:.2f}</b></p>",
            unsafe_allow_html=True
        )

# ----- Column 2: Plot + Table -----
with col2:
    st.subheader("📈 Forecast Plot")
    if result_df.empty:
        st.info("Nothing to plot for the selected window.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(result_df['date'], result_df['actual_sales'],
                label='Actual Sales', color='blue', marker='o')
        ax.plot(result_df['date'], result_df['predicted_sales'],
                label='Predicted Sales', color='red', marker='x')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.set_title(
            f"Forecast for Store {store_nbr}, Item {item_nbr} "
            f"({start_date.date()} to {end_date.date()})"
        )
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("📋 Forecast Table")
    st.dataframe(result_df, height=400)
