# 📊 Corporación Favorita Sales Forecasting

<!-- badges: start -->
[![Project Status: Active – The project has reached a stable, usable
state and is being
activelydeveloped](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
<!-- badges: end -->

This is a **Streamlit web application** that forecasts retail demand for Corporación Favorita, one of Ecuador's largest grocery retail chains. The analysis and forecasting within this app are specifically for the Guayas region. The app uses an XGBoost model trained on historical sales data from this region to predict future demand.

---

## ✨ Features

- **Interactive Forecasting**: Get instant sales predictions for various products and stores.
- **Data Visualization**: Explore historical sales trends and visualize forecast results.
- **Robust Model**: The core of the application is a powerful XGBoost model, known for its performance in tabular data forecasting.
- **Intuitive UI**: A user-friendly interface built with Streamlit, making it easy for anyone to use without technical expertise.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Onyi-RICH/Retail_Demand_Forcast.git
    cd Retail_Demand_Forcast
    ```

2.  **Install dependencies:**
    Ensure you have Python installed, then install the required packages using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    Launch the Streamlit application from your terminal.
    ```bash
    streamlit run app.py
    ```

---

## 📂 Project Structure

-   `app.py`: The main script that runs the Streamlit application and handles user interactions.
-   `requirements.txt`: Lists all the necessary Python dependencies for the project.
-   `model/`: Contains the trained machine learning model and related utility functions.
    -   `model/model_utils.py`: Functions for loading and managing the XGBoost model.
    -   `model/model.pkl`: The trained XGBoost model file.
-   `data/`: Stores the dataset and data-related processing scripts.
    -   `data/data_utils.py`: Helper functions for loading and preprocessing the data.
    -   `data/sales_data.csv`: The historical sales dataset used for training and forecasting.
-   `config.py`: Configuration file for application settings, such as titles and file paths.

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

