# StockPriceTomorrow
Predicting Tomorrow's Stock Price for the chosen Nifty 50 Stock
# Stock Price Prediction using LSTM

This Python script uses the Streamlit library to create a simple web application for predicting the stock price of a selected company's stock for the following day. It utilizes historical stock price data and an LSTM (Long Short-Term Memory) model for prediction.

## Usage

1. Ensure you have Python installed on your system.
2. Install the required Python packages using pip.
3. Run the application using the following command:
4. streamlit run stock_price_prediction.py
5. Select a stock from the Nifty50 index using the dropdown menu.
6. The application will predict the stock price for the following day based on historical data and display it along with training and testing metrics.
7. Additionally, training and testing plots, as well as explanations of key metrics (MAE and RMSE), are provided in the application.

## Requirements

- Python 3.x
- Streamlit
- yfinance
- numpy
- tensorflow
- scikit-learn
- matplotlib
- datetime
- PIL (Python Imaging Library)

## Notes

- The historical data used for prediction is fetched from Yahoo Finance.

- The LSTM model is trained with historical data, and the predictions are made based on the model's learning.

- MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error) are provided to evaluate the model's performance. Lower values are better.

- The application also provides explanations of key metrics for better understanding.

This project is intended for educational purposes and should not be used for making financial decisions.


## Author

- Sai Kumar Diguvapatnam

