# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 17:56:20 2023

@author: Sai
"""

import streamlit as st
import datetime
today = datetime.date.today()

def itc(selected_stock):
    import yfinance as yf
    import numpy as np
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    df_stock = yf.download(selected_stock, start="2018-01-01", end="2023-10-10")

    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_stock["Close"].values.reshape(-1, 1))

    # Create training and testing data sets
    train_size = int(len(scaled_data) * 0.7)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Create sequences of data
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    seq_length = 30
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    # Build the LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(X_train, y_train, epochs=5, batch_size=1)

    # Make predictions on the training data
    train_predictions = model.predict(X_train)
    train_predictions = scaler.inverse_transform(train_predictions)
    
    # Calculate MAE and RMSE for the training data
    train_mae = mean_absolute_error(scaler.inverse_transform(y_train), train_predictions)
    train_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_train), train_predictions))
    
    # Make predictions on the test data
    test_predictions = model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions)
    
    # Calculate MAE and RMSE for the test data
    test_mae = mean_absolute_error(scaler.inverse_transform(y_test), test_predictions)
    test_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test), test_predictions))


    import matplotlib.pyplot as plt

    # Make predictions on the training data
    train_predictions = model.predict(X_train)
    train_predictions = scaler.inverse_transform(train_predictions)

    # Make predictions on the test data
    test_predictions = model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions)
    
 
    # Plot the actual vs predicted values for the training data
    plt.figure(figsize=(12, 6))
    plt.plot(df_stock.iloc[seq_length:train_size].index, df_stock.iloc[seq_length:train_size]["Close"], label="Actual")
    plt.plot(df_stock.iloc[seq_length:train_size].index, train_predictions, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted (Training Data)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    
    plt.savefig("training_plot.png")
    plt.close()  # Close the plot to release resources

    # Plot the actual vs predicted values for the test data
    plt.figure(figsize=(12, 6))
    plt.plot(df_stock.iloc[train_size+seq_length:].index, df_stock.iloc[train_size+seq_length:]["Close"], label="Actual")
    plt.plot(df_stock.iloc[train_size+seq_length:].index, test_predictions, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted (Test Data)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    
    
    plt.savefig("test_plot.png")
    plt.close()  # Close the plot to release resources


    # Get the historical data for the last 30 days
    history_data = yf.Ticker(selected_stock).history(period="30d")

    # Extract the closing price values from the historical data
    history_close = history_data["Close"].values
    

    # Normalize the historical data
    history_scaled_data = scaler.transform(history_close.reshape(-1, 1))

    # Create a sequence of data for the historical data
    history_sequence = history_scaled_data[-seq_length:].reshape(1, seq_length, 1)

    # Make a prediction for the next day's stock price based on the historical data
    history_prediction = model.predict(history_sequence)
    history_prediction = scaler.inverse_transform(history_prediction)

    #training_plot_path = training_plot_path.replace("\\","//")
    #test_plot_path = test_plot_path.replace("\\","//")
    
    return history_prediction[0][0], round(train_mae,2),round(train_rmse,2),round(test_mae,2),round(test_rmse,2),



def main():
    from PIL import Image
    st.title("Stock Price Prediction for the following day")
    st.write("")
    st.write("")
    st.text("Pls hold for sometime, as we are predicting the Stock price, \nbased on the historical data(from 2018).")
    st.write("")
    
    nifty_50_stocks = ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
                  'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS',
                  'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS','HDFCBANK.NS',
                  'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS',
                  'ITC.NS', 'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LTIM.NS', 'LT.NS',
                  'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS',
                  'SBILIFE.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS',
                  'TECHM.NS', 'TITAN.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']
    
    selected_stock = st.sidebar.selectbox('Select Nifty50 stock(Ignore .NS)', nifty_50_stocks)

    pred_price,train_mae,train_rmse,test_mae,test_rmse = itc(selected_stock)
     
    st.text("Today's date is")
    st.write(today)

    stock_name = selected_stock.replace(".NS","")
    st.text("Selected Stock is : {}".format(stock_name))
    
    st.text("Predicted Stock price of {} for tomorrow based on the historical data".format(stock_name))
    
    st.write(pred_price)
    
    train_image = Image.open('training_plot.png')
    st.image(train_image, caption='Training Data')
    st.text("Mean Absolute Error(MAE) for Training Data:")
    st.write(train_mae)
    st.text("Root Mean Square Error(RMSE) for Training Data:")
    st.write(train_rmse) 
    
    test_image = Image.open('test_plot.png')
    st.image(test_image, caption='Test Data')  
    st.text("Mean Absolute Error(MAE) for Test Data:")
    st.write(test_mae)
    st.text("Root Mean Square Error(RMSE) for Test Data:")
    st.write(test_rmse) 
  
    st.markdown(
        """
        
        Key Metrics Explanation:
            
        Mean Absolute Error (MAE): MAE measures how far, on average, your predictions are from the actual values. It provides a simple, average difference between predicted and actual values.
    
        Root Mean Squared Error (RMSE): RMSE is like MAE but gives more weight to larger prediction errors. It's a measure of how spread out those errors are, taking both small and large errors into account.
    
        In both cases, lower values are better, indicating more accurate predictions.
        """
    )
    
if __name__ == '__main__':
    main()

