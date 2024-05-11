import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objs as go
import ta
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID")

if stock:  # Check if stock input is not empty
    try:
        end = datetime.now()
        start = datetime(end.year - 20, end.month, end.day)

        google_data = yf.download(stock, start, end)

        # Add today's data to the DataFrame
        today_data = yf.download(stock, end, end)
        google_data = pd.concat([google_data, today_data])

        st.subheader("Stock Data")
        st.write(google_data)

        # Get live price
        live_price = yf.Ticker(stock).history(period="1d")["Close"].iloc[-1]
        st.subheader("Live Price")
        st.write(f"The current live price for {stock} is: {live_price}")

        splitting_len = int(len(google_data) * 0.7)
        x_test = pd.DataFrame(google_data.Close[splitting_len:])

        st.subheader("Moving Averages")

        st.subheader('Original Close Price and MA for 250 days')
        google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
        fig_ma_250 = go.Figure()
        fig_ma_250.add_trace(go.Scatter(x=google_data.index, y=google_data['Close'], mode='lines', name='Close Price'))
        fig_ma_250.add_trace(go.Scatter(x=google_data.index, y=google_data['MA_for_250_days'], mode='lines', name='MA for 250 days'))
        st.plotly_chart(fig_ma_250)

        # Add technical indicators
        google_data['RSI'] = ta.momentum.RSIIndicator(google_data['Close'], window=14).rsi()
        macd = ta.trend.MACD(google_data['Close'])
        google_data['MACD'] = macd.macd()
        google_data['MACD_signal'] = macd.macd_signal()
        google_data['SMA_50'] = google_data['Close'].rolling(window=50).mean()

        st.subheader("Technical Indicators")

        # Plot RSI
        st.subheader("Relative Strength Index (RSI)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=google_data.index, y=google_data['RSI'], mode='lines', name='RSI'))
        st.plotly_chart(fig_rsi)

        # Plot MACD and Signal Line
        st.subheader("Moving Average Convergence Divergence (MACD)")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=google_data.index, y=google_data['MACD'], mode='lines', name='MACD'))
        fig_macd.add_trace(go.Scatter(x=google_data.index, y=google_data['MACD_signal'], mode='lines', name='Signal Line'))
        st.plotly_chart(fig_macd)

        # Plot SMA 50
        st.subheader("Simple Moving Average (SMA) - 50 Days")
        fig_sma_50 = go.Figure()
        fig_sma_50.add_trace(go.Scatter(x=google_data.index, y=google_data['SMA_50'], mode='lines', name='SMA 50'))
        st.plotly_chart(fig_sma_50)

        # Load MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(x_test[['Close']])

        x_data = []
        y_data = []

        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i-100:i])
            y_data.append(scaled_data[i])

        x_data, y_data = np.array(x_data), np.array(y_data)

        model = load_model("Latest_stock_price_model.keras")

        predictions = model.predict(x_data)

        inv_pre = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_data)

        ploting_data = pd.DataFrame(
          {
            'original_test_data': inv_y_test.reshape(-1),
              'predictions' : inv_pre.reshape(-1) 
          } , 
            index = google_data.index[splitting_len+100:]
        )
        st.subheader("Original values vs Predicted values")
        st.write(ploting_data)

        st.subheader('Original Close Price vs Predicted Close Price')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=google_data.index[:splitting_len+100], y=google_data.Close[:splitting_len+100], mode='lines', name='Original Test data'))
        fig.add_trace(go.Scatter(x=ploting_data.index, y=ploting_data['predictions'], mode='lines', name='Predicted Test data'))
        st.plotly_chart(fig)

        st.subheader("Last Predicted Closing Price")
        last_predicted_price = ploting_data['predictions'].iloc[-1]
        st.write(f"The last predicted closing price for {stock} is: {last_predicted_price}")

        # Predicting the stock price for the next 5 days
        future_dates = [end + timedelta(days=i) for i in range(1, 6)]
        future_predictions = []

        for _ in range(5):
            last_100_days = scaled_data[-100:]
            last_100_days = np.reshape(last_100_days, (1, 100, 1))
            prediction = model.predict(last_100_days)
            future_predictions.append(prediction[0][0])
            scaled_data = np.append(scaled_data, prediction[0].reshape(-1,1))

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        future_price_df = pd.DataFrame(future_predictions, index=future_dates, columns=["Predicted Close Price"])
        st.subheader("Predicted Close Prices for the Next 5 Days")
        st.write(future_price_df)

        # Candlestick Chart
        candlestick_data = google_data.iloc[splitting_len:]

        fig = go.Figure(data=[go.Candlestick(x=candlestick_data.index,
                                             open=candlestick_data['Open'],
                                             high=candlestick_data['High'],
                                             low=candlestick_data['Low'],
                                             close=candlestick_data['Close'])])
        st.subheader("Candlestick Chart")
        st.plotly_chart(fig)

        # OHLC Bars
        ohlc_data = candlestick_data.resample('1W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})

        fig = go.Figure(data=go.Ohlc(x=ohlc_data.index,
                                      open=ohlc_data['Open'],
                                      high=ohlc_data['High'],
                                      low=ohlc_data['Low'],
                                      close=ohlc_data['Close']))
        st.subheader("OHLC Bars")
        st.plotly_chart(fig)

        # Area Chart
        st.subheader("Area Chart")
        area_chart = go.Figure(go.Scatter(x=google_data.index, y=google_data['Close'], fill='tozeroy'))
        st.plotly_chart(area_chart)

        # Generating Buy/Sell signals based on Moving Averages (MA) and Relative Strength Index (RSI)
        google_data['Buy_Signal_MA'] = np.where(google_data['Close'] > google_data['MA_for_250_days'], 1, 0)
        google_data['Sell_Signal_MA'] = np.where(google_data['Close'] < google_data['MA_for_250_days'], -1, 0)

        # Define thresholds for RSI for buy and sell signals
        buy_threshold_rsi = 30
        sell_threshold_rsi = 70

        google_data['Buy_Signal_RSI'] = np.where(google_data['RSI'] < buy_threshold_rsi, 1, 0)
        google_data['Sell_Signal_RSI'] = np.where(google_data['RSI'] > sell_threshold_rsi, -1, 0)

        # Combine buy/sell signals
        google_data['Combined_Buy_Signal'] = google_data['Buy_Signal_MA'] + google_data['Buy_Signal_RSI']
        google_data['Combined_Sell_Signal'] = google_data['Sell_Signal_MA'] + google_data['Sell_Signal_RSI']

        # Plot Buy/Sell signals
        st.subheader("Buy/Sell Signals")
        fig_buy_sell = go.Figure()
        fig_buy_sell.add_trace(go.Scatter(x=google_data.index, y=google_data['Close'], mode='lines', name='Close Price'))
        fig_buy_sell.add_trace(go.Scatter(x=google_data.loc[google_data['Combined_Buy_Signal'] == 1].index, y=google_data.loc[google_data['Combined_Buy_Signal'] == 1]['Close'], mode='markers', name='Buy Signal', marker=dict(color='green', size=10)))
        fig_buy_sell.add_trace(go.Scatter(x=google_data.loc[google_data['Combined_Sell_Signal'] == -1].index, y=google_data.loc[google_data['Combined_Sell_Signal'] == -1]['Close'], mode='markers', name='Sell Signal', marker=dict(color='red', size=10)))
        st.plotly_chart(fig_buy_sell)

        from sklearn.metrics import mean_absolute_error, mean_squared_error

        # Get the actual closing prices for the test data
        actual_prices = ploting_data['original_test_data']

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(actual_prices, ploting_data['predictions'])

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(actual_prices, ploting_data['predictions'])

        st.subheader("Evaluation Metrics")
        st.write(f"Mean Absolute Error (MAE): {mae}")
        st.write(f"Mean Squared Error (MSE): {mse}")

    except ValueError:
        st.write("Please enter a valid stock symbol from Yahoo Finance, including (NS) or (BO) for stocks from the National Stock Exchange (NSE) or Bombay Stock Exchange (BSE) respectively.")

else:
    st.write("Enter a stock symbol to view data and predictions.")
