import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from datetime import date
import streamlit as st
from prettytable import PrettyTable
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter


# Parameters for data
weekly_start_date = '2018-01-01'
weekly_end_date = '2024-12-29'

# Function to download historical stock data (weekly data)
def weekly_download_stock_data(ticker, weekly_start_date, weekly_end_date):
    weekly_stock_data = yf.download(ticker, start=weekly_start_date, end=weekly_end_date, interval='1wk')
    if weekly_stock_data.empty:
        st.error(f"No data available for {ticker} from {weekly_start_date} to {weekly_end_date}")
        return None
    weekly_stock_data.index = pd.to_datetime(weekly_stock_data.index)
    return weekly_stock_data

# Add technical indicators
def add_technical_indicators(weekly_df):
    # Ichimoku Cloud calculation
    high_9 = weekly_df['High'].rolling(window=9).max()
    low_9 = weekly_df['Low'].rolling(window=9).min()
    high_26 = weekly_df['High'].rolling(window=26).max()
    low_26 = weekly_df['Low'].rolling(window=26).min()
    high_52 = weekly_df['High'].rolling(window=52).max()
    low_52 = weekly_df['Low'].rolling(window=52).min()
    
    weekly_df['Tenkan-sen'] = (high_9 + low_9) / 2
    weekly_df['Kijun-sen'] = (high_26 + low_26) / 2
    weekly_df['Senkou_Span_A'] = ((weekly_df['Tenkan-sen'] + weekly_df['Kijun-sen']) / 2).shift(26)
    weekly_df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
    weekly_df['Chikou_Span'] = weekly_df['Close'].shift(-26)

    # EMA (Exponential Moving Averages)
    weekly_df['EMA_20'] = ta.ema(weekly_df['Close'], length=20)
    weekly_df['EMA_50'] = ta.ema(weekly_df['Close'], length=50)
    weekly_df['EMA_100'] = ta.ema(weekly_df['Close'], length=100)
    weekly_df['EMA_200'] = ta.ema(weekly_df['Close'], length=200)
    
    # MACD
    macd = ta.macd(weekly_df['Close'])
    weekly_df['MACD'] = macd['MACD_12_26_9']
    weekly_df['MACD_signal'] = macd['MACDs_12_26_9']
    
    # RSI (Relative Strength Index)
    weekly_df['RSI'] = ta.rsi(weekly_df['Close'], length=14)
    weekly_df['RSI_30'] = ta.rsi(weekly_df['Close'], length=30)
  
    # ROC (Rate of Change)
    weekly_df['ROC'] = ta.roc(weekly_df['Close'], length=14)
    
    # CMF (Chaikin Money Flow)
    weekly_df['CMF'] = ta.cmf(weekly_df['High'], weekly_df['Low'], weekly_df['Close'], weekly_df['Volume'])
    
    # OBV (On Balance Volume)
    weekly_df['OBV'] = ta.obv(weekly_df['Close'], weekly_df['Volume'])
    return weekly_df


# Create PrettyTable for the last 10 rows
def create_pretty_table(weekly_df, y_pred_combined):
    # Check if there are enough predictions to display
    if len(y_pred_combined) < 10:
        st.write("Not enough predictions to display the last 10. Showing what is available.")
        predictions_to_show = len(y_pred_combined)
    else:
        predictions_to_show = 10
    
    # Extract the last `predictions_to_show` closing prices and corresponding predictions
    last_closing_prices = weekly_df['Close'].tail(predictions_to_show).values
    last_predictions = y_pred_combined[-predictions_to_show:]

    # Create a PrettyTable object
    table = PrettyTable()

    # Define the columns
    table.field_names = ["Date", "Closing Price", "Weekly Prediction Score"]

    # Fill the table with data from the last available predictions
    for date, close_price, prediction in zip(weekly_df.index[-predictions_to_show:], last_closing_prices, last_predictions):
        table.add_row([date.strftime("%Y-%m-%d"), round(close_price, 2), round(prediction, 4)])

    st.text(table)  # Use st.text() to display the PrettyTable in Streamlit
   

# Function to prepare data for GRU
def prepare_data(weekly_df, n_future_weeks=4):
    weekly_features = ['EMA_100', 'EMA_200', 'EMA_50', 'MACD', 'RSI', 'ROC', 'CMF', 'OBV']
    weekly_target = 'Price_Change_30w'

    # Calculate 10-week continuous price change as the target
    weekly_df['Price_Change_30w'] = weekly_df['Close'].pct_change(periods=4) * 100

    # Safely assign -9999 to the most recent n_future_weeks rows
    weekly_df.loc[weekly_df.index[-n_future_weeks:], 'Price_Change_30w'] = -9999

    # Convert to categorical (increase/decrease)
    weekly_df['Price_Change_Categorical'] = np.where(weekly_df['Price_Change_30w'] > 0, 1, 0)

    # Fill missing values
    weekly_df.fillna(method='ffill', inplace=True)
    weekly_df.fillna(method='bfill', inplace=True)

    # Standardize features
    scaler = StandardScaler()
    weekly_scaled_features = scaler.fit_transform(weekly_df[weekly_features])

    return weekly_df, weekly_scaled_features

# Create GRU model
def build_gru_model_weekly(input_shape):
    model = Sequential()
    model.add(GRU(15, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(20, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # For binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Plot function
def plot_weekly_stock_and_predictions(weekly_df, y_pred_gru_weekly):
    # Create a rolling 5-week average for predictions
    y_pred_gru_weekly_rolling = pd.Series(y_pred_gru_weekly).rolling(window=5).mean().fillna(0)

    # Determine the number of weeks to plot based on the length of available prediction data
    available_weeks = len(y_pred_gru_weekly_rolling)

    # Adjust the date range for the available prediction period
    date_range_weekly = weekly_df.index[-available_weeks:]  # Use all the available weeks for combined predictions

    # --- Create subplots for 3 plots on the same page ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18))  # 3 rows, 1 column

    # --- Plot 1: Stock prices and Ichimoku Cloud ---
    ax1.plot(date_range_weekly, weekly_df['Close'][-available_weeks:], label='Close Price', color='black')
    ax1.plot(date_range_weekly, weekly_df['Senkou_Span_A'][-available_weeks:], label='Senkou Span A', linestyle='--', color='orange')
    ax1.plot(date_range_weekly, weekly_df['Senkou_Span_B'][-available_weeks:], label='Senkou Span B', linestyle='--', color='green')

    ax1.fill_between(date_range_weekly, weekly_df['Senkou_Span_A'][-available_weeks:], weekly_df['Senkou_Span_B'][-available_weeks:], 
                     where=weekly_df['Senkou_Span_A'][-available_weeks:] >= weekly_df['Senkou_Span_B'][-available_weeks:], 
                     color='lightgreen', label='Ichimoku Cloud (Bullish)', alpha=0.3)

    ax1.fill_between(date_range_weekly, weekly_df['Senkou_Span_A'][-available_weeks:], weekly_df['Senkou_Span_B'][-available_weeks:], 
                     where=weekly_df['Senkou_Span_A'][-available_weeks:] < weekly_df['Senkou_Span_B'][-available_weeks:], 
                     color='lightcoral', label='Ichimoku Cloud (Bearish)', alpha=0.3)

    # Plot additional lines (Signal Line and Baseline)
    ax1.plot(date_range_weekly, weekly_df['Tenkan-sen'][-available_weeks:], label='Signal Line (Tenkan-sen)', linestyle='--', color='blue')
    ax1.plot(date_range_weekly, weekly_df['Kijun-sen'][-available_weeks:], label='Baseline (Kijun-sen)', linestyle='--', color='red')

    # Customize the first plot
    ax1.set_title(f"Weekly Stock Prices and Ichimoku Cloud for the Last {available_weeks} Weeks")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Stock Price")
    ax1.legend()
    ax1.grid(True)

    # --- Plot 2: Close price and EMAs ---
    ax2.plot(date_range_weekly, weekly_df['Close'][-available_weeks:], label='Close Price', color='black')
    ax2.plot(date_range_weekly, weekly_df['EMA_20'][-available_weeks:], label='EMA20', linestyle='--', color='blue')
    ax2.plot(date_range_weekly, weekly_df['EMA_50'][-available_weeks:], label='EMA50', linestyle='--', color='red')

    # Customize the second plot
    ax2.set_title(f"Close Prices and EMAs for the Last {available_weeks} Weeks")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Stock Price and EMAs")
    ax2.legend()
    ax2.grid(True)

    # --- Plot 3: GRU Model predictions ---
    ax3.plot(date_range_weekly, y_pred_gru_weekly_rolling, label='GRU Model Predictions (5-week avg)', color='blue')

    # Customize the third plot
    ax3.set_title(f"GRU Model Predictions for the Last {available_weeks} Weeks")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("GRU Model Predictions")
    ax3.set_ylim([0, 1])  # Predictions should be between 0 and 1 for classification
    ax3.legend()
    ax3.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    st.pyplot(fig)



# Function to download daily stock data
def download_daily_stock_data(ticker, daily_start_date, daily_end_date):
    # Download hourly data
    daily_stock_data = yf.download(ticker, start=daily_start_date, end=daily_end_date, interval='1h')
    if daily_stock_data.empty:
        raise ValueError(f"No data available for {ticker} from {daily_start_date} to {daily_end_date}")
    
    daily_stock_data.index = pd.to_datetime(daily_stock_data.index)
    return daily_stock_data



# Function to calculate the slope using linear regression over a rolling window
def calculate_slope(series, window_size):
    slopes = np.full(series.shape, np.nan)  # Create an empty array to store slopes
    x = np.arange(window_size).reshape(-1, 1)  # Create an array of indices as the independent variable

    for i in range(window_size, len(series)):
        y = series[i-window_size:i].values.reshape(-1, 1)
        
        if np.isnan(y).any():
            continue  # Skip if there are NaN values
        
        model = LinearRegression().fit(x, y)
        slopes[i] = model.coef_[0].item()  # Extract the first element as a scalar

    return slopes



#
# Add technical indicators to daily data
def add_daily_technical_indicators(daily_df):
    # Ichimoku Cloud calculation
    daily_high_9 = daily_df['High'].rolling(window=9).max()
    daily_low_9 = daily_df['Low'].rolling(window=9).min()
    daily_high_26 = daily_df['High'].rolling(window=26).max()
    daily_low_26 = daily_df['Low'].rolling(window=26).min()
    daily_high_52 = daily_df['High'].rolling(window=52).max()
    daily_low_52 = daily_df['Low'].rolling(window=52).min()
    
    daily_df['Tenkan-sen'] = (daily_high_9 + daily_low_9) / 2  # Conversion line (Signal Line)
    daily_df['Kijun-sen'] = (daily_high_26 + daily_low_26) / 2  # Base line (Baseline)
    daily_df['Senkou_Span_A'] = ((daily_df['Tenkan-sen'] + daily_df['Kijun-sen']) / 2).shift(26)  # Leading Span A
    daily_df['Senkou_Span_B'] = ((daily_high_52 + daily_low_52) / 2).shift(26)  # Leading Span B
    daily_df['Chikou_Span'] = daily_df['Close'].shift(-26)  # Lagging Span
    
    # Define whether the price is above, below, or within the cloud
    daily_df['Above_Cloud'] = (daily_df['Close'] > daily_df[['Senkou_Span_A', 'Senkou_Span_B']].max(axis=1)).astype(int)
    daily_df['Below_Cloud'] = (daily_df['Close'] < daily_df[['Senkou_Span_A', 'Senkou_Span_B']].min(axis=1)).astype(int)
    daily_df['In_Cloud'] = ((daily_df['Close'] >= daily_df[['Senkou_Span_A', 'Senkou_Span_B']].min(axis=1)) & 
                            (daily_df['Close'] <= daily_df[['Senkou_Span_A', 'Senkou_Span_B']].max(axis=1))).astype(int)
    
    # Entering the cloud from above or below
    daily_df['Enter_Cloud_From_Above'] = ((daily_df['Close'].shift(1) > daily_df[['Senkou_Span_A', 'Senkou_Span_B']].max(axis=1).shift(1)) &
                                          (daily_df['In_Cloud'] == 1)).astype(int)
    daily_df['Enter_Cloud_From_Below'] = ((daily_df['Close'].shift(1) < daily_df[['Senkou_Span_A', 'Senkou_Span_B']].min(axis=1).shift(1)) &
                                          (daily_df['In_Cloud'] == 1)).astype(int)
    
    # Signal line (Tenkan-sen) crossing the baseline (Kijun-sen)
    daily_df['Signal_Cross_Above_Baseline'] = ((daily_df['Tenkan-sen'].shift(1) <= daily_df['Kijun-sen'].shift(1)) & 
                                               (daily_df['Tenkan-sen'] > daily_df['Kijun-sen'])).astype(int)
    daily_df['Signal_Cross_Below_Baseline'] = ((daily_df['Tenkan-sen'].shift(1) >= daily_df['Kijun-sen'].shift(1)) & 
                                               (daily_df['Tenkan-sen'] < daily_df['Kijun-sen'])).astype(int)
    
    # Price above/below the baseline and signal line
    daily_df['Price_Above_Baseline'] = (daily_df['Close'] > daily_df['Kijun-sen']).astype(int)
    daily_df['Price_Below_Baseline'] = (daily_df['Close'] < daily_df['Kijun-sen']).astype(int)
    daily_df['Price_Above_Signal'] = (daily_df['Close'] > daily_df['Tenkan-sen']).astype(int)
    daily_df['Price_Below_Signal'] = (daily_df['Close'] < daily_df['Tenkan-sen']).astype(int)
    
    # Cloud color (Green/Bullish or Red/Bearish)
    daily_df['Green_Cloud'] = (daily_df['Senkou_Span_A'] > daily_df['Senkou_Span_B']).astype(int)
    daily_df['Red_Cloud'] = (daily_df['Senkou_Span_A'] < daily_df['Senkou_Span_B']).astype(int)
    
    # CCI
    daily_df['CCI'] = ta.cci(daily_df['High'], daily_df['Low'], daily_df['Close'])
    
    # TSI (True Strength Index) - Handle it dynamically
    daily_tsi = ta.tsi(daily_df['Close'])
    if isinstance(daily_tsi, pd.DataFrame):
        daily_df[['TSI', 'TSI_signal']] = daily_tsi.iloc[:, :2]  # Dynamically grab the first two columns if multiple
    else:
        daily_df['TSI'] = daily_tsi  # If only one column is returned

    # RSI
    daily_df['RSI'] = ta.rsi(daily_df['Close'])
    daily_df['RSI_30'] = ta.rsi(daily_df['Close'], length=30)
  
    # Rate of Change (ROC)
    daily_df['ROC'] = ta.roc(daily_df['Close'], length=14)

    # Stochastic Oscillator - Handle it dynamically
    daily_stoch = ta.stoch(daily_df['High'], daily_df['Low'], daily_df['Close'])
    if isinstance(daily_stoch, pd.DataFrame):
        daily_df[['Stoch_%K', 'Stoch_%D']] = daily_stoch.iloc[:, :2]  # Dynamically grab the first two columns if multiple
    else:
        daily_df['Stoch'] = daily_stoch  # If only one column is returned

    # MACD
    daily_df[['MACD', 'MACD_signal', 'MACD_hist']] = ta.macd(daily_df['Close'])
    
    # PPO
    daily_df[['PPO', 'PPO_signal', 'PPO_hist']] = ta.ppo(daily_df['Close'])
    
    # PVO (Price Volume Oscillator)
    daily_df[['PVO', 'PVO_signal', 'PVO_hist']] = ta.pvo(daily_df['Volume'])
    
    # CMF (Chaikin Money Flow)
    daily_df['CMF'] = ta.cmf(daily_df['High'], daily_df['Low'], daily_df['Close'], daily_df['Volume'])
    daily_df['CMF_50'] = ta.cmf(daily_df['High'], daily_df['Low'], daily_df['Close'], daily_df['Volume'], length=50)
    
    
    # OBV (On Balance Volume)
    daily_df['OBV'] = ta.obv(daily_df['Close'], daily_df['Volume'])
    
    # ATR (Average True Range)
    daily_df['ATR'] = ta.atr(daily_df['High'], daily_df['Low'], daily_df['Close'])
    
    # Bollinger Bands
    daily_bbands = ta.bbands(daily_df['Close'])
    daily_df = daily_df.join(daily_bbands)
    
    daily_df['ROC'] = ta.roc(daily_df['Close'], length=26)
    
    # Keltner Channels - Handle it dynamically
    daily_keltner = ta.kc(daily_df['High'], daily_df['Low'], daily_df['Close'])
    daily_df = daily_df.join(daily_keltner)

    # Let's print the column names to check the structure of Keltner Channels output
    print("Keltner Channels columns:", daily_keltner.columns)

    # Bollinger Bands inside Keltner Bands (Yes/No)
    if 'BBL_5_2.0' in daily_df.columns and 'BBU_5_2.0' in daily_df.columns and 'KC_L_20_2' in daily_df.columns and 'KC_U_20_2' in daily_df.columns:
        daily_df['BB_inside_KC'] = (daily_df['BBL_5_2.0'] > daily_df['KC_L_20_2']).astype(int) & (daily_df['BBU_5_2.0'] < daily_df['KC_U_20_2']).astype(int)
    else:
        print("Bollinger Bands or Keltner Channels columns not found for calculation.")

    # EMA (Exponential Moving Averages)
    daily_df['EMA_20'] = ta.ema(daily_df['Close'], length=20)
    daily_df['EMA_50'] = ta.ema(daily_df['Close'], length=50)
    daily_df['EMA_100'] = ta.ema(daily_df['Close'], length=100)
    daily_df['EMA_200'] = ta.ema(daily_df['Close'], length=200)
    
    # Price differences from EMAs
    daily_df['Price-EMA20'] = daily_df['Close'] - daily_df['EMA_20']
    daily_df['Price-EMA50'] = daily_df['Close'] - daily_df['EMA_50']
    daily_df['Price-EMA200'] = daily_df['Close'] - daily_df['EMA_200']
    
    # EMA differences
    daily_df['EMA20-EMA50'] = daily_df['EMA_20'] - daily_df['EMA_50']
    daily_df['EMA20-EMA200'] = daily_df['EMA_20'] - daily_df['EMA_200']
    daily_df['EMA50-EMA200'] = daily_df['EMA_50'] - daily_df['EMA_200']
    
    # Optional smoothing using Savitzky-Golay filter
    daily_df['EMA_20_Smoothed'] = savgol_filter(daily_df['EMA_20'], window_length=15, polyorder=2)
    daily_df['EMA_50_Smoothed'] = savgol_filter(daily_df['EMA_50'], window_length=15, polyorder=2)
    daily_df['EMA_200_Smoothed'] = savgol_filter(daily_df['EMA_200'], window_length=15, polyorder=2)

    # Calculate slopes for EMAs
    daily_df['Slope_EMA_20'] = calculate_slope(daily_df['EMA_20_Smoothed'], window_size=20)
    daily_df['Slope_EMA_50'] = calculate_slope(daily_df['EMA_50_Smoothed'], window_size=50)
    daily_df['Slope_EMA_200'] = calculate_slope(daily_df['EMA_200_Smoothed'], window_size=200)
    
    # Add Price 30 days ago column
    daily_df['Price_30d_Ago'] = daily_df['Close'].shift(30)
    
    # Calculate the price difference (current price - price 30 days ago)
    daily_df['Price_Diff_30d'] = daily_df['Close'] - daily_df['Price_30d_Ago']


    # Add 30-day high and low
    daily_df['30d_High'] = daily_df['High'].rolling(window=30).max()
    daily_df['30d_Low'] = daily_df['Low'].rolling(window=30).min()

    # Add 50-day high and low
    daily_df['50d_High'] = daily_df['High'].rolling(window=50).max()
    daily_df['50d_Low'] = daily_df['Low'].rolling(window=50).min()

    # Add 100-day high and low
    daily_df['100d_High'] = daily_df['High'].rolling(window=100).max()
    daily_df['100d_Low'] = daily_df['Low'].rolling(window=100).min()
  
    return daily_df


# Function to prepare daily data for GRU1
def prepare_daily_data(daily_df, n_future_days=5):
    daily_features = ['EMA_100', 'EMA_200', 'EMA_50', 'MACD', 'EMA20-EMA50', 'EMA20-EMA200', 
                'MACD_signal', 'RSI_30', 'RSI', 'ROC', 'CMF', 'OBV']
    daily_target = 'Price_Change_30d'

    # Calculate 30-day continuous price change as the target
    daily_df['Price_Change_30d'] = daily_df['Close'].pct_change(periods=30) * 100


    # Safely assign -9999 to the most recent n_future_days rows
    daily_df.loc[daily_df.index[-n_future_days:], 'Price_Change_30d'] = -9999

    # Convert to categorical (increase/decrease)
    daily_df['Price_Change_Categorical'] = np.where(daily_df['Price_Change_30d'] > 0, 1, 0)

    # Fill missing values
    daily_df.fillna(method='ffill', inplace=True)
    daily_df.fillna(method='bfill', inplace=True)

    # Standardize features
    scaler = StandardScaler()
    daily_scaled_features = scaler.fit_transform(daily_df[['EMA_100', 'EMA_200', 'EMA_50', 'MACD', 'RSI', 'ROC', 'CMF', 'OBV']])

    return daily_df, daily_scaled_features

# 4. Create GRU model for daily data
def build_daily_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(20, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # For binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

###plot the daily prices
def plot_daily_stock_and_predictions(daily_df, y_pred_gru_daily, available_days_daily=250):
    # Create a rolling 5-day average for predictions
    y_pred_gru_daily_rolling = pd.Series(y_pred_gru_daily).rolling(window=5).mean().fillna(0)

    # Limit to the last 250 days
    date_range_daily = daily_df.tail(available_days_daily)

    # Ensure that y_pred_gru_daily_rolling has the same length as date_range_daily
    y_pred_gru_daily_rolling = y_pred_gru_daily_rolling[-available_days_daily:]

    # --- Create subplots for 3 plots on the same page ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18))  # 3 rows, 1 column

    # --- Plot 1: Stock prices and Ichimoku Cloud ---
    ax1.plot(date_range_daily.index, date_range_daily['Close'], label='Close Price', color='black')
    ax1.plot(date_range_daily.index, date_range_daily['Senkou_Span_A'], label='Senkou Span A', linestyle='--', color='orange')
    ax1.plot(date_range_daily.index, date_range_daily['Senkou_Span_B'], label='Senkou Span B', linestyle='--', color='green')

    ax1.fill_between(date_range_daily.index, date_range_daily['Senkou_Span_A'], date_range_daily['Senkou_Span_B'],
                     where=date_range_daily['Senkou_Span_A'] >= date_range_daily['Senkou_Span_B'], 
                     color='lightgreen', label='Ichimoku Cloud (Bullish)', alpha=0.3)

    ax1.fill_between(date_range_daily.index, date_range_daily['Senkou_Span_A'], date_range_daily['Senkou_Span_B'], 
                     where=date_range_daily['Senkou_Span_A'] < date_range_daily['Senkou_Span_B'], 
                     color='lightcoral', label='Ichimoku Cloud (Bearish)', alpha=0.3)

    # Plot additional lines (Signal Line and Baseline)
    ax1.plot(date_range_daily.index, date_range_daily['Tenkan-sen'], label='Signal Line (Tenkan-sen)', linestyle='--', color='blue')
    ax1.plot(date_range_daily.index, date_range_daily['Kijun-sen'], label='Baseline (Kijun-sen)', linestyle='--', color='red')

    # Customize the first plot
    ax1.set_title(f"Stock Prices and Ichimoku Cloud for the Last {available_days_daily} Days")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Stock Price")
    ax1.legend()
    ax1.grid(True)

    # --- Plot 2: Close price and EMAs ---
    ax2.plot(date_range_daily.index, date_range_daily['Close'], label='Close Price', color='black')
    ax2.plot(date_range_daily.index, date_range_daily['EMA_20'], label='EMA20', linestyle='--', color='blue')
    ax2.plot(date_range_daily.index, date_range_daily['EMA_50'], label='EMA50', linestyle='--', color='red')

    # Customize the second plot
    ax2.set_title(f"Close Prices and EMAs for the Last {available_days_daily} Days")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Stock Price and EMAs")
    ax2.legend()
    ax2.grid(True)

    # --- Plot 3: GRU Model predictions ---
    ax3.plot(date_range_daily.index, y_pred_gru_daily_rolling, label='GRU Model Predictions (5-day avg)', color='blue')

    # Customize the third plot
    ax3.set_title(f"GRU Model Predictions for the Last {available_days_daily} Days")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("GRU Model Predictions")
    ax3.set_ylim([0, 1])  # Predictions should be between 0 and 1 for classification
    ax3.legend()
    ax3.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show all plots on the same page
    st.pyplot(fig)

# Function to create and display PrettyTable for the last 10 days with EMAs
def create_pretty_table(daily_df, y_pred_combined):
    # Ensure there are at least 10 data points available
    if len(y_pred_combined) < 10:
        print("Not enough predictions to display the last 10 days. Showing what is available.")
        predictions_to_show = len(y_pred_combined)
    else:
        predictions_to_show = 10
    
    # Extract the last `predictions_to_show` values
    last_dates = daily_df.index[-predictions_to_show:].strftime("%Y-%m-%d")
    last_closing_prices = daily_df['Close'].tail(predictions_to_show).values
    last_highs = daily_df['High'].tail(predictions_to_show).values
    last_lows = daily_df['Low'].tail(predictions_to_show).values
    last_EMA20 = daily_df['EMA_20'].tail(predictions_to_show).values
    last_EMA50 = daily_df['EMA_50'].tail(predictions_to_show).values
    last_EMA200 = daily_df['EMA_200'].tail(predictions_to_show).values
    last_predictions = y_pred_combined[-predictions_to_show:]

    # Create a PrettyTable object
    table = PrettyTable()

    # Define the columns
    table.field_names = ["Date", "Close", "High", "Low", "EMA20", "EMA50", "EMA200", "Prediction"]

    # Fill the table with data from the last available predictions
    for date, close_price, high, low, ema20, ema50, ema200, prediction in zip(
        last_dates, last_closing_prices, last_highs, last_lows, last_EMA20, last_EMA50, last_EMA200, last_predictions):
        
        # Add a row for each of the last 10 days
        table.add_row([
            date, round(close_price, 2), round(high, 2), round(low, 2), 
            round(ema20, 2), round(ema50, 2), round(ema200, 2), round(prediction, 4)
        ])

    st.text(table)   # Print the table in the console


































############%%%%%%%%%%%%%%%%%%%%%%%%%%%%#####################################################
#################################################
def create_hourly_pretty_table(hourly_df, y_pred_combined):
    # Check if there are enough predictions to display the last 10 hours
    if len(y_pred_combined) < 10:
        print("Not enough predictions to display the last 10 hours. Showing what is available.")
        predictions_to_show = len(y_pred_combined)
    else:
        predictions_to_show = 10
    
    # Extract the last `predictions_to_show` closing prices and corresponding predictions
    last_closing_prices = hourly_df['Close'].tail(predictions_to_show).values
    last_predictions = y_pred_combined[-predictions_to_show:]

    # Create a PrettyTable object
    table = PrettyTable()

    # Define the columns
    table.field_names = ["Date", "Closing Price", "Hourly Prediction Score"]

    # Fill the table with data from the last available predictions
    for date, close_price, prediction in zip(hourly_df.index[-predictions_to_show:], last_closing_prices, last_predictions):
        table.add_row([date.strftime("%Y-%m-%d %H:%M"), round(close_price, 2), round(prediction, 4)])

    # Print the table
    st.text(table) 

# Define hourly functions before calling them in the weekly main

# Function to download hourly stock data
def download_hourly_stock_data(ticker, hourly_start_date, hourly_end_date):
    # Download hourly data
    hourly_stock_data = yf.download(ticker, start=hourly_start_date, end=hourly_end_date, interval='1h')
    if hourly_stock_data.empty:
        raise ValueError(f"No data available for {ticker} from {hourly_start_date} to {hourly_end_date}")
    
    hourly_stock_data.index = pd.to_datetime(hourly_stock_data.index)
    return hourly_stock_data


# Add technical indicators to hourly data
def add_hourly_technical_indicators(hourly_df):
   # Ichimoku Cloud calculation
   high_9 = hourly_df['High'].rolling(window=9).max()
   low_9 = hourly_df['Low'].rolling(window=9).min()
   high_26 = hourly_df['High'].rolling(window=26).max()
   low_26 = hourly_df['Low'].rolling(window=26).min()
   high_52 = hourly_df['High'].rolling(window=52).max()
   low_52 = hourly_df['Low'].rolling(window=52).min()
   
   hourly_df['Tenkan-sen'] = (high_9 + low_9) / 2
   hourly_df['Kijun-sen'] = (high_26 + low_26) / 2
   hourly_df['Senkou_Span_A'] = ((hourly_df['Tenkan-sen'] + hourly_df['Kijun-sen']) / 2).shift(26)
   hourly_df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
   hourly_df['Chikou_Span'] = hourly_df['Close'].shift(-26)

   # EMA (Exponential Moving Averages)
   hourly_df['EMA_20'] = ta.ema(hourly_df['Close'], length=20)
   hourly_df['EMA_50'] = ta.ema(hourly_df['Close'], length=50)
   hourly_df['EMA_100'] = ta.ema(hourly_df['Close'], length=100)
   hourly_df['EMA_200'] = ta.ema(hourly_df['Close'], length=200)
   
   # MACD
   macd = ta.macd(hourly_df['Close'])
   hourly_df['MACD'] = macd['MACD_12_26_9']
   hourly_df['MACD_signal'] = macd['MACDs_12_26_9']
   
   # RSI (Relative Strength Index)
   hourly_df['RSI'] = ta.rsi(hourly_df['Close'], length=14)
   hourly_df['RSI_30'] = ta.rsi(hourly_df['Close'], length=30)
 
   # ROC (Rate of Change)
   hourly_df['ROC'] = ta.roc(hourly_df['Close'], length=14)
   
   # CMF (Chaikin Money Flow)
   hourly_df['CMF'] = ta.cmf(hourly_df['High'], hourly_df['Low'], hourly_df['Close'], hourly_df['Volume'])
   
   # OBV (On Balance Volume)
   hourly_df['OBV'] = ta.obv(hourly_df['Close'], hourly_df['Volume'])
   
    # Price differences from EMAs
   hourly_df['Price-EMA20'] = hourly_df['Close'] - hourly_df['EMA_20']
   hourly_df['Price-EMA50'] = hourly_df['Close'] - hourly_df['EMA_50']
   hourly_df['Price-EMA200'] = hourly_df['Close'] - hourly_df['EMA_200']
   
   # EMA differences
   hourly_df['EMA20-EMA50'] =  hourly_df['EMA_20'] -  hourly_df['EMA_50']
   hourly_df['EMA20-EMA200'] =  hourly_df['EMA_20'] -  hourly_df['EMA_200']
   hourly_df['EMA50-EMA200'] =  hourly_df['EMA_50'] -  hourly_df['EMA_200']
    
   
   return hourly_df

# Function to prepare hourly data for GRU
def prepare_hourly_data(hourly_df, n_future_days=5):
    hourly_features = ['EMA_100', 'EMA_200', 'EMA_50', 'MACD', 'EMA20-EMA50', 'EMA20-EMA200', 
                'MACD_signal', 'RSI_30', 'RSI', 'ROC', 'CMF', 'OBV']

    # Calculate 30-day continuous price change as the target
    hourly_df['Price_Change_30d'] = hourly_df['Close'].pct_change(periods=40) * 100

    # Assign -9999 to the last rows as dummy targets (we still want predictions here)
    hourly_df.loc[hourly_df.index[-n_future_days:], 'Price_Change_30d'] = -9999

    # Convert to categorical (increase/decrease)
    hourly_df['Price_Change_Categorical'] = np.where(hourly_df['Price_Change_30d'] > 0, 1, 0)

    # Fill missing values
    hourly_df.fillna(method='ffill', inplace=True)
    hourly_df.fillna(method='bfill', inplace=True)

    # Standardize features
    scaler = StandardScaler()
    hourly_scaled_features = scaler.fit_transform(hourly_df[hourly_features])

    return hourly_df, hourly_scaled_features


# Build GRU model for hourly data
def build_hourly_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(20, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # For binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Plot function for hourly data
def plot_hourly_stock_and_predictions(hourly_df, y_pred_class_gru_hourly, y_pred_gru_hourly, n_future_days=5):
    # Create a rolling 5-day average for predictions
    y_pred_gru_hourly_rolling = pd.Series(y_pred_gru_hourly).rolling(window=5).mean().fillna(0)

    # Limit to the last 100 hours
    available_hours = 95
    date_range_hourly = hourly_df.index[-available_hours:]
    
    # Adjust the date range for the available prediction period, excluding future dates
    date_range_hourly = hourly_df.index[-(available_hours + n_future_days):-n_future_days]

    # Create subplots for 3 plots on the same page
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18))

    # Plot 1: Stock prices and Ichimoku Cloud
    ax1.plot(date_range_hourly, hourly_df['Close'][-(available_hours + n_future_days):-n_future_days], label='Close Price', color='black')
    ax1.plot(date_range_hourly, hourly_df['Senkou_Span_A'][-(available_hours + n_future_days):-n_future_days], label='Senkou Span A', linestyle='--', color='orange')
    ax1.plot(date_range_hourly, hourly_df['Senkou_Span_B'][-(available_hours + n_future_days):-n_future_days], label='Senkou Span B', linestyle='--', color='green')

    ax1.fill_between(date_range_hourly, hourly_df['Senkou_Span_A'][-(available_hours + n_future_days):-n_future_days], 
                     hourly_df['Senkou_Span_B'][-(available_hours + n_future_days):-n_future_days], 
                     where=hourly_df['Senkou_Span_A'][-(available_hours + n_future_days):-n_future_days] >= hourly_df['Senkou_Span_B'][-(available_hours + n_future_days):-n_future_days], 
                     color='lightgreen', label='Ichimoku Cloud (Bullish)', alpha=0.3)

    ax1.fill_between(date_range_hourly, hourly_df['Senkou_Span_A'][-(available_hours + n_future_days):-n_future_days], 
                     hourly_df['Senkou_Span_B'][-(available_hours + n_future_days):-n_future_days], 
                     where=hourly_df['Senkou_Span_A'][-(available_hours + n_future_days):-n_future_days] < hourly_df['Senkou_Span_B'][-(available_hours + n_future_days):-n_future_days], 
                     color='lightcoral', label='Ichimoku Cloud (Bearish)', alpha=0.3)

    # Plot additional lines (Signal Line and Baseline)
    ax1.plot(date_range_hourly, hourly_df['Tenkan-sen'][-(available_hours + n_future_days):-n_future_days], label='Signal Line (Tenkan-sen)', linestyle='--', color='blue')
    ax1.plot(date_range_hourly, hourly_df['Kijun-sen'][-(available_hours + n_future_days):-n_future_days], label='Baseline (Kijun-sen)', linestyle='--', color='red')

    ax1.set_title(f"Hourly Stock Prices and Ichimoku Cloud for the Last {available_hours} Hours")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Stock Price")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Close price and EMAs
    ax2.plot(date_range_hourly, hourly_df['Close'][-(available_hours + n_future_days):-n_future_days], label='Close Price', color='black')
    ax2.plot(date_range_hourly, hourly_df['EMA_20'][-(available_hours + n_future_days):-n_future_days], label='EMA20', linestyle='--', color='blue')
    ax2.plot(date_range_hourly, hourly_df['EMA_50'][-(available_hours + n_future_days):-n_future_days], label='EMA50', linestyle='--', color='red')

    ax2.set_title(f"Close Prices and EMAs for the Last {available_hours} Hours")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Stock Price and EMAs")
    ax2.legend()
    ax2.grid(True)

    # Plot 3: GRU Model predictions
    ax3.plot(date_range_hourly, y_pred_gru_hourly_rolling[-available_hours:], label='GRU Model Predictions (5-hour avg)', color='blue')

    ax3.set_title(f"GRU Model Predictions for the Last {available_hours} Hours")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("GRU Model Predictions")
    ax3.set_ylim([0, 1])
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    st.pyplot(fig)


# PrettyTable function to display the results
def create_hourly_pretty_table(hourly_df, y_pred_combined):
    # Check if there are enough predictions to display the last 10 hours
    if len(y_pred_combined) < 10:
        st.write("Not enough predictions to display the last 10 hours. Showing what is available.")
        predictions_to_show = len(y_pred_combined)
    else:
        predictions_to_show = 10
    
    # Extract the last `predictions_to_show` closing prices and corresponding predictions
    last_closing_prices = hourly_df['Close'].tail(predictions_to_show).values
    last_predictions = y_pred_combined[-predictions_to_show:]

    # Create a PrettyTable object
    table = PrettyTable()

    # Define the columns
    table.field_names = ["Date", "Closing Price", "Hourly Prediction Score"]

    # Fill the table with data from the last available predictions
    for date, close_price, prediction in zip(hourly_df.index[-predictions_to_show:], last_closing_prices, last_predictions):
        table.add_row([date.strftime("%Y-%m-%d %H:%M"), round(close_price, 2), round(prediction, 4)])

    st.text(table) 






def main_daily(ticker):
    # Parameters for data
 daily_start_date = '2019-01-01'
 daily_end_date = '2024-12-29'
    
 #def download_daily_stock_data(ticker, daily_start_date, daily_end_date):
 daily_stock_data = yf.download(ticker, start=daily_start_date, end=daily_end_date, interval='1d')
        
        # Check if data is empty
 if daily_stock_data.empty:
            raise ValueError(f"No data available for {ticker} from {daily_start_date} to {daily_end_date}")
        
 daily_stock_data.index = pd.to_datetime(daily_stock_data.index)


    # Add technical indicators
 daily_stock_data = add_daily_technical_indicators(daily_stock_data)

    # Prepare data for GRU model
 daily_stock_data, daily_scaled_features_GRU = prepare_daily_data(daily_stock_data)

    # Prepare input for GRU model
 time_steps = 30
 X_gru_daily, y_gru_daily = [], []
 for i in range(len(daily_scaled_features_GRU) - time_steps):
        X_gru_daily.append(daily_scaled_features_GRU[i:i+time_steps])
        y_gru_daily.append(daily_stock_data['Price_Change_Categorical'].values[i + time_steps])
 X_gru_daily, y_gru_daily = np.array(X_gru_daily), np.array(y_gru_daily)

    # Split data into training and testing sets for GRU model
 X_train_gru_daily, X_test_gru_daily, y_train_gru_daily, y_test_gru_daily = train_test_split(X_gru_daily, y_gru_daily, test_size=0.2, shuffle=False)

    # Build and train GRU model
 gru_model_daily = build_daily_gru_model(input_shape=(X_train_gru_daily.shape[1], X_train_gru_daily.shape[2]))
 gru_model_daily.fit(X_train_gru_daily, y_train_gru_daily, epochs=10, batch_size=32, validation_data=(X_test_gru_daily, y_test_gru_daily))

    # Get predictions for GRU model
 y_pred_train_gru_daily = gru_model_daily.predict(X_train_gru_daily).flatten()
 y_pred_test_gru_daily = gru_model_daily.predict(X_test_gru_daily).flatten()

    # Combine predictions for both training and test sets
 y_pred_combined_daily = np.concatenate([y_pred_train_gru_daily, y_pred_test_gru_daily])

    # Filter the dataset for the last 250 days
 last_250_days_df = daily_stock_data.tail(250)
 y_pred_combined_last_250_daily = y_pred_combined_daily[-250:]  # Filter predictions for the last 250 days
 

    # Plot the results for the last 250 days (corrected function call)
 plot_daily_stock_and_predictions(last_250_days_df, y_pred_combined_last_250_daily)
     
 print("Shape of daily_df:", last_250_days_df.shape)
 print("Head of daily_df:\n", last_250_days_df.head())
 print("Length of y_pred_gru_daily:", len( y_pred_combined_last_250_daily))


    # Create and display the PrettyTable with the last 10 days of closing prices and predictions
 create_pretty_table(last_250_days_df, y_pred_combined_last_250_daily)
    







# Main function for hourly predictions
def main_hourly(ticker):
    # Define the start and end dates for hourly data
    hourly_start_date = '2024-01-01'
    hourly_end_date = '2024-12-29'

    # Download stock data
    hourly_stock_data = download_hourly_stock_data(ticker, hourly_start_date, hourly_end_date)

    if hourly_stock_data is not None:
        # Add technical indicators
        hourly_stock_data = add_hourly_technical_indicators(hourly_stock_data)

        # Prepare data for GRU model
        hourly_stock_data, hourly_scaled_features_GRU = prepare_hourly_data(hourly_stock_data)

        # Prepare input for GRU model 1
        time_steps = 30
        X_gru_hourly, y_gru_hourly = [], []
        for i in range(len(hourly_scaled_features_GRU) - time_steps):
            X_gru_hourly.append(hourly_scaled_features_GRU[i:i+time_steps])
            y_gru_hourly.append(hourly_stock_data['Price_Change_Categorical'].values[i + time_steps])
        X_gru_hourly, y_gru_hourly = np.array(X_gru_hourly), np.array(y_gru_hourly)

        # Split data into training and testing sets for GRU model 1
        X_train_gru_hourly, X_test_gru_hourly, y_train_gru_hourly, y_test_gru_hourly = train_test_split(X_gru_hourly, y_gru_hourly, test_size=0.2, shuffle=False)

        # Build and train GRU model 1
        gru_model_hourly = build_hourly_gru_model(input_shape=(X_train_gru_hourly.shape[1], X_train_gru_hourly.shape[2]))
        gru_model_hourly.fit(X_train_gru_hourly, y_train_gru_hourly, epochs=10, batch_size=32, validation_data=(X_test_gru_hourly, y_test_gru_hourly))

        # Get predictions for GRU model
        y_pred_gru_hourly = gru_model_hourly.predict(X_test_gru_hourly).flatten()

        # Combine predictions (for plotting and table creation purposes)
        y_pred_combined_hourly = np.concatenate([gru_model_hourly.predict(X_train_gru_hourly).flatten(), y_pred_gru_hourly])

        # Filter the dataset for the last 100 hours
        last_100_hours_df = hourly_stock_data.tail(100)
        y_pred_combined_last_100 = y_pred_combined_hourly[-100:]  # Filter predictions for the last 100 hours

        # Plot the results for the last 100 hours
        plot_hourly_stock_and_predictions(last_100_hours_df, None, y_pred_combined_last_100, n_future_days=5)

        # Create and display the PrettyTable with the last 10 hours of closing prices and predictions
        create_hourly_pretty_table(hourly_stock_data, y_pred_combined_hourly)



# Main function for Streamlit app
# Main function for Streamlit app
def main_weekly():
    st.title("Ichimoku Charts")

    # Get stock ticker input from the user
    ticker = st.text_input("Enter a stock ticker (e.g., AAPL, SPY)", "SPY")

    # If user enters a ticker, fetch and process data
    if ticker:
        st.write(f"Fetching data for {ticker}...")
        weekly_stock_data = weekly_download_stock_data(ticker, weekly_start_date, weekly_end_date)

        # Daily predictions
        st.header("Daily Chart")
        main_daily(ticker)

        # Hourly predictions
        st.header("Hourly Chart")
        main_hourly(ticker)

        if weekly_stock_data is not None:
            # Add technical indicators
            weekly_stock_data = add_technical_indicators(weekly_stock_data)

            # Prepare data for GRU model
            weekly_stock_data, weekly_scaled_features_GRU = prepare_data(weekly_stock_data)

            # Prepare input for GRU model
            time_steps = 15
            X_gru_weekly, y_gru_weekly = [], []
            for i in range(len(weekly_scaled_features_GRU) - time_steps):
                X_gru_weekly.append(weekly_scaled_features_GRU[i:i+time_steps])
                y_gru_weekly.append(weekly_stock_data['Price_Change_Categorical'].values[i + time_steps])

            X_gru_weekly, y_gru_weekly = np.array(X_gru_weekly), np.array(y_gru_weekly)

            # Split data into training and testing sets for GRU model (80% train, 20% test)
            X_train_gru_weekly, X_test_gru_weekly, y_train_gru_weekly, y_test_gru_weekly = train_test_split(
                X_gru_weekly, y_gru_weekly, test_size=0.2, shuffle=False)

            # Build and train GRU model
            gru_model_weekly = build_gru_model_weekly(
                input_shape=(X_train_gru_weekly.shape[1], X_train_gru_weekly.shape[2]))
            gru_model_weekly.fit(X_train_gru_weekly, y_train_gru_weekly,
                                 epochs=10, batch_size=32,
                                 validation_data=(X_test_gru_weekly, y_test_gru_weekly))

            # Predict on the training set
            y_pred_train_gru_weekly = gru_model_weekly.predict(X_train_gru_weekly).flatten()

            # Predict on the testing set (including most recent rows where targets are -9999)
            y_pred_test_gru_weekly = gru_model_weekly.predict(X_test_gru_weekly).flatten()

            y_pred_combined_weekly = np.concatenate([y_pred_train_gru_weekly, y_pred_test_gru_weekly])
           
            # Add this header for the weekly predictions:
            st.header("Weekly Charts")  

            # Plot the results using the combined predictions
            plot_weekly_stock_and_predictions(weekly_stock_data, y_pred_combined_weekly)

            # Create and display the PrettyTable with the last 10 closing prices and predictions
            create_pretty_table(weekly_stock_data, y_pred_combined_weekly)

if __name__ == "__main__":
    main_weekly()




#############################################################################

