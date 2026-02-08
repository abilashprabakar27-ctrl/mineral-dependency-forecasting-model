import pandas as pd
import matplotlib.pyplot as plt
import os

HISTORY_FILE = 'data/final.csv'
ARIMA_FILE = 'forecasts/arima_final_forecast.csv'
LSTM_FILE = 'forecasts/lstm_final_forecast.csv'

def generate_comparison_chart():
    history = pd.read_csv(HISTORY_FILE, parse_dates=['date'], index_col='date')
    
    arima_exists = os.path.exists(ARIMA_FILE)
    lstm_exists = os.path.exists(LSTM_FILE)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(history.index, history['import_value'], label='Historical Data', color='#1f77b4', linewidth=2)
    
    if arima_exists:
        arima = pd.read_csv(ARIMA_FILE, parse_dates=['Date'], index_col='Date')
        plt.plot(arima.index, arima['Forecast'], label='ARIMA (Baseline)', color='#2ca02c', linestyle='--', linewidth=2)
        plt.fill_between(arima.index, arima['Lower_CI'], arima['Upper_CI'], color='#2ca02c', alpha=0.1)
        
    if lstm_exists:
        lstm = pd.read_csv(LSTM_FILE, parse_dates=['Date'], index_col='Date')
        plt.plot(lstm.index, lstm['LSTM_Forecast'], label='LSTM (Deep Learning)', color='#d62728', linestyle='-.', linewidth=2)

    plt.title('Model Benchmark: ARIMA vs LSTM Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Import Value (Normalized/USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('forecasts/model_comparison.png', dpi=300)
    print("Benchmark Graph saved to: forecasts/model_comparison.png")

if __name__ == "__main__":
    generate_comparison_chart()