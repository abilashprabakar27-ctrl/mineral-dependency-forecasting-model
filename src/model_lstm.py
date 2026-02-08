import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Attention, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import os

DATA_PATH = 'data/final.csv'
OUTPUT_DIR = 'forecasts/'
SEQ_LENGTH = 12       
FORECAST_STEPS = 24   

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_advanced_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm1 = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    dropout1 = Dropout(0.2)(lstm1)
    lstm2 = Bidirectional(LSTM(64, return_sequences=True))(dropout1)
    dropout2 = Dropout(0.2)(lstm2)
    attention = Attention()([lstm2, lstm2])
    concat = Concatenate()([lstm2, attention])
    lstm3 = LSTM(32, return_sequences=False)(concat)
    dropout3 = Dropout(0.2)(lstm3)
    outputs = Dense(1)(dropout3)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def run_lstm_pipeline():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH, index_col='date', parse_dates=True)
    raw_data = df['import_value'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(raw_data)

    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    
    split = len(X) - 12
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print("Training Advanced LSTM Model...")
    model = build_advanced_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_data=(X_test, y_test))
    
    preds_scaled = model.predict(X_test, verbose=0)
    preds = scaler.inverse_transform(preds_scaled)
    y_test_actual = scaler.inverse_transform(y_test)
    
    metrics = {
        'MAPE': mean_absolute_percentage_error(y_test_actual, preds),
        'RMSE': mean_squared_error(y_test_actual, preds) ** 0.5,
        'R2': r2_score(y_test_actual, preds)
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame([metrics]).assign(Model='LSTM_Attention').to_csv(f"{OUTPUT_DIR}lstm_metrics.csv", index=False)
    print(f"LSTM Metrics: MAPE={metrics['MAPE']:.2%}, RMSE={metrics['RMSE']:.2f}")

    print("Generating 24-Month Future Forecast...")
    curr_seq = scaled_data[-SEQ_LENGTH:]
    forecast_scaled = []
    
    for _ in range(FORECAST_STEPS):
        next_pred = model.predict(curr_seq.reshape(1, SEQ_LENGTH, 1), verbose=0)
        forecast_scaled.append(next_pred[0, 0])
        curr_seq = np.append(curr_seq[1:], next_pred[0, 0]).reshape(SEQ_LENGTH, 1)
        
    forecast_values = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))
    
    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=FORECAST_STEPS, freq='M')
    
    out_df = pd.DataFrame({
        'Date': future_dates,
        'LSTM_Forecast': forecast_values.flatten()
    })
    
    out_df.to_csv(f"{OUTPUT_DIR}lstm_final_forecast.csv", index=False)
    print(f"Success! LSTM Forecast saved to {OUTPUT_DIR}lstm_final_forecast.csv")

if __name__ == "__main__":
    run_lstm_pipeline()