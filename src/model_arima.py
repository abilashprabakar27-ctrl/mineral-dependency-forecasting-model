import pandas as pd
import numpy as np
import os

# Configuration
OUTPUT_FILE = 'data/final.csv'
START_DATE = '2018-04-01'
PERIODS = 60  

def create_synthetic_data():
    """
    Generates a synthetic timeseries dataset for Copper imports.
    Simulates linear trend, seasonality, and market volatility (noise).
    """
    print(f"Generating synthetic training data ({PERIODS} months)...")
    
    dates = pd.date_range(start=START_DATE, periods=PERIODS, freq='MS')
    
    # Component Generation
    trend = np.linspace(10, 50, PERIODS) 
    seasonality = 8 * np.sin(np.linspace(0, 2 * np.pi * 5, PERIODS))
    noise = np.random.normal(0, 3, PERIODS)
    
    # Composition
    values = np.maximum(trend + seasonality + noise, 0)

    df = pd.DataFrame({
        'date': dates,
        'import_value': values
    })
    
    os.makedirs('data', exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success. Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    create_synthetic_data()