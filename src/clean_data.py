import pandas as pd
import os
import re

RAW_DATA_FOLDER = 'raw_data/'
OUTPUT_FILE = 'data/cleaned_trade_data.csv'

def extract_metadata(filename):
    year_match = re.search(r'20\d{2}', filename)
    year = year_match.group(0) if year_match else None
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month = next((m for m in months if m.lower() in filename.lower()), None)
    
    if 'import' in filename.lower(): flow = 'import'
    elif 'export' in filename.lower(): flow = 'export'
    else: flow = 'unknown'
    
    return year, month, flow

def clean_and_stitch():
    print(f"Scanning {RAW_DATA_FOLDER} for raw files...")
    
    if not os.path.exists(RAW_DATA_FOLDER):
        print(f"Warning: '{RAW_DATA_FOLDER}' not found. Creating it for demonstration.")
        os.makedirs(RAW_DATA_FOLDER, exist_ok=True)
        return

    data_map = {} 

    for root, _, files in os.walk(RAW_DATA_FOLDER):
        for filename in files:
            if not filename.endswith('.csv'): continue
            
            year, month, flow = extract_metadata(filename)
            if not year or not month: continue
            
            filepath = os.path.join(root, filename)
            
            try:
                df = pd.read_csv(filepath)
                val_col = next((c for c in df.columns if 'Value' in str(c) or 'US' in str(c)), None)
                
                if val_col:
                    val = df[val_col].iloc[0] 
                    key = (year, month, flow)
                    data_map[key] = data_map.get(key, 0.0) + float(val)
                    
            except Exception as e:
                print(f"Skipping {filename}: {e}")

    records = []
    for (year, month, flow), total_val in data_map.items():
        date_str = f"{year}-{month}-01"
        records.append({
            'date': pd.to_datetime(date_str, format='%Y-%b-%d'),
            'flow': flow,
            'value': total_val
        })
    
    if not records:
        print("No valid data found to stitch.")
        return

    df_final = pd.DataFrame(records)
    
    pivot_df = df_final.pivot_table(index='date', columns='flow', values='value', aggfunc='sum')
    pivot_df.columns = [f"{c}_value" for c in pivot_df.columns]
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    pivot_df.sort_index().fillna(0).to_csv(OUTPUT_FILE)
    print(f"ETL Complete. Cleaned data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_and_stitch()