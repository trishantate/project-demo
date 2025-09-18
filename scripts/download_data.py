# scripts/download_data.py

import yfinance as yf
import pandas as pd
import datetime
import os

def download_and_save_data(symbols, save_dir="data/yfinance_data", start_date="2020-01-01", end_date="2023-01-01"):
    """
    Download stock data and save to CSV files.
    
    Args:
        symbols: List of stock symbols
        save_dir: Directory to save CSV files
        start_date: Start date for data download (YYYY-MM-DD)
        end_date: End date for data download (YYYY-MM-DD)
    """
    # ✅ ensure save folder exists
    os.makedirs(save_dir, exist_ok=True)

    # Convert string dates to datetime objects
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    print(f"📅 Downloading data from {start_date.date()} to {end_date.date()}")
    print(f"📂 Save directory: {os.path.abspath(save_dir)}")
    print(f"🎯 Symbols: {symbols}\n")

    for symbol in symbols:
        print(f"⬇️ Downloading {symbol} data...")
        
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if data.empty:
                print(f"⚠️ No data found for {symbol}")
                continue

            # reset index so Date becomes a column
            data.reset_index(inplace=True)

            # ✅ match naming convention: SYMBOL_historical_data.csv
            file_path = os.path.join(save_dir, f"{symbol}_historical_data.csv")

            # save CSV
            data.to_csv(file_path, index=False)

            print(f"✅ Saved {symbol} data: {len(data)} rows")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")
            print(f"   File: {os.path.abspath(file_path)}\n")
            
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {str(e)}\n")

def create_sample_sentiment_data(save_dir="data/yfinance_data", start_date="2020-01-01", end_date="2023-01-01"):
    """
    Create sample sentiment data for testing if you don't have real sentiment data.
    """
    import numpy as np
    
    # Convert string dates to datetime objects
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create sample headlines and sentiment data
    sample_headlines = [
        "Company reports strong quarterly earnings",
        "Stock price reaches new highs amid positive outlook",
        "Market volatility affects stock performance", 
        "Analysts upgrade stock rating following good results",
        "Company faces regulatory challenges",
        "Innovation drives company growth",
        "Market uncertainty impacts investor confidence"
    ]
    
    sentiment_data = []
    
    for date in date_range:
        # Skip weekends (stock market closed)
        if date.weekday() < 5:  # Monday = 0, Sunday = 6
            # Generate 1-3 random headlines per day
            num_headlines = np.random.randint(1, 4)
            
            for _ in range(num_headlines):
                headline = np.random.choice(sample_headlines)
                # Add some variation to headlines
                if np.random.random() > 0.7:
                    headline = headline.replace("Company", np.random.choice(["Tech giant", "Major firm", "Leading company"]))
                
                sentiment_data.append({
                    'date': date,
                    'headline': headline
                })
    
    # Create DataFrame
    sentiment_df = pd.DataFrame(sentiment_data)
    
    # Save to CSV
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "raw_analyst_ratings.csv")
    sentiment_df.to_csv(file_path, index=False)
    
    print(f"✅ Created sample sentiment data: {len(sentiment_df)} rows")
    print(f"   File: {os.path.abspath(file_path)}")
    print(f"   Date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")

if __name__ == "__main__":
    # ✅ FIXED: Include all 7 stock symbols that your Streamlit app expects
    stock_symbols = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
    
    # ✅ FIXED: Use the same date range as your Streamlit app (2020-2023)
    save_directory = "data/yfinance_data"
    
    # Download stock data
    download_and_save_data(
        symbols=stock_symbols, 
        save_dir=save_directory,
        start_date="2020-01-01", 
        end_date="2023-01-01"
    )
    
    # Create sample sentiment data if you don't have real sentiment data
    print("\n" + "="*50)
    print("Creating sample sentiment data...")
    create_sample_sentiment_data(
        save_dir=save_directory,
        start_date="2020-01-01", 
        end_date="2023-01-01"
    )
    
    print("\n🎉 Data download complete!")
    print(f"📂 All files saved to: {os.path.abspath(save_directory)}")