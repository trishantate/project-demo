import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from textblob import TextBlob

import yfinance as yf
import datetime

import os
import yfinance as yf
import pandas as pd

def download_stock_data(ticker, data_directory, start_date='2020-01-01', end_date='2021-01-01'):
    """
    Download stock data from Yahoo Finance and save as a CSV file.
    """
    # Download stock data from Yahoo Finance
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Define the file path where to save the data
    file_path = os.path.join(data_directory, f"{ticker}_historical_data.csv")
    
    # Save the data as a CSV file
    stock_data.to_csv(file_path)
    
    return stock_data

def merge_stocks(data_directory):
    """
    Merge stock data for a list of tickers, downloading missing files.
    """
    # Ensure the directory exists
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # Define the list of stock tickers you are interested in
    stock_tickers = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
    
    # Create an empty DataFrame to hold all stock data
    stock_data = pd.DataFrame()

    # Loop through each ticker, check if data exists, if not, download it
    for ticker in stock_tickers:
        file_path = os.path.join(data_directory, f"{ticker}_historical_data.csv")

        # If file doesn't exist, download the stock data
        if not os.path.exists(file_path):
            print(f"Downloading data for {ticker}...")
            download_stock_data(ticker, data_directory)  # Download and save the data
        
        # Load the stock data after downloading
        df = pd.read_csv(file_path)
        df['Stock'] = ticker  # Add stock ticker column to the data
        
        # Check for the correct date column
        possible_date_cols = ['Date', 'date', 'Datetime', 'timestamp']
        for col in possible_date_cols:
            if col in df.columns:
                df.rename(columns={col: 'Date'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'])  # Convert to datetime
                break
        else:
            raise KeyError(f"No date column found in {file_path}. Columns: {df.columns}")

        # Concatenate with the overall stock data
        stock_data = pd.concat([stock_data, df], ignore_index=True)

    # Sort the merged data by Date and Stock
    stock_data.sort_values(['Date', 'Stock'], inplace=True)
    
    return stock_data

# load stock data 
def load_stock_data(file_path, ticker=None):
    df = pd.read_csv(file_path)

    # Normalize the Date column to 'Date'
    possible_date_cols = ['Date', 'date', 'Datetime', 'datetime', 'timestamp']
    for col in possible_date_cols:
        if col in df.columns:
            df.rename(columns={col: 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            break
    else:
        raise KeyError(f"No date column found in {file_path}. Columns: {df.columns}")

    # Normalize Stock/Ticker column
    possible_stock_cols = ['Stock', 'stock', 'Symbol', 'Ticker']
    for col in possible_stock_cols:
        if col in df.columns:
            df.rename(columns={col: 'Stock'}, inplace=True)
            break
    else:
        if ticker:
            df['Stock'] = ticker
        else:
            df['Stock'] = "Unknown"

    return df


# # merge stocks data 
# def merge_stocks(data_directory):
#     # Define the stock tickers and their associated file names
#     stock_files = {
#         'AAPL': 'AAPL_historical_data.csv',
#         'AMZN': 'AMZN_historical_data.csv',
#         'GOOGL': 'GOOG_historical_data.csv',
#         'META': 'META_historical_data.csv',
#         'MSFT': 'MSFT_historical_data.csv',
#         'NVDA': 'NVDA_historical_data.csv',
#         'TSLA': 'TSLA_historical_data.csv'
#     }
    
#     stock_data = pd.DataFrame()  # Empty DataFrame to store all stock data
    
#     # Check if each stock file exists or needs to be downloaded
#     for ticker, file_name in stock_files.items():
#         file_path = os.path.join(data_directory, file_name)
        
#         # If file doesn't exist, download the stock data
#         if not os.path.exists(file_path):
#             print(f"Downloading data for {ticker}...")
#             download_stock_data(ticker)  # Download the data for the missing stock
            
#         # Load the stock data after download
#         df = load_stock_data(file_path, ticker)
#         stock_data = pd.concat([stock_data, df], ignore_index=True)
    
#     # Sort the merged data by Date and Stock
#     stock_data.sort_values(['Date', 'Stock'], inplace=True)
    
#     return stock_data

# calculate sentiment
def calculate_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity


# calculate daily returns 
def calculate_daily_returns(df):
    # Convert Close column to numeric to avoid errors during pct_change
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Daily_Return'] = df.groupby('Stock')['Close'].pct_change()
    return df



# sentiment analysis on 'headline'
def perform_sentiment_analysis(df):
    if 'headline' not in df.columns:
        raise KeyError("Expected a 'headline' column for sentiment analysis.")
    df['Sentiment_Score'] = df['headline'].apply(calculate_sentiment)
    return df


# calculate mean daily sentiment 
def aggregate_daily_sentiment(df):
    return df.groupby('Date')['Sentiment_Score'].mean().reset_index()


# calculate correlation between Sentiment Score and Daily return
def calculate_correlation(df):
    """Calculate correlation between sentiment scores and stock returns."""
    correlations = df.groupby('Stock').apply(
        lambda x: x['Sentiment_Score'].corr(x['Daily_Return'], method='pearson')
    )
    return correlations


# test correlation significance 
def test_correlation_significance(df, column1, column2):
    correlation, p_value = stats.pearsonr(df[column1], df[column2])
    return pd.Series({'correlation': correlation, 'p_value': p_value})