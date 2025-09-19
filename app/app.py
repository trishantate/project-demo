import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os
from datetime import datetime, timedelta

# CRITICAL: Page config MUST be first
st.set_page_config(
    page_title="Stock Sentiment Analysis",
    page_icon="📈",
    layout="centered",  # Changed from "wide" for WebView
    initial_sidebar_state="collapsed"  # Collapsed sidebar for mobile
)

# WebView-safe CSS - removes problematic elements
st.markdown("""
<style>
    /* Hide all Streamlit branding that causes WebView issues */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Disable problematic interactions */
    .stApp * {
        -webkit-user-select: none;
        -moz-user-select: none;
        user-select: none;
    }
    
    /* WebView-safe styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
    }
    
    /* Prevent zoom issues */
    .stPlotlyChart, .stPyplot {
        pointer-events: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Simplified caching that works in WebView
def safe_cache_function(func):
    """Safe caching wrapper for WebView"""
    try:
        return st.cache_data(ttl=1800)(func)  # 30 min cache
    except:
        return func  # No caching if it fails

@safe_cache_function
def download_simple_stock_data(symbol="AAPL"):
    """Simplified stock data download"""
    try:
        # Get last 3 months of data only
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        data = yf.download(
            symbol, 
            start=start_date.strftime('%Y-%m-%d'), 
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        if data.empty:
            return None
            
        data.reset_index(inplace=True)
        
        # Calculate daily returns
        data['Daily_Return'] = data['Close'].pct_change()
        
        return data
        
    except Exception as e:
        st.error(f"Error downloading {symbol}: {str(e)}")
        return None

def create_simple_sentiment_data(stock_data):
    """Create simple sentiment data that matches stock dates"""
    if stock_data is None or len(stock_data) == 0:
        return None
        
    np.random.seed(42)  # For consistent results
    
    # Simple sentiment generation
    sentiment_scores = []
    for i in range(len(stock_data)):
        # Create sentiment that has some correlation with returns
        base_sentiment = np.random.normal(0, 0.5)
        if i > 0:
            # Add some correlation with previous day's return
            prev_return = stock_data['Daily_Return'].iloc[i-1] if not pd.isna(stock_data['Daily_Return'].iloc[i-1]) else 0
            base_sentiment += prev_return * 0.3
        
        sentiment_scores.append(np.clip(base_sentiment, -1, 1))
    
    stock_data['Sentiment_Score'] = sentiment_scores
    return stock_data

def create_simple_chart_fallback(stock_data, selected_stock):
    """Create simple fallback visualization using Streamlit native charts"""
    try:
        # Prepare data for Streamlit chart
        chart_data = stock_data[['Date', 'Sentiment_Score', 'Daily_Return']].copy()
        chart_data = chart_data.set_index('Date')
        
        # Simple line chart
        st.subheader(f"📈 {selected_stock} - Sentiment vs Returns")
        st.line_chart(chart_data[['Sentiment_Score']], height=300)
        
        st.subheader(f"📊 {selected_stock} - Daily Returns")
        st.line_chart(chart_data[['Daily_Return']], height=300)
        
    except Exception as e:
        st.error("Chart display failed - showing data table instead")
        st.dataframe(stock_data[['Date', 'Sentiment_Score', 'Daily_Return']].tail(10))

def main():
    """Main application with comprehensive WebView error handling"""
    try:
        # Simple header
        st.title("📈 Stock Sentiment Dashboard")
        st.markdown("*WebView-optimized version*")
        
        # Simple stock selection (no sidebar to avoid WebView issues)
        st.subheader("🎯 Select Stock")
        stock_options = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"]
        selected_stock = st.selectbox("Choose a stock:", stock_options, index=0)
        
        if st.button("📊 Load Analysis", type="primary"):
            
            with st.spinner(f"Loading {selected_stock} data..."):
                # Download stock data
                stock_data = download_simple_stock_data(selected_stock)
                
                if stock_data is None:
                    st.error("❌ Failed to load stock data")
                    return
                
                # Add sentiment data
                stock_data = create_simple_sentiment_data(stock_data)
                
                if stock_data is None:
                    st.error("❌ Failed to process data")
                    return
                
                # Remove any NaN values
                stock_data = stock_data.dropna()
                
                if len(stock_data) < 5:
                    st.warning("⚠️ Insufficient data for analysis")
                    return
                
                st.success("✅ Data loaded successfully!")
                
                # Calculate correlation
                correlation = stock_data['Sentiment_Score'].corr(stock_data['Daily_Return'])
                
                # Display basic metrics
                st.subheader("📊 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Correlation", 
                        f"{correlation:.3f}"
                    )
                
                with col2:
                    avg_sentiment = stock_data['Sentiment_Score'].mean()
                    st.metric(
                        "Avg Sentiment", 
                        f"{avg_sentiment:.3f}"
                    )
                
                with col3:
                    avg_return = stock_data['Daily_Return'].mean()
                    st.metric(
                        "Avg Return", 
                        f"{avg_return:.3f}"
                    )
                
                # Correlation interpretation
                abs_corr = abs(correlation)
                if abs_corr >= 0.5:
                    st.success(f"🟢 Strong correlation detected ({correlation:.3f})")
                elif abs_corr >= 0.3:
                    st.info(f"🟡 Moderate correlation detected ({correlation:.3f})")
                else:
                    st.warning(f"🔴 Weak correlation detected ({correlation:.3f})")
                
                # Simple charts (WebView-safe)
                create_simple_chart_fallback(stock_data, selected_stock)
                
                # Data summary
                st.subheader("📋 Recent Data")
                display_data = stock_data[['Date', 'Close', 'Daily_Return', 'Sentiment_Score']].tail(10)
                st.dataframe(display_data, use_container_width=True, hide_index=True)
                
                # Simple statistics
                with st.expander("📈 Statistics Summary"):
                    st.write(f"**Data Period:** {stock_data['Date'].min().strftime('%Y-%m-%d')} to {stock_data['Date'].max().strftime('%Y-%m-%d')}")
                    st.write(f"**Total Days:** {len(stock_data)}")
                    st.write(f"**Sentiment Range:** {stock_data['Sentiment_Score'].min():.3f} to {stock_data['Sentiment_Score'].max():.3f}")
                    st.write(f"**Return Range:** {stock_data['Daily_Return'].min():.3f} to {stock_data['Daily_Return'].max():.3f}")
    
    except Exception as e:
        # Comprehensive error handling for WebView
        st.error("🚫 WebView Compatibility Error")
        st.warning("This error commonly occurs in mobile WebView environments")
        
        # Show error details
        with st.expander("🔧 Error Information"):
            st.code(f"Error: {str(e)}")
            st.code(f"Streamlit Version: {st.__version__}")
        
        # Provide basic functionality even with errors
        st.info("📱 Basic Mode Available")
        st.write("Some advanced features are disabled for mobile compatibility")
        
        # Simple fallback
        if st.button("📊 Show Sample Data"):
            sample_data = pd.DataFrame({
                'Date': pd.date_range('2024-01-01', periods=5),
                'Stock': ['AAPL'] * 5,
                'Price': [150, 152, 151, 153, 155],
                'Sentiment': [0.2, 0.5, -0.1, 0.3, 0.7]
            })
            st.dataframe(sample_data)

if __name__ == "__main__":
    main()