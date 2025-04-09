from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for thread safety
import matplotlib.pyplot as plt
from functools import lru_cache

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from tests.model_testing.test_lstm import test_lstm
from utils.data_fetcher import StockDataFetcher
from utils.sentiment_analyzer import SentimentAnalyzer
from dotenv import load_dotenv
import yfinance as yf
from newsapi import NewsApiClient

# Set up template and static folder paths
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')
static_dir = os.path.join(current_dir, 'static')
plots_dir = os.path.join(static_dir, 'plots')

# Ensure directories exist
os.makedirs(static_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir)

# List of companies to analyze
COMPANIES = [
    {'symbol': 'CRM', 'name': 'Salesforce Inc'},
    {'symbol': 'AAPL', 'name': 'Apple Inc'},
    {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
    {'symbol': 'GOOGL', 'name': 'Alphabet Inc'},
    {'symbol': 'AMZN', 'name': 'Amazon.com Inc'}
]

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Initialize components
sentiment_analyzer = SentimentAnalyzer()

@lru_cache(maxsize=10)
def fetch_stock_data(symbol, period='5y'):
    """Fetch stock data using yfinance directly with caching"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        df = df.reset_index()
        return df
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {str(e)}")
        return None

def fetch_news_data(company_name):
    """Fetch news data using NewsAPI directly"""
    if not NEWS_API_KEY:
        print("NewsAPI key not found")
        return None
        
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        news = newsapi.get_everything(
            q=company_name,
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='publishedAt'
        )
        
        if 'articles' in news and news['articles']:
            print(f"Successfully fetched {len(news['articles'])} news articles for {company_name}")
            return news['articles']
        else:
            print(f"No news articles found for {company_name}")
            return []
    except Exception as e:
        print(f"Error fetching news for {company_name}: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_companies')
def get_companies():
    return jsonify(COMPANIES)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symbol = data.get('symbol')
    
    if not symbol:
        return jsonify({"error": "No symbol provided"}), 400
    
    try:
        # Create plot path
        plot_filename = f'lstm_predictions_{symbol}.png'
        plot_path = os.path.join(plots_dir, plot_filename)
        
        print(f"Generating predictions for {symbol}...")
        
        # Fetch stock data only when needed
        stock_data = fetch_stock_data(symbol)
        if stock_data is None:
            raise Exception(f"Failed to fetch stock data for {symbol}")
            
        # Save stock data
        data_dir = os.path.join('data', 'stock_data')
        os.makedirs(data_dir, exist_ok=True)
        stock_data.to_csv(os.path.join(data_dir, f'{symbol}_stock_data.csv'))
        
        print(f"Plot will be saved to: {plot_path}")
        
        # Generate predictions using test_lstm
        results = test_lstm(symbol, save_path=plot_path)
        
        # Ensure the plot was created
        if not os.path.exists(plot_path):
            raise FileNotFoundError(f"Plot file was not created at {plot_path}")
            
        print(f"Successfully generated predictions for {symbol}")
        
        return jsonify({
            "metrics": results['metrics'],
            "plot_url": f"/static/plots/{plot_filename}",
            "dates": results['dates'],
            "actual": results['actual'],
            "predictions": results['predictions']
        })
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_sentiment', methods=['POST'])
def get_sentiment():
    data = request.get_json()
    symbol = data.get('symbol')
    page = data.get('page', 1)
    
    if not symbol:
        return jsonify({"error": "No symbol provided"}), 400
    
    try:
        # Get company data
        company = next((c for c in COMPANIES if c['symbol'] == symbol), None)
        if not company:
            return jsonify({"error": "Invalid company symbol"}), 400
        
        # Fetch news data
        articles = fetch_news_data(company['name'])
        if articles is None:
            if not NEWS_API_KEY:
                return jsonify({
                    "error": "NewsAPI key not configured. Please add NEWS_API_KEY to your .env file."
                }), 500
            return jsonify({"error": "Failed to fetch news data"}), 500
        
        # Process and analyze sentiment for each news item
        processed_news = []
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            if title or description:  # Only process if we have either title or description
                text = f"{title} {description}".strip()
                if text:
                    sentiment = sentiment_analyzer.analyze_text(text)
                    processed_news.append({
                        'date': article.get('publishedAt', ''),
                        'title': title or 'No title available',
                        'sentiment_score': sentiment['compound']
                    })
            
        # Sort by date (most recent first) and paginate
        processed_news.sort(key=lambda x: x['date'], reverse=True)
        start_idx = (page - 1) * 5
        end_idx = start_idx + 5
        page_news = processed_news[start_idx:end_idx] if processed_news else []
        
        return jsonify({
            'news': page_news,
            'has_more': end_idx < len(processed_news)
        })
        
    except Exception as e:
        print(f"Error getting sentiment data: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 