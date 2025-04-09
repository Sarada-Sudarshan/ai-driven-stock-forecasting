from utils.data_fetcher import StockDataFetcher
from utils.sentiment_analyzer import SentimentAnalyzer
import pandas as pd

def test_sentiment_analysis():
    """Test the sentiment analysis functionality"""
    
    # Initialize components
    fetcher = StockDataFetcher()
    analyzer = SentimentAnalyzer()
    
    # Fetch combined stock and news data
    print("\nFetching combined stock and news data...")
    combined_data = fetcher.get_combined_data(
        symbol='RELIANCE.NS',
        company_name='Reliance Industries',
        days=30
    )
    
    if combined_data is None:
        print("Error: Could not fetch combined data")
        return
    
    print(f"\nAnalyzing sentiment for {len(combined_data)} days of data...")
    
    # Perform sentiment analysis
    analyzed_data = analyzer.analyze_combined_data(combined_data)
    
    # Get summary statistics
    summary = analyzer.get_sentiment_summary(analyzed_data)
    
    # Print summary
    print("\nSentiment Analysis Summary:")
    print(f"Total days analyzed: {summary['total_days']}")
    print(f"Days with news: {summary['days_with_news']}")
    print(f"Average sentiment: {summary['avg_sentiment']:.4f}")
    print(f"Bullish days: {summary['bullish_days']}")
    print(f"Bearish days: {summary['bearish_days']}")
    print(f"Neutral days: {summary['neutral_days']}")
    
    print("\nMost Positive Day:")
    print(f"Date: {summary['most_positive_day']['date']}")
    print(f"Sentiment: {summary['most_positive_day']['sentiment']:.4f}")
    print(f"News: {summary['most_positive_day']['title']}")
    
    print("\nMost Negative Day:")
    print(f"Date: {summary['most_negative_day']['date']}")
    print(f"Sentiment: {summary['most_negative_day']['sentiment']:.4f}")
    print(f"News: {summary['most_negative_day']['title']}")
    
    # Create visualization
    print("\nGenerating sentiment analysis plots...")
    analyzer.plot_sentiment_analysis(analyzed_data)
    
    print("\nSentiment analysis complete!")

if __name__ == "__main__":
    test_sentiment_analysis() 