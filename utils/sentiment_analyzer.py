import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with NLTK's VADER"""
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            print("Downloading VADER lexicon...")
            nltk.download('vader_lexicon')
        
        self.sia = SentimentIntensityAnalyzer()
    
    def analyze_text(self, text):
        """Analyze sentiment of a single text"""
        if not isinstance(text, str) or not text.strip():
            return {
                'compound': 0,
                'pos': 0,
                'neu': 0,
                'neg': 0
            }
        
        return self.sia.polarity_scores(text)
    
    def analyze_combined_data(self, combined_df):
        """
        Analyze sentiment from combined stock and news data
        
        Parameters:
        - combined_df: DataFrame containing stock data with news columns
        
        Returns:
        - DataFrame with additional sentiment columns
        """
        # Create copy to avoid modifying original
        df = combined_df.copy()
        
        # Initialize sentiment columns
        df['sentiment_compound'] = 0.0
        df['sentiment_positive'] = 0.0
        df['sentiment_neutral'] = 0.0
        df['sentiment_negative'] = 0.0
        
        # Analyze sentiment for each day's news
        for idx in df.index:
            # Combine title and description for better context
            text = f"{df.loc[idx, 'title']} {df.loc[idx, 'description']}"
            sentiment = self.analyze_text(text)
            
            # Store sentiment scores
            df.loc[idx, 'sentiment_compound'] = sentiment['compound']
            df.loc[idx, 'sentiment_positive'] = sentiment['pos']
            df.loc[idx, 'sentiment_neutral'] = sentiment['neu']
            df.loc[idx, 'sentiment_negative'] = sentiment['neg']
        
        # Calculate rolling averages for smoothing
        df['sentiment_compound_ma5'] = df['sentiment_compound'].rolling(window=5).mean()
        df['sentiment_compound_ma10'] = df['sentiment_compound'].rolling(window=10).mean()
        
        # Calculate sentiment signals
        df['sentiment_signal'] = np.where(df['sentiment_compound'] > 0.2, 1,  # Bullish
                                np.where(df['sentiment_compound'] < -0.2, -1,  # Bearish
                                0))  # Neutral
        
        # Save analyzed data
        symbol = combined_df.iloc[0]['Date'].strftime('%Y-%m-%d')  # Use first date as identifier
        df.to_csv(f'data/sentiment_analysis_{symbol}.csv', index=False)
        print(f"Sentiment analysis saved to data/sentiment_analysis_{symbol}.csv")
        
        return df
    
    def get_sentiment_summary(self, df):
        """
        Generate summary statistics for sentiment analysis
        
        Parameters:
        - df: DataFrame with sentiment columns
        
        Returns:
        - dict containing summary statistics
        """
        summary = {
            'total_days': len(df),
            'days_with_news': len(df[df['title'] != '']),
            'avg_sentiment': df['sentiment_compound'].mean(),
            'bullish_days': len(df[df['sentiment_signal'] == 1]),
            'bearish_days': len(df[df['sentiment_signal'] == -1]),
            'neutral_days': len(df[df['sentiment_signal'] == 0]),
            'most_positive_day': {
                'date': df.loc[df['sentiment_compound'].idxmax(), 'Date'],
                'sentiment': df['sentiment_compound'].max(),
                'title': df.loc[df['sentiment_compound'].idxmax(), 'title']
            },
            'most_negative_day': {
                'date': df.loc[df['sentiment_compound'].idxmin(), 'Date'],
                'sentiment': df['sentiment_compound'].min(),
                'title': df.loc[df['sentiment_compound'].idxmin(), 'title']
            }
        }
        
        return summary
    
    def plot_sentiment_analysis(self, df):
        """
        Create visualization of sentiment analysis results
        
        Parameters:
        - df: DataFrame with sentiment columns
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            
            # Plot 1: Stock price with sentiment overlay
            ax1.plot(df['Date'], df['Close'], label='Stock Price', color='blue')
            ax1.set_title('Stock Price with Sentiment Analysis')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Stock Price', color='blue')
            
            # Add sentiment moving average on secondary y-axis
            ax1_twin = ax1.twinx()
            ax1_twin.plot(df['Date'], df['sentiment_compound_ma5'], 
                         label='5-day Sentiment MA', color='green', alpha=0.6)
            ax1_twin.plot(df['Date'], df['sentiment_compound_ma10'],
                         label='10-day Sentiment MA', color='red', alpha=0.6)
            ax1_twin.set_ylabel('Sentiment Score', color='green')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Plot 2: Daily sentiment scores
            ax2.bar(df['Date'], df['sentiment_compound'], 
                   label='Daily Sentiment', alpha=0.6)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            ax2.set_title('Daily Sentiment Scores')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Sentiment Score')
            
            # Add color bands for sentiment zones
            ax2.axhspan(0.2, 1.0, alpha=0.2, color='green', label='Bullish Zone')
            ax2.axhspan(-0.2, 0.2, alpha=0.2, color='gray', label='Neutral Zone')
            ax2.axhspan(-1.0, -0.2, alpha=0.2, color='red', label='Bearish Zone')
            ax2.legend()
            
            plt.tight_layout()
            
            # Save plot
            symbol = df.iloc[0]['Date'].strftime('%Y-%m-%d')
            plt.savefig(f'data/sentiment_analysis_{symbol}.png')
            print(f"Sentiment analysis plot saved to data/sentiment_analysis_{symbol}.png")
            
            plt.close()
            
        except ImportError:
            print("Matplotlib and/or seaborn not installed. Skipping visualization.")
