import sys
import argparse
import unicodedata
import warnings

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# suppress pandas future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def download_resources():
    """Ensure VADER lexicon is available."""
    nltk.download('vader_lexicon', quiet=True)


def load_data(path: str) -> pd.DataFrame:
    """Load tweet data and validate required columns."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        sys.exit(1)

    required = {'Date', 'Tweet', 'Stock Name'}
    missing = required - set(df.columns)
    if missing:
        print(f"Error: Missing columns in input CSV: {missing}")
        sys.exit(1)

    # Parse dates (keep only the date part)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
    if df['Date'].isnull().any():
        print("Error: Some dates could not be parsed.")
        sys.exit(1)

    return df


def prompt_ticker(df: pd.DataFrame) -> str:
    """Prompt the user to choose a ticker from the dataset."""
    available = sorted(df['Stock Name'].unique())
    print("Available tickers:")
    for t in available:
        print(f"  • {t}")
    chosen = input("Enter one of the above stock tickers: ").strip()
    if chosen not in available:
        print(f"Error: '{chosen}' is not in the available tickers.")
        sys.exit(1)
    return chosen


def filter_by_stock(df: pd.DataFrame, stock: str) -> pd.DataFrame:
    """Return subset of DataFrame for the given stock symbol."""
    return df[df['Stock Name'] == stock].copy()


def compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Compute VADER compound sentiment score for each tweet."""
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for text in df['Tweet'].astype(str):
        norm = unicodedata.normalize('NFKD', text)
        vs = analyzer.polarity_scores(norm)
        scores.append(vs['compound'])
    df = df.copy()
    df['sentiment_score'] = scores
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment scores by date (daily average)."""
    return df.groupby('Date', as_index=False)['sentiment_score'].mean()


def parse_args():
    """Parse CLI args for custom input/output paths."""
    parser = argparse.ArgumentParser(
        description="Interactive sentiment analysis for stock tweets using VADER."
    )
    parser.add_argument(
        '--input', '-i', default='stock_tweets.csv',
        help='Path to input tweets CSV (allows custom input path)'
    )
    parser.add_argument(
        '--output', '-o', default=None,
        help='Path to output CSV (allows custom output path)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    download_resources()

    # 1. Load   
    df = load_data(args.input)

    # 2. Prompt  
    ticker = prompt_ticker(df)

    # 3. Filter  
    df_stock = filter_by_stock(df, ticker)
    if df_stock.empty:
        print(f"Error: No tweets found for {ticker}.")
        sys.exit(1)

    # 4. Compute & Aggregate  
    df_sent  = compute_sentiment(df_stock)
    df_daily = aggregate_daily(df_sent)

    # 5. Save  
    out_file = args.output or f"{ticker}_twitter_sentiment.csv"  # custom or default
    df_daily.to_csv(out_file, index=False)
    print(f"\nSaved daily sentiment to '{out_file}'")


if __name__ == "__main__":
    main()
