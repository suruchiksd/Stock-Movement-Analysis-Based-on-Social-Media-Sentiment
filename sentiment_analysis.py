import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")


def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)["compound"]


def perform_sentiment_analysis(input_file, output_file):
    # Load preprocessed data
    df = pd.read_csv(input_file)

    # Perform sentiment analysis
    df["sentiment_score"] = df["cleaned_text"].apply(analyze_sentiment)

    # Categorize sentiment
    df["sentiment"] = pd.cut(
        df["sentiment_score"],
        bins=[-1, -0.1, 0.1, 1],
        labels=["negative", "neutral", "positive"],
    )

    # Save results
    df.to_csv(output_file, index=False)
    print(f"Sentiment analysis results saved to {output_file}")


if __name__ == "__main__":
    perform_sentiment_analysis(
        "preprocessed_stocks_data.csv", "sentiment_stocks_data.csv"
    )
