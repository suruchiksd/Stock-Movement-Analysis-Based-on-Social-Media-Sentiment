# data_preprocessing.py
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import re


nltk.download("stopwords", quiet=True)


def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords and stem
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]

    # Join tokens back into a string
    cleaned_text = " ".join(tokens)

    return cleaned_text


def preprocess_data(input_file, output_file):
    # Load data
    df = pd.read_csv(input_file)

    # Combine title and body
    df["text"] = df["title"] + " " + df["body"].fillna("")

    # Clean text
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Save preprocessed data
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")


if __name__ == "__main__":
    preprocess_data("reddit_stocks_data.csv", "preprocessed_stocks_data.csv")
