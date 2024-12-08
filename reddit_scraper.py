import os
from dotenv import load_dotenv
import praw
import pandas as pd
from datetime import datetime

# Load environment variables
load_dotenv()


def scrape_reddit(subreddit_name, time_filter="week", limit=1000):
    # Initialize Reddit API client
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="Sentiment_analysis:v1.0 (by /u/YOUR_REDDIT_USERNAME)",
    )

    # Access the specified subreddit
    subreddit = reddit.subreddit(subreddit_name)

    # Scrape posts
    posts = []
    try:
        for post in subreddit.top(time_filter=time_filter, limit=limit):
            posts.append(
                {
                    "title": post.title,
                    "body": post.selftext,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "created_utc": datetime.fromtimestamp(post.created_utc),
                }
            )
    except Exception as e:
        print(f"An error occurred while scraping: {e}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(posts)
    return df


if __name__ == "__main__":
    # Scrape data from r/stocks
    df = scrape_reddit("stocks")

    if df is not None:
        # Save to CSV
        df.to_csv("reddit_stocks_data.csv", index=False)
        print(f"Scraped {len(df)} posts and saved to reddit_stocks_data.csv")
    else:
        print("Failed to scrape data.")
