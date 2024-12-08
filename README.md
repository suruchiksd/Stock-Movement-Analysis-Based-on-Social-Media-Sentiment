# Stock Movement Analysis Based on Social Media Sentiment

This project develops a machine learning model to predict stock movements based on sentiment analysis of Reddit posts from r/stocks.
## Features
- Scrapes and analyzes sentiment from Reddit posts
- Preprocesses text data for sentiment analysis
- Builds a predictive model for stock movement based on sentiment data
- Provides visualizations, such as confusion matrix and feature importance

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/stock-sentiment-analysis.git
   cd stock-sentiment-analysis

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt

3. Set up your Reddit API credentials:
   create .env file and add Client ID, Secret, and User-Agent.
   ```bash
   REDDIT_CLIENT_ID='YOUR_CLIENT_ID'
   REDDIT_CLIENT_SECRET='YOUR_CLIENT_SECRET'
   
## Usage
Follow these steps to run the project:

1. Scrape data from Reddit:
   ```bash
   python reddit_scraper.py

2. Preprocess the scraped data:
   ```bash
   python data_preprocessing.py

2. Perform sentiment analysis:
   ```bash
   python sentiment_analysis.py

2. Build and evaluate the prediction model:
   ```bash
   python prediction_model.py

## Results
- Model performance metrics (e.g., accuracy, precision, recall) will be displayed in the terminal.
- Visualizations such as the confusion matrix and feature importance plots will be saved in the project directory.

## Future Improvements
Some potential enhancements for the project:
- Feature Expansion: Incorporate additional features such as historical stock prices, trading volume, or market indicators.
- Algorithm Optimization: Experiment with advanced machine learning models or fine-tune hyperparameters.
- Real-Time Predictions: Develop real-time sentiment scraping and prediction capabilities.
