#!/usr/bin/env python
# coding: utf-8

# In[4]:


# REDDITT SENTIMENT


# In[11]:
















import praw
from textblob import TextBlob
import pandas as pd
import requests
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import os
import plotly.express as px
import plotly.graph_objects as go
import time

# Initialize Reddit API


# Use environment variables for Reddit API credentials if available
reddit_client_id = os.environ.get("REDDIT_CLIENT_ID", "GSl5ASGK24eNVwIKQAm8CQ")
reddit_client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "PmRh_hoFbgCrGOaEomX54-_QkYMfyQ")

# Initialize Reddit API with these credentials
reddit = praw.Reddit(
    client_id=reddit_client_id,
    client_secret=reddit_client_secret,
    user_agent="your_user_agent"
)

# Analyze sentiment
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Returns polarity (-1 to 1)

# Fetch Reddit posts and calculate sentiment
def fetch_reddit_sentiment():
    posts = []
    subreddit_name = "Bitcoin"  # Subreddit
    search_term = "Bitcoin"  # Search term

    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.search(search_term, limit=100, sort="new"):
        posts.append({
            "Title": post.title,
            "Score": post.score,
            "Text": post.selftext,
            "URL": post.url
        })

    # Analyze sentiment
    results = []
    for post in posts:
        content = f"{post['Title']} {post['Text']}"  # Combine title and text
        polarity = analyze_sentiment(content)  # Get polarity score (-1 to 1)
        results.append({
            "Title": post["Title"],
            "Score": post["Score"],
            "Polarity": polarity,
            "URL": post["URL"]
        })

    # Create a DataFrame and calculate the average sentiment
    df = pd.DataFrame(results)
    avg_polarity = df["Polarity"].mean() if not df.empty else 0
    sentiment_score = int((avg_polarity + 1) * 50)  # Scale -1 to 1 into 0 to 100
    print(f"Reddit Bitcoin Sentiment Score: {sentiment_score}")
    return sentiment_score

# Volatility Index function
def fetch_volatility_sentiment():
    try:
        # Fetch Bitcoin market data from CoinGecko (24-hour price data)
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": "1"}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            prices = [entry[1] for entry in data["prices"]]

            if not prices:
                print("No price data available.")
                return 50  # Neutral if data is missing

            # Calculate High-Low Range Volatility
            highest_price = max(prices)
            lowest_price = min(prices)
            mean_price = np.mean(prices)

            price_volatility = (highest_price - lowest_price) / mean_price  # High-Low Range %
            start_price = prices[0]
            end_price = prices[-1]
            price_change = (end_price - start_price) / start_price  # % Change from Start to End

            # Print Debugging Information
            print(f"Start Price: {start_price:.2f}, End Price: {end_price:.2f}, Change: {price_change:.2%}")
            print(f"Highest Price: {highest_price:.2f}, Lowest Price: {lowest_price:.2f}")
            print(f"Price Volatility (High-Low %): {price_volatility:.4f}")

            # Updated Volatility Thresholds
            high_volatility = 0.06  # 6% = high volatility
            low_volatility = 0.01   # 1% = low volatility

            # Sentiment Calculation
            if price_volatility > high_volatility:
                if price_change < 0:
                    return int(0 + 40 * (1 - min(price_volatility / high_volatility, 1)))  # High Fear
                else:
                    return int(60 + 40 * min(price_volatility / high_volatility, 1))  # High Greed
            elif price_volatility < low_volatility:
                return 50  # Neutral for low volatility
            else:
                # Gradual adjustment for mid-range volatility
                base_sentiment = 50
                sentiment_adjustment = price_change * 50 / 10  # Scale change (-10% to +10%) to sentiment range
                return int(base_sentiment + sentiment_adjustment)

    except Exception as e:
        print(f"Error fetching CoinGecko volatility: {e}")
    
    return 50  # Default neutral sentiment if API call fails

# Fear and Greed Index
def fetch_fear_greed_index(retries=3, delay=5):
    """
    Fetch the Fear and Greed Index from the alternative.me website.
    Includes retry logic for handling temporary server issues.
    """
    url = "https://alternative.me/crypto/fear-and-greed-index/"
    
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            
            soup = BeautifulSoup(response.content, "html.parser")
            fear_greed_index_online = soup.find("div", class_="fng-circle").text.strip()
            
            # Convert the value to an integer
            fear_greed_index = int(fear_greed_index_online)
            print(f"Fear and Greed Index: {fear_greed_index}")
            return fear_greed_index
        
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                print("❌ Error fetching Fear and Greed Index after multiple attempts.")
                return 50  # Default neutral value if retries fail

# Calculate combined sentiment
def calculate_combined_sentiment():
    # Fetch sentiment scores
    reddit_sentiment = fetch_reddit_sentiment()
    volatility_sentiment = fetch_volatility_sentiment()
    alternative_me_sentiment = fetch_fear_greed_index()

    # Apply weights (50% Alternative.me, 30% Volatility, 20% Reddit)
    combined_sentiment_score = (
        (alternative_me_sentiment * 0.5) +
        (volatility_sentiment * 0.3) +
        (reddit_sentiment * 0.2)
    )

    # Print results for debugging
    print(f"Reddit Sentiment: {reddit_sentiment}")
    print(f"Volatility Sentiment: {volatility_sentiment}")
    print(f"Alternative.me Sentiment: {alternative_me_sentiment}")
    print(f"Weighted Combined Sentiment Score: {combined_sentiment_score:.2f} (0 = Negative, 100 = Positive)")

    rounded_score = round(combined_sentiment_score)
    return rounded_score

# Fetch historical sentiment data
def fetch_historical_sentiment(start_date=None):
    """
    Fetch historical data from alternative.me.
    If start_date is provided, only fetch data after that date.
    """
    url = "https://api.alternative.me/fng/?limit=365"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()["data"]
        sentiment_data = []
        for entry in data:
            date = datetime.utcfromtimestamp(int(entry["timestamp"])).strftime("%Y-%m-%d")
            # Filter by start_date if provided
            if start_date is None or date >= start_date:
                score = int(entry["value"])
                sentiment_data.append({
                    "Date": date, 
                    "Sentiment Score": score, 
                    "Source": "alternative.me"
                })
        
        df = pd.DataFrame(sentiment_data)
        return df
    else:
        print(f"Error fetching Alternative.me data: {response.status_code}")
        return None

# Initialize historical data
def initialize_historical_data():
    """
    Function to initialize the CSV with historical data from alternative.me.
    """
    df = fetch_historical_sentiment()
    if df is not None:
        df.to_csv("sentiment_scores.csv", index=False)
        print(f"✅ Historical data initialized with {len(df)} records")
    else:
        print("❌ Failed to initialize historical data")

# Clean historical data
def clean_historical_data():
    """
    Ensure historical data has no future dates and is properly formatted.
    """
    current_date = datetime.now()
    try:
        df = pd.read_csv("sentiment_scores.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[df["Date"] <= current_date]
        
        # Sort by date to ensure chronological order
        df = df.sort_values("Date")
        
        # Remove duplicates, keeping the latest entry for each date
        df = df.drop_duplicates(subset="Date", keep="last")
        
        df.to_csv("sentiment_scores.csv", index=False)
        print(f"✅ Historical data cleaned up to current date: {current_date.strftime('%Y-%m-%d')}")
    except FileNotFoundError:
        print("❌ No historical data file found. Starting fresh.")

# Save sentiment to CSV
def save_sentiment_to_csv(score):
    """
    Save today's custom calculation, never overwriting historical data
    """
    today = datetime.now().strftime("%Y-%m-%d")
    sentiment_data = {
        "Date": today,
        "Sentiment Score": score,
        "Source": "custom_calculation"
    }

    try:
        # Load existing data
        df = pd.read_csv("sentiment_scores.csv")
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        
        # Remove any existing entry for today (if exists)
        df = df[df["Date"] != today]
        
        # Add new custom calculation
        new_data = pd.DataFrame([sentiment_data])
        df = pd.concat([df, new_data], ignore_index=True)
        
        # Sort by date
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        
        df.to_csv("sentiment_scores.csv", index=False)
        print(f"✅ Custom Sentiment Score saved for {today}: {score}")
    except FileNotFoundError:
        pd.DataFrame([sentiment_data]).to_csv("sentiment_scores.csv", index=False)
        print(f"✅ Created new file with custom sentiment score for {today}: {score}")

# Plot interactive sentiment chart
def plot_interactive_sentiment():
    try:
        # Load historical data
        df = pd.read_csv("sentiment_scores.csv")
        df["Date"] = pd.to_datetime(df["Date"])  # Convert Date column to datetime
        df = df.sort_values(by="Date")  # Sort data in chronological order

        # Create the line chart
        fig = px.line(
            df,
            x="Date",
            y="Sentiment Score",
            title="Bitcoin Fear & Greed Sentiment Over Time",
            labels={"Sentiment Score": "Sentiment (0 = Fear, 100 = Greed)"},
            markers=True
        )

        # Get the max date (end_date) and set up time ranges
        end_date = df["Date"].max()

        # Add buttons for different time periods
        fig.update_layout(
            updatemenus=[{
                'buttons': [
                    {
                        'args': [{'xaxis.range': [end_date - timedelta(days=7), end_date]}],
                        'label': '7D',
                        'method': 'relayout'
                    },
                    {
                        'args': [{'xaxis.range': [end_date - timedelta(days=30), end_date]}],
                        'label': '1M',
                        'method': 'relayout'
                    },
                    {
                        'args': [{'xaxis.range': [end_date - timedelta(days=90), end_date]}],
                        'label': '3M',
                        'method': 'relayout'
                    },
                    {
                        'args': [{'xaxis.range': [end_date - timedelta(days=365), end_date]}],
                        'label': '1Y',
                        'method': 'relayout'
                    },
                    {
                        'args': [{'xaxis.range': [df["Date"].min(), end_date]}],
                        'label': 'Max',
                        'method': 'relayout'
                    }
                ],
                'direction': 'right',
                'showactive': True,
                'type': 'buttons',
                'x': 0.2,
                'y': 2
            }],
            yaxis=dict(range=[0, 100]),  # Fix y-axis range
            plot_bgcolor="white",
            yaxis_gridcolor="lightgrey",
            xaxis_gridcolor="lightgrey"
        )

        # Save the chart to HTML
        fig.write_html("line_index.html")
        print("✅ line_index.html has been updated.")
        
        # Also display the chart if in an interactive environment
        fig.show()

    except Exception as e:
        print(f"❌ Error plotting sentiment: {e}")
        print(f"Detailed error: {str(e)}")

# Plot fear and greed gauge
def plot_fear_greed_gauge(sentiment_score):
    # Define gauge chart with Bitcoin logo as indicator
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        title={
            "text": "Bitcoin Fear & Greed Index<br><span style='font-size:20px;color:gray'>Crypto Market Sentiment Analysis</span>",
            "font": {"size": 40}
        },
        number={"font": {"size": 50}, "prefix": ""},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "black"},
            "bar": {"color": "black"},  # This will be our indicator
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 2,
            "bordercolor": "white",
            "steps": [
                {"range": [0, 25], "color": "red"},
                {"range": [25, 50], "color": "orange"},
                {"range": [50, 75], "color": "yellow"},
                {"range": [75, 100], "color": "green"}
            ],
            "shape": "angular"
        }
    ))

    # Add Bitcoin logo
    fig.update_layout(
        images=[
            dict(
                source="https://cryptologos.cc/logos/bitcoin-btc-logo.png",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                sizex=0.15,
                sizey=0.15,
                xanchor="center",
                yanchor="middle",
                layer="above"
            )
        ]
    )

    # Update layout
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor='rgba(0,0,0,0)',
        font={"color": "black", "family": "Arial"},
        width=600,
        height=400,
        margin=dict(t=120, b=40, l=40, r=40)
    )
    
    # Save to HTML file
    fig.write_html("index_index.html")
    print("✅ index_index.html has been created.")
    
    # Also display the chart if in an interactive environment
    fig.show()

# Push updates to GitHub
def push_to_github():
    repo_path = "/Users/tubaltheodros/Desktop/btc-fear-greed-index"  # Use explicit path
    
    try:
        # Change to the repository directory
        os.chdir(repo_path)
        
        # Pull the latest changes from the remote repository
        os.system("git pull origin main")
        
        # Add, commit, and push updates
        os.system("git add sentiment_scores.csv index_index.html line_index.html")
        today_date = datetime.now().strftime('%Y-%m-%d')
        os.system(f"git commit -m 'Updated sentiment data and visualizations for {today_date}'")
        os.system("git push origin main")
        
        print("✅ Successfully pushed updates to GitHub.")
    except Exception as e:
        print(f"❌ Failed to push updates to GitHub: {e}")

# Function to rebuild the dataset with complete history
def rebuild_complete_dataset():
    """
    Rebuild the dataset with:
    1. All alternative.me historical data
    2. Preserve existing custom algorithm data for dates that have it
    3. Ensure no gaps in the timeline
    """
    try:
        # Check if we have existing data
        try:
            existing_df = pd.read_csv("sentiment_scores.csv")
            has_existing_data = True
            
            # Backup the existing file just in case
            existing_df.to_csv("sentiment_scores_backup.csv", index=False)
            print("✅ Backed up existing data to sentiment_scores_backup.csv")
            
            # Filter to only keep custom algorithm data
            custom_data = existing_df[existing_df["Source"].str.contains("custom", na=False)]
            print(f"Found {len(custom_data)} records with custom algorithm data")
        except FileNotFoundError:
            has_existing_data = False
            custom_data = pd.DataFrame(columns=["Date", "Sentiment Score", "Source"])
            print("No existing data found, starting fresh")
        
        # Get all historical data from alternative.me
        alt_me_data = fetch_historical_sentiment()
        if alt_me_data is None or alt_me_data.empty:
            print("❌ Failed to fetch alternative.me data")
            return
            
        print(f"Fetched {len(alt_me_data)} records from alternative.me")
        
        # Convert dates to datetime for merging
        alt_me_data["Date"] = pd.to_datetime(alt_me_data["Date"])
        if not custom_data.empty:
            custom_data["Date"] = pd.to_datetime(custom_data["Date"])
        
        # For each date that exists in custom_data, remove it from alt_me_data
        if not custom_data.empty:
            custom_dates = custom_data["Date"].dt.strftime("%Y-%m-%d").tolist()
            alt_me_data = alt_me_data[~alt_me_data["Date"].dt.strftime("%Y-%m-%d").isin(custom_dates)]
            
        # Combine the datasets
        combined_df = pd.concat([alt_me_data, custom_data], ignore_index=True)
        
        # Sort by date
        combined_df = combined_df.sort_values("Date")
        
        # Save the combined dataset
        combined_df.to_csv("sentiment_scores.csv", index=False)
        print(f"✅ Rebuilt dataset with {len(combined_df)} total records")
        
        # Show summary of date range
        oldest_date = combined_df["Date"].min().strftime("%Y-%m-%d")
        newest_date = combined_df["Date"].max().strftime("%Y-%m-%d")
        print(f"Dataset now spans from {oldest_date} to {newest_date}")
        
    except Exception as e:
        print(f"❌ Error rebuilding dataset: {e}")

# Main script execution
if __name__ == "__main__":
    # Choose whether to use your algorithm or alternative.me data
    USE_CUSTOM_ALGORITHM = True  # Set to False to always use alternative.me
    
    # Rebuild the complete dataset with no gaps
    rebuild_complete_dataset()
    
    # Clean historical data
    clean_historical_data()
    
    # Determine today's date
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Get today's sentiment
    if USE_CUSTOM_ALGORITHM:
        print(f"Using custom algorithm for today's sentiment ({today})...")
        today_sentiment_score = calculate_combined_sentiment()
    else:
        print(f"Using alternative.me for today's sentiment ({today})...")
        # Get today's sentiment from alternative.me API
        alt_me_index = fetch_fear_greed_index()
        today_sentiment_score = alt_me_index
    
    # Save today's sentiment score
    save_sentiment_to_csv(today_sentiment_score)
    
    # Generate updated visualizations
    plot_interactive_sentiment()
    plot_fear_greed_gauge(today_sentiment_score)
    
    # Push updates to GitHub
    push_to_github()
from textblob import TextBlob
import pandas as pd
import requests
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import os
import plotly.express as px
import plotly.graph_objects as go
import time

