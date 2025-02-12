#!/usr/bin/env python
# coding: utf-8

# In[4]:


# REDDITT SENTIMENT


# In[11]:


import praw
from textblob import TextBlob
import pandas as pd
import requests

# Initialize Reddit API
reddit = praw.Reddit(
    client_id="GSl5ASGK24eNVwIKQAm8CQ",
    client_secret="PmRh_hoFbgCrGOaEomX54-_QkYMfyQ",
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

# Call the function to calculate the sentiment
(fetch_reddit_sentiment())



# In[100]:


# Volatility Index


# In[15]:


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

            # **Calculate High-Low Range Volatility**
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

            # **Updated Volatility Thresholds**
            high_volatility = 0.06  # **6% = high volatility**
            low_volatility = 0.01   # **1% = low volatility**

            # **Sentiment Calculation**
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

# Call and print the result
sentiment_score = fetch_volatility_sentiment()
print(f"Final Volatility Sentiment Score: {sentiment_score}")


# In[16]:



# In[17]:


from bs4 import BeautifulSoup
import requests

from bs4 import BeautifulSoup
import requests
import time

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


# In[18]:


import numpy as np

def calculate_combined_sentiment():
    # Fetch sentiment scores
    reddit_sentiment = fetch_reddit_sentiment()  # This now returns a numeric score
    volatility_sentiment = fetch_volatility_sentiment()
    alternative_me_sentiment = fetch_fear_greed_index()

    # Apply weights (35% Alternative.me, 15% Volatility, 15% Reddit)
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

# Call the function
calculate_combined_sentiment()


# In[19]:


import requests
import pandas as pd
from datetime import datetime

# 📌 Step 1: Fetch historical sentiment from Alternative.me
def fetch_historical_sentiment(cutoff_date):
    """
    Fetch historical data only up to the cutoff date.
    After cutoff_date, only custom calculations will be used.
    """
    url = "https://api.alternative.me/fng/?limit=365"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()["data"]
        sentiment_data = []
        for entry in data:
            date = datetime.utcfromtimestamp(int(entry["timestamp"])).strftime("%Y-%m-%d")
            # Only include data before the cutoff date
            if date < cutoff_date:
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
    
def initialize_historical_data(cutoff_date):
    """
    One-time function to initialize the CSV with historical data up to cutoff_date.
    Should only be run once when setting up the system.
    """
    df = fetch_historical_sentiment(cutoff_date)
    if df is not None:
        df.to_csv("sentiment_scores.csv", index=False)
        print(f"✅ Historical data initialized up to {cutoff_date}")
    else:
        print("❌ Failed to initialize historical data")

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


# In[21]:


import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# 📌 Step 1: Ensure Historical Data Has No Future Dates
def clean_historical_data():
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
        


# 📌 Step 3: Create the Interactive Line Chart

import plotly.express as px
from datetime import timedelta

def plot_interactive_sentiment():
    try:
        # Load historical data
        df = pd.read_csv("sentiment_scores.csv")
        df["Date"] = pd.to_datetime(df["Date"])  # Convert Date column to datetime
        df = df.sort_values(by="Date")  # Sort data in chronological order

        # 📈 Create the line chart
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
        fig.write_html("/Users/tubaltheodros/Desktop/btc-fear-greed-index/line_index.html")
        print("✅ line_index.html has been updated.")
        fig.show()

    except Exception as e:
        print(f"❌ Error plotting sentiment: {e}")
        print(f"Detailed error: {str(e)}")




# 📌 Step 4: Main Script
if __name__ == "__main__":
    # Clean historical data
    clean_historical_data()
    
    # Calculate today's sentiment score using your formula
    today_sentiment_score = calculate_combined_sentiment()

    # Save today's sentiment score (if it's a new calendar day)
    save_sentiment_to_csv(today_sentiment_score)

    # Plot the interactive sentiment chart
    plot_interactive_sentiment()


# 📌 Step 4: Main Script
# 📌 Step 4: Main Script
if __name__ == "__main__":
    # Set your cutoff date here
    CUTOFF_DATE = "2024-02-12"  # Or whatever date you want to start your custom calculations
    
    # Check if we need to initialize historical data
    try:
        pd.read_csv("sentiment_scores.csv")
    except FileNotFoundError:
        print("Initializing historical data...")
        initialize_historical_data(CUTOFF_DATE)
    
    # Calculate and save today's sentiment score
    today_sentiment_score = calculate_combined_sentiment()
    save_sentiment_to_csv(today_sentiment_score)
    
    # Plot the interactive sentiment chart
    plot_interactive_sentiment()


# In[130]:


import plotly.graph_objects as go

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
    fig.write_html("/Users/tubaltheodros/Downloads/index_index.html")
    print("line_index.html has been created.")
    fig.show()

# Use the calculated sentiment score
combined_sentiment_score = calculate_combined_sentiment()
plot_fear_greed_gauge(combined_sentiment_score)


# In[ ]:

import os
from datetime import datetime
import pandas as pd
import plotly.express as px

# Your existing functions (clean_historical_data, calculate_combined_sentiment, save_sentiment_to_csv, etc.)

if __name__ == "__main__":
    # Step 1: Clean historical data
    clean_historical_data()
    
    # Step 2: Calculate today's sentiment score
    today_sentiment_score = calculate_combined_sentiment()
    
    # Step 3: Save today's sentiment score to CSV
    save_sentiment_to_csv(today_sentiment_score)
    
    # Step 4: Generate updated visualizations
    plot_interactive_sentiment()  # Saves the updated plot to a file, like index.html

    # Step 5: Push updated files to GitHub
    
    
    repo_path = "/Users/tubaltheodros/Desktop/btc-fear-greed-index"  # Your GitHub repo path

try:
    # Step: Pull the latest changes from the remote repository
    os.system(f"git -C {repo_path} pull origin main")

    # Then proceed with adding, committing, and pushing your updates:
    os.system(f"git -C {repo_path} add sentiment_scores.csv index_index.html line_index.html")
    os.system(f"git -C {repo_path} commit -m 'Updated sentiment data and visualizations'")
    os.system(f"git -C {repo_path} push origin main")

    print("✅ Successfully pushed updates to GitHub.")

except Exception as e:
    # Handle and display the error if something goes wrong
    print(f"❌ Failed to push updates to GitHub: {e}")













