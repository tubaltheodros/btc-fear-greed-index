# Bitcoin Fear & Greed Index

## **Overview**
This project implements a custom **Bitcoin Fear & Greed Index** that aggregates multiple data sources to quantify market sentiment. The index runs on an automated daily pipeline, combining real-time data analysis with historical patterns to provide a comprehensive view of market psychology.

## **Methodology**
This index captures market sentiment by analyzing and weighting four key data sources:

- **Market Volatility Analysis (40%)**: Quantifies Bitcoin's recent price movements against historical volatility patterns to identify periods of market uncertainty or confidence. The algorithm calculates high-low ranges and price momentum to detect abnormal market conditions.
- **Alternative.me Sentiment Index (30%)**: Incorporates established sentiment metrics that analyze discussions across multiple social platforms and forums to detect shifting market psychology.
- **Reddit Sentiment Analysis (20%)**: Uses natural language processing with **TextBlob** to evaluate the tone, frequency, and engagement of Bitcoin-related discussions on the platform where cryptocurrency enthusiasts gather.
- **Google Trends Analysis (10%)**: Tracks search term popularity to identify periods of increasing public interest or concern around Bitcoin.

## **Technical Implementation**

### **Data Pipeline Architecture**
- **Data Collection**: Automated API integrations with **CoinGecko, Reddit, and Alternative.me**
- **NLP Processing**: Sentiment analysis of Reddit post content using **TextBlob**
- **Time Series Analysis**: Detection of volatility anomalies against 24-hour price data
- **Data Storage**: Persistent **CSV storage** with automatic backfilling for missed dates
- **Visualization**: Interactive **Plotly dashboards** with time-period selection

### **Automation Framework**
The system is fully automated with:
- **Midnight UTC cron job scheduling** for daily updates
- **System reboot detection** for maintaining continuity
- **GitHub integration** for version control and web hosting
- **Robust error handling and logging**

## **Results & Insights**
The **Fear & Greed Index** reveals several interesting patterns:
- Market sentiment tends to **lag price movements by 2-3 days**
- **Extreme fear (<20)** has historically been a strong **contrarian buying indicator**
- **Sentiment cycles** average **6-8 weeks** from peak to trough
- Correlation between **sentiment and volatility increases** during market downturns

## **Technologies Used**
- **Python**: Core implementation language
- **Pandas & NumPy**: Data processing and numerical analysis
- **PRAW**: Reddit API integration
- **TextBlob**: Natural language processing for sentiment analysis
- **Requests & BeautifulSoup**: Web scraping and API interactions
- **Plotly**: Interactive data visualization
- **Git**: Version Control and Deployment
