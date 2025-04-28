# Financial Trading Bot Suite
A modular Python project for end-to-end stock analysis and trading simulation.  
It guides you through four stages:

1. **Sentiment Analysis** (SA)  
2. **Technical & Fundamental Analysis** (TA/FA)  
3. **LSTM + PPO Trading Bot**  
4. **Trend Follower Simulation**  

## Repository Structure
```
├── Sentiment Analysis/           # Twitter sentiment extraction with VADER
│   ├── stock_tweets.csv          # Sample input CSV of tweets
│   └── Sentiment Analyser.py     # Computes daily sentiment scores
│
├── Technical & Fundamental Analysis/  # Jupyter notebooks and scripts to extract TA & FA indicators
│   ├── Extract_FA_AMZN.ipynb
│   ├── Extract_FA_TSLA.ipynb
│   ├── Extract_FA_TSM.ipynb
│   ├── Technical Analysis.ipynb
│   ├── AMZN Feature Engineering.rmd
│   ├── TSLA Feature Engineering.rmd
│   └── TSM Feature Engineering.rmd
│
├── Trading Bot/                  # LSTM forecasting + PPO trading simulation
│   ├── training.py               # Trains LSTM + PPO, computes SHAP
│   └── test_forecast.py          # Runs out-of-sample simulation
│   └── AMZN_Significant.csv
│   └── TSLA_Significant.csv
│   └── TSM_Significant.csv
│
└── TrendFollower.py              # Standalone trend-following strategy script
```

## Requirements

- **Python** 3.7 or higher (3.11 recommended)  
- **Key libraries**:  
  `pandas`, `nltk`, `numpy`, `yfinance`, `talib`, `pdfplumber`,  
  `matplotlib`, `tensorflow`, `stable_baselines3`,  
  `prettytable`, `tabulate`  

## Overall Workflow

1. **Sentiment Analysis**  
   Run `Sentiment Analyser.py` to compute daily Twitter sentiment for your chosen ticker, producing `<TICKER>_twitter_sentiment.csv`.

2. **Technical & Fundamental Analysis**  
   Execute the notebooks in `Technical & Fundamental Analysis/` in order to extract technical indicators (e.g., ATR, MACD) and fundamental metrics (e.g., EPS, margins) for each company. The outputs are compiled manually into `<TICKER>_Significant.csv`.

3. **Trading Bot**  
   - **training.py**: loads `<TICKER>_Significant.csv`, builds and evaluates an LSTM forecasting model, generates a 7‑day forecast, computes SHAP explanations, defines a custom Gym environment, and trains a PPO agent.  
   - **test_forecast.py**: loads the trained PPO model and runs a backtest, printing a tabulated report of Date, Action, Balance, Holdings, Net Worth, and Close price.

4. **TrendFollower Simulation**  
   Run `TrendFollower.py` independently to simulate a simple trend‑following strategy and compare results.

## Quickstart

1. Clone the repo:
   ```bash
   git clone https://github.com/SEAN02-EDU/Financial-Trading-Bot
   cd financial-trading-bot
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies for all components:
   ```bash
   pip install pandas numpy matplotlib yfinance nltk talib tensorflow stable-baselines3 gymnasium prettytable shap scikit-learn
   ```
4. Run each component in sequence:
   - Sentiment Analysis:
     ```bash
     cd "Sentiment Analysis"
     python "Sentiment Analyser.py" -i stock_tweets.csv
     ```
   - Technical & Fundamental Analysis:
     ```bash
     cd "../Technical & Fundamental Analysis"
     # open each notebook in order and execute
     ```
   - Trading Bot:
     ```bash
     cd "../Trading Bot"
     python training.py
     python test_forecast.py
     ```
   - TrendFollower:
     ```bash
     cd ..
     python TrendFollower.py --start YYYY-MM-DD --end YYYY-MM-DD
     ```

## Configuration

- **Tickers & Date Ranges** can be modified inside each script or notebook (see header sections).
- **Output Paths**: by default, CSVs and models are saved in the same folder as their scripts; update path variables as needed.

## Acknowledgements
-  Twitter dataset by **Hanna Yukhymenko** (Kaggle): https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction

## Contributions

Feel free to submit issues or pull requests for enhancements. For any questions, contact the maintainer.

