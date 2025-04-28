import os
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from tabulate import tabulate

# Import the custom environment from training.py
from training import TradingEnvWithForecast

from stable_baselines3 import PPO

# -------------------------------
# Load Test Data and Dates from CSV files.
# -------------------------------
df_test = pd.read_csv("test.csv")
test_dates = pd.read_csv("test_dates.csv")["Date"]
test_dates = pd.to_datetime(test_dates)

# For historical test simulation, compute forecast as next day's Close.
forecasted_test = df_test["Close"].shift(-1).ffill()

# Create the test environment.
env_test = TradingEnvWithForecast(df_test, forecasted_test, test_dates, seq_length=1, initial_balance=1000)

# Load the trained PPO model.
ppo_model = PPO.load("ppo_trading_model",env = env_test)
action_map = {0: "Sell", 1: "Hold", 2: "Buy"}

# -------------------------------
# PART 1: Out-of-Sample Historical Trading Simulation
# -------------------------------
print("\n=== Out-of-Sample Historical Trading Simulation ===")
obs, _ = env_test.reset()
hist_table = []

while True:
    action, _ = ppo_model.predict(obs)
    action_val = int(action.item() if hasattr(action, 'item') else action)
    current_date = env_test.dates.iloc[env_test.current_step].strftime("%Y-%m-%d")
    current_close = env_test.df.iloc[env_test.current_step]["Close"]
    hist_table.append([
        current_date,
        action_map[action_val],
        f"${env_test.balance:,.2f}",
        env_test.shares_held,
        f"${env_test.net_worth:,.2f}",
        current_close
    ])
    obs, reward, terminated, truncated, info = env_test.step(action_val)
    if terminated:
        break

print(tabulate(hist_table, headers=["Date", "Action", "Balance", "Shares Held", "Net Worth", "Close"], tablefmt="pretty"))

print("test_forecast.py completed.")
