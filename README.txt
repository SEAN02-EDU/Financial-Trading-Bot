README.txt

Project Overview

This repository provides two Python scripts for stock-price forecasting and reinforcement-learning-based trading simulation:

training.py
Preprocesses and splits financial time-series data
Builds, trains, and evaluates an LSTM forecasting model
Generates future price forecasts
Computes SHAP explanations for the LSTM model
Defines a custom Gym trading environment
Trains and evaluates a PPO trading agent 

test_forecast.py
Loads saved PPO model and test data
Runs an out-of-sample historical trading simulation
Prints a tabulated report of Date, Action, Balance, Shares Held, Net Worth, and Close 

Prerequisites

Python: 3.7 or higher

Libraries:
Data handling: pandas, numpy
Plotting: matplotlib
Machine learning: tensorflow (Keras), scikit-learn
Reinforcement learning: stable-baselines3, gymnasium
Utilities: tabulate, shap 

Installation

pip install pandas numpy matplotlib tensorflow scikit-learn stable-baselines3 gymnasium tabulate shap

File Descriptions

training.py
Imports: data libs, TensorFlow/Keras, sklearn scaler, SHAP, Gymnasium, Stable-Baselines3 
Main Parts:
Data Loading & Filtering: specify file_path, start_date, end_date.
Feature Selection: selects technical indicators + sentiment, removes constant features.
Train/Test Split: 75% training, 25% testing → saves train.csv, test.csv, *_dates.csv.
Sequence Generation: sliding window seq_length=60.
LSTM Model: two LSTM layers (50 units, dropout 0.2), Dense output, Adam lr=0.001.
Training: epochs=50, batch_size=32, callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard.
Evaluation & Plots: training vs. validation loss, actual vs. predicted.
Forecasting Horizon: 7-day ahead forecast, saves forecasted_prices.csv.
SHAP Analysis: KernelExplainer on flattened sequences.
PPO Agent: defines TradingEnvWithForecast, trains for 20000 timesteps, saves ppo_trading_model.zip.
Historical & Future Simulation: prints action logs on both past and forecasted data 

test_forecast.py
Imports: data libs, Matplotlib, Tabulate, Stable-Baselines3, custom TradingEnvWithForecast 
Workflow:
Load test.csv and test_dates.csv.
Compute one-step-ahead forecast by shifting Close.
Instantiate TradingEnvWithForecast(seq_length=1, initial_balance=1000).
Load PPO model via PPO.load("ppo_trading_model", env=env_test).
Loop through environment steps, record and tabulate results.
Prints a table of Date, Action (Sell/Hold/Buy), Balance, Shares Held, Net Worth, and Close.

Usage

Prepare data
Place your TSLA (or other ticker) CSV in the path specified by file_path in training.py.

Train models
python training.py
Outputs: train.csv, test.csv, train_dates.csv, test_dates.csv, forecasted_prices.csv, weights.weights.h5, ppo_trading_model.zip, and plots.

Simulate out-of-sample trading
python test_forecast.py
The script will load the PPO model, step through your test data, and print a table of Date, Action, Balance, Shares Held, Net Worth, and Close price.

Configuration Parameters

training.py
file_path: Input CSV file (default TSLA_Significant.csv)
start_date, end_date: Date range filters
selected_features: List of features used for modeling
seq_length: LSTM window size (default 60)
forecast_horizon: Days ahead to forecast (default 7)
batch_size, epochs, learning_rate, patience, factor: LSTM training hyperparameters
initial_balance: Starting cash for PPO env (default 1000)
total_timesteps: PPO training timesteps (default 20000)

test_forecast.py
seq_length: Sequence length for env (default 1)
initial_balance: Starting cash (default 1000)
Model path: ppo_trading_model.zip

Notes

Ensure correct working directory so that CSV and model files are found.
Customize hyperparameters and file paths as needed.
Review plots in Matplotlib windows or saved figures.

License & Contact

For questions or contributions, please contact the author or open an issue in the repository.

