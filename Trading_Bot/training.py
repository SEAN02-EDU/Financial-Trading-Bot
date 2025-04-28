import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# TensorFlow and Keras imports for the LSTM forecasting model
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard  # type: ignore

# Scikit-learn for normalization
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# PART 1: Data Loading, Merging, Filtering
# --------------------------------------------------
file_path = r"C:\Users\User\Downloads\Trading_Bot\TSLA_Significant.csv"

start_date = pd.to_datetime("2021-10-01")
end_date   = pd.to_datetime("2022-09-29")

# Load Financial dataset.
df_ta = pd.read_csv(file_path)
if "Unnamed: 0" in df_ta.columns:
    df_ta.drop("Unnamed: 0", axis=1, inplace=True)
if "Date" not in df_ta.columns:
    raise ValueError("TA dataset must include a 'Date' column.")
df_ta["Date"] = pd.to_datetime(df_ta["Date"])
df_ta = df_ta.sort_values("Date").reset_index(drop=True)
print("TA dataset shape:", df_ta.shape)

df_data = df_ta.copy()

# Filter the data to the desired date range.
df_filtered = df_data[(df_data["Date"] >= start_date) & (df_data["Date"] <= end_date)].copy()
print("Filtered dataset shape:", df_filtered.shape)

# Save dates (for later use) and then drop the Date column.
dates = df_filtered["Date"].copy()
# Reset the date series index for consistency.
dates.reset_index(drop=True, inplace=True)
df_filtered = df_filtered.drop("Date", axis=1)

# Now, select only the desired features.
selected_features = ['Close', 'Daily_ROI', 'OBV','ATR','MACD','ADX','EPS','Gross_Margin','sentiment_score']
df_filtered = df_filtered[selected_features]
print("DataFrame columns after selecting features:", df_filtered.columns.tolist())

# Ensure that "Close" is the first column.
if "Close" not in df_filtered.columns:
    raise ValueError("The 'Close' column is missing!")
cols = list(df_filtered.columns)
cols.insert(0, cols.pop(cols.index("Close")))
df_filtered = df_filtered[cols]
print("Reordered columns:", df_filtered.columns.tolist())

# Remove constant features.
features = list(df_filtered.columns)
constant_features = [col for col in features if df_filtered[col].nunique() == 1]
print("Constant features detected and removed:", constant_features)
features = [col for col in features if col not in constant_features]
print("Dynamic features to be used:", features)

# --------------------------------------------------
# Save the filtered dataset as two CSV files for PPO training and testing.
# --------------------------------------------------
n_total = len(dates)
n_train = int(n_total * 0.75)
train_dates = dates.iloc[:n_train].reset_index(drop=True)
test_dates = dates.iloc[n_train:].reset_index(drop=True)

df_train = df_filtered.iloc[:n_train].reset_index(drop=True)
df_test = df_filtered.iloc[n_train:].reset_index(drop=True)

df_train.to_csv("train.csv", index=False)
df_test.to_csv("test.csv", index=False)
train_dates.to_frame(name="Date").to_csv("train_dates.csv", index=False)
test_dates.to_frame(name="Date").to_csv("test_dates.csv", index=False)
print("Train and test CSV files created.")

# --------------------------------------------------
# (UNCHANGED) PART 2: Data Preprocessing and Sequence Generation
# --------------------------------------------------
data = df_filtered.values
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)


# Sliding window using previous 60 days to forecast 1 days
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x_seq = data[i:(i + seq_length)]
        y_seq = data[i + seq_length, 0]  # The target is the "Close" price
        xs.append(x_seq)
        ys.append(y_seq)
    return np.array(xs), np.array(ys)

# For LSTM, use the full data for now.
X, y = create_sequences(scaled_data, seq_length=60)
print("LSTM Input shape:", X.shape, "Target shape:", y.shape)

# --------------------------------------------------
# (UNCHANGED) PART 3: Building the TensorFlow LSTM Model
# --------------------------------------------------
num_features = X.shape[2]

lstm_model = Sequential([
    LSTM(50, return_sequences=True, dropout=0.2, input_shape=(60, num_features)),
    LSTM(50, dropout=0.2),
    Dense(1)
])
lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
lstm_model.summary()

# --------------------------------------------------
# (UNCHANGED) PART 4: Training the LSTM Model
# --------------------------------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=5, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1),
    ModelCheckpoint(filepath='weights.weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True),
    TensorBoard(log_dir='logs')
]

split_ratio = 0.75
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

history = lstm_model.fit(X_train, y_train, 
                         epochs=50, 
                         batch_size=32,
                         validation_data=(X_test, y_test),
                         callbacks=callbacks,
                         verbose=1)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('LSTM Model Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------
# (UNCHANGED) PART 5: Evaluating the LSTM Model and Plotting Predictions
# --------------------------------------------------
train_preds = lstm_model.predict(X_train)
test_preds = lstm_model.predict(X_test)

def inverse_transform(preds, actual, num_features):
    dummy_preds = np.zeros((preds.shape[0], num_features))
    dummy_actual = np.zeros((actual.shape[0], num_features))
    dummy_preds[:, 0] = preds.flatten()
    dummy_actual[:, 0] = actual.flatten()
    inv_preds = scaler.inverse_transform(dummy_preds)[:, 0]
    inv_actual = scaler.inverse_transform(dummy_actual)[:, 0]
    return inv_preds, inv_actual

train_pred_prices, train_actual_prices = inverse_transform(train_preds, y_train, num_features)
test_pred_prices, test_actual_prices   = inverse_transform(test_preds, y_test, num_features)

train_indices = np.arange(len(train_actual_prices))
test_indices = np.arange(len(train_actual_prices), len(train_actual_prices) + len(test_actual_prices))

plt.figure(figsize=(12, 6))
plt.plot(train_indices, train_actual_prices, label='Train Actual', color='blue')
plt.plot(train_indices, train_pred_prices, label='Train Prediction', color='cyan', linestyle='--')
plt.plot(test_indices, test_actual_prices, label='Test Actual', color='green')
plt.plot(test_indices, test_pred_prices, label='Test Prediction', color='orange', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Stock Close Price')
plt.title('Train/Test Actual vs. Prediction')
plt.legend()
plt.show()

# --------------------------------------------------
# (UNCHANGED) PART 6: Forecasting Future Prices and Plotting Forecast
# --------------------------------------------------

forecast_horizon = 7

last_seq = scaled_data[-60:]  # last available sequence from normalized data
print(last_seq)
forecast = []
current_seq = last_seq.copy()
for i in range(forecast_horizon):  # forecast horizon of 7 days
    seq_input = np.expand_dims(current_seq, axis=0)
    pred = lstm_model.predict(seq_input)[0, 0]
    forecast.append(pred)
    new_row = current_seq[-1].copy()
    new_row[0] = pred
    current_seq = np.vstack((current_seq[1:], new_row))

forecast = np.array(forecast).reshape(-1, 1)
dummy_forecast = np.zeros((forecast.shape[0], num_features))
dummy_forecast[:, 0] = forecast[:, 0]
forecasted_prices = scaler.inverse_transform(dummy_forecast)[:, 0]

last_date = dates.iloc[-1]
datelist_future = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq='D')
PREDICTIONS_FUTURE = pd.DataFrame(forecasted_prices, columns=['Close'], index=datelist_future)
print("Future Predictions DataFrame:")
print(PREDICTIONS_FUTURE)

plt.figure(figsize=(10, 6))
plt.plot(PREDICTIONS_FUTURE.index, forecasted_prices, marker='o')
plt.xlabel('Date')
plt.ylabel('Forecasted Stock Close Price')
plt.title('Forecasted Future Stock Prices')
plt.grid(True)
plt.show()

# NEW STEP: Save the forecasted "Close" prices DataFrame to a CSV file.
PREDICTIONS_FUTURE.to_csv("forecasted_prices.csv")
print("Forecasted prices saved to 'forecasted_prices.csv'.")

# --------------------------------------------------
# (UNCHANGED) PART 7: SHAP Explanation for the LSTM Model
# --------------------------------------------------

import shap

def keras_model_wrapper(flat_input):
    reshaped = flat_input.reshape(flat_input.shape[0], 60, num_features)
    preds = lstm_model.predict(reshaped)
    return preds.flatten()

background_flat = X_train[:100].reshape(100, -1)
explainer_kernel = shap.KernelExplainer(keras_model_wrapper, background_flat)
X_test_flat = X_test[:10].reshape(10, -1)
shap_values_kernel = explainer_kernel.shap_values(X_test_flat, nsamples=100)

flat_feature_names = []
for t in range(60):
    for fname in features:
        flat_feature_names.append(f"t-{60-t}_{fname}")

shap.summary_plot(shap_values_kernel, X_test_flat, feature_names=flat_feature_names)

#########################################
# Part 8: PPO Trading Agent Using TradingEnvWithForecast
#########################################
# We use Gymnasium and Stable-Baselines3.
# Make sure to install gymnasium, stable-baselines3, and tabulate:
#     pip install gymnasium stable-baselines3 tabulate

# Load the train CSV and associated dates.
df_train = pd.read_csv("train.csv")
train_dates = pd.read_csv("train_dates.csv")["Date"]
# Convert train_dates to datetime (if not already)
train_dates = pd.to_datetime(train_dates)

# For historical data, we compute a simple forecast: next day's actual Close.
forecasted_hist = df_train["Close"].shift(-1).ffill()

initial_balance = 1000
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# Define the custom trading environment.
class TradingEnvWithForecast(gym.Env):
    """
    A custom trading environment that uses:
         [balance, shares held, current Close, forecasted Close]
    Observation:
        - balance: Current cash available.
        - shares held: Number of shares currently held.
        - current Close: Current day's close price from df.
        - forecasted Close: For historical data, we use a precomputed forecast.
    Actions:
         0 = Sell (liquidate all shares),
         1 = Hold,
         2 = Buy (invest entire balance).
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, forecasted_prices, dates, seq_length=1, initial_balance=1000):
        super(TradingEnvWithForecast, self).__init__()
        self.df = df.reset_index(drop=True)  # Raw market data DataFrame (must contain "Close")
        # Fix the error: if dates is a DatetimeIndex, convert it to a Series.
        if isinstance(dates, pd.DatetimeIndex):
            self.dates = pd.Series(dates)
        else:
            self.dates = dates.reset_index(drop=True)
        self.forecasted_prices = forecasted_prices.reset_index(drop=True)  # Precomputed forecast (Series)
        self.seq_length = 0
        self.initial_balance = initial_balance
        self.current_step = 0  # Start when we have at least seq_length days.
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_steps = len(df)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        self.current_step = self.seq_length
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        return self._get_observation(), {}
    
    def _get_observation(self):
        current_price = self.df.iloc[self.current_step]["Close"]
        # For historical data, use precomputed forecast: we align forecasted price with the row index.
        forecasted_price = self.forecasted_prices.iloc[self.current_step - self.seq_length]
        return np.array([self.balance, self.shares_held, current_price, forecasted_price], dtype=np.float32)
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step]["Close"]
        if action == 0:  # Sell: liquidate all shares held.
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0
        elif action == 2:  # Buy: invest entire balance.
            shares_to_buy = int(self.balance / current_price)
            if shares_to_buy > 0:
                self.balance -= shares_to_buy * current_price
                self.shares_held += shares_to_buy
        # If Hold (action == 1), do nothing.
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        next_price = self.df.iloc[self.current_step]["Close"] if not done else self.df.iloc[-1]["Close"]
        self.net_worth = self.balance + self.shares_held * next_price
        reward = self.net_worth - self.initial_balance
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        info = {"net_worth": self.net_worth}
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        current_date = self.dates.iloc[self.current_step].strftime("%Y-%m-%d")
        print(f"Step: {self.current_step} | Balance: ${self.balance:,.2f} | Shares Held: {self.shares_held} | Net Worth: ${self.net_worth:,.2f}")

# For historical data, we compute a simple forecast.
# We'll use next day's actual Close as the forecast.
forecasted_hist = df_filtered["Close"].shift(-1).ffill()  # Use ffill() to fill missing values.

# Create the PPO training environment using the historical data.
print(df_filtered)
env = TradingEnvWithForecast(df_train, forecasted_hist, dates, seq_length=1, initial_balance=initial_balance)

# Train the PPO agent on historical data.
ppo_model = PPO("MlpPolicy", env, verbose=1)
ppo_model.learn(total_timesteps=20000)
ppo_model.save("ppo_trading_model")

print("PPO model saved as 'ppo_trading_model.zip'.")

# Mapping for action names.
action_names = {0: "Sell", 1: "Hold", 2: "Buy"}

# -------------------------------
# Evaluate PPO Agent on Historical Data
# -------------------------------
print("\n=== Evaluating PPO Trading Agent on Historical Data ===")
obs, _ = env.reset()
step_counter = 0
print("-" * 100)
print(f"{'Date':<12} | {'Action':<6} | {'Balance':>15} | {'Shares Held':>12} | {'Net Worth':>15} | {'Close':>8}")
print("-" * 100)

while True:
    action, _ = ppo_model.predict(obs)
    action_val = int(action.item() if hasattr(action, 'item') else action)
    current_date = env.dates.iloc[env.current_step].strftime("%Y-%m-%d")
    current_close = env.df.iloc[env.current_step]["Close"]
    print(f"{current_date:<12} | {action_names[action_val]:<6} | Balance: ${env.balance:15,.2f} | Shares Held: {env.shares_held:12d} | Net Worth: ${env.net_worth:15,.2f} | Close: {current_close:8.2f}")
    obs, reward, terminated, truncated, info = env.step(action_val)
    step_counter += 1
    if terminated:
        break

print("-" * 100)
print(f"Final Net Worth (Historical): ${env.net_worth:,.2f}")

# -------------------------------
# Simulate PPO on Forecasted Future Data (7 Days)
# -------------------------------
print("\n=== Simulating PPO Agent on Forecasted Future Data (Next 7 Days) ===")
# Use the precomputed forecasted future prices from Part 6.
PREDICTIONS_FUTURE = pd.read_csv("forecasted_prices.csv", index_col=0, parse_dates=True)
forecast_prices = PREDICTIONS_FUTURE['Close']
# For the forecast environment, we use the last row of df_filtered as a baseline for other features,
# and then replace the "Close" price with each forecasted value.
last_row = df_filtered.iloc[-1].copy()
forecast_rows = []
for fc in forecast_prices:
    new_row = last_row.copy()
    new_row["Close"] = fc
    forecast_rows.append(new_row)
forecast_df = pd.DataFrame(forecast_rows, columns=df_filtered.columns)

# Create forecast environment using forecast_df and the forecast dates from PREDICTIONS_FUTURE.
forecast_env = TradingEnvWithForecast(forecast_df, forecast_df["Close"], PREDICTIONS_FUTURE.index, seq_length=1, initial_balance=env.balance)
obs, _ = forecast_env.reset()
step_counter = 0

print("-" * 100)
print(f"{'Date':<12} | {'Action':<6} | {'Close':>8}")
print("-" * 100)

while True:
    action, _ = ppo_model.predict(obs)
    action_val = int(action.item() if hasattr(action, 'item') else action)
    current_date = forecast_env.dates.iloc[forecast_env.current_step].strftime("%Y-%m-%d")
    current_close = forecast_env.df.iloc[forecast_env.current_step]["Close"]
    print(f"{current_date:<12} | {action_names[action_val]:<6} | Close: {current_close:8.2f}")
    obs, reward, terminated, truncated, info = forecast_env.step(action_val)
    step_counter += 1
    if terminated:
        break

print("-" * 100)