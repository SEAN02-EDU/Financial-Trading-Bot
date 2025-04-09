import os
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import gym
from gym import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import shap

#########################################
# Part 1: Data Download, Feature Creation,
# LSTM Forecasting Model (TA & FA inputs)
#########################################

# =======================
# 1. Download and Prepare TSLA Data with TA & FA Features
# =======================
ticker_symbol = "TSLA"
ticker = yf.Ticker(ticker_symbol)
# Download historical data
data = ticker.history(start="2019-01-01", end="2023-01-01")
print("Raw data shape:", data.shape)

# Use only the 'Close' price initially
df = data[['Close']].copy()

# --- Compute Technical Analysis (TA) Features ---
# Moving Averages
df["MA20"] = df["Close"].rolling(window=20).mean()
df["MA50"] = df["Close"].rolling(window=50).mean()

# Relative Strength Index (RSI)
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["RSI"] = compute_RSI(df["Close"], period=14)

# --- Add Fundamental Analysis (FA) Features ---
fa_info = ticker.info
# In case FA values are missing, use 0
df["PE_Ratio"] = fa_info.get("trailingPE", 0)
df["Market_Cap"] = fa_info.get("marketCap", 0)
df["Dividend_Yield"] = fa_info.get("dividendYield", 0)

# Clean the DataFrame: drop rows with NaNs only from TA features (rolling windows)
df.dropna(subset=["MA20", "MA50", "RSI"], inplace=True)
df.reset_index(inplace=True)

print("DataFrame with TA & FA Features:")
print(df.head())
print("DataFrame shape after dropna:", df.shape)

# ===============================
# 2. Prepare Data for the LSTM Forecast Model
# ===============================
# Define the feature set (input for the LSTM)
features = ["Close", "MA20", "MA50", "RSI", "PE_Ratio", "Market_Cap", "Dividend_Yield"]

# Scale the features using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features].values)
df_scaled = pd.DataFrame(scaled_data, columns=features)

print("\nScaled DataFrame head:")
print(df_scaled.head())

def create_feature_sequences(data, seq_length, feature_cols):
    X, y = [], []
    arr = data[feature_cols].values  # Convert DataFrame to NumPy array
    for i in range(len(arr) - seq_length):
        X.append(arr[i:i+seq_length])       # Sequence of length seq_length
        y.append(arr[i+seq_length][0])        # Target: next day's 'Close' price
    return np.array(X), np.array(y)

seq_length = 10
X_seq, y_seq = create_feature_sequences(df_scaled, seq_length, features)
print("\nShape of LSTM feature sequences:", X_seq.shape)
print("Sample input sequence (first sample):")
print(pd.DataFrame(X_seq[0], columns=features))

# Convert sequences to PyTorch tensors
X_tensor = torch.FloatTensor(X_seq)         # Shape: (samples, seq_length, number_of_features)
y_tensor = torch.FloatTensor(y_seq).unsqueeze(-1)  # Shape: (samples, 1)

# ================================================
# 3. Define the PyTorch LSTM Forecasting Model
# ================================================
class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMForecast, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer (batch_first=True so input shape is (batch, seq, feature))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer to output the forecast
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # Use output from the last time step
        out = self.fc(out[:, -1, :])
        return out

input_size = len(features)   # Number of input features (7)
hidden_size = 50
num_layers = 2
output_size = 1              # Forecasting 'Close' price
lstm_model = LSTMForecast(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# ==========================
# 4. Train the LSTM Model
# ==========================
num_epochs = 10  # Increase epochs for better performance in practice
lstm_model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = lstm_model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
lstm_model.eval()

# Forecast function: Given a scaled sequence (shape: (seq_length, features)),
# returns the forecasted Close price in the original scale.
def lstm_forecast(sequence):
    with torch.no_grad():
        seq_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Shape: (1, seq_length, features)
        forecast_scaled = lstm_model(seq_tensor)                # Shape: (1, 1)
    # Create a dummy array where the first value is the forecast and the rest are zeros
    forecast_array = forecast_scaled.detach().numpy()
    dummy = np.zeros((1, len(features)))
    dummy[0, 0] = forecast_array[0, 0]
    inv = scaler.inverse_transform(dummy)
    return inv[0, 0]

# ============================================================== 
# 4.a Evaluate and Plot LSTM Forecast vs. Actual Prices
# ============================================================== 
test_range = 100
test_predictions = []
actual_prices = []
test_start = len(df_scaled) - test_range - seq_length
for i in range(test_start, len(df_scaled) - seq_length):
    sequence = df_scaled.iloc[i:i+seq_length].values
    pred_price = lstm_forecast(sequence)
    test_predictions.append(pred_price)
    actual_price = df.iloc[i+seq_length]["Close"]
    actual_prices.append(actual_price)

plt.figure(figsize=(10, 5))
plt.plot(actual_prices, label='Actual Close Price')
plt.plot(test_predictions, label='LSTM Forecast')
plt.xlabel('Time Step')
plt.ylabel('Stock Price')
plt.title('LSTM Forecast vs. Actual TSLA Stock Prices')
plt.legend()
plt.show()

############################################
# Part 2: Define Trading Environment & PPO
############################################

# ===============================================================
# 5. Define a Custom Gym Environment Using the LSTM Forecast
# ===============================================================
class TradingEnvWithForecast(gym.Env):
    """
    A custom trading environment that uses the current price and an LSTM forecast
    based on TA/FA features.
    Observation: [balance, shares held, current price, forecasted price]
    Actions: 0 = Sell, 1 = Hold, 2 = Buy
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, df_scaled, seq_length=10, initial_balance=10000):
        super(TradingEnvWithForecast, self).__init__()
        self.df = df                  # Raw DataFrame (for 'Close' price)
        self.df_scaled = df_scaled    # Scaled DataFrame with TA/FA features
        self.seq_length = seq_length
        self.initial_balance = initial_balance
        self.current_step = seq_length  # Start after we have enough past data
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_steps = len(df) - 1
        
        # Define action space: 0 = Sell, 1 = Hold, 2 = Buy.
        self.action_space = spaces.Discrete(3)
        # Define observation space: [balance, shares held, current price, forecasted price]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
    
    def reset(self):
        self.current_step = self.seq_length
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        return self._get_observation()
    
    def _get_observation(self):
        current_price = self.df.iloc[self.current_step]["Close"]
        recent_seq = self.df_scaled.iloc[self.current_step - self.seq_length:self.current_step].values
        forecasted_price = lstm_forecast(recent_seq)
        obs = np.array([self.balance, self.shares_held, current_price, forecasted_price], dtype=np.float32)
        return obs
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step]["Close"]
        
        # Execute action: Sell (0), Hold (1), Buy (2)
        if action == 0:  # Sell if holding shares
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0
        elif action == 2:  # Buy with all available balance
            shares_to_buy = int(self.balance / current_price)
            if shares_to_buy > 0:
                self.shares_held += shares_to_buy
                self.balance -= shares_to_buy * current_price
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        next_price = self.df.iloc[self.current_step]["Close"] if not done else self.df.iloc[-1]["Close"]
        self.net_worth = self.balance + self.shares_held * next_price
        reward = self.net_worth - self.initial_balance
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        info = {"net_worth": self.net_worth}
        return obs, reward, done, info
    
    def render(self, mode='human'):
        print(f"Step: {self.current_step} | Balance: {self.balance:.2f} | Shares: {self.shares_held} | Net Worth: {self.net_worth:.2f}")

# ========================================================
# 6. Train PPO Trading Agent Using the Custom Environment
# ========================================================
env = TradingEnvWithForecast(df, df_scaled, seq_length=seq_length, initial_balance=10000)
ppo_model = PPO("MlpPolicy", env, verbose=1)
ppo_model.learn(total_timesteps=10000)

# ========================================
# 7. Evaluate the Trained PPO Trading Agent
# ========================================
obs = env.reset()
net_worths = [env.net_worth]
done = False

while not done:
    action, _ = ppo_model.predict(obs)
    obs, reward, done, info = env.step(action)
    net_worths.append(env.net_worth)
    env.render()

print("Final Net Worth:", env.net_worth)

plt.figure(figsize=(10, 5))
plt.plot(net_worths, label="Net Worth")
plt.xlabel("Time Step")
plt.ylabel("Net Worth")
plt.title("Portfolio Value Over Time (PPO Trading Agent with TA/FA)")
plt.legend()
plt.show()

############################################
# Part 3: Use SHAP to Explain PPO’s Decisions
############################################

# Here we define a policy_predict function that feeds raw observations through
# the full PPO policy forward pass. This way the observations are processed as during training.
def policy_predict(observations):
    """
    Given a batch of observations (numpy array of shape (n_samples, 4)),
    returns the action probabilities as a numpy array of shape (n_samples, 3).
    """
    obs_tensor = torch.FloatTensor(observations).to(ppo_model.policy.device)
    with torch.no_grad():
        # Use the policy forward method to process the observation
        # The forward method returns a tuple (value, distribution, latent_pi)
        _, dist, _ = ppo_model.policy.forward(obs_tensor)
        probs = dist.probs
    return probs.cpu().numpy()

# Create a background dataset for SHAP (e.g., 10 sample observations)
background_obs = np.array([env.reset() for _ in range(10)])

# Initialize SHAP KernelExplainer using our policy_predict function
explainer = shap.KernelExplainer(policy_predict, background_obs)

# Choose a test observation (e.g., from env.reset()) to explain
test_obs = env.reset().reshape(1, -1)  # shape: (1, 4)
shap_values = explainer.shap_values(test_obs)

# shap_values is a list with one array per action (we have 3 actions: Sell, Hold, Buy).
# Here, we choose to explain the 'Buy' action (assumed index 2).
shap.initjs()  # Initialize javascript visualization for notebooks (if running in one)
force_plot = shap.force_plot(explainer.expected_value[2], shap_values[2][0],
                             feature_names=["Balance", "Shares Held", "Current Price", "Forecasted Price"])

# Save the force plot as an HTML file for viewing in a browser.
shap.save_html("ppo_shap_explanation.html", force_plot)
print("SHAP explanation has been saved as 'ppo_shap_explanation.html'. Open this file in a browser to view the force plot.")
