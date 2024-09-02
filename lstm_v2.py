import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, TensorDataset
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Load the new data
file_path = '/Users/peter/VS_Studio/Test/bitcoin_price_data.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Preprocessing
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Feature engineering
data['MA7'] = data['Close'].rolling(window=7).mean()
data['MA30'] = data['Close'].rolling(window=30).mean()
data['MA90'] = data['Close'].rolling(window=90).mean()
data['STD7'] = data['Close'].rolling(window=7).std()
data['STD30'] = data['Close'].rolling(window=30).std()
data['STD90'] = data['Close'].rolling(window=90).std()
data['Lag1'] = data['Close'].shift(1)
data['Lag7'] = data['Close'].shift(7)
data['Lag30'] = data['Close'].shift(30)
data['RSI'] = RSIIndicator(data['Close']).rsi()
data['MACD'] = MACD(data['Close']).macd()

# Define the prediction targets
data['NextMonth_Close'] = data['Close'].shift(-30)
data['Next3Months_Close'] = data['Close'].shift(-90)
data['Next6Months_Close'] = data['Close'].shift(-180)

# Drop rows with NaN values created by rolling calculations and target shifts
data = data.dropna()

# Verify that we have data
if data.empty:
    raise ValueError("The DataFrame is empty after dropping NaN values. Please check the data processing steps.")

# Features and labels
features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA30', 'MA90', 'STD7', 'STD30', 'STD90', 'Lag1', 'Lag7', 'Lag30', 'RSI', 'MACD']]
targets = data[['NextMonth_Close', 'Next3Months_Close', 'Next6Months_Close']]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Normalize targets
target_scaler = StandardScaler()
targets_scaled = target_scaler.fit_transform(targets)

# Convert to PyTorch tensors
features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
targets_tensor = torch.tensor(targets_scaled, dtype=torch.float32)

# Create sequences for LSTM
def create_sequences(features, targets, seq_length):
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        x = features[i:i+seq_length]
        y = targets[i+seq_length]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)

seq_length = 30  # Use 30 days of data to predict the future
X, y = create_sequences(features_tensor, targets_tensor, seq_length)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM network with more layers and dropout
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# Hyperparameter tuning
def train_model(X_train, y_train, input_size, output_size, params):
    hidden_layer_size = params['hidden_layer_size']
    num_layers = params['num_layers']
    dropout = params['dropout']
    learning_rate = params['learning_rate']
    
    model = LSTMModel(input_size, hidden_layer_size, output_size, num_layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_function(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
        
    return model

# Define the parameter grid for GridSearchCV
param_grid = {
    'hidden_layer_size': [50, 100, 150],
    'num_layers': [1, 2, 3],
    'dropout': [0.2, 0.5],
    'learning_rate': [0.001, 0.01]
}

best_mse = float('inf')
best_params = None
best_model = None

for hidden_layer_size in param_grid['hidden_layer_size']:
    for num_layers in param_grid['num_layers']:
        for dropout in param_grid['dropout']:
            for learning_rate in param_grid['learning_rate']:
                params = {
                    'hidden_layer_size': hidden_layer_size,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'learning_rate': learning_rate
                }
                model = train_model(X_train, y_train, X_train.shape[2], y_train.shape[1], params)
                model.eval()
                with torch.no_grad():
                    y_test_pred = model(X_test)
                    mse = mean_squared_error(y_test.numpy(), y_test_pred.numpy(), multioutput='raw_values').mean()
                if mse < best_mse:
                    best_mse = mse
                    best_params = params
                    best_model = model

print(f'Best Parameters: {best_params}')
print(f'Best MSE: {best_mse}')

# Example prediction for the next month, 3 months, and 6 months
best_model.eval()
with torch.no_grad():
    latest_data = features_tensor[-seq_length:].unsqueeze(0)  # Take the latest sequence
    future_predictions_scaled = best_model(latest_data).numpy()
    future_predictions = target_scaler.inverse_transform(future_predictions_scaled)  # Inverse transform the predictions
    print(f'Predicted closing price in 1 month: {future_predictions[0][0]}')
    print(f'Predicted closing price in 3 months: {future_predictions[0][1]}')
    print(f'Predicted closing price in 6 months: {future_predictions[0][2]}')


# Best Hyperparameters
# Hidden Layer Size: 100
# Number of Layers: 2
# Dropout: 0.2
# Learning Rate: 0.01
# Best Mean Squared Error (MSE)
# Best MSE: 0.009848046116530895
# Predicted Closing Prices
# Predicted closing price in 1 month: $44,256.13
# Predicted closing price in 3 months: $52,025.77
# Predicted closing price in 6 months: $65,364.21
