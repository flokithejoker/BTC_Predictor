import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime

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
features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA30', 'MA90', 'STD7', 'STD30', 'STD90', 'Lag1', 'Lag7', 'Lag30']]
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

seq_length = 30  # Use 100 days of data to predict the future
X, y = create_sequences(features_tensor, targets_tensor, seq_length)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the LSTM network with more layers and dropout
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=2, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

input_size = X_train.shape[2]
hidden_layer_size = 100  # Increase the number of units
output_size = 3  # Predicting 3 targets: next month, next 3 months, next 6 months

model = LSTMModel(input_size, hidden_layer_size, output_size)

# Loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 300  # Increase the number of epochs
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_function(y_pred, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Evaluation
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)
    mse = mean_squared_error(y_test.numpy(), y_test_pred.numpy(), multioutput='raw_values')
    print(f'Mean Squared Error for predicting next month: {mse[0]}')
    print(f'Mean Squared Error for predicting next 3 months: {mse[1]}')
    print(f'Mean Squared Error for predicting next 6 months: {mse[2]}')

# Example prediction for the next month, 3 months, and 6 months
with torch.no_grad():
    latest_data = features_tensor[-seq_length:].unsqueeze(0)  # Take the latest sequence
    future_predictions_scaled = model(latest_data).numpy()
    future_predictions = target_scaler.inverse_transform(future_predictions_scaled)  # Inverse transform the predictions
    print(f'Predicted closing price in 1 month: {future_predictions[0][0]}')
    print(f'Predicted closing price in 3 months: {future_predictions[0][1]}')
    print(f'Predicted closing price in 6 months: {future_predictions[0][2]}')

# Mean Squared Error for predicting next month: 0.8592191338539124
# Mean Squared Error for predicting next 3 months: 3.7291364669799805
# Mean Squared Error for predicting next 6 months: 0.702614426612854
# Predicted closing price in 1 month: 46849.30859375
# Predicted closing price in 3 months: 61908.80859375
# Predicted closing price in 6 months: 36400.83984375