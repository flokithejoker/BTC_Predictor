import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

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

# Adding Bollinger Bands features
bollinger = BollingerBands(data['Close'])
data['Bollinger_High'] = bollinger.bollinger_hband()
data['Bollinger_Low'] = bollinger.bollinger_lband()

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
features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA30', 'MA90', 'STD7', 'STD30', 'STD90', 'Lag1', 'Lag7', 'Lag30', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low']]
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

# Create sequences for the Transformer model
def create_sequences(features, targets, seq_length):
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        x = features[i:i+seq_length]
        y = targets[i+seq_length]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)

# Use multiple lookback periods
seq_lengths = [30, 60, 90]  # Different lookback periods
X_list, y_list = [], []
for seq_length in seq_lengths:
    X, y = create_sequences(features_tensor, targets_tensor, seq_length)
    X_list.append(X)
    y_list.append(y)

# Truncate sequences to the shortest length
min_length = min(len(X) for X in X_list)
X_list = [X[:min_length] for X in X_list]
y_list = [y[:min_length] for y in y_list]

# Ensure all sequences have the same length by padding shorter sequences with zeros
max_length = max(seq_lengths)
X_padded_list = [torch.cat([X, torch.zeros((X.shape[0], max_length - X.shape[1], X.shape[2]))], dim=1) if X.shape[1] < max_length else X for X in X_list]

# Combine sequences from different lookback periods
X_combined = torch.cat(X_padded_list, dim=2)
y_combined = y_list[0]  # Targets are the same for all sequences

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, embedding_size, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, output_size, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, embedding_size)
        self.transformer = nn.Transformer(d_model=embedding_size, nhead=num_heads, num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(embedding_size, output_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        transformer_out = self.transformer(src, tgt)
        output = self.fc_out(transformer_out[:, -1, :])
        return output

# Hyperparameters
input_size = X_train.shape[2]
embedding_size = 128  # Embedding size must be divisible by num_heads
num_heads = 4
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 128
dropout = 0.2
output_size = y_train.shape[1]
learning_rate = 0.001

# Initialize the model, loss function, and optimizer
model = TransformerModel(input_size, embedding_size, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, output_size, dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train, X_train)  # Using X_train as both src and tgt
    loss = loss_function(y_pred, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# Evaluate model predictions in the original scale
model.eval()
with torch.no_grad():
    y_test_pred_scaled = model(X_test, X_test).numpy()
    y_test_pred = target_scaler.inverse_transform(y_test_pred_scaled)
    y_test_true = target_scaler.inverse_transform(y_test.numpy())
    mse_true_scale = mean_squared_error(y_test_true, y_test_pred, multioutput='raw_values')
    print(f'Mean Squared Error in the original scale for predicting next month: {mse_true_scale[0]}')
    print(f'Mean Squared Error in the original scale for predicting next 3 months: {mse_true_scale[1]}')
    print(f'Mean Squared Error in the original scale for predicting next 6 months: {mse_true_scale[2]}')

# Example prediction for the next month, 3 months, and 6 months
with torch.no_grad():
    latest_data = features_tensor[-max(seq_lengths):].unsqueeze(0)  # Take the latest sequence
    future_predictions_scaled = model(latest_data, latest_data).numpy()
    future_predictions = target_scaler.inverse_transform(future_predictions_scaled)  # Inverse transform the predictions
    print(f'Predicted closing price in 1 month: {future_predictions[0][0]}')
    print(f'Predicted closing price in 3 months: {future_predictions[0][1]}')
    print(f'Predicted closing price in 6 months: {future_predictions[0][2]}')
