import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, Dropout, BatchNormalization
import torch
import torch.nn as nn
import torch.optim as optim

# Preprocessing function for CIC-IDS2017

def preprocess_data(df):
    # Drop empty or unnamed columns
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Drop rows with missing values
    df = df.dropna()

    # Binary encode the label
    df['Label'] = df['Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)

    # Separate features and target
    y = df['Label']
    X = df.drop(columns=['Label'])

    return X, y

# Load NSL-KDD dataset and preprocess (this should already exist in your code)
df = pd.read_csv("KDDTrain+.txt", header=None)
columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
           "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
           "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
           "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", 
           "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
           "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
           "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
           "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
           "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
           "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "difficulty"]
df.columns = columns

# Encode categorical and label columns
df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])
df['attack'] = df['attack'].apply(lambda x: 0 if x == 'normal' else 1)

# Split and normalize
X = df.drop(columns=['attack', 'difficulty'])
y = df['attack']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# CNN Model
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train_cnn = np.expand_dims(X_train, axis=-1)
X_test_cnn = np.expand_dims(X_test, axis=-1)
cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test))
cnn_model.save("cnn_model.h5")


# LSTM Model
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    LSTM(32),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train_lstm = np.expand_dims(X_train, axis=-1)
X_test_lstm = np.expand_dims(X_test, axis=-1)
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test))

# Evaluate on test set
cnn_loss, cnn_acc = cnn_model.evaluate(X_test_cnn, y_test)
lstm_loss, lstm_acc = lstm_model.evaluate(X_test_lstm, y_test)
print(f"CNN Accuracy: {cnn_acc:.2f}")
print(f"LSTM Accuracy: {lstm_acc:.2f}")

# Transformer Model
class TransformerIDS(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerIDS, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return torch.sigmoid(x)

input_dim = X_train.shape[1]
num_classes = 1
transformer_model = TransformerIDS(input_dim, num_classes)
criterion = nn.BCELoss()
optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

# Train Transformer
for epoch in range(10):
    optimizer.zero_grad()
    outputs = transformer_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/10, Loss: {loss.item():.4f}')
