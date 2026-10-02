import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, Dropout, BatchNormalization

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# LOAD AND PREPROCESS NSL-KDD DATASET
# ============================================================

df = pd.read_csv("data/KDDTrain+.txt", header=None)

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack", "difficulty"
]

df.columns = columns

# Encode categorical features
df = pd.get_dummies(
    df,
    columns=["protocol_type", "service", "flag"]
)

# Binary classification:
# 0 = Normal
# 1 = Attack
df["attack"] = df["attack"].apply(
    lambda x: 0 if x == "normal" else 1
)

# Separate features and target
X = df.drop(columns=["attack", "difficulty"])
y = df["attack"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Normalize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of features:", X_train.shape[1])


# ============================================================
# CNN MODEL
# ============================================================

print("\n================ CNN MODEL ================\n")

cnn_model = Sequential([
    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        input_shape=(X_train.shape[1], 1)
    ),
    BatchNormalization(),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

cnn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

X_train_cnn = np.expand_dims(X_train, axis=-1)
X_test_cnn = np.expand_dims(X_test, axis=-1)

cnn_model.fit(
    X_train_cnn,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_cnn, y_test)
)

# Save CNN model
cnn_model.save("models/cnn_model.h5")


# ============================================================
# LSTM MODEL
# ============================================================

print("\n================ LSTM MODEL ================\n")

lstm_model = Sequential([
    LSTM(
        64,
        return_sequences=True,
        input_shape=(X_train.shape[1], 1)
    ),
    Dropout(0.3),
    LSTM(32),
    Dense(1, activation="sigmoid")
])

lstm_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

X_train_lstm = np.expand_dims(X_train, axis=-1)
X_test_lstm = np.expand_dims(X_test, axis=-1)

lstm_model.fit(
    X_train_lstm,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_lstm, y_test)
)


# ============================================================
# CNN + LSTM EVALUATION
# ============================================================

print("\n================ MODEL EVALUATION ================\n")

cnn_loss, cnn_acc = cnn_model.evaluate(
    X_test_cnn,
    y_test,
    verbose=1
)

lstm_loss, lstm_acc = lstm_model.evaluate(
    X_test_lstm,
    y_test,
    verbose=1
)

print(f"CNN Accuracy: {cnn_acc:.2f}")
print(f"LSTM Accuracy: {lstm_acc:.2f}")


# ============================================================
# TRANSFORMER MODEL
# ============================================================

print("\n================ TRANSFORMER MODEL ================\n")


class TransformerIDS(nn.Module):

    def __init__(self, input_dim, num_classes):

        super(TransformerIDS, self).__init__()

        self.embedding = nn.Linear(
            input_dim,
            128
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=3
        )

        self.fc = nn.Linear(
            128,
            num_classes
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.transformer_encoder(x)

        x = x.mean(dim=1)

        x = self.fc(x)

        return torch.sigmoid(x)


# ============================================================
# PREPARE TRANSFORMER DATA
# ============================================================

input_dim = X_train.shape[1]
num_classes = 1

transformer_model = TransformerIDS(
    input_dim,
    num_classes
)

criterion = nn.BCELoss()

optimizer = optim.Adam(
    transformer_model.parameters(),
    lr=0.001
)


# Convert to tensors
X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
).unsqueeze(1)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).unsqueeze(1)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).unsqueeze(1)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
).unsqueeze(1)


# ============================================================
# MINI-BATCH TRAINING
# ============================================================

batch_size = 256

train_dataset = torch.utils.data.TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)


# ============================================================
# TRAIN TRANSFORMER
# ============================================================

epochs = 10

for epoch in range(epochs):

    transformer_model.train()

    total_loss = 0

    for batch_X, batch_y in train_loader:

        optimizer.zero_grad()

        outputs = transformer_model(batch_X)

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {average_loss:.4f}"
    )


# ============================================================
# TRANSFORMER EVALUATION
# ============================================================

transformer_model.eval()

with torch.no_grad():

    transformer_outputs = transformer_model(
        X_test_tensor
    )

    transformer_predictions = (
        transformer_outputs >= 0.5
    ).int().numpy().flatten()


transformer_accuracy = (
    transformer_predictions == y_test.values
).mean()

print(
    f"\nTransformer Accuracy: "
    f"{transformer_accuracy:.2f}"
)


# ============================================================
# TRANSFORMER CLASSIFICATION REPORT
# ============================================================

print("\n================ TRANSFORMER CLASSIFICATION REPORT ================\n")

print(
    classification_report(
        y_test,
        transformer_predictions,
        target_names=["Normal", "Attack"]
    )
)


# ============================================================
# SINGLE SAMPLE PREDICTION
# ============================================================

print("\n================ SAMPLE PREDICTION ================\n")

sample = X_test_tensor[0:1]

with torch.no_grad():

    sample_prediction = transformer_model(
        sample
    ).item()

print(
    f"Prediction probability: "
    f"{sample_prediction:.4f}"
)

if sample_prediction > 0.5:

    print("🔥 Intrusion Detected!")

else:

    print("✅ Normal Traffic")