import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization

# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
EPOCHS = 10
BATCH_SIZE = 256

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "KDDTrain+.txt"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------
# Load and preprocess NSL-KDD
# --------------------------------------------------

def load_data():
    columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes",
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
        "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate",
        "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "attack", "difficulty"
    ]

    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = columns

    # Convert attack labels to binary:
    # 0 = Normal, 1 = Attack
    df["attack"] = df["attack"].apply(
        lambda x: 0 if x == "normal" else 1
    )

    # One-hot encode categorical features
    df = pd.get_dummies(
        df,
        columns=["protocol_type", "service", "flag"]
    )

    y = df["attack"]
    X = df.drop(columns=["attack", "difficulty"])

    return X, y


# --------------------------------------------------
# CNN
# --------------------------------------------------

def build_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),

        Conv1D(64, kernel_size=3, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Conv1D(32, kernel_size=3, activation="relu"),
        Dropout(0.3),

        tf.keras.layers.Flatten(),

        Dense(128, activation="relu"),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# --------------------------------------------------
# LSTM
# --------------------------------------------------

def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),

        LSTM(64, return_sequences=True),
        Dropout(0.3),

        LSTM(32),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# --------------------------------------------------
# Transformer
# --------------------------------------------------

class TransformerIDS(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.embedding = nn.Linear(input_dim, 128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)

        # Transformer expects a sequence dimension
        x = x.unsqueeze(1)

        x = self.transformer(x)

        x = x.mean(dim=1)

        return torch.sigmoid(self.output(x))


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def main():

    print("\n================ DATA PREPARATION ================\n")

    X, y = load_data()

    print(f"Total samples: {len(X)}")
    print(f"Total features: {X.shape[1]}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    # Standardization
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save preprocessing artifacts for API
    import joblib

    joblib.dump(
        scaler,
        MODEL_DIR / "scaler.pkl"
    )

    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(list(X.columns), f, indent=2)

    # Reshape for CNN/LSTM
    X_train_dl = X_train_scaled.reshape(
        X_train_scaled.shape[0],
        X_train_scaled.shape[1],
        1
    )

    X_test_dl = X_test_scaled.reshape(
        X_test_scaled.shape[0],
        X_test_scaled.shape[1],
        1
    )

    # --------------------------------------------------
    # CNN
    # --------------------------------------------------

    print("\n================ CNN MODEL ================\n")

    cnn = build_cnn(
        (X_train_dl.shape[1], 1)
    )

    cnn.fit(
        X_train_dl,
        y_train,
        epochs=EPOCHS,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    cnn_predictions = (
        cnn.predict(X_test_dl, verbose=0) > 0.5
    ).astype(int).ravel()

    cnn_accuracy = accuracy_score(
        y_test,
        cnn_predictions
    )

    print(f"\nCNN Accuracy: {cnn_accuracy:.4f}")

    print("\nCNN Classification Report:\n")

    print(
        classification_report(
            y_test,
            cnn_predictions,
            target_names=["Normal", "Attack"]
        )
    )

    cnn.save(
        MODEL_DIR / "cnn_model.h5"
    )

    # --------------------------------------------------
    # LSTM
    # --------------------------------------------------

    print("\n================ LSTM MODEL ================\n")

    lstm = build_lstm(
        (X_train_dl.shape[1], 1)
    )

    lstm.fit(
        X_train_dl,
        y_train,
        epochs=EPOCHS,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    lstm_predictions = (
        lstm.predict(X_test_dl, verbose=0) > 0.5
    ).astype(int).ravel()

    lstm_accuracy = accuracy_score(
        y_test,
        lstm_predictions
    )

    print(f"\nLSTM Accuracy: {lstm_accuracy:.4f}")

    print("\nLSTM Classification Report:\n")

    print(
        classification_report(
            y_test,
            lstm_predictions,
            target_names=["Normal", "Attack"]
        )
    )

    # --------------------------------------------------
    # Transformer
    # --------------------------------------------------

    print("\n================ TRANSFORMER MODEL ================\n")

    X_train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train.to_numpy(),
        dtype=torch.float32
    ).unsqueeze(1)

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    )

    y_test_tensor = torch.tensor(
        y_test.to_numpy(),
        dtype=torch.float32
    ).unsqueeze(1)

    train_dataset = torch.utils.data.TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    transformer = TransformerIDS(
        input_dim=X_train_scaled.shape[1]
    )

    criterion = nn.BCELoss()

    optimizer = optim.Adam(
        transformer.parameters(),
        lr=0.001
    )

    for epoch in range(EPOCHS):

        transformer.train()

        total_loss = 0

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()

            outputs = transformer(batch_X)

            loss = criterion(
                outputs,
                batch_y
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(train_loader)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"Loss: {average_loss:.4f}"
        )

    # Transformer evaluation
    transformer.eval()

    with torch.no_grad():

        outputs = transformer(
            X_test_tensor
        )

        transformer_predictions = (
            outputs >= 0.5
        ).int().numpy().ravel()

    transformer_accuracy = accuracy_score(
        y_test,
        transformer_predictions
    )

    print(
        f"\nTransformer Accuracy: "
        f"{transformer_accuracy:.4f}"
    )

    print(
        "\nTransformer Classification Report:\n"
    )

    print(
        classification_report(
            y_test,
            transformer_predictions,
            target_names=["Normal", "Attack"]
        )
    )

    # --------------------------------------------------
    # Sample prediction
    # --------------------------------------------------

    sample = X_test_scaled[0]

    sample_tensor = torch.tensor(
        sample,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        probability = transformer(
            sample_tensor
        ).item()

    print("\n================ SAMPLE PREDICTION ================\n")

    print(
        f"Prediction probability: {probability:.4f}"
    )

    if probability >= 0.5:
        print("⚠️ Attack Traffic")
    else:
        print("✅ Normal Traffic")

    print("\nTraining and evaluation complete.")


if __name__ == "__main__":
    main()