# Intrusion Detection System

A deep-learning based network intrusion detection system that compares **CNN, LSTM, and Transformer** architectures for binary network traffic classification.

The system classifies network traffic as either **Normal** or **Attack** using the **NSL-KDD benchmark dataset** and exposes the trained CNN model through a lightweight **FastAPI** prediction service.

## Overview

This project explores different deep-learning architectures for network intrusion detection:

* **CNN** — learns local patterns from network traffic features.
* **LSTM** — models sequential relationships between features.
* **Transformer** — uses self-attention to learn feature relationships.

The models are trained and evaluated using a consistent preprocessing pipeline with categorical feature encoding and standardization.

## Results

Performance on the held-out NSL-KDD test set:

| Model       | Test Accuracy |
| ----------- | ------------: |
| CNN         |       ~98–99% |
| LSTM        |        98.60% |
| Transformer |    **99.17%** |

### Transformer Classification Report

| Class       | Precision |   Recall | F1-Score |
| ----------- | --------: | -------: | -------: |
| Normal      |      0.99 |     0.99 |     0.99 |
| Attack      |      0.99 |     0.99 |     0.99 |
| **Overall** |  **0.99** | **0.99** | **0.99** |

The Transformer achieved **99.17% accuracy** on the held-out test set.

## Architecture

```text
                    NSL-KDD Dataset
                           │
                           ▼
                Data Preprocessing
                           │
                ┌──────────┴──────────┐
                │                     │
          Categorical             Standard
           Encoding              Scaling
                │                     │
                └──────────┬──────────┘
                           │
                    Feature Matrix
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
         CNN              LSTM         Transformer
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                  Binary Classification
                     Normal / Attack
                           │
                           ▼
                    Model Evaluation
                           │
                           ▼
                     FastAPI Service
```

## Dataset

The primary dataset used in this project is **NSL-KDD**, a benchmark dataset for network intrusion detection research.

The dataset contains network connection records with numerical and categorical traffic features.

For this implementation, attack labels are converted into a binary classification task:

* `0` → Normal
* `1` → Attack

Categorical features such as protocol, service, and flag are converted using one-hot encoding before standardization.

## Preprocessing

The preprocessing pipeline consists of:

1. Loading the NSL-KDD training data.
2. Converting attack labels into binary classes.
3. One-hot encoding categorical features.
4. Splitting the dataset into training and test sets.
5. Standardizing features using `StandardScaler`.
6. Reshaping the feature matrix for CNN and LSTM models.
7. Saving the scaler and feature-column metadata for API inference.

The same scaler used during training is loaded by the API to ensure consistent preprocessing during prediction.

## Models

### CNN

The CNN uses one-dimensional convolutional layers to learn local patterns across the feature representation.

Architecture includes:

* Conv1D
* Batch Normalization
* Dropout
* Dense layers
* Sigmoid output

### LSTM

The LSTM model uses recurrent layers to learn relationships across the feature sequence.

Architecture includes:

* LSTM
* Dropout
* LSTM
* Dense layers
* Sigmoid output

### Transformer

The Transformer model uses self-attention to model relationships between network traffic features.

Architecture includes:

* Linear feature embedding
* Transformer Encoder
* 3 encoder layers
* 8 attention heads
* Binary classification output

Training uses mini-batches to keep memory usage manageable on local hardware.

## Project Structure

```text
intrusion-detectionsystem/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── KDDTrain+.txt
│   └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│
├── models/
│   ├── cnn_model.h5
│   ├── scaler.pkl
│   └── feature_columns.json
│
└── src/
    ├── train_and_evaluate.py
    └── api.py
```

Dataset and trained model artifacts are excluded from version control through `.gitignore`.

## Installation

Clone the repository:

```bash
git clone https://github.com/shanumsharief/intrusion-detectionsystem-.git
cd intrusion-detectionsystem-
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Training and Evaluation

Run the complete training and evaluation pipeline:

```bash
python3 src/train_and_evaluate.py
```

This trains the CNN, LSTM, and Transformer models and evaluates them on the held-out test set.

The pipeline also generates the preprocessing artifacts required by the API:

```text
models/
├── cnn_model.h5
├── scaler.pkl
└── feature_columns.json
```

## API

The project includes a FastAPI service for CNN-based traffic classification.

Start the API:

```bash
python3 -m uvicorn src.api:app --reload
```

The API will run locally at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Health Check

```http
GET /health
```

Example response:

```json
{
  "model_loaded": true,
  "scaler_loaded": true
}
```

### Prediction

```http
POST /predict
```

The endpoint accepts a preprocessed feature vector and returns the predicted class and attack probability.

Example response:

```json
{
  "prediction": "Attack",
  "attack_probability": 0.9993
}
```

## Technologies

* Python
* TensorFlow / Keras
* PyTorch
* Scikit-learn
* Pandas
* NumPy
* FastAPI
* Uvicorn

## Limitations

This project is an experimental intrusion detection system evaluated on a benchmark dataset.

The reported accuracy reflects performance on the NSL-KDD test split and should not be interpreted as real-world network detection accuracy.

The current API accepts a preprocessed feature vector rather than extracting network-flow features directly from live traffic.

The CIC-IDS2017 dataset is included locally for future experimentation but is not part of the reported evaluation results.

## Future Improvements

* Evaluate models across multiple intrusion-detection datasets.
* Extend the system to multiclass attack classification.
* Add real-time network traffic processing.
* Add automated feature extraction from network flows.
* Compare additional machine-learning and deep-learning architectures.
* Containerize the API for deployment.

## License

This project is intended for educational and portfolio purposes.
