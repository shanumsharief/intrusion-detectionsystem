# Network Intrusion Detection System

Compares **CNN, LSTM, and Transformer** models for binary network traffic classification (Normal vs. Attack) on the NSL-KDD benchmark, and serves the trained CNN through a **FastAPI** prediction service.

## Overview

- **CNN:** learns local patterns across the traffic features.
- **LSTM:** models relationships across the feature sequence.
- **Transformer:** uses self-attention to learn feature interactions.

All three models share one preprocessing pipeline (one-hot encoding of categorical features and standardization), so their results are directly comparable.

## Results

Evaluated on a random 20% split of `KDDTrain+` (25,195 test records):

| Model | Test accuracy |
|---|---:|
| CNN | ~98–99%% |
| LSTM | 98.60% |
| Transformer | **99.17%** |

Transformer classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Normal | 0.99 | 0.99 | 0.99 | 13,469 |
| Attack | 0.99 | 0.99 | 0.99 | 11,726 |

> **How to read these numbers:** the test records come from the same file as the training records, so the figures are optimistic. They are not comparable to published results on the official `KDDTest+` set, which is harder.

## Pipeline

```text
NSL-KDD (KDDTrain+)
        ↓
Binary labels (0 = Normal, 1 = Attack)
        ↓
One-hot encoding → train/test split → standard scaling
        ↓
   CNN  |  LSTM  |  Transformer
        ↓
Evaluation → FastAPI service (CNN)
```

## Data

NSL-KDD is a benchmark dataset for network intrusion detection, with 41 traffic features per connection record. Download `KDDTrain+.txt` from https://www.unb.ca/cic/datasets/nsl.html and place it in `data/`. The dataset is not included in this repository.

Preprocessing:

1. Load `KDDTrain+` and convert attack labels to binary classes.
2. One-hot encode `protocol_type`, `service`, and `flag`.
3. Split into training and test sets (80/20).
4. Standardize features with `StandardScaler`.
5. Reshape the matrix for the CNN and LSTM.
6. Save the scaler and feature-column list so the API applies identical preprocessing.

## Models

| Model | Architecture |
|---|---|
| CNN | Conv1D, batch normalization, dropout, dense layers, sigmoid output |
| LSTM | LSTM, dropout, LSTM, dense layers, sigmoid output |
| Transformer | Linear feature embedding, 3 encoder layers, 8 attention heads, binary output |

The CNN's hyperparameters were tuned with Keras Tuner . The Transformer is trained in mini-batches to keep memory use manageable on local hardware.

## Setup

```bash
git clone https://github.com/shanumsharief/intrusion-detectionsystem.git
cd intrusion-detectionsystem
python3 -m pip install -r requirements.txt
```

## Training and evaluation

```bash
python3 src/train_and_evaluate.py
```

This trains and evaluates all three models and saves the artifacts the API needs:

```text
models/
├── cnn_model.h5
├── scaler.pkl
└── feature_columns.json
```

## API

Start the service:

```bash
python3 -m uvicorn src.api:app --reload
```

Interactive docs are at `http://127.0.0.1:8000/docs`.

**`GET /health`**

```json
{ "model_loaded": true, "scaler_loaded": true }
```

**`POST /predict`** takes a preprocessed feature vector and returns the predicted class and attack probability.

```json
{ "prediction": "Attack", "attack_probability": 0.9993 }
```

## Project structure

```text
├── README.md
├── requirements.txt
├── data/               # NSL-KDD files (not in the repo)
├── models/             # Trained CNN, scaler, feature columns (not in the repo)
└── src/
    ├── train_and_evaluate.py
    └── api.py
```

## Limitations

- **Optimistic evaluation.** Accuracy comes from a random split of the training file, not the official `KDDTest+` set, which contains unseen attack types and gives much lower scores.
- **No baseline.** I did not compare against a classical model such as Random Forest, which often matches deep networks on tabular data.
- **Single run.** Results come from one training run with no confidence intervals, so small gaps between models (such as LSTM vs. Transformer) may not be meaningful.
- **Preprocessed API input.** The API expects an already-processed feature vector, not raw connection fields or live traffic.
- **Dated benchmark.** NSL-KDD derives from 1999-era traffic and does not represent modern networks.

## Future improvements

- Evaluate on the official `KDDTest+` set and add a Random Forest baseline
- Run multiple seeds and report mean ± standard deviation
- Report false positive rate, ROC-AUC, and a confusion matrix
- Accept raw connection records in the API and preprocess them server-side
- Extend to multiclass attack classification
- Containerize the API with Docker

## License

MIT
