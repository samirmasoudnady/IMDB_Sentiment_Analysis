# IMDB Sentiment Analysis with LSTM & Pretrained Embeddings

A TensorFlow/Keras NLP project for binary sentiment classification on the **IMDB movie reviews** dataset.

The notebook starts with a basic `SimpleRNN`, then experiments with pretrained **Universal Sentence Encoder (USE)** and **GloVe embeddings**, and finally compares `SimpleRNN`, `LSTM`, and `Bidirectional LSTM` architectures.

The final recorded model is a **Bidirectional LSTM with frozen 100D GloVe embeddings**, reaching **85.34% test accuracy**.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Experiments](#experiments)
- [Results](#results)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Reproduce the Notebook](#reproduce-the-notebook)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Known Issues](#known-issues)
- [Future Work](#future-work)

---

## Overview

### Objective

Build a sentiment classifier that maps movie reviews to:

```text
0 → Bad Review
1 → Good Review
```

### Main techniques

- TensorFlow / Keras
- TensorFlow Datasets
- `TextVectorization`
- Word embeddings
- GloVe 100D pretrained embeddings
- Universal Sentence Encoder
- SimpleRNN
- LSTM
- Bidirectional LSTM
- Binary classification

---

## Dataset

The notebook uses the `imdb_reviews` dataset from TensorFlow Datasets:

```python
raw_train_set, raw_valid_set, raw_test_set = tfds.load(
    "imdb_reviews",
    split=["train[:90%]", "train[90%:]", "test"],
    as_supervised=True
)
```

| Split | Samples |
|---|---:|
| Train | 22,500 |
| Validation | 2,500 |
| Test | 25,000 |
| Total | 50,000 |

The data is shuffled and batched with:

```python
train_ds = raw_train_set.shuffle(5000, seed=42).batch(32).prefetch(1)
valid_ds = raw_valid_set.batch(32).prefetch(1)
test_ds = raw_test_set.batch(32).prefetch(1)
```

---

## Workflow

```text
IMDB Reviews
     ↓
Train / Validation / Test
     ↓
TextVectorization
     ↓
Word IDs
     ↓
Embedding Representation
     ↓
RNN / LSTM
     ↓
Sigmoid
     ↓
0 = Bad | 1 = Good
```

The notebook also demonstrates the complete text-vectorization cycle:

```text
Text → Token IDs → Text
```

Example:

```python
token_ids = tokenizer("I love you")
# [10, 115, 23]

decode_to_text(token_ids, tokenizer)
# "i love you"
```

The tokenizer uses:

```python
max_tokens = 1000
standardize = "lower_and_strip_punctuation"
split = "whitespace"
```

---

## Experiments

### 1. Baseline — SimpleRNN

```text
TextVectorization
      ↓
Embedding(128)
      ↓
SimpleRNN(32, return_sequences=True)
      ↓
SimpleRNN(32)
      ↓
Dense(1, sigmoid)
```

Configuration:

```text
Optimizer: Adam
Learning rate: 1e-3
Loss: binary_crossentropy
Epochs: 5
Batch size: 32
```

---

### 2. Universal Sentence Encoder

The notebook tests Google's pretrained Universal Sentence Encoder through TensorFlow Hub.

```text
Text
 ↓
Universal Sentence Encoder (512D)
 ↓
Dense(32, ReLU)
 ↓
Dense(1, Sigmoid)
```

The USE layer is frozen:

```python
trainable = False
```

Optimizer:

```text
Nadam, learning rate = 1e-3
```

The notebook contains this experiment, but does not record a complete training result, so no metric is reported here.

---

### 3. GloVe + SimpleRNN

The notebook loads **GloVe 100D** vectors and builds an embedding matrix aligned with the tokenizer vocabulary.

```text
Word → Token ID
Word → GloVe Vector
Token ID → Embedding Vector
```

Recorded embedding information:

```text
Embedding matrix: (1000, 100)
Words without GloVe: 7
```

Architecture:

```text
TextVectorization
      ↓
Frozen GloVe Embedding(100)
      ↓
SimpleRNN(32)
      ↓
Dense(1, Sigmoid)
```

---

### 4. GloVe + LSTM

```text
TextVectorization
      ↓
Frozen GloVe Embedding(100)
      ↓
LSTM(100)
      ↓
Dense(1, Sigmoid)
```

---

### 5. GloVe + Bidirectional LSTM

Final architecture:

```text
TextVectorization
      ↓
Frozen GloVe Embedding(100)
      ↓
Bidirectional(LSTM(100))
      ↓
Dense(1, Sigmoid)
```

The GloVe embedding layer remains frozen.

---

## Results

Results below are taken directly from the notebook's recorded training output.

| Model | Embedding | Validation Accuracy |
|---|---|---:|
| SimpleRNN baseline | Trainable 128D | 57.68%* |
| SimpleRNN | Frozen GloVe 100D | 56.68% |
| LSTM | Frozen GloVe 100D | **85.56%** |
| Bidirectional LSTM | Frozen GloVe 100D | **85.88%** |

### Final Test Result

The final Bidirectional LSTM was evaluated on the 25,000-review test set:

```text
Test Loss     = 0.3332
Test Accuracy = 0.8534
```

**Test Accuracy: 85.34%**

Training configuration for the final model:

```text
Epochs: 5
Batch size: 32
Optimizer: Adam
Learning rate: 1e-3
Loss: Binary Cross-Entropy
```

\* The baseline's epoch-5 validation accuracy is not present in the stored output; `57.68%` is the last validation accuracy explicitly recorded (epoch 4).

---

## Requirements

The notebook uses:

```text
Python 3.12.13
TensorFlow
TensorFlow Datasets
TensorFlow Hub
NumPy
Keras
```

Install the main dependencies:

```bash
pip install tensorflow tensorflow-datasets tensorflow-hub numpy jupyter
```

The original notebook was run in a **Kaggle Notebook** environment.

---

## Quick Start

Open the notebook:

```text
imbd-sentiment-analysis-ipynb.ipynb
```

Then run the cells from top to bottom.

The core setup is:

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Nadam
import tensorflow_hub as hub
```

Load the dataset:

```python
raw_train_set, raw_valid_set, raw_test_set = tfds.load(
    "imdb_reviews",
    split=["train[:90%]", "train[90%:]", "test"],
    as_supervised=True
)
```

---

## Reproduce the Notebook

### 1. Prepare the data

```python
tf.random.set_seed(42)

train_ds = raw_train_set.shuffle(5000, seed=42).batch(32).prefetch(1)
valid_ds = raw_valid_set.batch(32).prefetch(1)
test_ds = raw_test_set.batch(32).prefetch(1)
```

### 2. Build the tokenizer

```python
tokenizer = TextVectorization(
    max_tokens=1000,
    standardize="lower_and_strip_punctuation",
    split="whitespace"
)

tokenizer.adapt(
    [X.numpy().decode() for X, y in raw_train_set]
)
```

### 3. Build the GloVe matrix

The notebook expects:

```text
glove.6B.100d.txt
```

and uses a Kaggle-specific path:

```text
/kaggle/input/models/rudra29/glove/keras/default/1/glove.6B.100d.txt
```

For another environment, change this path.

### 4. Train the final model

```python
optimizer = Adam(learning_rate=1e-3)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=5
)
```

### 5. Evaluate

```python
model.evaluate(test_ds)
```

Expected result based on the notebook:

```text
loss     ≈ 0.3332
accuracy ≈ 0.8534
```

---

## Inference

The notebook saves the final model in both formats:

```text
Bidirectional_Lstm_model.h5
Bidirectional_Lstm_model.keras
```

The recommended native Keras format is:

```python
model.save("Bidirectional_Lstm_model.keras")
```

Load it:

```python
from tensorflow.keras.models import load_model

loaded_model = load_model(
    "Bidirectional_Lstm_model.keras"
)
```

Convert predictions to labels:

```python
label_map = {
    0: "Bad Review",
    1: "Good Review"
}

predictions = loaded_model.predict(test_ds)

predicted_classes = (predictions > 0.5).astype(int)

results = [
    label_map[int(x)]
    for x in predicted_classes.flatten()
]
```

For a new review, the model can receive raw text because `TextVectorization` is included inside the model:

```python
review = ["This movie was excellent and enjoyable."]

prediction = loaded_model.predict(review)

label = label_map[int(prediction[0][0] > 0.5)]

print(label)
```

---

## Project Structure

Recommended GitHub structure:

```text
imdb-sentiment-analysis/
│
├── README.md
├── imbd-sentiment-analysis-ipynb.ipynb
├── Bidirectional_Lstm_model.keras
├── requirements.txt
├── LICENSE
└── .gitignore
```

The current project is notebook-based, so the repository does not need unnecessary source-code folders unless the project is later converted into a package/application.

---

## Known Issues

- The GloVe path is Kaggle-specific.
- The USE model depends on TensorFlow Hub/Kaggle model resources.
- Exact TensorFlow package version was not recorded.
- Vocabulary size is limited to 1,000 tokens.
- GloVe embeddings are frozen rather than fine-tuned.
- Only 5 training epochs are used.
- The notebook mainly evaluates accuracy and binary cross-entropy.
- No confusion matrix, precision, recall, F1, or ROC-AUC is included.
- HDF5 saving produces a legacy-format warning; `.keras` is preferred.
- The USE experiment has no complete recorded training metrics.

---

## Future Work

- Increase vocabulary size.
- Fine-tune GloVe embeddings.
- Add dropout and regularization.
- Add early stopping/checkpointing.
- Compare with transformer-based models.
- Add precision, recall, F1-score, ROC-AUC, and confusion matrix.
- Add training/validation plots.
- Create a clean inference script or API.
- Pin dependency versions for reproducibility.

---

## Notebook Summary

```text
Load IMDB
   ↓
Explore samples / labels / shapes
   ↓
Batch & shuffle data
   ↓
TextVectorization
   ↓
Encoding / Decoding
   ↓
Baseline SimpleRNN
   ↓
Universal Sentence Encoder
   ↓
GloVe 100D
   ↓
GloVe + SimpleRNN
   ↓
GloVe + LSTM
   ↓
GloVe + Bidirectional LSTM
   ↓
Evaluate
   ↓
Save .h5 / .keras
   ↓
Load saved model
   ↓
Predict Good / Bad Review
```
