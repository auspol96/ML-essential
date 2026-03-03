# Neural Network (Keras) – RSU Version (Using Google Drive Dataset)

เอกสารนี้ใช้ไฟล์จริงจาก Google Drive ตาม config ที่ใช้ในคลาส

---

## 1) Mount Google Drive (Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 2) Import Libraries

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
```

---

## 3) Load Dataset (ตาม path ที่ใช้ในคลาส)

```python
df = pd.read_csv(
    '/content/drive/MyDrive/Customer_data/Customer_Churn.csv'
)

df.head()
```

---

## 4) Prepare Data

สมมติว่า target column ชื่อ `churn` (0/1)

```python
# Clean column names first (safe practice)
df.columns = df.columns.str.strip()

# Target
y = df["Churn"]

# Convert Yes/No → 1/0
y = y.map({"Yes": 1, "No": 0})

# Features
X = df.drop(columns=["Churn"])

# Convert categorical variables
X = pd.get_dummies(X, drop_first=True)
```

---

## 5) Train/Test Split + Scaling

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

---

## 6) Build Neural Network

```python
model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train_s.shape[1],)),
    Dropout(0.2),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])
```

---

## 7) Compile Model

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)
```

---

## 8) Train Model

```python
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train_s, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)
```

---

## 9) Evaluate Model

```python
loss, acc, auc = model.evaluate(X_test_s, y_test, verbose=0)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test AUC: {auc:.4f}")
```

---

## 10) Predict

```python
y_prob = model.predict(X_test_s).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("First 10 probabilities:", y_prob[:10])
print("First 10 predictions:", y_pred[:10])
```

---

## Quick Checklist Before Run

- ✅ target column = 0/1
- ✅ ทำ get_dummies แล้วถ้ามี categorical
- ✅ ทำ scaling ก่อนเข้า model
- ✅ output layer = sigmoid
- ✅ loss = binary_crossentropy
