# Session 1 – Practice Notes (Part C)
## Defining Features (X) and Target Variable (y)

This note guides **PhD and Master students** through the first **machine learning–specific step**:
**separating input variables (X) and the outcome variable (y)**.

> Goal of Part C:  
> Clearly define **what the model learns from** and **what the model tries to predict**.

---

## Prerequisite

Before starting Part C, you should have completed:

- Part A: Google Colab & Google Drive connection
- Part B: Basic Python commands for data understanding

Your dataset should already be loaded as a DataFrame named `df`.

---

## 1. Why We Must Define X and y

In supervised machine learning:
- **X (features)** = information we use to make predictions
- **y (target)** = outcome we want to predict

> If X and y are not clearly defined, machine learning cannot work.

In our dataset:
- **X** = customer characteristics
- **y** = whether the customer churned

---

## 2. Identify the Target Variable (y)

From Part B, we already saw the column names:

```python
df.columns
```

In the Customer Churn dataset:
- The target variable is **`Churn`**
- It contains values: `Yes` or `No`

Define `y` as follows:

```python
y = df['Churn']
```

---

## 3. Why We Do NOT Include the Target in X

Important rule in machine learning:

> The model must never see the correct answer while learning.

If we include `Churn` inside X:
- The model will appear extremely accurate
- The result is invalid
- This is called **data leakage**

---

## 4. Define Feature Variables (X)

We define X by **removing the target column** from the dataset.

```python
X = df.drop('Churn', axis=1)
```

At this stage:
- X contains all customer information
- y contains only the churn outcome

---

## 5. Remove Non-Informative Identifiers

Some columns identify customers but do not help prediction.

Example:
- `customerID`

We remove it from X:

```python
X = X.drop('customerID', axis=1)
```

> IDs identify individuals but do not describe behavior.

---

## 6. Check X and y Shapes

Always verify your result:

```python
X.shape
```

```python
y.shape
```

Expected result:
- Number of rows in X = number of rows in y
- Number of columns in X = total variables minus target and ID

---

## 7. Confirm X and y Content

Preview X:

```python
X.head()
```

Preview y:

```python
y.head()
```

This confirms:
- X contains input variables only
- y contains the outcome only

---

## 8. Conceptual Check (Very Important)

Ask yourself:

- Do all X variables exist **before** churn happens?
  What this means 
    “Are we only using information that was available before the customer decided to leave?”
  Why this matters
    Machine learning is used to predict the future, not to explain the past using future information.
    If we use information that appears after churn, the model is cheating.
- Does y represent something we want to predict?
  What this means
    “Is y actually the outcome we care about, or are we predicting the wrong thing?”
- Could any X variable accidentally reveal y?
  Very simple explanation
    “Does any input variable secretly contain the answer?”

> These questions are critical in research methodology.

---

## 9. What We Have Achieved So Far

At the end of Part C, we have:

- Cleanly defined features (X)
- Clearly defined target variable (y)
- Avoided data leakage
- Prepared the dataset for model training

No machine learning model has been trained yet.

---

## 10. Transition to Next Step

Now that X and y are defined, the next steps will be:

- Train / test split
- Model training
- Model evaluation

This is where machine learning officially begins.

---

**End of Practice Notes – Session 1 (Part C)**
