# Session 1 – Practice Notes  
## Google Colab Introduction & Connecting Google Drive

This note is for **PhD and Master students** to follow step-by-step during **Session 1 practice**.

---

## 1. What Is Google Colab?

Google Colab is an online Python notebook environment provided by Google.

**Key points:**
- Runs in a web browser
- No installation required
- Uses Google’s servers
- Commonly used in academic research and prototyping

In this course, Colab is used to **experiment with machine learning**, not for production systems.

---

## 2. Why We Use Google Drive for Data

In real research work:
- Data and code are usually stored separately
- Data should be organized and reusable
- Code should not depend on local machines

In this class:
- **Data is stored in Google Drive**
- **Colab reads data from Drive**

This improves reproducibility and good research practice.

---

## 3. Dataset Location

The dataset has been organized as follows:

```
My Drive/
└── Customer_data/
    └── Customer_Churn.csv
```

**Important notes:**
- Folder and file names are case-sensitive
- Avoid spaces in file names
- Keep datasets in clearly named folders

---

## 4. Creating a New Colab Notebook

1. Go to: https://colab.research.google.com
2. Click **New Notebook**
3. Rename the notebook (top-left):

```
Session1_Customer_Churn.ipynb
```

Naming files properly is part of good academic practice.

---

## 5. Connecting Google Drive to Colab

Run the following code cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Steps:
1. Click the authorization link
2. Choose your Google account
3. Click **Allow**

This gives Colab permission to read files from your Drive.

---

## 6. Verifying Drive Connection

Check that Google Drive is connected:

```python
!ls /content/drive/MyDrive/
```

Then check the dataset folder:

```python
!ls /content/drive/MyDrive/Customer_data/
```

You should see:

```
Customer_Churn.csv
```

If you see the file name, the connection is successful.

---

## 7. Understanding the File Path

The full file path used in Colab is:

```
/content/drive/MyDrive/Customer_data/Customer_Churn.csv
```

Explanation:
- `/content/drive/MyDrive/` → Google Drive root
- `Customer_data/` → dataset folder
- `Customer_Churn.csv` → data file

Colab requires the **full path** to read files.

---

## 8. Loading the Dataset (No Machine Learning Yet)

Now load the dataset using pandas:

```python
import pandas as pd

df = pd.read_csv(
    '/content/drive/MyDrive/Customer_data/Customer_Churn.csv'
)

df.head()
```

At this stage:
- We are only checking data access
- No machine learning is performed yet

---

## 9. Common Errors and Troubleshooting

If you see an error, check the following:

- File name spelling (capital letters matter)
- Folder name spelling
- Google Drive is mounted
- Dataset is in **My Drive**, not **Shared Drive**

Most problems are related to incorrect file paths.

---

## 10. Key Takeaway

Before building any machine learning model:
- Ensure your environment works
- Ensure your data can be accessed correctly
- Understand where your data is stored

Once this is complete, we are ready to define:
- Features (X)
- Target variable (y)

---

**End of Practice Notes – Session 1**
