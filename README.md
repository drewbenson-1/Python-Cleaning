# Python-Cleaning
A data cleaning process using Python to transformation messy data into analysis ready data.
Markdown
# 🧹 Automated Data Cleaning & Preprocessing Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Cleaning-orange.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

## 📌 Project Overview
Real-world datasets are rarely ready for analysis straight out of the box. This project demonstrates an automated data cleaning and preprocessing pipeline built in Python. It takes a raw, highly unformatted dataset containing missing values, structural inconsistencies, and outliers, and transforms it into a clean, standardized format ready for downstream machine learning or business intelligence dashboards.

## 🛠️ Tools & Libraries Used
* **Language:** Python
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization (EDA):** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / VS Code

---

## 🔍 The Data Cleaning Workflow

The script systematically executes the following transformation steps:

1. **Structural Audit & Schema Enforcement:** Standardized column naming conventions (snake_case), corrected data types (converting strings to datetimes and floats), and dropped redundant unindexed columns.
2. **Missing Value Treatment:** Evaluated missingness patterns. Imputed numerical columns using median values and categorical columns using mode, while safely dropping rows where critical target identifiers were absent.
3. **Outlier Detection & Handling:** Utilized the Interquartile Range (IQR) method to flag and cap extreme outliers in financial/numerical fields without skewing the core distribution.
4. **Text Standardization & Deduplication:** Stripped leading/trailing whitespaces, corrected capitalization inconsistencies in categorical fields, and removed duplicate rows based on unique primary keys.

---

## 💻 Code Highlight: Handling Missing Data & Outliers

```python
import pandas as pd
import numpy as np

def clean_dataset(df):
    # 1. Standardize column names (lowercase, replace spaces with underscores)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # 2. Drop exact duplicate rows
    df = df.drop_duplicates()
    
    # 3. Impute missing numerical values with the median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # 4. Cap outliers using IQR method for financial columns
    Q1 = df['revenue'].quantile(0.25)
    Q3 = df['revenue'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    df['revenue'] = np.where(df['revenue'] > upper_bound, upper_bound, df['revenue'])
    
    return df
