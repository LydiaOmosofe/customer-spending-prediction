# Customer Spending Prediction using PySpark
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PySpark](https://img.shields.io/badge/PySpark-3.4.1-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)

## Project Overview
This project predicts total customer spending using transaction data from a UK-based online retailer. The dataset contains **541,909 records** across 8 columns.

##  Business Problem
Understanding customer spending patterns helps businesses:
- Identify high-value customers
- Personalize marketing campaigns
- Optimize inventory and pricing strategies

##  Technologies Used
- **PySpark** (Data processing & ML)
- **Python** (Pandas, Matplotlib, Seaborn)
- **MLlib** (Linear Regression, CrossValidator)

##  Dataset
**Online Retail Dataset** (UCI Machine Learning Repository)
- 541,909 transactions
- 8 columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

##  Project Workflow

### 1. Data Loading & Exploration
- Loaded CSV with schema inference
- Checked for missing values: 1,454 missing descriptions, 135,080 missing CustomerIDs
- Analyzed data distribution by country

### 2. Data Cleaning
- Removed rows with null CustomerIDs
- Converted InvoiceDate to timestamp format
- Filtered out negative quantities and prices
- Created `TotalAmount` column (Quantity × UnitPrice)

### 3. Feature Engineering
Aggregated data at customer level:
- **TotalOrders** — Number of unique invoices
- **TotalQuantity** — Total items purchased
- **AvgUnitPrice** — Average price per item
- **TotalSpent** — Target variable

### 4. Model Building
- **Linear Regression** with StandardScaler
- 80/20 train-test split
- Features: TotalOrders, TotalQuantity, AvgUnitPrice

### 5. Hyperparameter Tuning
- 5-fold CrossValidator
- Grid search over:
  - `regParam`: [0.01, 0.1, 0.5]
  - `elasticNetParam`: [0.0, 0.5, 1.0]
  - `maxIter`: [50, 100]
- **90 model fits** total

### 6. Model Evaluation
| Metric | Baseline | Tuned |
|--------|----------|-------|
| **RMSE** | £2,216.61 | £2,216.52 |
| **MAE**  | £537.41   | £537.34   |
| **R²**   | 0.8738    | 0.8738    |

##  Visualizations

| Plot | Description |
|------|-------------|
| **Actual vs Predicted** | Scatter plot showing model performance |
| **Residuals Distribution** | Histogram of prediction errors |
| **Feature Coefficients** | Feature importance from the model |
| **Metrics Comparison** | Baseline vs Tuned performance |

##  Repository Structure
