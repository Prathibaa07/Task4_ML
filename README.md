# Task 4 - Household Energy Consumption Prediction

## Project Overview

This project uses **Machine Learning** to predict household energy consumption based on household and usage-related factors.

A **Polynomial Regression** model is used to predict:

**Energy Consumption (kWh)**

using the following input features:

* Household Size
* Average Temperature (°C)
* Peak Hours Usage (kWh)

The project also evaluates the model using MAE, MSE, RMSE, and R² score.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / Google Colab

---

## Dataset

The dataset used in this project is:

```text
household_energy_consumption.csv
```

The dataset contains household energy consumption information.

### Input Features

```text
Household_Size
Avg_Temperature_C
Peak_Hours_Usage_kWh
```

### Target Variable

```text
Energy_Consumption_kWh
```

---

# Steps Performed

## Step 1 - Import Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

These libraries are used for:

* Data manipulation
* Numerical calculations
* Data visualization

---

## Step 2 - Load the Dataset

The dataset is loaded using Pandas.

```python
df = pd.read_csv("/content/household_energy_consumption.csv")
```

The dataset is stored in the DataFrame `df`.

---

## Step 3 - Display First Rows

```python
df.head()
```

This displays the first few records of the dataset.

It helps to understand the structure and values in the dataset.

---

## Step 4 - Display Last Rows

```python
df.tail()
```

This displays the last few records of the dataset.

---

## Step 5 - Check Dataset Shape

```python
df.shape
```

This returns the number of rows and columns in the dataset.

---

## Step 6 - Check Dataset Information

```python
df.info()
```

This provides information about:

* Column names
* Number of records
* Data types
* Non-null values

---

## Step 7 - Generate Statistical Summary

```python
df.describe()
```

This provides statistical information about the numerical columns, including:

* Count
* Mean
* Standard deviation
* Minimum
* Maximum
* Quartiles

---

## Step 8 - Check Data Types

```python
df.dtypes
```

This displays the data type of each column.

---

## Step 9 - Display Column Names

```python
df.columns
```

This displays all the column names available in the dataset.

---

## Step 10 - Check Missing Values

```python
df.isnull().sum()
```

This checks the number of missing values in each column.

---

## Step 11 - Remove Missing Values

Missing rows are removed using:

```python
df = df.dropna()
```

This ensures that the Machine Learning model does not contain missing values.

---

# Machine Learning

## Step 12 - Import Machine Learning Libraries

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

These libraries are used for:

* Splitting the dataset
* Polynomial feature generation
* Linear Regression
* Model evaluation

---

## Step 13 - Select Input Features and Target

The input features are:

```python
x = df[
    [
        "Household_Size",
        "Avg_Temperature_C",
        "Peak_Hours_Usage_kWh"
    ]
]
```

The target variable is:

```python
y = df["Energy_Consumption_kWh"]
```

Therefore:

```text
Input:
    Household Size
    Average Temperature
    Peak Hours Usage

        ↓

Polynomial Regression

        ↓

Output:
    Energy Consumption
```

---

## Step 14 - Split the Dataset

The dataset is divided into training and testing data.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
```

### Explanation

```text
80% → Training Data
20% → Testing Data
```

The model learns from the training data and is evaluated using the testing data.

---

## Step 15 - Create Polynomial Features

Polynomial features are generated using:

```python
poly = PolynomialFeatures(degree=2)

x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
```

The polynomial degree is:

```text
Degree = 2
```

This allows the model to learn non-linear relationships between the input variables and energy consumption.

---

## Step 16 - Create and Train the Model

A Linear Regression model is created:

```python
model = LinearRegression()
```

The model is trained using the polynomial features:

```python
model.fit(x_train_poly, y_train)
```

---

## Step 17 - Make Predictions

The trained model predicts energy consumption for the test data.

```python
y_pred = model.predict(x_test_poly)
```

The predicted values are stored in:

```text
y_pred
```

---

# Model Evaluation

## Step 18 - Calculate MAE

```python
mae = mean_absolute_error(y_test, y_pred)
```

**Mean Absolute Error (MAE)** measures the average absolute difference between actual and predicted energy consumption.

---

## Step 19 - Calculate MSE

```python
mse = mean_squared_error(y_test, y_pred)
```

**Mean Squared Error (MSE)** calculates the average squared difference between actual and predicted values.

---

## Step 20 - Calculate RMSE

```python
rmse = mse ** 0.5
```

**Root Mean Squared Error (RMSE)** is the square root of MSE.

---

## Step 21 - Calculate R² Score

```python
r2 = r2_score(y_test, y_pred)
```

**R² score** measures how well the model explains the variation in the target variable.

---

## Step 22 - Display Model Results

```python
print("\nPolynomial Regression Results")
print("-----------------")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE :", rmse)
print("R2 :", r2)
```

The output contains:

```text
Polynomial Regression Results
-----------------
MAE  : ...
MSE  : ...
RMSE : ...
R2   : ...
```

The actual values are generated when the notebook is executed.

---

# Actual vs Predicted Values

## Step 23 - Create Comparison Table

```python
result = pd.DataFrame({
    "Actual Energy": y_test.values,
    "Predicted Energy": y_pred
})

print("\nActual vs Predicted:")
print(result.head(10))
```

This compares the actual energy consumption with the values predicted by the model.

Example format:

```text
Actual Energy    Predicted Energy
-------------    ----------------
...              ...
...              ...
...              ...
```

---

# Visualization

## Step 24 - Actual vs Predicted Energy Consumption

The notebook creates a scatter plot to compare actual and predicted energy consumption.

```python
plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred,
    color="blue"
)

plt.xlabel("Actual Energy Consumption (kWh)")
plt.ylabel("Predicted Energy Consumption (kWh)")

plt.title("Actual vs Predicted Energy Consumption")

plt.show()
```

### Interpretation

* X-axis → Actual Energy Consumption
* Y-axis → Predicted Energy Consumption
* Each point represents a test observation.
* Points closer to the expected relationship indicate closer agreement between actual and predicted values.

---

# Colorful Visualization

If you want the output graph in color, use:

```python
plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred,
    color="blue",
    edgecolor="black",
    alpha=0.7
)

plt.xlabel("Actual Energy Consumption (kWh)")
plt.ylabel("Predicted Energy Consumption (kWh)")

plt.title(
    "Actual vs Predicted Energy Consumption",
    color="darkblue"
)

plt.grid(True, alpha=0.3)

plt.show()
```

To save the colorful graph as an image:

```python
plt.savefig(
    "actual_vs_predicted_energy.png",
    dpi=300,
    bbox_inches="tight"
)
```

This creates:

```text
actual_vs_predicted_energy.png
```

---

# Project Workflow

```text
Household Energy Dataset
          ↓
     Load Dataset
          ↓
   Explore Dataset
          ↓
 Check Missing Values
          ↓
 Remove Missing Values
          ↓
 Select Input Features
          ↓
   Select Target Value
          ↓
    Train-Test Split
          ↓
 Polynomial Features
     Degree = 2
          ↓
 Polynomial Regression
          ↓
      Prediction
          ↓
 Actual vs Predicted
          ↓
   Model Evaluation
          ↓
 MAE / MSE / RMSE / R²
```

---

# Project Structure

```text
Task4_ML/
│
├── Task4_ML.ipynb
├── household_energy_consumption.csv
├── README.md
└── actual_vs_predicted_energy.png
```

---

# How to Run

## Using Google Colab

1. Open `Task4_ML.ipynb` in Google Colab.
2. Upload `household_energy_consumption.csv`.
3. Make sure the dataset path is:

```text
/content/household_energy_consumption.csv
```

4. Run the cells from top to bottom.
5. Check the dataset information and statistical results.
6. Check the Polynomial Regression results.
7. Check the Actual vs Predicted visualization.

---

## Using Jupyter Notebook

Install the required libraries:

```bash
pip install numpy pandas matplot
```
