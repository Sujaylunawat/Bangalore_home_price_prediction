# 🏡 Bengaluru House Price Prediction using Machine Learning

An end-to-end Machine Learning project focused on predicting real estate prices in Bengaluru using advanced data preprocessing, feature engineering, outlier detection, and regression modeling techniques.

This project demonstrates the complete workflow of a real-world Data Science pipeline — from raw dataset cleaning to model evaluation and price prediction.

---

## 🚀 Project Highlights

* 📊 Data Cleaning & Preprocessing
* 🧠 Feature Engineering
* 📉 Outlier Detection & Removal
* 🏘️ Location-Based Price Analysis
* 🤖 Multiple ML Models Comparison
* 📈 Cross Validation & GridSearchCV
* 🔮 Real-Time House Price Prediction

---

## 📌 Problem Statement

Real estate prices vary significantly based on:

* Location
* Square feet area
* Number of BHKs
* Bathrooms
* Market demand

The goal of this project is to build a Machine Learning model capable of accurately predicting house prices in Bengaluru using historical housing data.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* Jupyter Notebook

---

## 📂 Workflow

### 1️⃣ Data Cleaning

* Removed unnecessary columns
* Handled missing values
* Converted ranges in `total_sqft` into numeric values

### 2️⃣ Feature Engineering

* Extracted BHK values from size column
* Created Price Per Square Foot (PPSQ)
* Reduced dimensionality in location feature

### 3️⃣ Outlier Detection

Removed:

* Unrealistic square feet per BHK entries
* PPSQ anomalies using standard deviation
* Irregular bathroom configurations

### 4️⃣ Model Building

Implemented:

* Linear Regression
* Lasso Regression
* Decision Tree Regressor

### 5️⃣ Hyperparameter Tuning

Used:

* GridSearchCV
* ShuffleSplit Cross Validation

### 6️⃣ Prediction System

Created a custom prediction function to estimate property prices based on:

* Location
* Square Feet
* Bathrooms
* BHK

---

## 📈 Model Performance

The Linear Regression model achieved strong predictive performance after extensive preprocessing and outlier handling.

Evaluation techniques used:

* Train-Test Split
* Cross Validation
* Model Comparison using GridSearchCV

---

## 🔮 Example Prediction

```python
predict_price('Indira Nagar',1000,3,3)
```

Predicts the estimated price of a 3 BHK house in Indira Nagar with 1000 sqft area and 3 bathrooms.

---

## 📁 Dataset

Dataset used:

* Bengaluru House Prices Dataset

Contains:

* Area Type
* Location
* Total Square Feet
* BHK
* Bathrooms
* Price

---

## 🎯 Key Learnings

Through this project, I learned:

* Real-world data preprocessing
* Feature engineering techniques
* Handling missing & inconsistent data
* Outlier detection strategies
* Regression modeling
* Hyperparameter tuning
* End-to-end ML workflow implementation

---

## 📌 Future Improvements

* Deploy as a web application using Streamlit
* Add interactive UI
* Integrate advanced ensemble models
* Improve prediction accuracy
* Add visualization dashboard

---

## 🤝 Connect With Me

If you liked this project or have suggestions, feel free to connect and collaborate.

---

