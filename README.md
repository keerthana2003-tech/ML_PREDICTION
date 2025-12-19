# ML_PREDICTION 📊🤖

ML_PREDICTION is a Machine Learning–based web application that demonstrates the practical implementation of multiple supervised and unsupervised learning algorithms using **Python**, **Flask**, and **Scikit-learn**.  
The project provides an interactive user interface where users can input real-world data and obtain predictions from different ML models.

---

## 🔍 Project Overview

This project integrates **seven popular machine learning algorithms**, each applied to a meaningful real-world problem.  
The application allows users to explore how different algorithms work by entering input features and viewing predicted outputs instantly through a web interface.
<img width="1097" height="439" alt="front" src="https://github.com/user-attachments/assets/ec498d08-2821-4781-b4e6-4f5751ac758b" />


---

## 🧠 Machine Learning Models Implemented

### 1️⃣ Linear Regression – *House Price Prediction*
- **Type:** Supervised Learning (Regression)
- **Objective:** Predict house prices based on property features
- **Input Features:**
  - Number of bedrooms
  - Number of bathrooms
  - Square footage
  - Lot size
  - Number of floors
- **Description:**  
  Linear Regression models the relationship between dependent and independent variables using a straight-line equation. It is used here to estimate house prices based on numerical inputs.
<img width="586" height="622" alt="linear" src="https://github.com/user-attachments/assets/8d819b3b-9ae3-4c6e-b00d-a33e5182246b" />

---

### 2️⃣ Logistic Regression – *10-Year CHD Risk Prediction*
- **Type:** Supervised Learning (Classification)
- **Objective:** Predict the risk of Coronary Heart Disease (CHD)
- **Input Features:**
  - Age
  - Smoking status
  - Cigarettes per day
  - Cholesterol level
  - Blood pressure
  - BMI
- **Description:**  
  Logistic Regression is used for binary classification. The model predicts whether a person is at risk of developing heart disease within 10 years based on clinical data.
<img width="471" height="635" alt="logistic" src="https://github.com/user-attachments/assets/d10334c4-f19e-4b2d-8ffb-58f6762b680e" />

---

### 3️⃣ Decision Tree Regressor – *Gold Closing Rate Prediction*
- **Type:** Supervised Learning (Regression)
- **Objective:** Predict gold closing prices
- **Input Features:**
  - Opening price
  - High price
  - Low price
- **Description:**  
  Decision Trees split data into branches based on feature values, making them easy to interpret. This model predicts gold prices based on OHLC (Open, High, Low) data.
<img width="489" height="464" alt="decision" src="https://github.com/user-attachments/assets/a9717813-b3b1-4d2d-bba1-a5175d588cbf" />

---

### 4️⃣ Support Vector Machine (SVM) – *Breast Cancer Classification*
- **Type:** Supervised Learning (Classification)
- **Objective:** Classify tumors as benign or malignant
- **Input Features:**
  - Radius mean
  - Texture mean
  - Perimeter mean
  - Area mean
  - Smoothness mean
- **Description:**  
  SVM finds an optimal hyperplane that separates data into different classes. It is widely used in medical diagnosis due to its high accuracy in classification tasks.
<img width="476" height="621" alt="svm" src="https://github.com/user-attachments/assets/04dbdf26-9133-411e-953e-c83e1a8a241d" />

---

### 5️⃣ K-Nearest Neighbors (KNN) – *Wine Quality Prediction*
- **Type:** Supervised Learning (Classification)
- **Objective:** Predict wine quality score
- **Input Features:**
  - Volatile acidity
  - Citric acid
  - Density
  - pH
  - Sulphates
  - Alcohol
- **Description:**  
  KNN classifies data based on similarity to its nearest neighbors. Wine quality is predicted by comparing chemical properties with similar wine samples.
<img width="502" height="634" alt="knn" src="https://github.com/user-attachments/assets/992881f7-68e5-41df-b20b-bd42d1f89e97" />

---

### 6️⃣ K-Means Clustering – *Animal Classification*
- **Type:** Unsupervised Learning (Clustering)
- **Objective:** Group animals based on physical traits
- **Input Features:**
  - Hair
  - Feathers
  - Eggs
  - Milk
  - Airborne
  - Aquatic
  - Predator
- **Description:**  
  K-Means clusters data into groups without predefined labels. Animals are grouped based on shared characteristics, demonstrating unsupervised learning.
<img width="485" height="638" alt="k means" src="https://github.com/user-attachments/assets/09c7667d-a55f-4b03-8b94-f44428d9f24a" />

---

### 7️⃣ Random Forest – *BMI Prediction*
- **Type:** Supervised Learning (Regression)
- **Objective:** Predict Body Mass Index (BMI)
- **Input Features:**
  - Gender
  - Height
  - Weight
- **Description:**  
  Random Forest is an ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting. It is used here to estimate BMI reliably.
<img width="457" height="499" alt="random" src="https://github.com/user-attachments/assets/e551daf0-1f5f-4f12-99f3-60d417c00bd2" />

---

## 🛠️ Technologies Used
- Python
- Flask
- Scikit-learn
- HTML5
- CSS3
- Git & GitHub

---

## 🌐 Application Features
- User-friendly web interface
- Real-time predictions
- Multiple ML models in one application
- Clean and modular project structure
- Practical real-world datasets

---

## ▶️ How to Run the Project

```bash
git clone https://github.com/your-username/ML_PREDICTION.git
cd ML_PREDICTION
pip install -r requirements.txt
python main.py
