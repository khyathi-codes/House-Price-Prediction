# House Price Prediction
A Machine Learning-based web application that predicts house prices using property characteristics and geographical information. The project trains and compares multiple regression models and provides interactive predictions through a Streamlit interface.
##  Project Overview
This project aims to estimate house prices based on various features such as median income, housing age, number of rooms, population, and ocean proximity.
Three machine learning algorithms are implemented and evaluated to identify the best-performing prediction model.
##  Features
- Predict house prices using property details.
- Compare predictions from three machine learning models.
- Evaluate model performance using R², MAE, and RMSE.
- Visualize feature importance.
- Explore the California Housing dataset.
- Interactive web interface built using Streamlit.
##  Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib / Data Visualization
- Joblib
##  Machine Learning Algorithms
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
## Model Performance
The models were evaluated using an 80:20 train-test split.
| Model | R² Score | MAE | RMSE |
|---|---:|---:|---:|
| Linear Regression | 0.5970 | $50,889 | $72,669 |
| Decision Tree | 0.7306 | $39,034 | $59,414 |
| Random Forest | **0.8092** | **$32,212** | **$49,997** |
**Best Performing Model: Random Forest Regressor**
## Project Structure
```text
House-Price-Prediction/
│
├── app.py
├── train.py
├── models.joblib
├── metrics.json
├── requirements.txt
├── README.md
│
└── data/
    └── housing.csv
```
##  Installation and Execution
### 1. Clone the repository
```bash
git clone https://github.com/khyathi-codes/House-Price-Prediction.git
```
### 2. Navigate to the project directory

```bash
cd House-Price-Prediction
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the Streamlit application
```bash
streamlit run app.py
```
### 5. Open the application
Open the local URL displayed in your terminal.
##  Dataset
The project uses the California Housing dataset containing information about California housing districts from the 1990 census.
Dataset source: https://github.com/ageron/handson-ml2/tree/master/datasets/housing
## Disclaimer
The predictions are based on historical California housing data from 1990 and are intended for educational and demonstration purposes only. They should not be considered actual current property valuations.
##  Author
**Khyathi Adusumalli**
B.Sc. Data Science Student | VIT-AP
- GitHub: https://github.com/khyathi-codes
