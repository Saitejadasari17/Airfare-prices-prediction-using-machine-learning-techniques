# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# 2. Create Synthetic Dataset
np.random.seed(42)

# Simulate a dataset
n_samples = 2000

data = {
    'Airline': np.random.choice(['IndiGo', 'Air India', 'SpiceJet', 'Vistara', 'GoAir'], n_samples),
    'Source': np.random.choice(['Delhi', 'Kolkata', 'Mumbai', 'Chennai', 'Bangalore'], n_samples),
    'Destination': np.random.choice(['Cochin', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata'], n_samples),
    'Total_Stops': np.random.choice(['non-stop', '1 stop', '2 stops', '3 stops'], n_samples, p=[0.4, 0.4, 0.15, 0.05]),
    'Duration': np.random.randint(60, 720, n_samples), # in minutes
    'Journey_day': np.random.randint(1, 31, n_samples),
    'Journey_month': np.random.randint(1, 13, n_samples),
    'Dep_hour': np.random.randint(0, 24, n_samples),
    'Price': np.random.randint(3000, 20000, n_samples)
}

df = pd.DataFrame(data)

print(df.head())

# 3. Preprocessing

# Label Encoding for categorical features
categorical_cols = ['Airline', 'Source', 'Destination', 'Total_Stops']

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Features and Target
X = df.drop('Price', axis=1)
y = df['Price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (needed for SVMs)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Different Models

models = {
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf'),
    'Linear SVR': LinearSVR(random_state=42),
    'Bagging Regressor': BaggingRegressor(n_estimators=50, random_state=42)
}

results = {}

for name, model in models.items():
    if 'SVR' in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2 Score': r2}

# 5. Results

results_df = pd.DataFrame(results).T
print("\n\n📊 Model Performance Comparison:\n")
print(results_df.sort_values('R2 Score', ascending=False))

# 6. Visualization

plt.figure(figsize=(10,6))
sns.barplot(x=results_df.index, y=results_df['R2 Score'])
plt.title('Model R² Scores')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.grid()
plt.show()
