import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('climate_crop_data.csv')
print(df.head())  # Debugging: Print first few rows

# Check if DataFrame is empty
if df.empty:
    print("Error: The dataset is empty. Check if the CSV file is correctly loaded.")
    exit()

# Data Preprocessing
def preprocess_data(df):
    df = df.dropna()  
    df = df.select_dtypes(include=[np.number])  
    return df

# Feature Selection
def feature_target_split(df):
    df.columns = df.columns.str.strip()  
    X = df.iloc[:, :-1]  
    y = df["Crop Yield (tons/ha)"]  # Explicitly select target column
    return X, y

# Model Training and Evaluation
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Model Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:\nMAE: {mae}\nRMSE: {rmse}\nR2 Score: {r2}")
    return model

# Visualization
def plot_feature_importance(model, X):
    feature_importances = model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.show()

# Main Execution
if __name__ == "__main__":
    df = preprocess_data(df)
    X, y = feature_target_split(df)
    model = train_model(X, y)
    plot_feature_importance(model, X)

