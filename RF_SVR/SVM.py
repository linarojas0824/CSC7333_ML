import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


'''
Load/ preprocess
'''
data = pd.read_csv('data/clean_RawData.csv')
if 'INDEX' in data.columns:
    data = data.drop('INDEX', axis=1)
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)
data.fillna(data.mean(), inplace=True)
print(data.head())

X = data.drop('XMEAS10', axis=1) 
y = data['XMEAS10']   

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

'''
SVR
'''
svr = SVR(kernel='rbf')


param_grid_svr = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    'epsilon': [0.1, 0.2, 0.5, 1]  # Epsilon in the epsilon-SVR model
}



grid_search_svr = GridSearchCV(estimator=svr, param_grid=param_grid_svr, cv=3, n_jobs=-1, verbose=2)
grid_search_svr.fit(X_train, y_train)
best_svr = grid_search_svr.best_estimator_


y_train_pred = best_svr.predict(X_train)
y_test_pred = best_svr.predict(X_test)

# Calculate RMSE and R-squared
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Training RMSE: {rmse_train}")
print(f"Testing RMSE: {rmse_test}")
print(f"Training R2: {r2_train}")
print(f"Testing R2: {r2_test}")

# Scatter plot for testing data
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Scatter Plot of Predicted vs Actual Values (Test Data)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.text(0.1, 0.9, f'R-squared: {r2_test:.4f}', transform=plt.gca().transAxes, 
         fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))
plt.grid(True)
plt.savefig('Kyle/SVR_test_scatter_plot.png', format='png')
plt.show()

# Scatter plot for training data
plt.figure(figsize=(10, 10))
plt.scatter(y_train, y_train_pred, alpha=0.5)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
plt.title('Scatter Plot of Predicted vs Actual Values (Training Data)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.text(0.1, 0.9, f'R-squared: {r2_train:.4f}', transform=plt.gca().transAxes, 
         fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))
plt.grid(True)
plt.savefig('Kyle/SVR_training_scatter_plot.png', format='png')
plt.show()

