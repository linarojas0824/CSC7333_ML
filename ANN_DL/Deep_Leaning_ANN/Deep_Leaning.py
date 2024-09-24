# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:04:37 2023

@author: mattia1
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Record the start time
start_time = time.time()

# Load data into a DataFrame for VA1 model
df_VA1 = pd.read_csv("TEP0_1_time.csv")
df_VA1 = df_VA1.drop("INDEX", axis=1)


# Assume 'VA1' is the target column to predict
target_column = 'VA1'

# Extract input features and target variable
X = df_VA1.drop(target_column, axis=1)
y = df_VA1[target_column]

# Scale the input features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define a neural network model for VA1 prediction
class VA1Model(nn.Module):
    def __init__(self):
        super(VA1Model, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 32)  # Input size based on the number of features
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 64)
        self.fc6 = nn.Linear(64, 1)  # Output size is 1 for a regression task

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

# Create and train a VA1 model
model_VA1 = VA1Model()
criterion_VA1 = nn.MSELoss()
optimizer_VA1 = optim.Adam(model_VA1.parameters(), lr=0.001)

n_epochs = 200
loss_values = []

for epoch in range(n_epochs):
    inputs_VA1 = torch.tensor(X_train, dtype=torch.float32)
    labels_VA1 = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    optimizer_VA1.zero_grad()
    outputs_VA1 = model_VA1(inputs_VA1)
    loss_VA1 = criterion_VA1(outputs_VA1, labels_VA1)
    loss_VA1.backward()
    optimizer_VA1.step()

    # Append the loss value for plotting
    loss_values.append(loss_VA1.item())

    # Plot the loss curve for this iteration for VA1 model
    plt.figure(figsize=(8, 4))
    plt.plot(loss_values, label=f"VA1 Iteration {epoch+1}")
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title(f"Training Loss Curve (VA1 Iteration {epoch+1})")
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.svg', format='svg')
    plt.show()

# Evaluate the model on the test data
model_VA1.eval()
with torch.no_grad():
    test_inputs_VA1 = torch.tensor(X_test, dtype=torch.float32)
    predictions_VA1 = model_VA1(test_inputs_VA1).numpy()

# Calculate RMSE for the model on the test data
rmse_VA1 = np.sqrt(mean_squared_error(y_test, predictions_VA1))
print(f"Root Mean Squared Error on Test Data for VA1 Model: {rmse_VA1:.2f}")

# Save the model
torch.save(model_VA1.state_dict(), 'best_model_VA1.pth')

# Predict on training data for the VA1 model
train_inputs_VA1 = torch.tensor(X_train, dtype=torch.float32)
train_predictions_VA1 = model_VA1(train_inputs_VA1).detach().numpy()


# Calculate RMSE for training data for the VA1 model
rmse_train_VA1 = np.sqrt(mean_squared_error(y_train, train_predictions_VA1))
print(f"Root Mean Squared Error on Training Data for VA1 Model: {rmse_train_VA1:.2f}")

# Check for overfitting by comparing training and test RMSE for the VA1 model
if rmse_train_VA1 < rmse_VA1:
    print("VA1 Model may be overfitting (Training RMSE < Test RMSE)")
else:
    print("VA1 Model is not overfitting (Training RMSE >= Test RMSE)")

# Calculate R2 score for the VA1 model
r2_test_VA1 = r2_score(y_test, predictions_VA1)
r2_train_VA1 = r2_score(y_train, train_predictions_VA1)

# Print R2 score for the VA1 model
print(f"R-squared (R2) Score on Test Data for the VA1 Model: {r2_test_VA1:.2f}")
print(f"R-squared (R2) Score on Training Data for the VA1 Model: {r2_train_VA1:.2f}")

# Scatter plot for VA1 test data
plt.figure(figsize=(10, 10))
plt.scatter(y_test, predictions_VA1, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Scatter Plot of VA1 Values (Test Data)')
plt.xlabel('Actual Ln VA1 at Groundwater Wells')
plt.ylabel('Predicted Ln VA1 at Groundwater Wells')
plt.grid(True)
plt.savefig('test_scatter_plot_VA1.svg', format='svg')
plt.show()

# Scatter plot for VA1 training data
plt.figure(figsize=(10, 10))
plt.scatter(y_train, train_predictions_VA1, alpha=0.5)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
plt.title('Scatter Plot of VA1 Values (Training Data)')
plt.xlabel('Actual Ln VA1 at Groundwater Wells')
plt.ylabel('Predicted Ln VA1 at Groundwater Wells')
plt.grid(True)
plt.savefig('training_scatter_plot_VA1.svg', format='svg')
plt.show()



# Accessing the weights from the first layer (fc1)
weights_fc1 = model_VA1.fc1.weight.detach().numpy()

# Calculate the absolute values of the weights for relative importance
absolute_weights_fc1 = np.abs(weights_fc1)

# Calculate the relative importance by dividing each weight by the sum of all weights
relative_importance_fc1 = absolute_weights_fc1 / absolute_weights_fc1.sum()

# Get the feature names
feature_names = X.columns

# Rank the features based on their relative importance
ranked_features = sorted(zip(feature_names, relative_importance_fc1[0]), key=lambda x: x[1], reverse=False)

# Separate the names and importance values for plotting
ranked_feature_names, ranked_importance_values = zip(*ranked_features)

# Plotting
plt.figure(figsize=(12, 18))
plt.barh(ranked_feature_names, ranked_importance_values)
plt.xlabel('Relative Importance')
plt.title('Relative Influence of Variables (Ranked)')
plt.grid(axis='x')
plt.savefig('Relative_importance_VA1.svg', format='svg')
plt.show()

