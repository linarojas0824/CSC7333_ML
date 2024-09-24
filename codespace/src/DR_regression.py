'''
CODE SOURCE
'''

# Import necessary libraries and modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pacmap
from hdbscan import HDBSCAN

'''
PREPROCESSING DATA
'''

# Load the data from a CSV file 
datafile = pd.read_csv('../../data/TEP0_1_time.csv')
print(datafile.head())

# Check if 'INDEX' column exists and drop it if it does
if 'INDEX' in datafile.columns:
    datafile = datafile.drop('INDEX', axis=1)

# Preprocess data
datafile.fillna(datafile.mean(), inplace=True)

# Define target and predictor variables
targetlist = ["VA1"]
predictorlist = ["VA2", "VA3", "VA4", "VA5"]

# Separate predictors (X) and target (y)
X = datafile[predictorlist].values
y = datafile[targetlist].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.ravel()
y_test = y_test.ravel()

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

'''
DR AND CLUSTERING (based on X_train_scaled only)
'''
sns.set(style="whitegrid")
colors = sns.color_palette("Set1")


'''
(1) PCA and KMEANS
'''
# PCA
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_train_scaled)

# K-Means
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
kmeans.fit(X_pca) 
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Combined PCA and K-Means Plot
plt.figure(figsize=(10, 8))
for i, cluster in enumerate(np.unique(labels)):
    plt.scatter(X_pca[labels == cluster, 0], 
                X_pca[labels == cluster, 1], 
                c=[colors[i % len(colors)]], 
                label=f'Cluster {cluster}',
                alpha=0.7)

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Dataset with K-Means Clustering')
plt.legend()
sns.despine()
plt.tight_layout()
plt.show()

'''
(2) PaCMAP and HDBSCAN
'''
# PaCMAP
embedding = pacmap.PaCMAP() 
X_pacmap = embedding.fit_transform(X_train_scaled)

# HDBSCAN
clusterer = HDBSCAN(min_cluster_size=30, min_samples=20)
cluster_labels = clusterer.fit_predict(X_pacmap)

# HDBSCAN Plot
plt.figure(figsize=(10, 8))
unique_clusters = np.unique(cluster_labels)
for i, cluster in enumerate(unique_clusters):
    if cluster == -1: 
        color = 'grey'
        label = 'Noise'
    else:
        color = colors[i % len(colors)]
        label = f'Cluster {cluster}'
    plt.scatter(X_pacmap[cluster_labels == cluster, 0], 
                X_pacmap[cluster_labels == cluster, 1], 
                c=[color], 
                label=label,
                alpha=0.7)

plt.xlabel('PaCMAP Dimension 1')
plt.ylabel('PaCMAP Dimension 2')
plt.title('HDBSCAN Clustering on PaCMAP-Reduced Data')
plt.legend(markerscale=2)
sns.despine()
plt.tight_layout()
plt.show()


'''
PREDICTIVE MODELING
'''
# List of regression models to try
regression_models = [
    LinearRegression(),
    Ridge(),
    Lasso(),
    ElasticNet(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    AdaBoostRegressor(),
    SVR()
]

# Initialize dictionaries to store results
accuracy_metrics = {}
mse_results = {}
r_squared_results = {}

# Train and evaluate each model
for model in regression_models:
    model_name = model.__class__.__name__
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    mse_results[model_name] = mse
    r_squared = model.score(X_test_scaled, y_test)
    r_squared_results[model_name] = r_squared
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    mpe = np.mean((y_test - predictions) / y_test) * 100
    
    # Store accuracy metrics for each model
    accuracy_metrics[model_name] = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R-squared': r_squared,
        'MAPE': mape,
        'MPE': mpe
    }

# Rank models by MSE and R-squared
sorted_models_mse = sorted(mse_results.items(), key=lambda x: x[1])
top_models_mse = [model_name for model_name, _ in sorted_models_mse[:5]]
sorted_models_r_squared = sorted(r_squared_results.items(), key=lambda x: x[1], reverse=True)
top_models_r_squared = [model_name for model_name, _ in sorted_models_r_squared[:3]]

# Format and print evaluation metrics for the top models
m_outputs = []
for model_name in top_models_mse:
    metrics = accuracy_metrics[model_name]
    metrics_output = (
        f"Model: {model_name}\n"
        f"R-squared: {metrics['R-squared']:.5f}\n"
        f"MSE: {metrics['MSE']:.5f}\n"
        f"MAE: {metrics['MAE']:.5f}\n"
        f"RMSE: {metrics['RMSE']:.5f}\n"
        f"MAPE: {metrics['MAPE']:.5f}%\n"
        f"MPE: {metrics['MPE']:.5f}%\n"
        + "=" * 60 + "\n"
    )
    m_outputs.append(metrics_output)

for output in m_outputs[:5]:
    print(output)

# Visualizations
sns.set(style="whitegrid")
colors = sns.color_palette("Set1")

# Scatter Plot for Actual vs. Predicted Values
plt.figure(figsize=(10, 6))
for i, model_name in enumerate(top_models_mse):
    model = [m for m in regression_models if m.__class__.__name__ == model_name][0]
    predictions = model.predict(X_test_scaled)
    plt.scatter(y_test, predictions, label=model_name, color=colors[i], alpha=0.7)

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='gray')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.legend()
sns.despine()
plt.tight_layout()
plt.show()

# Residual Plot for the top models
plt.figure(figsize=(10, 6))
for i, model_name in enumerate(top_models_mse):
    model = [m for m in regression_models if m.__class__.__name__ == model_name][0]
    predictions = model.predict(X_test_scaled)
    residuals = predictions - y_test
    plt.scatter(y_test, residuals, label=model_name, color=colors[i], alpha=0.7)

plt.axhline(y=0, color='gray', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.legend()
sns.despine()
plt.tight_layout()
plt.show()