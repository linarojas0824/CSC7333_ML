# CSC7333_ML
**Data-Driven Predictive Modeling for Chemical Plants: A Comparative Analysis of ML models** <br>
<br>
This project focuses on the application of machine learning (ML) to enhance operations in chemical plants. Employing unsupervised learning techniques such as dimension reduction and
clustering, our goal is to simplify multivariable datasets containing information on operational conditions such as flow, rate, temperature, pressure, and level, while identifying inherent patterns in the data. Following that, supervised learning models for classification and regression will be developed. <br>
<br>
Simulated plant data was used for training and evaluation, with ML tools and libraries, such as scikit-learn, PyTorch, TensorFlow, and Python for model development and analysis. <br>

**Design Implementation** <br>
<br>
**Dimensionality Reduction** <br>
- Pairwise Controlled Manifold Approximation Projection (PaCMAP)<br>
- Principal Component Analysis (PCA)<br>

**Clustering** <br>
- HDBSCAN <br>
- K- means <br>

**Supervised Learning** <br>
Four representative algorithms were thoroughly investigated; support vector regression (SVR), decision-based regression (DT), Artificial Neural Networks (ANN), and Deep Learning (DL).<br>
Hyperparameter optimization was conducted through a grid search.<br>
  Parameters:<br>
  - Number of estimators<br>
  - Maximum tree depth<br>
  - Minimum samples required for node splitting<br>
Performance evaluation employing Root Mean Squared Error (RMSE) and R-squared (R2)    





