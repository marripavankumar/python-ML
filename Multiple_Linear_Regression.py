# Multiple Linear Regression using scikit-learn

import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1 - Create a synthetic dataset with multiple features
np.random.seed(0)
n_samples = 100
X1 = np.random.rand(n_samples) * 10
X2 = np.random.rand(n_samples) * 5
# True relationship: y = 4*X1 + 2*X2 + noise
y = 4 * X1 + 2 * X2 + np.random.normal(0, 2, n_samples)

# Combine features into a single matrix
X = np.column_stack((X1, X2))

# Step 2 - Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3 - Create and train the multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4 - Make predictions
y_pred = model.predict(X_test)

# Step 5 - Visualize the results (scatter plot: actual vs predicted)
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicted vs Actual'))
fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal Fit', line=dict(dash='dash')))
fig.update_layout(
    title='Multiple Linear Regression: Actual vs Predicted',
    xaxis_title='Actual y',
    yaxis_title='Predicted y'
)
fig.show()

# Step 6 - Print model coefficients and score
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
print(f"Model Score (R^2): {model.score(X_test, y_test)}")

# Step 7 - Save the model
import joblib
joblib.dump(model, "multiple_regression_model.pkl")