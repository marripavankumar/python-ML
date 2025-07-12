import pandas as pd 
from sklearn.datasets import load_diabetes

data = load_diabetes()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['target'])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import Ridge
reggressor = Ridge(alpha=1.0)
reggressor.fit(x_train, y_train)

# Generate predictions on the test set
y_pred = reggressor.predict(x_test)


import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test['target'], y = y_pred.flatten(), mode='markers'))
fig.add_trace(go.Scatter(x=[0,350],y= [0, 350] ,mode= 'lines' ))
fig.update_layout(title='Ridge Regression Predictions vs Actual',
                  xaxis_title='Actual Values',
                  yaxis_title='Predicted Values')
fig.show()
# Print model coefficients and score
print(f"Model Coefficients: {reggressor.coef_}")
print(f"Model Intercept: {reggressor.intercept_}")
print(f"Model Score (R^2): {reggressor.score(x_test, y_test)}")

# Save the model
import joblib
joblib.dump(reggressor, "ridge_regression_model.pkl")

