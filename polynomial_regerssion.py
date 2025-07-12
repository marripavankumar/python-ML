import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(0)
x= np.linspace(0,10,100)
y = 3*x**2+2*x+ np.random.normal(0,10,100) 

x = x.reshape(-1, 1)  # Reshape for sklearn
y = y.reshape(-1, 1)  # Reshape for sklearn

#split the data into training and testing sets
x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  

#Transform the features to polynomial features  
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

#Create and train the polynomial regression model
model = LinearRegression()
model.fit(x_train_poly, y_train)

#Generate predictions on the test set
y_train_pred = model.predict(x_test_poly)
y_test_pred = model.predict(x_test_poly)    

#Visualize the results
import plotly.graph_objects as go

# Create a scatter plot for the training data
trace_train = go.Scatter(x=x_train.flatten(), y=y_train.flatten(), mode='markers', name='Train Data', marker=dict(color='blue'))

# Create a trace for the test data
trace_test = go.Scatter(x=x_test.flatten(), y=y_test.flatten(), mode='markers', name='Test Data', marker=dict(color='green'))

# line for the polynomial regression model
trace_regression = go.Scatter(x=x_test.flatten(),y=y_test_pred.flatten(), mode='lines', name='Polynomial Regression', line=dict(color='red', width=2 ))

# Create the layout
layout = go.Layout(
    title='Polynomial Regression',
    xaxis=dict(title='X'),
    yaxis=dict(title='Y')
)

# comnbine all traces into a figure
figure = go.Figure(data=[trace_train, trace_test, trace_regression], layout=layout)
figure.show()

# Print model coefficients and score
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
print(f"Model Score (R^2): {model.score(x_test_poly, y_test)}") 
# Save the model
import joblib
joblib.dump(model, "polynomial_regression_model.pkl")



