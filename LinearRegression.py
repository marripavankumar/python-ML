# implement Linerar regression using scikit learning  
import numpy  as np 
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as linearRegression


#step1 - create a complex dataset   
np.random.seed(0)
x = np.linspace(0, 10, 100) # indepenent variable
y = 3 * x**2 + 2 * x + np.random.normal(0, 10,100) # dependent variable with noise


#step2 - split the dataset into training and testing sets   
X_train, X_test, y_train, y_test = train_test_split(x , y, test_size=0.2, random_state=42)

#step3 - create a linear regression model
model = linearRegression()
X_train = X_train.reshape(-1, 1)  # Reshape X_train to 2D array
model.fit(X_train, y_train)

#step4 - make predictions
X_test = X_test.reshape(-1, 1)  # Reshape X_test to 2D array
y_pred = model.predict(X_test)

#step5 - visualize the results
fig = go.Figure()
fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_test, mode='markers', name='Actual Data'))
fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_pred, mode='lines', name='Linear Regression Fit'))

fig.show()
#step6 - print the model coefficients
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
print(f"Model Score: {model.score(X_test, y_test)}")
#step7 - save the model
import joblib