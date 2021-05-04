# model.py
#   orkney data pipeline
# by: Noah Syrkis

# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from director import Director
from prep import X, y, P
from icecream import ic
import mlflow
from matplotlib import pyplot as plt
import pickle

X_train, X_test = X[:600], X[600:]
y_train, y_test = y[:600], y[600:]

# pipeline definition
mlflow.sklearn.autolog()
params = {

        "poly__degree": [i for i in range(1, 10)]
    }

pipe = Pipeline([
        ('director', Director()),
        ('scalar', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('model', LinearRegression())
    ])

tscv = TimeSeriesSplit(n_splits=5)

regr = GridSearchCV(estimator=pipe, cv=tscv, param_grid=params)

regr.fit(X_train, y_train)
p = regr.predict(X_test)
mse = mean_squared_error(y_test, p)
ic(mse)
mlflow.sklearn.log_model(regr, 'model')
