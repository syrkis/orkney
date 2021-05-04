# model.py
#   orkney data pipeline
# by: Noah Syrkis

# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from director import Director
from prep import X, y, P
from icecream import ic
import mlflow
from matplotlib import pyplot as plt
import pickle

X_train, X_test = X[:500], X[500:]
y_train, y_test = y[:500], y[500:]

# pipeline definition
mlflow.sklearn.autolog()
params = {

        "model__max_depth": [i for i in range(1, 7)],
        "model__n_estimators": [100, 150, 200, 250]
    }

pipe = Pipeline([
        ('director', Director()),
        ('scalar', StandardScaler()),
        ('model', RandomForestRegressor())
    ])

tscv = TimeSeriesSplit(n_splits=5)

regr = GridSearchCV(estimator=pipe, cv=tscv, param_grid=params)

regr.fit(X_train, y_train)
p = regr.predict(X_test)
mse = mean_squared_error(y_test, p)
ic(mse)
pickle.dump(regr, open('models/model.pkl', 'wb'))
