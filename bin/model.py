# model.py
#   orkney data pipeline
# by: Noah Syrkis

# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
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
        "model__max_depth": [10, 13, 15, 30],
        "model__n_estimators": [100, 200, 300, 400]
    }


pipe = Pipeline([
        ('director', Director()),
        ('scalar', StandardScaler()),
        ('model', RandomForestRegressor())
    ])

clf = GridSearchCV(pipe, params)

clf.fit(X_train, y_train)
p = clf.predict(X_test)
mse = mean_squared_error(y_test, p) ** (1 / 2)
pickle.dump(pipe, open('../models/model.pkl', 'wb'))
