# pred.py
#   makes predictions about orkney yay!
# by: Noah Syrkis

# imports
from prep import X, y, P
from sklearn.metrics import mean_squared_error
import pickle
from icecream import ic

# test split
X_test, y_test = X[500:], y[500:]

# load pipe
regr = pickle.load(open('../models/model.pkl', 'rb'))
p = regr.predict(X_test)
mse = mean_squared_error(y_test, p)

ic(mse)
