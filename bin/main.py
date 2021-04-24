####################################################
## Getting the data

from influxdb import InfluxDBClient # install via "pip install influxdb"
import pandas as pd

client = InfluxDBClient(host='influxus.itu.dk', port=8086, username='lsda', password='icanonlyread')
client.switch_database('orkney')

def get_df(results):
    values = results.raw["series"][0]["values"]
    columns = results.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns=columns).set_index("time")
    df.index = pd.to_datetime(df.index) # Convert to datetime-index
    return df

# Get the last 90 days of power generation data
generation = client.query(
    "SELECT * FROM Generation where time > now()-90d"
    ) # Query written in InfluxQL

# Get the last 90 days of weather forecasts with the shortest lead time
wind  = client.query(
    "SELECT * FROM MetForecasts where time > now()-90d and time <= now() and Lead_hours = '1'"
    ) # Query written in InfluxQL

forecasts = client.query(
    "SELECT * FROM MetForecasts where time > now()"
) # Query written in InfluxQL

gen_df = get_df(generation)
win_df = get_df(wind)
for_df = get_df(forecasts)

####################################################
## Pre pipeline processing

from icecream import ic
import numpy as np

def tminus(X, y):
    ic(X, y)

D = gen_df['Total']
D = win_df.resample('1Min').pad().merge(gen_df['Total'], how='left', right_index=True, left_index=True)
y = D['Total']
X = D[D.columns[D.columns != 'Total']]


P = for_df.to_numpy()
ic(P[:,1])
####################################################
## Preprocessing classes

from matplotlib import pyplot as plt
import seaborn as sns
import math

class Prepro:

    def __init__(self, piperun=True):
        self.piperun = piperun
 
    def fit(self, X, y=None): 
        return self

    def transform(self, X, y = None): 
        if 't0' not in X.columns:
            X = self.__augment(X, y)
        X = self.__vectorize(X)
        return X

    def __augment(self, X, y = None, n = 10): 
        if type(y) is type(None):
            y = get_df(generation)['Total'] 
        D = X.resample('1Min').pad() 
        D = pd.merge(D, y, how='right', right_index=True, left_index=True)
        D.rename(columns = {'Total': 't0'}, inplace=True)
        X = D.copy().iloc[n:, :]
        for i in range(1, n):
            delta = D['t0'][n - i : - i] 
            X[f"t-{i}"] = list(delta) 
        return X.dropna()

    def __vectorize(self, X):
        directions = ['E', 'ENE', 'NE', 'NNE', 'N', 'NNW', 'NW', 'WNW',
                  'W', 'WSW', 'SW', 'SSW', 'S', 'SSE', 'SE', 'ESE']
        radians = {d: (idx / 16) * 2 * math.pi for idx, d in enumerate(directions)}
        long = lambda direction: round(math.cos(radians[direction]), 3)
        lat  = lambda direction: round(math.sin(radians[direction]), 3) 
        X['long'] = X['Direction'].apply(long)
        X['lat'] = X['Direction'].apply(lat)
        X['long'] = X['long'].multiply(X['Speed'])
        X['lat'] = X['lat'].multiply(X['Speed']) 
        if self.piperun:
            X = X[X.columns[X.columns != 'Direction']]  
        return X
        

####################################################
## Construct Pipeline and declare params

from sklearn.pipeline import Pipeline
import pickle
from sklearn.svm import SVR

pipeline = Pipeline([
    ('prepro', Prepro(True)),
    ('model', SVR(max_iter=1000))
])

# Fit the pipeline
####################################################
## Running

#X, y = setup(wind_df, gen_df)

# gs.fit(X_train, y_train)
# print(gs.best_params_)
"""
X, y = wind_df, gen_df['Total'].dropna()
prepro = Prepro(False)
X = prepro.transform(X, y)
y = X['t0']
X = X[X.columns[X.columns != 't0']]
pipeline.fit(X, y)
"""
# Load stored model and compare with newly trained
# and store the best one
"""
p_new = pipeline.predict(X_test)

pipeline_old = pickle.load(open('../models/model.pkl', 'rb'))
p_old = pipeline_old.predict(X_test)

if mean_squared_error(p_new, y_test) < mean_squared_error(p_old, y_test):
pickle.dump(pipeline, open('../models/model.pkl', 'wb'))
print(mean_squared_error(p_old, y_test))
print(mean_squared_error(p_new, y_test))
print('NEW')
else:
pipeline = pipeline_old
print(mean_squared_error(p_old, y_test))
print(mean_squared_error(p_new, y_test))
print('OLD')
"""
####################################################
## Do forecasting with the best one

# Get all future forecasts regardless of lead time
forecasts = client.query(
"SELECT * FROM MetForecasts where time > now()"
) # Query written in InfluxQL
#forecast = get_df(forecasts).to_numpy()
# Limit to only the newest source time
#newest_source_time = for_df["Source_time"].max()
#newest_forecasts = for_df.loc[for_df["Source_time"] == newest_source_time].copy()

# Preprocess the forecasts and do predictions in one fell swoop 
# using your best pipeline.

