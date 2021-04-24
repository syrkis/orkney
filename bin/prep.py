# prepare.py
#   preproceses data frames for training and predicting
# by: Noah Syrkis

# imports
from read import wind_df, gen_df, fore_df
import numpy as np
from icecream import ic
from matplotlib import pyplot as plt
import math


class Prepare:

    def __init__(self, weather, power):
        self.weather = weather
        self.power = power
        self.X = None

    def filter(self):
        weather = self.weather[self.weather['Source_time'] == self.weather['Source_time'].max()]
        self.weather = weather

    def augment(self, n = 10):
        power = self.power.resample('3H').mean()
        D = power.merge(self.weather, how='outer', left_index=True, right_index=True)
        X = self.weather.copy()
        for i in range(n):
            delta = np.array(D['Total'][-(X.shape[0] + (n - i)) : - (n - i)])
            X[f"t-{n - i}"] = delta # confirm that for wind_data this indeed ads right t minus i Total values   
        
        y = X.join(power)['Total']
        return X, y

    def enhance(self, X):
        directions = ['E', 'ENE', 'NE', 'NNE', 'N', 'NNW', 'NW', 'WNW',
                  'W', 'WSW', 'SW', 'SSW', 'S', 'SSE', 'SE', 'ESE']
        radians = {d: (idx / 16) * 2 * math.pi for idx, d in enumerate(directions)}
        long = lambda direction: round(math.cos(radians[direction]), 3)
        lat  = lambda direction: round(math.sin(radians[direction]), 3)
        X['lat'] = X['Direction'].apply(lat); X['lat'] = X['lat'].multiply(X['Speed']) 
        X['long'] = X['Direction'].apply(long); X['long'] = X['long'].multiply(X['Speed'])
        X = X.drop(['Direction', 'Lead_hours', 'Source_time', 'Speed'], axis=1)
        return X    
                 
T = Prepare(wind_df, gen_df)
X, y = T.augment()
X = T.enhance(X)

P = Prepare(fore_df, gen_df)
P.filter()
tmp_X, _ = P.augment()
P = P.enhance(tmp_X)
