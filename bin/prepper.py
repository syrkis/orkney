# prepper.py
#   preproceses data frames for training and predicting
# by: Noah Syrkis

# imports
from reader import wind_df, gen_df, fore_df
import numpy as np
from icecream import ic
from matplotlib import pyplot as plt


class Prepper:

    def __init__(self, weather, power):
        self.weather = weather
        self.power = power

    def pertinize(self):
        weather = self.weather[self.weather['Source_time'] == self.weather['Source_time'].max()]
        self.weather = weather

    def augment(self, n = 20):
        D = self.power.resample('3H').mean().merge(self.weather, how='outer', left_index=True, right_index=True)
        X = self.weather.copy()
        for i in range(n):
            delta = np.array(D['Total'][-(X.shape[0] + (n - i)) : - (n - i)])
            X[f"t-{n - i}"] = delta
        ic(X)

    def enhance(self):
        pass

T = Prepper(wind_df, gen_df)
P = Prepper(fore_df, gen_df)


P.pertinize()
P.augment()
T.augment()

