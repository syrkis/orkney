# director.py
#   turns direction and speed into vectors
# by: Noah Syrkis

# imports
import math

class Director:

    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        directions = ['E', 'ENE', 'NE', 'NNE', 'N', 'NNW', 'NW', 'WNW',
                  'W', 'WSW', 'SW', 'SSW', 'S', 'SSE', 'SE', 'ESE']
        radians = {d: (idx / 16) * 2 * math.pi for idx, d in enumerate(directions)}

        long = lambda direction: round(math.cos(radians[direction]), 3)
        lat  = lambda direction: round(math.sin(radians[direction]), 3)

        X['lat'] = X['Direction'].apply(lat).multiply(X['Speed']) 
        X['long'] = X['Direction'].apply(long).multiply(X['Speed'])

        X = X.drop(['Direction', 'Speed'], axis=1)
        return X
