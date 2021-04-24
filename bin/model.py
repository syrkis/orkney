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
wind_df = get_df(wind)

####################################################
## Preliminaries

from icecream import ic
import math

gen_df = gen_df.resample('3H').mean()
D = gen_df.merge(wind_df, how='right', left_index=True, right_index=True)
y = D['Total']

def __director(X):
        directions = ['E', 'ENE', 'NE', 'NNE', 'N', 'NNW', 'NW', 'WNW',
                  'W', 'WSW', 'SW', 'SSW', 'S', 'SSE', 'SE', 'ESE']
        radians = {d: (idx / 16) * 2 * math.pi for idx, d in enumerate(directions)}
        long = lambda direction: round(math.cos(radians[direction]), 3)
        lat  = lambda direction: round(math.sin(radians[direction]), 3) 
        X['long'] = X['Direction'].apply(long)
        X['lat'] = X['Direction'].apply(lat)
        X['long'] = X['long'].multiply(X['Speed'])
        X['lat'] = X['lat'].multiply(X['Speed']) 
        X = X[X.columns[X.columns != 'Direction' & X.columns != 'Total']]   
        return X

X = __director(D)
ic(X)
####################################################
## Predicting

for_df = get_df(forecasts).to_numpy()
# vectorize direction
# for row in forecast augment  with past
