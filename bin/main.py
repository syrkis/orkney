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


gen_df = get_df(generation)
wind_df = get_df(wind)

####################################################
## Preprocessing class
from matplotlib import pyplot as plt
import seaborn as sns
import torch
from torch import nn
import math
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet

gen_df = gen_df[gen_df.columns[2]]
wind_df = wind_df[wind_df.columns[[0, 3]]]


directions = ['E', 'ENE', 'NE', 'NNE', 'N', 'NNW', 'NW', 'WNW',
          'W', 'WSW', 'SW', 'SSW', 'S', 'SSE', 'SE', 'ESE']
radians = {d: (idx / 16) * 2 * math.pi for idx, d in enumerate(directions)}    
long = lambda direction: round(math.cos(radians[direction]), 3)
lat  = lambda direction: round(math.sin(radians[direction]), 3)

wind_df['long'] = wind_df[wind_df.columns[0]].apply(long)
wind_df['lat'] = wind_df[wind_df.columns[0]].apply(lat)

wind_df['long'] = wind_df['long'].multiply(wind_df['Speed'])
wind_df['lat'] = wind_df['lat'].multiply(wind_df['Speed'])

wind_df = wind_df[wind_df.columns[2:]].resample('1Min', kind='timestamp').pad()
data = gen_df.to_frame().join(wind_df)
data = data.asfreq(pd.infer_freq(data.index))
out = int(data.index.apply(lambda t: t * 10 ** 9))
print(out)
#M = TimeSeriesDataSet(data, 'time', 'Total', ['long', 'lat'])


####################################################
## Train model
from torch.distributed.pipeline.sync import Pipe
