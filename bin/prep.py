# prep.py
#   preps orkeny data
# by: Astrid & Noah

# imports
from read import wind_df, gen_df, fore_df
from matplotlib import pyplot as plt
from icecream import ic

D = wind_df.join(gen_df.resample('3H').mean()).dropna()
X, y = D[D.columns[[0, 3]]], D['Total']
P = fore_df[fore_df.columns[[0, -1]]]
