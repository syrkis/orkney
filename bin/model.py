# trainer.py
#   fits to pipeline with prepared data
# by: Noah Syrkis

# imports
from prep import X, y, P
from pipe import pipe
from icecream import ic
import numpy as np

pipe.fit(X, y)

for i in range(P.shape[0]):
    entry = np.array(ic(P.iloc[i, :])).reshape(1, -1)         
    pred = pipe.predict(entry)
    cols = [col for col in P.columns if col[:2] == 't-'][::-1]
    for j in range(len(cols)):
        P.iloc[j, cols[j]] = pred
