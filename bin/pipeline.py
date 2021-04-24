# pipeline.py
#   sklearn pipeline for orkeny project
# by: Noah Syrkis

# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

pipe = Pipeline([
        ('scale', StandardScaler()),
        ('model', SVR())
    ])
