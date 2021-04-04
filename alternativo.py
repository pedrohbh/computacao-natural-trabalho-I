import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

iris = pd.read_csv('iris.csv')

#Processo de normatização dos dados
for c in iris.columns:
    if c not in ['class']:
        c_max = max(iris[c])
        c_min = min(iris[c])
        iris[c] = (iris[c] - c_min) / (c_max - c_min)

for c in iris.columns:
    if c in ['class']:
        dummies = pd.get_dummies(iris[c], prefix=c)
        iris = iris.join(dummies)

print('head():'); print(iris.head())
print('-----------------------------------------------')

print('tail():'); print(iris.tail())
print('-----------------------------------------------')

def fuzzificar(w):
    sepal_length = np.arange(0, 1.01, 0.01)
    sepal_width = np.arange(0, 1.01, 0.01)
    petal_length = np.arange(0, 1.01, 0.01)
    petal_width = np.arange(0, 1.01, 0.01)

    sepal_length_short = fuzz.trimf(sepal_length, [0, 0, w])
    sepal_length_middle = fuzz.trimf(sepal_length, [0, w, 1])
    sepal_length_long = fuzz.trimf(sepal_length, [w, 1, 1])

    sepal_width_short = fuzz.trimf(sepal_width, [0, 0, w])
    sepal_width_middle = fuzz.trimf(sepal_width, [0, w, 1])
    sepal_width_long = fuzz.trimf(sepal_width, [w, 1, 1])

    





