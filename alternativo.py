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

    petal_length_short = fuzz.trimf(petal_length, [0, 0, w])
    petal_length_middle = fuzz.trimf(petal_length, [0, w, 1])
    petal_length_long = fuzz.trimf(petal_length, [w, 1, 1])

    petal_width_short = fuzz.trimf(petal_width, [0, 0, w])
    petal_width_middle = fuzz.trimf(petal_width, [0, w, 1])
    petal_width_long = fuzz.trimf(petal_width, [w, 1, 1])

    N = iris.shape[0]

    for i in range(0, 2):
        participante = iris.iloc[[i],:]

        #Sepal Length
        participante_sepal_length = participante['sepal_length']
        print(participante_sepal_length)

        x1_short = fuzz.interp_membership(sepal_length, sepal_length_short, participante_sepal_length)
        x1_middle = fuzz.interp_membership(sepal_length, sepal_length_middle, participante_sepal_length)
        x1_long = fuzz.interp_membership(sepal_length, sepal_length_long, participante_sepal_length)


        participante_sepal_width = participante['sepal_width']
        print(participante_sepal_width)





fuzzificar(0.6)



