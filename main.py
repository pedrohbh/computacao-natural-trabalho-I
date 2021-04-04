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

#Salva arquivo para teste
#iris.to_csv("iris_teste.csv", sep=',', index=False)

#Fim do processo de normatização dos dados
sepal_length = ctrl.Antecedent(np.arange(0.0, 1.1, 0.1), 'sepal_length')
sepal_width = ctrl.Antecedent(np.arange(0.0, 1.1, 0.1), 'sepal_width')
petal_length = ctrl.Antecedent(np.arange(0.0, 1.1, 0.1), 'petal_length')
petal_width = ctrl.Antecedent(np.arange(0.0, 1.1, 0.1), 'petal_width')

classe_flor = ctrl.Consequent(np.arange(0, 11, 1), 'classe_flor')

classe_flor['setosa'] = fuzz.trimf(classe_flor.universe, [0, 0, 5])
classe_flor['versicolor'] = fuzz.trimf(classe_flor.universe, [0, 5, 10])
classe_flor['virginica'] = fuzz.trimf(classe_flor.universe, [5, 10, 10])

classe_flor.view()