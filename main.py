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

#Início do código classificador Fuzzy
sepal_length = ctrl.Antecedent(np.arange(0.0, 1.1, 0.1), 'sepal_length')
sepal_width = ctrl.Antecedent(np.arange(0.0, 1.1, 0.1), 'sepal_width')
petal_length = ctrl.Antecedent(np.arange(0.0, 1.1, 0.1), 'petal_length')
petal_width = ctrl.Antecedent(np.arange(0.0, 1.1, 0.1), 'petal_width')

classe_flor = ctrl.Consequent(np.arange(0, 11, 1), 'classe_flor')

classe_flor['setosa'] = fuzz.trimf(classe_flor.universe, [0, 0, 5])
classe_flor['versicolor'] = fuzz.trimf(classe_flor.universe, [0, 5, 10])
classe_flor['virginica'] = fuzz.trimf(classe_flor.universe, [5, 10, 10])

classe_flor.view()

for elemento in [sepal_length, sepal_width, petal_length, petal_width]:
    elemento['short'] = fuzz.trimf(elemento.universe, [0.0, 0.0, 0.6])
    elemento['middle'] = fuzz.trimf(elemento.universe, [0.0, 0.6, 1.0])
    elemento['long'] = fuzz.trimf(elemento.universe, [0.6, 1.0, 1.0])

figura = sepal_length.view()
#figura.savefig('exemplo.png')

x1 = sepal_length
x2 = sepal_width
x3 = petal_length
x4 = petal_width

rule1 = ctrl.Rule( ( x1['short'] | x1['long'] ) & ( x2['middle'] | x2['long'] ) & ( x3['middle'] | x3['long'] ) & ( x4['middle'] ), classe_flor['versicolor'] )
rule2 = ctrl.Rule( ( x3['short'] | x3['middle'] ) & ( x4['short'] ), classe_flor['setosa'] )
rule3 = ctrl.Rule( ( x2['short'] | x2['middle'] ) & ( x3['long'] ) & ( x4['long'] ), classe_flor['virginica'] )
rule4 = ctrl.Rule( ( x1['middle'] ) & ( x2['short'] | x2['middle'] ) & ( x3['short'] ) & ( x4['long'] ), classe_flor['versicolor'] )

classe_flores_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])

classe_resposta = ctrl.ControlSystemSimulation(classe_flores_ctrl)

N = iris.shape[0]
print(N)

for i in range(0, N):
    participante = iris.iloc[[i], :]
    for nome in [ 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
        classe_resposta.input[nome] = participante[nome]
    print(classe_resposta.input)

    classe_resposta.compute()
    print(classe_resposta.output)
