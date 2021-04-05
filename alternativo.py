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

    for i in range(0, N):
        participante = iris.iloc[[i],:]

        #Sepal Length
        participante_sepal_length = participante['sepal_length']
        print(participante_sepal_length)

        x1_short = fuzz.interp_membership(sepal_length, sepal_length_short, participante_sepal_length)
        x1_middle = fuzz.interp_membership(sepal_length, sepal_length_middle, participante_sepal_length)
        x1_long = fuzz.interp_membership(sepal_length, sepal_length_long, participante_sepal_length)


        participante_sepal_width = participante['sepal_width']
        print(participante_sepal_width)

        x2_short = fuzz.interp_membership(sepal_width, sepal_width_short, participante_sepal_width)
        x2_middle = fuzz.interp_membership(sepal_width, sepal_width_middle, participante_sepal_width)
        x2_long = fuzz.interp_membership(sepal_width, sepal_width_long, participante_sepal_width)

        ##Prints para teste
        #print(x2_short)
        #print(x2_middle)
        #print(x2_long)

        participante_petal_length = participante['petal_length']
        print(participante_petal_length)

        x3_short = fuzz.interp_membership(petal_length, petal_length_short, participante_petal_length)
        x3_middle = fuzz.interp_membership(petal_length, petal_length_middle, participante_petal_length)
        x3_long = fuzz.interp_membership(petal_length, petal_length_long, participante_petal_length)

        ##Prints para teste
        #print(x3_short)
        #print(x3_middle)
        #print(x3_long)

        participante_petal_width = participante['petal_width']
        print(participante_petal_width)

        x4_short = fuzz.interp_membership(petal_width, petal_width_short, participante_petal_width)
        x4_middle = fuzz.interp_membership(petal_width, petal_width_middle, participante_petal_width)
        x4_long = fuzz.interp_membership(petal_width, petal_width_long, participante_petal_width)

        rule1 = min(max(x1_short, x1_long), max(x2_middle, x2_long), max(x3_middle, x3_long), x4_middle)
        rule2 = min(max(x3_short,x3_middle),x4_short)
        rule3 = min(max(x2_short, x2_middle),x3_long,x4_long)
        rule4 = min(x1_middle, max(x2_short, x2_middle), x3_short, x4_long)
        print(rule1)
        print(rule2)
        print(rule3)
        print(rule4)

        vetor_resposta = [rule1, rule2, rule3, rule4]
        indice_maximo = vetor_resposta.index(max(vetor_resposta))
        print(indice_maximo)
        classe_resposta = ""
        if indice_maximo == 0 or indice_maximo == 3:
            classe_resposta = "class_Iris-versicolor"
        elif indice_maximo == 1:
            classe_resposta = "class_Iris-setosa"
        elif indice_maximo == 2:
            classe_resposta = 'class_Iris-virginica'
        print(classe_resposta)

        






fuzzificar(0.6)



