# Modelo de Machine Learning utilizando un modelo
# de regresión linear, usando librerías o framewoeks
# previamente implementados. (sklearn)

# Manolo Ramírez Pintor
# A01706155

# Como hemos visto antes, en el modelo que hicimos
# a manita, nos basamos mucho en la fórmula de
# regresión linear, en este caso funciona igual que
# (y = a + bx_1 + cx_2 + ... cx_n. )

# En Machine learning, usamos esta regresión para
# entrenar un modelo y "enseñarle" los valores "m"
# y "b" (o más) para crear la línea que más se a- 
# juste a nuestro conjunto de datos.

# En esta ocasión también trabajaremos con lo mismo
# pero en forma multivariable, tomando dos variables.

# Nuvamente, tenemos parámetros como y, m, x, b.

# Ahora todo lo de machine learning se va a una
# simple función de sklearn: LinearRegression.fit(x, y)


# Importar librerías
from random import randrange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

print("--------------------------------------------------------------------------------------------------------------")
print("\t\t\t*** MODELO DE REGRESIÓN LINEAR USANDO SKLEARN: ***")
print("Importando datos...", end='')

# Importar dataset que vamos a utilizar
# Este dataset contiene datos de 500 personas y un índice que
# define la obesidad de una persona...
# Las columnas tienen ['Sexo | Altura | Peso | Índice']
# La escala define que 0 es poco obesa y 5 muy obesa.

# Mi propósito de obtener ese índice a través de ML es tomar
# una predicción general y más confiable obteniendo resultados
# que quedan entre las escalas.
df = pd.read_csv("500_Person_Gender_Height_Weight_Index.csv")

# De acuerdo a Kaggle, estos datos ya se encuentran lo más
# limpios posibles, igual es bueno tomar lo que más importa...

# Como observación, antes quería tomar el puro peso pero las
# predicciones salían con mucho error y la variación era alta.

# Cortamos los datos que sólo necesitamos importar al modelo
x = df.drop(columns = ['Gender', 'Index']) # Peso y altura sólamente
y = df['Index'] # Índice de obesidad

print("\t¡Listo!\n")

# Visualizamos los datos que tenemos
# plt.scatter(df_y, df.iloc[:, 2:3], df.iloc[:, 1:2])
# plt.show()

# Mediante sklearn y la función de train_test split vamos a
# tomar cierta candidad de datos para entrenar y para probar
# en el modelo de regresión linear multivariable que haremos...

t_s = 0.35 # Definimos el porcentaje de datos de prueba

print("Obteniendo " + str(t_s*100) + " % " + "de datos para usarlos de prueba...", end='')

# test_size define el porcentaje que tomaremos para TEST
# random_state nos permite recrear nuevamente un corte
# de datos específico y revisar que todo esté bien
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = t_s)

print("\t¡Listo!")

print("Entrenando el modelo con sklearn LinearRegression.fit(x,y)...", end='')

# Sklearn ya tiene implementada la regresión linear
# múltiple, así que podemos meter dos columnas en
# él sin problema, las interpretará como y = mn*Xn + b
# Finalmente como: y = m1*x1 + m2*x2 + ... + mn*xn + b
LR = LinearRegression()

# Entrenamos el modelo utilizando .fit(x,y)
# Aquí lo que hace es usar otra técnica distinta
# al Gradiente Descendiente, a diferencia del
# que implementé a mano...
LR.fit(x_train, y_train)

print("\t¡Listo!\n")

# Una vez que el modelo se ha entrenado, podemos obtener
# el parametro b con sklearn, usando .intercept_ y
# los coeficientes con sklearn, usando .coef_
b = LR.intercept_ # b
c = LR.coef_      # c

print("La pendiente obtenida fue de", b)
print("Los coeficientes obtenidos fueron", c, "\n")

print("La fórmula de regresión final es:")
print("y = " + str(c[0]) + " * x1 + " + str(c[1]) + " * x2 + " + str(b) + "\n")

# Con el modelo ya entrenado, procedemos a predecir resultados
# primero usando los datos que ya ha visto el modelo entrenando.
y_pred_train = LR.predict(x_train)

# Obtenemos el error con el mismo principio que hemos visto
# del Mean Squared Error, encontrando la diferencia entre "Y" y
# el valor predecido. (y_i - ý_i)^2 / n, esto ya implementado
# en la librería de sklearn...
train_score = r2_score(y_train, y_pred_train)

print("Porcentaje de predicción con datos de ENTRENAMIENTO: " + str("{:.2f}".format(train_score*100)) + "%")

# Plot prediction for TRAIN
# plt.plot(y_pred_train, x["Width"], label = 'Linear Regression', color = 'b')
# plt.scatter(y_pred_train, x["Width"], x["Height"], label = 'Actual test data', color='g', alpha = 0.7)
# plt.legend()
# plt.show()

# Ahora, con el mismo modelo, procedemos a predecir resultados
# usando ahora los datos que nunca ha visto el modelo...
y_pred_test = LR.predict(x_test)

# Volvemos a usar el Mean Squared Error de sklearn...
test_score = r2_score(y_test, y_pred_test)

print("Porcentaje de predicción con datos de PRUEBA: " + str("{:.2f}".format(test_score*100)) + "% \n")

# Por útimo creamos un gráfico que nos permita hacer
# una comparativa de la regresión multivariable con
# los datos de prueba y los datos predecidos por el modelo

# Entre más unidos los puntos como en modo de "correlación",
# mejores serán los resultados de predicción. :)
print("Creando gráfico con pyplot... ", end='')

plt.scatter(y_test, y_pred_test, antialiased = 'true', color = 'pink', label='Test vs. Pred.')
plt.legend(loc='upper left')
plt.xlabel("Test data")
plt.ylabel("Predicted data")
plt.title("Test data vs. Predicted data")

print("¡Listo!")
print("--------------------------------------------------------------------------------------------------------------")

plt.show()