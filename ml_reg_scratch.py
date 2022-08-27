# Modelo de Machine Learning utilizando un modelo
# de regresión linear, sin librerías o frameworks
# previamente implementados.

# Manolo Ramírez Pintor
# A01706155

# La ecuación de regresión nos sirve para realizar 
# predicciones a partir de una o más variables.
# (y = a + bx_1 + cx_2 + ... cx_n. )

# A través de la regresión linear modelamos la re-
# lación que tienen las variables dependientes de
# las independientes.

# La ecuación de regresión se define por
# y = m * x + b
# y = variable dependiente
# m = la pendiente de la línea
# x = variable independiente
# b = la ordenada al origen

# En Machine learning, usamos esta regresión para
# entrenar un modelo y "enseñarle" los valores "m"
# y "b" (o más) para crear la línea que más se a- 
# juste a nuestro conjunto de datos.


# Importar librerías

import pandas as pd
import matplotlib.pyplot as plt

__errors__= []  # Variable global que almacena los errores



# Funcion de Hipótesis

# Con la hipótesis podemos obtener una evaluación
# de una función linear para más adelante poder
# realizar el cálculo del costo por MSE.

# Al obtener la hipótesis, estamos obteniendo el
# los resultados actuales con los params actuales.
def hipotesis(parametros, vars_ind):
	"""
        A partir de la regresión linear, creamos una hipótesis
		la definimos como h(x), usamos las variables iniciales
		de m, b y el dataset que tenemos, de usaremos un acumulador.
    """
	acum = 0
	for i in range(len(parametros)):
		acum += parametros[i] * vars_ind[i]  # h(x) = a+bx1+cx2+ ... nxn. 
	return acum



# Funcion de Costo por MSE

# A través de la función de pérdida, encontraremos
# el error que tiene nuestro modelo de ML e inten-
# taremos minimizarlo lo más posible para obtener
# predicción más correcta o acertada.

# El Mean Squared Error Function nos permite cal-
# cularlo, encontrando la diferencia entre "Y" y
# el valor predecido. (y_i - ý_i)^2 / n
def costo_MSE(parametros, vars_ind, vars_dep):
	"""
        Acumula las pérdidas a través del cálculo del MSE, el uso de
		los valores de la hipótesis de x y el valor real de "y".
		
		vars_ind = Variables Independientes
		vars_dep = Variables Dependientes
    """
	error_acum = 0

	for i in range(len(vars_ind)):
		hip = hipotesis(parametros, vars_ind[i])
		# Imprime el avance en tiempo real del modelo
		#print( "Hipotesis:  %f  , y = %f " % (hip,  vars_dep[i])) 
		error_acum =+ (hip - vars_dep[i])**2 # Función de costo MSE (y_i - ý_i)^2
	
	__errors__.append( error_acum / len(vars_ind) ) # Agregar (y_i - ý_i)^2 / n
	
	# Imprime el error nuevo después del proceso de la función (debug)
	# new = (__errors__[-1]) * 100
	# print("Error: ", new)



# Funcion de Gradiente Descendiente

# El Gradiente Descendiente nos va a servir para
# ir calculando las variables más óptimas,
# cada ciclo iterativo se define como una época.

# La velocidad (o forma) en la que va aprendiendo
# depende de una variable (Lr) de aprendizaje que
# establece el cambio que va a tener en la pendi-
# ente y la ordenada al origen. (m y b)
def GradienteDescendiente(parametros, vars_ind, vars_dep, Lr):
	"""
		El Gradiente descendiente recibe los parámetros actuales
		del aprendizaje, las variables independientes y dependi-
		entes a analizar para obtener el error y los steps que
		debe de dar a partir del Learning Rate (Lr)
	"""
	
	gd_list = list(parametros) # Lista temporal que guardará nuevos parametros
	
	for j in range(len(parametros)):
		
		acum = 0
		
		for i in range(len(vars_ind)):
			error = hipotesis(parametros, vars_ind[i]) - vars_dep[i]
			acum = acum + error * vars_ind[i][j]  # Acumulador del gradiente a partir de la fórmula de regresión.
		gd_list[j] = parametros[j] - Lr*(1/len(vars_ind))*acum  # Resta de los params originales con el learning rate.
	
	return gd_list


# ----------------------------------------------------------------------------


# Importar dataset que vamos a utilizar
df = pd.read_csv("500_Person_Gender_Height_Weight_Index.csv")

# Cortar los datos que sólo necesitamos aprender
df_x = df.iloc[:, 1:3].to_numpy().tolist() # Peso y altura (sin sexo).
df_y = df["Index"].to_numpy().tolist() # Valores del índice de masa.

# Ejemplo base de uso del modelo en modo multivariable
# parametros = [0,0,0]
# vars_ind = [[1,1],[2,2],[3,3],[4,4],[5,5]]
# vars_dep = [2,4,6,8,10]

parametros = [0,0,0] # Establecer los ceros por variable.
# 					   ej: [0, 0] = a, b | [0, 0, 0] = a, b, c
vars_ind = df_x # Entrada de datos a analizar
vars_dep = df_y # Entrada de las salidas esperadas

Lr = 0.000001  #  Tasa de aprendizaje (Learning rate):
#				  Entre más pequeño, más tarda pero da más precisión.

# Añadir un (1) como primer elemento de cada conjunto de parámetros.
for i in range(len(vars_ind)):
	if isinstance(vars_ind[i], list):
		vars_ind[i]=  [1]+vars_ind[i]
		# Comprobación de los arreglos (debug)
		# print(vars_ind)
	else:
		vars_ind[i]=  [1,vars_ind[i]]


# ----------------------------------------------------------------------------

epocas_obj = 1000 # Valor objetivo de épocas (EDITABLE)
epoca_actual = 0 # Valor inicial de las épocas (NO MOVER)

while True:  #  Corre el ciclo hasta que se acaban las épocas o se alcanza el error objetivo
	oldparams = list(parametros)
	
	parametros = GradienteDescendiente(parametros, vars_ind,vars_dep, Lr) # Aprendizaje del modelo
	costo_MSE(parametros, vars_ind, vars_dep)  # Para calcular el error del modelo
	# Imprime información relevante durante proceso del modelo:
	print("Época actual:", epoca_actual, "\tError actual:", "{:.4f}".format(__errors__[-1]*100) + "%", "\tFórmula: y =", str(round(parametros[0], 10)) + " + (" + str("{:.8f}".format(parametros[1])) + " * x1) + (" + str("{:.8f}".format(parametros[2])) + " * x2)") 
	# (y = a + bx_1 + cx_2 + ... cx_n. ); Formula base para imprimir el proceso.
	epoca_actual += 1 # Sumamos un ciclo
	if(oldparams == parametros or epoca_actual == epocas_obj): # Aquí se establece el número de épocas a recorrer...
		# Al finalizar, detenemos el ciclo						 alternativamente el error objetivo que queremos tener.
		break

# Una vez que el proceso principal del programa finalice, 
# vamos a imprimir los resultados que obtuvimos de nues- 
# modelo de Machine Learning.

print("--------------------------------------------------------------------------------------------------------------")
print("\t\t\t*** EL PROGRAMA HA FINALIZADO, RESULTADOS: ***")
print("Error final: ", "{:.2f}".format(__errors__[-1]*100) + "%", "\tFórmula final: y =", str("{:.15f}".format(parametros[0], -60)) + " + (" + str("{:.15f}".format(parametros[1])) + " * x1) + (" + str("{:.15f}".format(parametros[2])) + " * x2)\n")
print("Creando gráfico con pyplot... ", end='')

# Creación del gráfico con el error y parámetros de visualización personalizados
plt.plot(__errors__, antialiased = 'true', color = 'hotpink', marker = 'o', mfc = 'pink', label='Error')
plt.legend(loc='upper right')
plt.title("Aprendizaje del modelo de regresión con G.D.")
plt.xlabel("Época")
plt.ylabel("Min Squared Error")

print("¡Listo! :D")
print("--------------------------------------------------------------------------------------------------------------")

plt.show()
