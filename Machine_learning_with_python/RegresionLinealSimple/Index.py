import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

# La función train_test_split de sklearn.model_selection divide tu conjunto de datos en dos partes:
# Entrenamiento (train): Para construir (entrenar) el modelo.
# Prueba (test): Para evaluar qué tan bien funciona con datos que no ha visto.
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df = read_csv(url)
# print(df.columns)
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
# print(cdf.sample(5))


# vic = cdf[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY']]
# vic.hist()
'''
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('CO2EMISSIONS')
plt.show()


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.show()


plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel('CYLINDERS')
plt.ylabel('CO2EMISSIONS')
plt.show()
'''

#Convierte las columnas en arreglos de Numpy, que es lo que los modelos de sckit-learn esperan
X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

#Usamos "train_test_split" para separa nuestros datos en dos partes (pruebas y entrenamiento)
#test_size = usarmos el 20% de datos para pruebas y 80% para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

type(X_train), np.shape(X_train), np.shape(X_train)


# Entrenar el modelo con los datos de entrenamiento

#linearRegression = crea el modelo de regresion lineal con .LinearRegression()
regressor = linear_model.LinearRegression()


# X_train es un array unidimensional, pero los modelos de sklearn esperan un array bidimensional como entrada para los datos de entrenamiento, con forma (n_observaciones, n_características).
# Por lo tanto, necesitamos remodelarlo. Podemos permitir que infiera el número de observaciones usando '-1'.

# .fit = entrena el modelo con tus datos de entrenamiento.
# .reshape(-1, 1) convierte X_train en un arrego 2D ya que es unidimensional
regressor.fit(X_train.reshape(-1, 1), y_train)

# Print the coefficients
# Con la regresión lineal simple solo hay un coeficiente; aquí lo extraemos del array 1x1.
print ('Coefficientss: ', regressor.coef_[0])  #Theta 0 = Pendiente
print ('Intercepts: ',regressor.intercept_) #Theta 1 = intercepto

plt.scatter(X_train, y_train,  color='blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#Modelo de Evaluacion
y_test_ = regressor.predict(X_test.reshape(-1,1))
