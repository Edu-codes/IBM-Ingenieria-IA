import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)

print(churn_df)
print("----------------------------------------")


# churn_df = churn_df[["antigüedad", "edad", "dirección", "ingresos", "educación", "emplear", "equipo"  "abandono"]]

# Filtramos nuestros datos
churn_df = churn_df[['tenure', 'callcard', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]

# Pasamos nuestra columna a predecir a numero entero
churn_df['churn'] = churn_df['churn'].astype('int')
print(churn_df)

print("-------------Valores de X---------------")

X = np.asarray(churn_df[['tenure', 'callcard', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
print(X[0:5]) #print the first 5 values

print("-------------Valores de Y---------------")

# Convertimos las columnas en arrays que es lo que espera, sckelearn
Y = np.asarray(churn_df['churn'])
print(Y[0:5])

# Normalizamos o Estandarizamos nuestras variables independientes de tal modo que:
# media = 0 y desviacion estandar = 1
X_norm = StandardScaler().fit(X).transform(X)
print(X_norm[0:5])

# Sepramaos la los datos de pruebas y de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X_norm, Y, test_size=0.2, random_state=42)

#Entrenamos nuestro modelo
LR = LogisticRegression().fit(X_train,y_train)

yhat = LR.predict(X_test)
print("----------------Yhat-------------------")
print(yhat[:10])

print("----------------YhatProba-------------------")
yhat_proba = LR.predict_proba(X_test)
print(yhat_proba[:10])


coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
print(plt.show())