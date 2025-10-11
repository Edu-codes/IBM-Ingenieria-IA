
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

import warnings

file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
print(data.head())
print("------------------------")



# Distribution of target variable
sns.countplot(y='NObeyesdad', data=data)
plt.title('Distribution of Obesity Levels')
# plt.show()


# print(data.isnull().sum())
# print(data.info())
# print(data.describe())


# Standardizing continuous numerical features

#Selecciona las columnas que tienen datos numéricos decimales (float64), y las guarda como una lista.
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()


# Crea un escalador usando StandardScaler() de sklearn.
# Aplica ese escalador a las columnas numéricas seleccionadas.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])
print("---scaled-features-----")
print(scaled_features)
# Converting to a DataFrame

# Convierte los datos ya escalados en un nuevo DataFrame, y le pone los mismos nombres de columnas que tenían antes.
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
print("---scaled-df-----")
print(scaled_df)

# Elimina del DataFrame original las columnas numéricas originales (drop(columns=...)).
# Luego combina (concatena) ese DataFrame sin números con el scaled_df que ya está estandarizado.
scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)
print("---scaled-data-----")
print(scaled_data)

#Su objetivo es transformar las columnas categóricas
# (que son de tipo texto) en una forma que los modelos
# de machine learning puedan entender: números. Esta técnica se llama One-Hot Encoding.

# Detecta todas las columnas de tipo texto (object) del dataset scaled_data, y las guarda en una lista.
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
print("---categorical_columns-----")
print(categorical_columns)

# Luego elimina la columna "NObeyesdad" de esa lista porque es la columna objetivo (target) y no debe codificarse.
categorical_columns.remove('NObeyesdad')  # Exclude target column

#Crea el codificador OneHotEncoder.
# sparse_output=False: asegura que la salida será un array común (no un objeto disperso).
# drop='first': evita la colinealidad eliminando la primera categoría de cada columna (muy común en modelos lineales).
encoder = OneHotEncoder(sparse_output=False, drop='first')

# Aplica el codificador a las columnas categóricas.
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
print("---encoded_features-----")
print(encoded_features)

# Convierte el resultado del one-hot encoding en un nuevo DataFrame con nombres de columnas adecuado
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
print("---encoded_df-----")
print(encoded_df)

# Elimina las columnas categóricas originales (texto) del scaled_data.
# Añade las nuevas columnas codificadas (ya en formato numérico binario)
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)
print("---prepped_data-----")
print(prepped_data)

# Encoding the target variable
# .astype, convierte los datos de la columna en tipo categoria y .cat.codes asigna a cada categoria un numero entero
# (1, 2 ,3...)
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
print("---prepped_data.head-----")
print(prepped_data.head())

# Preparing final dataset

# X = Todas las columnas menos la columna objetivo "etiqueta"
# y = Nuestra columna a predecir "etiqueta"
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']

print("---x----")
print(X)
print("---y----")
print(y)


# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Training logistic regression model using One-vs-All (default)
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)

# Predictions
y_pred_ova = model_ova.predict(X_test)

# Evaluation metrics for OvA
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")


# Training logistic regression model using One-vs-One
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)

# Predictions
y_pred_ovo = model_ovo.predict(X_test)

# Evaluation metrics for OvO
print("One-vs-One (OvO) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")