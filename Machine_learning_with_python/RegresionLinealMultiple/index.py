import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

from RegresionLinealSimple.Index import y_train

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df = pd.read_csv(url)
# print (df.sample(5))
# print(df.columns)
df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1)
df= df.drop(['CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB'], axis=1)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#plotting.scatter.matrix genera una matriz de diagramas de dispercion con los valores numerido de "df"
axes = pd.plotting.scatter_matrix(df, alpha= 0.2)

for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
# print(plt.show())


# Agregamos en valriables nuestras variables dependientes e independientes
x = df.iloc[:,[0,1]]
y = df.iloc[:,[2]]

# Con sklearn y preprocessing se ESTANDARIZA en nuestras caracteristicas de entrada evitamos que el modelo
# favorezca ninguna caractiristica debido a su magintud
from sklearn import preprocessing

std_scaler = preprocessing.StandardScaler()
x_std = std_scaler.fit_transform(x)

# una variable estandarizada tiene una media de cero y una desviación estándar de uno.
# print(pd.DataFrame(x_std).describe().round(2))

# Separamos nuestros datos en datos de entreamiento y datos de prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_std, y, test_size=0.2, random_state=42)

#Seleccionamos el tipo de modelo (Lineal)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

#Entrenamos nuestro modelo
regressor.fit(X_train, y_train)

coef_ = regressor.coef_
intercept_ = regressor.intercept_
print("---------------------------------------------------------")

print('Coefficients: ', coef_)
print('Intercept: ', intercept_)
print("---------------------------------------------------------")


#Si no estuviera estandarizado se veria asi:
# Get the standard scaler's mean and standard deviation parameters
means_ = std_scaler.mean_
std_devs_ = np.sqrt(std_scaler.var_)

# The least squares parameters can be calculated relative to the original, unstandardized feature space as:
coef_original = coef_ / std_devs_
intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_)

print ('Coefficients: ', coef_original)
print ('Intercept: ', intercept_original)


X1 = X_test[:, 0] if X_test.ndim > 1 else X_test
X2 = X_test[:, 1] if X_test.ndim > 1 else np.zeros_like(X1)

# Create a mesh grid for plotting the regression plane
x1_surf, x2_surf = np.meshgrid(np.linspace(X1.min(), X1.max(), 100),
                               np.linspace(X2.min(), X2.max(), 100))

y_surf = intercept_ +  coef_[0,0] * x1_surf  +  coef_[0,1] * x2_surf

# Predict y values using trained regression model to compare with actual y_test for above/below plane colors
y_pred = regressor.predict(X_test.reshape(-1, 1)) if X_test.ndim == 1 else regressor.predict(X_test)
above_plane = y_test >= y_pred
below_plane = y_test < y_pred
above_plane = np.ravel(above_plane)
below_plane = np.ravel(below_plane)

# Plotting
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points above and below the plane in different colors
ax.scatter(X1[above_plane], X2[above_plane], y_test[above_plane],  label="Above Plane",s=70,alpha=.7,ec='k')
ax.scatter(X1[below_plane], X2[below_plane], y_test[below_plane],  label="Below Plane",s=50,alpha=.3,ec='k')

# Plot the regression plane
ax.plot_surface(x1_surf, x2_surf, y_surf, color='k', alpha=0.21,label='plane')

# Set view and labels
ax.view_init(elev=10)

ax.legend(fontsize='x-large',loc='upper center')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(None, zoom=0.75)
ax.set_xlabel('ENGINESIZE', fontsize='xx-large')
ax.set_ylabel('FUELCONSUMPTION', fontsize='xx-large')
ax.set_zlabel('CO2 Emissions', fontsize='xx-large')
ax.set_title('Multiple Linear Regression of CO2 Emissions', fontsize='xx-large')
plt.tight_layout()
print(plt.show())


plt.scatter(X_train[:,0], y_train,  color='blue')
plt.plot(X_train[:,0], coef_[0,0] * X_train[:,0] + intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
print(plt.show())

plt.scatter(X_train[:,1], y_train,  color='blue')
plt.plot(X_train[:,1], coef_[0,1] * X_train[:,1] + intercept_[0], '-r')
plt.xlabel("FUELCONSUMPTION_COMB_MPG")
plt.ylabel("Emission")
print(plt.show())
