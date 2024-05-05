# Importa las librerías pandas y statsmodels, que son útiles para manejar datos y realizar análisis estadístico, respectivamente.
import pandas as pd
import statsmodels.api as sm

# Define dos listas con valores numéricos que se usarán más adelante para hacer predicciones.
valor1 = [1.88, 2.08, 1.88, 2.13,2.08]
valor2 = [153, 155, 187, 193,189]

# Carga un conjunto de datos desde un archivo CSV. Este archivo debe contener al menos las columnas 'Duracion', 'Capacity' y 'Passengers'.
datos = pd.read_csv('C:\\Users\\uriel\\Documents\\python\\baselímpia.csv')

# Selecciona las columnas 'Duracion' y 'Capacity' del DataFrame para usarlas como variables independientes en el modelo.
X = datos[['Duracion', 'Capacity']]  
# Selecciona la columna 'Passengers' como la variable dependiente.
y = datos['Passengers']

# Agrega una columna constante a las variables independientes, lo cual es necesario para incluir el intercepto en el modelo.
X = sm.add_constant(X)

# Crea y ajusta un modelo de regresión Poisson usando las variables definidas.
# La regresión Poisson es útil para modelar conteos o cantidades como la cantidad de pasajeros.
modelo_poisson = sm.GLM(y, X, family=sm.families.Poisson()).fit()

# Imprime un resumen del modelo ajustado para revisar estadísticas como coeficientes y significancia.
print(modelo_poisson.summary())

# Realiza predicciones usando nuevos datos basados en los valores en las listas 'valor1' y 'valor2'.
for i in range(5):  # Itera a través de las cuatro parejas de valores en 'valor1' y 'valor2'.
    # Crea un DataFrame para cada nueva observación con valores de 'Duracion' y 'Capacity'.
    nuevos_datos = pd.DataFrame({'Duracion': [valor1[i]], 'Capacity': [valor2[i]]})
    # Añade una constante al DataFrame para poder hacer predicciones correctas.
    nuevos_datos = sm.add_constant(nuevos_datos, has_constant='add')
    
    # Usa el modelo ajustado para predecir el número de pasajeros basado en los nuevos datos.
    predicciones = modelo_poisson.predict(nuevos_datos)
    # Imprime las predicciones para cada conjunto de datos nuevos.
    print("Predicciones:")
    print(predicciones)
