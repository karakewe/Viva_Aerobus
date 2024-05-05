# Importa las librerías necesarias para el manejo de datos y análisis estadístico.
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Prepara un diccionario con dos listas de datos, una para pasajeros y otra para la media de viajes.
# Algunos valores para 'Passengers' faltan, indicados con 'None'.
data = {
    'Passengers': [138.43, 138.70, 159.43, 162.22, 179.79, None, None],
    'mediaViajes': [0.38, 2.83, 3.08, 3.87, 4.54, 5.06, 6.35]
}

# Crea un DataFrame de pandas a partir del diccionario para manipular los datos fácilmente.
df_new = pd.DataFrame(data)
# Imprime el DataFrame para ver qué aspecto tiene.
print(df_new)

# Carga datos desde un archivo CSV que debe tener las columnas 'Quantity', 'Passengers' y 'mediaViajes'.
df = pd.read_csv("C:\\Users\\uriel\\Documents\\Python productos\\PoisonQuantityMatricula.csv")

# Configura y ajusta un modelo de regresión utilizando la API de fórmula de statsmodels.
# 'Quantity' es la variable que queremos predecir, basada en 'Passengers' y 'mediaViajes'.
# Se especifica un modelo de regresión binomial negativa con un parámetro alpha.
model = smf.glm('Quantity ~ Passengers + mediaViajes', data=df,
                family=sm.families.NegativeBinomial(alpha=8.0)).fit()
# Imprime un resumen del modelo para ver los resultados estadísticos.
print(model.summary())

# Utiliza el modelo para hacer predicciones sobre el DataFrame 'df_new'.
# Antes de predecir, elimina las filas que contienen 'None' para evitar errores.
predicciones = model.predict(df_new.dropna())
# Imprime las predicciones para revisarlas.
print(predicciones)

# Si es necesario examinar cómo el modelo se ajusta a cada dato individual, calcula los residuos.
# Los residuos de Pearson y de devianza pueden ayudar a identificar datos que no se ajustan bien al modelo.
pearson_residuals = model.resid_pearson
# Imprime los residuos de Pearson para analizarlos.
print(pearson_residuals)

deviance_residuals = model.resid_deviance
# Imprime los residuos de devianza para una evaluación adicional.
print(deviance_residuals)
