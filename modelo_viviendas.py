import pandas as pd #trbbaja con tablas
import numpy as np #trabaja con arrays numericos
from sklearn.model_selection import train_test_split #separa los datos de entrenamiento yy de prueba
from sklearn.linear_model import LinearRegression #es el algoritmo que aprenderá a predecir precios.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score #son metricas las cualues nos dicen como estan de bien las predicciones de los precios
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline



datos=pd.read_csv("viviendas.csv") #leemos el csv

columnas_numericas = [
    "metros",
    "habitaciones",
    "banos",
    "antiguedad",
    "planta",
    "ascensor",
    "terraza",
    "parking",
    "distancia_centro_km"
] # definimos las columnas de entrada que sriven para calcular el precio

columnas_categoricas = [
    "ciudad",
    "zona"
]

X=datos[columnas_numericas+columnas_categoricas] #seleccionamos las columnas de entrada y las guardamos en una variable

y=datos["precio"] #seleccionamos la columna precio y la guardamos en una variable

print("Dimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)


print("\nEntradas:")
print(X.head())


print("\nRespuestas:")
print(y.head())

#X_entrenamiento: características utilizadas para aprender.
#y_entrenamiento: precios correctos correspondientes.
#X_prueba: características de viviendas que el modelo no ha visto.
#y_prueba: precios reales de esas viviendas.

X_entrenamiento , X_prueba, y_entrenamiento, y_prueba = train_test_split( #80% de los datos para entrenamiento y 20% para prueba
    X, 
    y,
    test_size=0.2, # se selecciona el 20% de los datos para prueba
    random_state=42 # se selecciona el numero 42 para que los resultados sean reproducibles, es decir, que siempre se obtenga la misma división de datos al ejecutar el código varias veces
    stratify=datos["ciudad"] #intenta repoartir por igual las ciudades de los pisos en los conjuntos de entrenamiento y prueba
)

print("Viviendas para entrenar:", X_entrenamiento.shape)
print("Viviendas para probar:", X_prueba.shape)

#Creamos un procesamiento para que las columnas categóricas se conviertan en numeros (OneHotEncoding)

preprocesador=ColumnTransformer( #Permite convertir las columnas de textto como ciudad y zona a numericas
    transformers=[
        (
            "categorias", #nombre que le damos a la transformación
            OneHotEncoder(handle_unknown="ignore"), #convierte los valores de texto en columnas numericas
            columnas_categoricas
        )
    ],
    remainder="passthrough" #las columnas que no se transforman, se dejan tal
)

modelo=Pipeline([# creamos el modelo que primero preprocesa los datos y luego aplica la regresión lineal 
    ("preprocesamiento", preprocesador),
    ("regresion", LinearRegression())
])

modelo.fit(X_entrenamiento, y_entrenamiento) #entrenamos el modelo, ahora si aprende

print("Modelo entrenado con éxito!")


predicciones = modelo.predict(X_prueba) #Le damos al modelo las características de las viviendas de prueba y este devuelve sus precios estimados. 

print("Primeras cinco predicciones:") #el modelo predice el precio de las 400 viviendas de prueba
print(predicciones[:5])

print(type(predicciones)) 
print(predicciones.shape) #predicciones es un array de NumPy. Si hay 400 viviendas de prueba


comparacion= pd.DataFrame({ # creamos un DataFrame para comparar los valores reales y predichos
    "Real": y_prueba.to_numpy(), # Lo convertimos a NumPy para que Real y Predicho tengan el mismo formato y se emparejen únicamente por su posición.
    "Predicho": predicciones
})

comparacion["diferencia"] = (
    comparacion["Real"]
    - comparacion["Predicho"]
)

print("\nCOMPARACIÓN")
print(comparacion.head(10).round(2))

predicciones = modelo.predict(X_prueba) #Le damos al modelo las características de las viviendas que no habia visto durante el entrenamiento.




mae=mean_absolute_error(y_prueba, predicciones) #Calculamos el error absoluto medio (MAE) entre los precios reales y los predichos.
rmse=np.sqrt(mean_squared_error(y_prueba, predicciones)) #Calculamos el error cuadrático medio (RMSE) entre los precios reales y los predichos.
r2=r2_score(y_prueba, predicciones) #Calculamos el coeficiente de determinación (R²) entre los precios reales y los predichos.

print("\nMétricas de evaluación del modelo:")
print("MAE (Error absoluto medio):", mae)
print("RMSE (Raíz del error cuadrático medio):", rmse)
print("R² (Coeficiente de determinación):", r2)
