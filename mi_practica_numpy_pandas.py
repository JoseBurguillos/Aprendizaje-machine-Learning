import numpy as np
import pandas as pd

datos=pd.read_csv("viviendas.csv")

print(datos.head())

print("Dimensiones")
print(datos.shape) #shape indica el tamaño del DataFrame: 2000 columnas y 13 filas

print("Columnas")
print(datos.columns) #datos.columns muestra los nombres de todas las columnas:

print("Tipos de datos")
print(datos.dtypes) #dtypes muestra qué tipo de información contiene cada columna. Por ejemplo


print("Estadisticas pandas")

precios=datos["precio"] #seleccionamos la columna precio y la guardamos en una variable

print("Precio Max")
print(precios.max()) #max() nos devuelve el valor máximo de la columna precio

print("Precio Min")
print(precios.min()) #min() nos devuelve el valor mínimo de la columna precio

print("Precio Mediano")
print(precios.median()) #median() nos devuelve el valor mediano de la columna precio

print("Media")
print(precios.mean())


print("Numpy")

precios_numpy=precios.to_numpy() #convertimos la columna precio a un array de numpy

print("Precio Max")
print(np.max(precios_numpy)) #np.max() nos devuelve el valor máximo del array de numpy

print("Precio Min")
print(np.min(precios_numpy)) #np.min() nos devuelve el valor mínimo del array de numpy

print("Precio Mediano")
print(np.median(precios_numpy)) #np.median() nos devuelve el valor mediano del array de numpy

print("Media")
print(np.mean(precios_numpy)) #np.mean() nos devuelve la media del array de numpy

print("Filtros")

filtro=(
    (datos["precio"] < 250000) & (datos["habitaciones"] < 3)
)

viviendas_filtradas=datos[filtro] #aplicamos el filtro


print("\nVIVIENDAS FILTRADAS")
print(viviendas_filtradas.head(10))


print("Cantidad encontrada:", len(viviendas_filtradas)) #len sirve para contar(Lenght)
