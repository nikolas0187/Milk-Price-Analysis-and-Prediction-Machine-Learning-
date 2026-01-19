# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 11:36:32 2025

@author: camil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("anex-SIPSALeche-SerieHistoricaPrecios-2025.xlsx")

df4 = df.copy()

#primeras 5 filas
print(df4.head())

#ultimas 5 filas
print(df4.tail())

#dimensiones del data set
print(df4.shape)

#nombre de las columnas
print(df4.columns)

#tipo de datos y valores nulos
print(df4.info())

#estadistica basica
print(df4.describe())

# Elimino columnas no útiles para análisis inicial
df4.drop(columns=['Nombre departamento', 'Nombre municipio '], inplace=True)

print(df4.columns)

print(df4.duplicated().sum()) 

##Renombrar una Columna 
df4.rename (columns={"Mes y año": "Mes_y_año"}, inplace=True)
df4.rename (columns={"Código departamento": "Código_departamento"}, inplace=True) 
df4.rename (columns={"Código municipio": "Código_municipio"}, inplace=True)
df4.rename (columns={"Precio promedio por litro": "Precio_promedio_por_litro"}, inplace=True)
print(df4.columns)

#nombre de las columnas
print(df4.columns)


df4.to_csv("precioleche2025_limpio.csv", index=False)

