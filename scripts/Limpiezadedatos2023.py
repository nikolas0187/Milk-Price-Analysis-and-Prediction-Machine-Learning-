# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 09:34:45 2025

@author: Admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("anex-SIPSALeche-SerieHistoricaPrecios-2023.xlsx")

df4 = df.copy()

print(df4.info())

df4.drop(columns=['Nombre departamento', 'Nombre municipio '], inplace=True)

print(df4.columns)

print(df4.isnull().sum())

df4.rename (columns={"Mes y año": "Mes_y_año"}, inplace=True)
df4.rename (columns={"Código departamento": "Código_departamento"}, inplace=True) 
df4.rename (columns={"Código municipio": "Código_municipio"}, inplace=True)
df4.rename (columns={"Precio promedio por litro": "Precio_promedio_por_litro"}, inplace=True)
print(df4.columns)

df4.to_csv("precioleche2023_limpio.csv", index=False)

print(df4["Código_municipio"])

plt.hist(df4["Precio_promedio_por_litro"], bins=70, color=["red"], edgecolor="black", alpha=0.7)
plt.title("Histograma de Precio Promedio de Leche Cruda 2023")
plt.xlabel("Precio por litro")
plt.ylabel("Frecuencia")
plt.show()


promedio_precio_mes = df4.groupby('Mes_y_año')['Precio_promedio_por_litro'].mean()

print(promedio_precio_mes)

meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun","Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

plt.bar(meses, promedio_precio_mes, color = "green")
plt.title("Precio promedio de Leche Cruda por Mes 2023")
plt.xlabel("Mes")
plt.ylabel("Promedio Precio")
plt.legend()
plt.grid(True)
plt.show()