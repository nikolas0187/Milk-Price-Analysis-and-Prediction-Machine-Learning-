# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 09:20:33 2025

@author: Admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("anex-SIPSALeche-SerieHistoricaPrecios-2024.xlsx")

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

df4.to_csv("precioleche2024_limpio.csv", index=False)


