# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 22:35:26 2025

@author: nikol
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df1 = pd.read_csv('precioleche2025_limpio.csv')
df2 = pd.read_csv('precioleche2024_limpio.csv')
df3 = pd.read_csv('precioleche2023_limpio.csv')

df_final = pd.concat([df1, df2, df3], ignore_index=True)

df_final.to_csv("base_unida.csv", index=False)

df_final.isnull().sum()
df_final.dtypes

print(df_final.columns)

# Convierto la fecha que estaba en object a data time.
df_final['Mes_y_año'] = pd.to_datetime(df_final['Mes_y_año'])

df_final.dtypes

# Extraigo año, mes y día de la fecha para facilitar el machine learning.
df_final['año'] = df_final['Mes_y_año'].dt.year
df_final['mes'] = df_final['Mes_y_año'].dt.month
df_final['dia'] = df_final['Mes_y_año'].dt.day

df_final = df_final.drop('Mes_y_año', axis=1)

df_final.to_csv("base_unida.csv", index=False)

#Visualizar outliers
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=df_final["Precio_promedio_por_litro"])
plt.title("Boxplot Precio Promedio por Litro")

plt.tight_layout()
plt.show()


