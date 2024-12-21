#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:23:02 2024

@author: diegofmeijide
"""

# %% Importando módulos

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import statsmodels.api as sm     

from stargazer.stargazer import Stargazer
from sklearn.preprocessing import scale
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Evita la notación científica en numpy
pd.options.display.float_format = '{:.2f}'.format

# configurando directorio de trabajo
os.chdir("/Users/diegofmeijide/Documents/MAE/Big data/tp4")


# %% importando datos

h_2024 = pd.read_excel("datos/usu_hogar_T124.xlsx")
i_2024 = pd.read_excel("datos/usu_individual_T124.xlsx")
h_2004 = pd.read_stata("datos/Hogar_t104.dta", convert_categoricals=False)
i_2004 = pd.read_stata("datos/Individual_t104.dta", convert_categoricals=False)


# %% filtramos Bahía Blanca en 2024

h_2024.info(verbose = True)# verbose imprime resumen completo

h_2024["AGLOMERADO"]=pd.Categorical(h_2024["AGLOMERADO"])

h_2024 = h_2024[h_2024["AGLOMERADO"]==3]

i_2024.info(verbose = True)# verbose imprime resumen completo

i_2024["AGLOMERADO"]=pd.Categorical(i_2024["AGLOMERADO"])

i_2024 = i_2024[i_2024["AGLOMERADO"]==3]

# %% Mergeamos la base de individuos 2024 con la de hogares 2024

# Merge manteniendo columnas comunes de la izquierda
df_2024 = i_2024.merge(h_2024, on=["CODUSU", "NRO_HOGAR"], how='left', suffixes=('', '_right'))

# Eliminar columnas duplicadas provenientes de la derecha
for col in h_2024.columns:
    if col in i_2024.columns and col != "CODUSU" and col != "NRO_HOGAR":  # Evitar eliminar las claves de unión
        df_2024.drop(columns=[f"{col}_right"], inplace=True)


# %% filtramos Bahía Blanca en 2004

h_2004.info(verbose = True)# verbose imprime resumen completo

h_2004["aglomerado"]=pd.Categorical(h_2004["aglomerado"])

# Convertir la variable categórica de flotante a entero y luego a categórica
h_2004['aglomerado'] = h_2004['aglomerado'].astype(int).astype('category')

h_2004["aglomerado"].value_counts()

h_2004 = h_2004[h_2004["aglomerado"]==3]

i_2004.info(verbose = True)# verbose imprime resumen completo

h_2004["aglomerado"]=pd.Categorical(h_2004["aglomerado"])

# Convertir la variable categórica de flotante a entero y luego a categórica
h_2004['aglomerado'] = h_2004['aglomerado'].astype(int).astype('category')

i_2004["aglomerado"].value_counts()

i_2004 = i_2004[i_2004["aglomerado"]==3]


# %% Mergeamos la base de individuos 2004 con la de hogares 2004

# Merge manteniendo columnas comunes de la izquierda
df_2004 = i_2004.merge(h_2004, 
                       on=["CODUSU", "nro_hogar"], 
                       how='left', 
                       suffixes=('', '_right'))

# Eliminar columnas duplicadas provenientes de la derecha
for col in h_2004.columns:
    if col in i_2004.columns and col != "CODUSU" and col != "nro_hogar":  # Evitar eliminar las claves de unión
        df_2004.drop(columns=[f"{col}_right"], inplace=True)

# %% paso todas las columnas a minúscula en 2024 para que quede como en 2004

df_2024.columns = df_2024.columns.str.lower()

# %% modificando codusu

# Cambiar el nombre de la columna "CODUSU" a "codusu"
df_2004 = df_2004.rename(columns={"CODUSU": "codusu"})

# %% creando la variable año

# Asignar una nueva columna "año" con valor categórico "2004"
df_2004["año"] = pd.Categorical(["2004"] * len(df_2004))
df_2024["año"] = pd.Categorical(["2024"] * len(df_2024))

# %% concatenando ambas bases

# Concatenar 2004 y 2024 verticalmente
df = pd.concat([df_2004, df_2024], axis=0, ignore_index=True)

print(df.shape)


# %% Filtrando variables relevantes y modificando nombres de variables

df_clean = df[["año","codusu","nro_hogar","componente","pondera","ch04", "ch06", "ch07", 
               "ch08","ch03",
               "nivel_ed", "estado", "cat_inac", "ipcf",
               "v1","v2","v3","v4","v5","v6","v7","v8","v9",
               "v10","v11","v12","v13","v14","v15","v16",
               "v17","v18","v19_a","v19_b", 
               "iv2", "t_vi"]]

df_clean = df_clean[df_clean["t_vi"]!=-9]

df_clean = df_clean.rename(columns={'ch03': 'parentesco'})
df_clean = df_clean.rename(columns={'ch04': 'genero'})
df_clean = df_clean.rename(columns={'ch06': 'edad'})
df_clean = df_clean.rename(columns={'ch07': 'estado_civil'})
df_clean = df_clean.rename(columns={'ch08': 'cobertura_medica'})

# %% 1.3.missing values

print(df_clean.isnull().sum())

# no se presentan missing values

# %% 1.3.Outliers

plt.boxplot(df_clean['ipcf'], vert=False)
plt.title("Detección de Outliers en ipcf")
plt.xlabel("ipcf")
plt.show()

# Asegurarse de que 'ipcf' esté en formato correcto
data_x = df_clean['ipcf']

# 1. Gráfico de caja expandido
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x=data_x, ax=ax, color='yellowgreen', flierprops={'marker': 'o', 'color': 'red', 'alpha': 0.5})
ax.set_title('Boxplot para detectar outliers')
ax.set_xlabel('Valores de IPCF')
plt.show()

# 2. Gráfico de dispersión log-transformado
fig, ax = plt.subplots(figsize=(8, 4))
log_data_x = np.log1p(data_x)  # Log-transformar los datos
ax.scatter(range(len(log_data_x)), log_data_x, 
           alpha=0.6, color='tomato', s=5)
ax.set_title('Gráfico de dispersión con escala logarítmica')
ax.set_xlabel('Índice de Observación')
ax.set_ylabel('Log(1+IPCF)')
plt.show()

# 3. Histograma truncado
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(data_x, bins=30, color='thistle', alpha=0.7, range=(0, np.percentile(data_x, 95)))  # Truncado al percentil 95
ax.set_title('Histograma truncado para IPCF')
ax.set_xlabel('Valores de IPCF (truncados)')
ax.set_ylabel('Frecuencia')
plt.show()

# 4. Gráfico ECDF
fig, ax = plt.subplots(figsize=(8, 4))
sns.ecdfplot(data_x, ax=ax, color='orchid')
ax.set_title('Gráfico ECDF para IPCF')
ax.set_xlabel('Valores de IPCF')
ax.set_ylabel('Probabilidad acumulada')
plt.show()

# 5. Violin plot con escala logarítmica
fig, ax = plt.subplots(figsize=(8, 4))
sns.violinplot(x=np.log1p(data_x), ax=ax, color='yellowgreen', alpha=0.7)
ax.set_title('Violin Plot con escala logarítmica para IPCF')
ax.set_xlabel('Log(1+IPCF)')
plt.show()


# %% 1.4. Contrucción de variables relevantes que predigan desocupación

# 1. Cantidad de personas inactivas que viven en el hogar

# Filtrar las personas inactivas (estado == 3)
df_clean['inactiva'] = (df_clean['estado'] == 3).astype(int)

# Agrupar por codusu y nro_hogar y contar las personas inactivas
df_clean['cantidad_inactivos'] = df_clean.groupby(['codusu', 'nro_hogar'])['inactiva'].transform('sum')

# Eliminar la columna auxiliar si no la necesitas
df_clean.drop(columns=['inactiva'], inplace=True)


# 2. Variable de hacinamiento

# Contar la cantidad de miembros en cada hogar
df_clean['miembros_hogar'] = df_clean.groupby(['codusu', 'nro_hogar'])['componente'].transform('count')

# Crear la variable 'hacinamiento' basada en la condición de más de 3 miembros por habitación
df_clean['hacinamiento'] = (df_clean['miembros_hogar'] / df_clean['iv2'] > 3).astype(int)


# 3. inegreso no laboral per cápita, la no respuesta de t_vi se identifica con -9 y hay 74

# Calcular el ingreso no laboral del hogar per cápita
df_clean['ingreso_no_laboral_pc'] = df_clean['t_vi'] / df_clean['miembros_hogar']



# %% 1.5. Estadísticas descriptivas

# Supongamos que ya tienes tu DataFrame 'df_clean'
# Seleccionar las variables que quieres incluir en la tabla
variables = ['cantidad_inactivos', 'hacinamiento', 'ingreso_no_laboral_pc', 'genero', 'edad']
df_subset = df_clean[variables]

# Calcular estadísticas descriptivas
descriptives = df_subset.describe(include='all').transpose()

descriptives

# Crear la nueva variable
# df_clean['diferencia_ingreso'] = df_clean['ipcf'] - df_clean['ingreso_no_laboral_pc']
# Calcular estadísticas descriptivas de la nueva variable
# estadisticas_diferencia = df_clean['diferencia_ingreso'].describe()
# Imprimir las estadísticas descriptivas
# print("Estadísticas descriptivas de la variable 'diferencia_ingreso':")
# print(estadisticas_diferencia)

# %% 1.6. Tasa de desocupación para el aglomerado

# Crear la variable 'desocupado' en función de la columna 'estado'
df_clean['desocupado'] = (df_clean['estado'] == 2).astype(int)

df_hogares = df_clean[df_clean["parentesco"]==1]

# Calcular la tasa de desocupación a nivel general
hogares_totales = df_hogares['pondera'].sum()
hogares_desocupados = df_hogares.loc[df_hogares['desocupado'] == 1, 'pondera'].sum()
tasa_desocupacion = (hogares_desocupados / hogares_totales) * 100

# Mostrar el resultado
print(f"Hogares totales ponderados: {hogares_totales}")
print(f"Hogares desocupados ponderados: {hogares_desocupados}")
print(f"Tasa de desocupación: {tasa_desocupacion:.2f}%")


# %% 2.1 Entrenamiento y test

df_04 = df_clean[df_clean["año"]=="2004"]

df_24 = df_clean[df_clean["año"]=="2024"]

# Identificar las columnas que comienzan con 'v'
columns_starting_with_v = [col for col in df_04.columns if col.startswith('v')]

# Convertir esas columnas a categóricas
for col in columns_starting_with_v:
    df_04[col] = pd.Categorical(df_04[col])
    
df_04.info(verbose = True)# verbose imprime resumen completo

# Creamos variables dummies para las variables string
dummies_04 = pd.get_dummies(df_04[[
                                   'estado_civil',
                                   'cobertura_medica',
                                   'nivel_ed']], drop_first=True)

dummies_24 = pd.get_dummies(df_24[[
                                   'estado_civil',
                                   'cobertura_medica',
                                   'nivel_ed']], drop_first=True)

# Definimos las variables que incluiremos en el set de X
# Eliminamos salarios (porque es nuestra y) y las columnas de strings
X_04_ = df_04.drop(['estado_civil', 'cobertura_medica', 'nivel_ed', 
                   'iv2', 't_vi', 'año', 'codusu', 'nro_hogar', 'componente',
                   'pondera'], axis=1).astype('float64')

X = pd.concat([X_04_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)

# seleccionando X e y 2004
y_04 = df_04.desocupado
X_04 = df_04.

# seleccionando X e y 2024
y_24 = df_24.desocupado
X_24 = df_24.

# Train test split 2004
X_train_04, X_test_04, y_train_04, y_test_04 = train_test_split(X, y, test_size=0.3, random_state=101)

# Train test split 2024
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)





































