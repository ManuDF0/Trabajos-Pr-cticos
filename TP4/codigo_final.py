#______________________________________________________________________________________________________________________________#
#
#                                            TRABAJO PRÁCTICO N°4 MACHINE LEARNING 
#                           Alumnos: Manuel Díaz de la Fuente, Diego Fernández Meijide y Sofía Kastika 
#                                               Profesor: Walter Sosa Escudero
#                                                 Asistente: Tomás Pacheco 
#______________________________________________________________________________________________________________________________#

#______________________________________________________________________________________________________________________________#
#
#                                          PARTE I: Análisis de la base de hogares y tipo de ocupación
#______________________________________________________________________________________________________________________________#

# %% Importamos módulos
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import statsmodels.api as sm     


from stargazer.stargazer import Stargazer
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from typing import Literal

# Evitamos la notación científica en numpy
pd.options.display.float_format = '{:.2f}'.format

# configurando directorio de trabajo
os.chdir("/Users/diegofmeijide/Documents/GitHub/Trabajos-Pr-cticos/TP4")
#os.chdir("C:/Users/sofia/Desktop/Maestría/Tercer trimestre/Machine Learning/Trabajos-Pr-cticos/TP4")

# %% importamos datos

h_2024 = pd.read_excel("./input/usu_hogar_T124.xlsx")
i_2024 = pd.read_excel("./input/usu_individual_T124.xlsx")
h_2004 = pd.read_stata("./input/Hogar_t104.dta", convert_categoricals=False)
i_2004 = pd.read_stata("./input/Individual_t104.dta", convert_categoricals=False)

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

# %% Pasamos todas las columnas a minúscula en 2024 para que quede como en 2004

df_2024.columns = df_2024.columns.str.lower()

# %% Modificando codusu

# Cambiar el nombre de la columna "CODUSU" a "codusu"
df_2004 = df_2004.rename(columns={"CODUSU": "codusu"})

# %% creando la variable año

# Asignar una nueva columna "año" con valor categórico "2004"
df_2004["año"] = pd.Categorical(["2004"] * len(df_2004))
df_2024["año"] = pd.Categorical(["2024"] * len(df_2024))

# %% Concatenando ambas bases

# Concatenar 2004 y 2024 verticalmente
df = pd.concat([df_2004, df_2024], axis=0, ignore_index=True)

print(df.shape)

# %% Filtrando variables relevantes y modificando nombres de variables

df_clean = df[["año","codusu","nro_hogar","componente","pondera","ch04", "ch06", "ch07", 
               "ch08","ch03",
               "nivel_ed", "estado", "cat_inac", "ipcf",
               "v2", "v5","v6","v7","v8","v9",
               "v10","v11","v12","v13","v14","v15","v16",
               "v17","v18","v19_a","v19_b", 
               "iv2", "t_vi", "iv3", "iv4", "iv5", "iv6", "iv7", "iv8", "iv9", "iv10", "iv11", "ii3", "iv12_3" ]]

df_clean = df_clean.rename(columns={'ch03': 'parentesco'})
df_clean = df_clean.rename(columns={'ch04': 'genero'})
df_clean = df_clean.rename(columns={'ch06': 'edad'})
df_clean = df_clean.rename(columns={'ch07': 'estado_civil'})
df_clean = df_clean.rename(columns={'ch08': 'cobertura_medica'})

# %% 1.3.Missing values

print(df_clean.isnull().sum()) # no se presentan missing values

# Missing values para ingreso no laboral individual (se identifica con -9)
count_9= df['t_vi'].value_counts().get(-9, 0)
print(f"Cantidad de -9: {count_9}") # vemos que hay 74 missing values en ingreso no laboral per capita
df_clean = df_clean[df_clean["t_vi"]!=-9] # eliminamos los missing values del ingreso no laboral 

# Chequeamos que se hayan eliminado los missing values 
count_9_nuevo= df_clean['t_vi'].value_counts().get(-9, 0)
print(f"Cantidad de -9: {count_9_nuevo}")  
df_clean = df_clean[df_clean["t_vi"]!=-9] 

# %% 1.3. Valores negativos 

# Edad 
print(df_clean['edad'].dtype) # variable del tipo float
obs = df_clean.shape[0] # cantidad de observaciones 2090
obs
# Nos quedamos con los valores que son mayores o iguales a cero 
df_clean = df_clean.loc[(df_clean['edad'] >= 0)] 
obs = df_clean.shape[0] # pasamos a 2068 observaciones. Hay 22 edades negativas
obs

# Ingreso per cápita (ipcf)
print(df_clean['ipcf'].dtype) # variable del tipo numérica
# Nos quedamos con los valores que son mayores o iguales a cero 
df_clean = df_clean.loc[(df_clean['ipcf'] >= 0)] 
obs = df_clean.shape[0] 
obs # seguimos con 2068, no hay valores negativos 

# Ingreso no laboral per cápita (t_vi)
print(df_clean['t_vi'].dtype) # variable del tipo numérica
# Nos quedamos con los valores que son mayores o iguales a cero 
df_clean = df_clean.loc[(df_clean['t_vi'] >= 0)] 
obs = df_clean.shape[0] 
obs # seguimos con 2068, no hay valores negativos 

# %% 1.3.Outliers

# 1. Ingreso per cápita familiar 
# Nos aseguramos de que 'ipcf' esté en formato correcto
data_x = df_clean['ipcf']
# Gráfico de caja expandido
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x=data_x, ax=ax, color='yellowgreen', flierprops={'marker': 'o', 'color': 'red', 'alpha': 0.5})
ax.set_title('')
ax.set_xlabel('Ingreso Per Cápita Familiar (en millones de pesos)', fontsize=14)
fig.savefig("./output/boxplot_ipcf.png")
plt.show()

# 2. Ingreso no laboral per cápita 
# Nos aseguramos de que 't_vi' esté en formato correcto
data_x2 = df_clean['t_vi']
# Gráfico de caja expandido
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x=data_x2, ax=ax, color='yellowgreen', flierprops={'marker': 'o', 'color': 'red', 'alpha': 0.5})
ax.set_title('')
ax.set_xlabel('Ingreso No Laboral Per Cápita (en millones de pesos)', fontsize=14)
fig.savefig("./output/boxplot_tvi.png")
plt.show()


# 3. Edad
# Nos aseguramos de que 't_vi' esté en formato correcto
data_x3 = df_clean['edad']
# Gráfico de caja expandido
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x=data_x3, ax=ax, color='yellowgreen', flierprops={'marker': 'o', 'color': 'red', 'alpha': 0.5})
ax.set_title('')
ax.set_xlabel('Edad', fontsize=14)
fig.savefig("./output/boxplot_edad.png")
plt.show()

# %% 1.3 Generamos dummys para variables categóricas 

# Pasamos a formato categórico las variables de fuentes alternativas de ingreso
print(df_clean['v2'].dtype) 
# Identificar las columnas que comienzan con 'v'
columns_starting_with_v = [col for col in df_clean.columns if col.startswith('v')]
# Convertir esas columnas a categóricas
for col in columns_starting_with_v:
    df_clean[col] = pd.Categorical(df_clean[col])
df_clean.info(verbose = True)# verbose imprime resumen completo

# Pasamos a formato categórico las otras variables 
other_columns = ["cobertura_medica", "nivel_ed", "estado_civil", "iv3", "iv4", "iv5", "iv6", "iv7", "iv8", "iv9", "iv10", "iv11", "ii3", "iv12_3"]
for col in other_columns:
    df_clean[col] = pd.Categorical(df_clean[col])
df_clean.info(verbose = True)# verbose imprime resumen completo

# Generamos dummys para variables con más de una categoría 
# Primero pasamos a formato categórico a las variables
categoricas = ["cobertura_medica", "nivel_ed", "estado_civil", "iv3", "iv4", "iv5", "iv6", "iv7", "iv8", "iv9", "iv10", "iv11", "ii3", "iv12_3"]
for var in categoricas:
    df_clean[var] = pd.Categorical(df_clean[var])

# Generamos las variables dummy
dummies = pd.get_dummies(df_clean[categoricas], 
                          drop_first=True,
                          dtype=int)

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


# 3. ingreso no laboral per cápita

# Calcular el ingreso no laboral del hogar per cápita
df_clean['sum_t_vi_hogar'] = df_clean.groupby(['codusu', 'nro_hogar'])['t_vi'].transform('sum')
df_clean['ingreso_no_laboral_pc'] = df_clean['sum_t_vi_hogar'] / df_clean['miembros_hogar']

# %% 1.5. Estadísticas descriptivas

# Calculamos para cada año los porcentajes para cada variable 
# 1) Subsidios 
data_2004_counts1 = df_clean[df_clean['año'] == "2004"]['v5'].value_counts(normalize=True) * 100 
data_2024_counts1 = df_clean[df_clean['año'] == "2024"]['v5'].value_counts(normalize=True) * 100
# 2) Ahorros 
data_2004_counts2 = df_clean[df_clean['año'] == "2004"]['v15'].value_counts(normalize=True) * 100 
data_2024_counts2 = df_clean[df_clean['año'] == "2024"]['v15'].value_counts(normalize=True) * 100
# 3) Lugar de trabajo 
df_clean = df_clean[df_clean['ii3'] != 0]
data_2004_counts3 = df_clean[df_clean['año'] == "2004"]['ii3'].value_counts(normalize=True) * 100 
data_2024_counts3 = df_clean[df_clean['año'] == "2024"]['ii3'].value_counts(normalize=True) * 100
# 4) Villa de emergencia 
data_2004_counts4 = df_clean[df_clean['año'] == "2004"]['iv12_3'].value_counts(normalize=True) * 100 
data_2024_counts4 = df_clean[df_clean['año'] == "2024"]['iv12_3'].value_counts(normalize=True) * 100




# Armamos una función para hacer gráficos de barras que comparen entre años para cuando hay 2 categorías 
def composicion_bar(val_2004, val_2024, x_label, output_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # creamos una figura con 2 subplots 

    # Gráfico de barras para 2004
    bars_2004 = axs[0].bar(val_2004.index, val_2004.values, color='skyblue')
    axs[0].set_title('Composición en 2004', fontsize=13)  # título del gráfico 
    axs[0].set_xlabel(x_label, fontsize=12)  # título eje x con tamaño más grande
    axs[0].set_ylabel('Porcentaje', fontsize=12)  # título eje y c
    axs[0].set_ylim(0, 100)  # establecemos los límites del eje y entre 0 y 100
    axs[0].set_xticks([])  # quitar números del eje x

    for bar, value, label in zip(bars_2004, val_2004.values, ["No", "Sí"]):
        axs[0].text(
            bar.get_x() + bar.get_width() / 2, 
            value + 2, 
            f'{label}: {value:.1f}%', 
            ha='center', 
            fontsize=10  # tamaño de la fuente de los textos sobre las barras
        )

    # Gráfico de barras para 2024
    bars_2024 = axs[1].bar(val_2024.index, val_2024.values, color='salmon')
    axs[1].set_title('Composición en 2024', fontsize=14)  # título del gráfico 
    axs[1].set_xlabel(x_label, fontsize=12)  # título eje x 
    axs[1].set_ylim(0, 100)  # establecemos los límites del eje y entre 0 y 100
    axs[1].set_xticks([])  # sacamos números del eje x

    for bar, value, label in zip(bars_2024, val_2024.values, ["No", "Sí"]):
        axs[1].text(
            bar.get_x() + bar.get_width() / 2, 
            value + 2, 
            f'{label}: {value:.1f}%', 
            ha='center', 
            fontsize=10  # tamaño de la fuente de los textos sobre las barras
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    plt.show() 

# Graficamos las composiciones por año
composicion_bar(data_2004_counts1, data_2024_counts1, 'Subsidios', "./output/subsidios.png")
composicion_bar(data_2004_counts2, data_2024_counts2, 'Préstamos', "./output/ahorros.png")
composicion_bar(data_2004_counts3, data_2024_counts3, 'Espacio de Trabajo', "./output/trabajo.png")
composicion_bar(data_2004_counts4, data_2024_counts4, 'Villa de Emergencia', "./output/villa.png")

# Variable Pisos. Más de 2 categorías
# 5) Pisos 
data_2004_counts5 = df_clean[df_clean['año'] == "2004"]['iv3'].value_counts(normalize=True) * 100 
data_2024_counts5 = df_clean[df_clean['año'] == "2024"]['iv3'].value_counts(normalize=True) * 100

# Definimos función
def composicion_barras_3categorias(data_2004_counts, data_2024_counts, labels, output_path=None):
    # Definir la posición de las barras
    x = np.arange(len(labels))  # la ubicación de las categorías en el eje x
    width = 0.35  # el ancho de las barras

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # creamos una figura con 2 subplots (uno al lado del otro)

    # Gráfico de barras para 2004
    bars_2004 = axs[0].bar(x - width/2, data_2004_counts, width, color='skyblue')
    axs[0].set_title('Composición en 2004', fontsize=16)  # Título 
    axs[0].set_ylabel('Porcentaje', fontsize=14)  # Título del eje Y
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels, fontsize=12)  # Etiquetas del eje X
    axs[0].set_ylim(0, 100)  # Limitamos el eje y entre 0 y 100

    # Agregamos texto con los porcentajes sobre las barras para 2004
    for bar in bars_2004:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2, height + 2, f'{height:.1f}%', ha='center', fontsize=10)

    # Gráfico de barras para 2024
    bars_2024 = axs[1].bar(x + width/2, data_2024_counts, width, color='salmon')
    axs[1].set_title('Composición en 2024', fontsize=16)  # Título más grande
    axs[1].set_ylabel('Porcentaje', fontsize=14)  # Título del eje Y más grande
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, fontsize=12)  # Etiquetas del eje X más grandes
    axs[1].set_ylim(0, 100)  # Limitar el eje y entre 0 y 100

    # Agregamos texto con los porcentajes sobre las barras para 2024
    for bar in bars_2024:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width() / 2, height + 2, f'{height:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    plt.show()

# Ejemplo de uso con datos ficticios
labels = ['Baldosa', 'Cemento', 'Tierra']
composicion_barras_3categorias(data_2004_counts5, data_2024_counts5, labels, "./output/pisos.png")

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

#______________________________________________________________________________________________________________________________#
#
#                                          PARTE II: Clasificación y Regularización
#______________________________________________________________________________________________________________________________#

# %% 2.1. Base de prueba y base de entrenamiento 
           
# Generamos bases respondieron y no respondieron
respondieron = df_clean.loc[df_clean['estado'] != 0] # creamos base con personas que respondieron su condición de actividad
norespondieron = df_clean.loc[df_clean['estado'] == 0] # creamos base con personas que no respondieron su condición de actividad

# Generamos variable desocupada
desocupada = pd.DataFrame(
    np.where(respondieron['estado'].isin([2]), 1, 0),  # Creamos la columna "desocupada"
    columns=['desocupada'],
    index=respondieron.index  # Le ponemos el index del df respondieron para que haga bien el concat
)
respondieron = pd.concat([respondieron, desocupada], axis=1) # Le agregamos la columna al df

# Definimos las columnas de interés
columnas = ["genero", "edad", "estado_civil", "cobertura_medica","parentesco",
            "nivel_ed", "ipcf","v2", "v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15",
            "v16", "v17","v18","v19_a","v19_b","iv2", "t_vi", "iv3", "iv4", "iv5", "iv6", "iv7", "iv8", "iv9", "iv10", 
            "iv11", "ii3", "iv12_3", "hacinamiento", "ingreso_no_laboral_pc", "cantidad_inactivos"]

# Definimos para cada año variable explicada
y_2004 = respondieron[respondieron["año"]=="2004"].desocupada
y_2024 = respondieron[respondieron["año"]=="2024"].desocupada

# Definimos para cada año vector de variables explicativas (en formato dummy) 
x_2004 = pd.get_dummies(respondieron[respondieron['año']=="2004"][columnas])
x_2024 = pd.get_dummies(respondieron[respondieron['año']=="2024"][columnas])


# Agregamos constantes
x_2004['constante'] = 1
x_2024.loc[:,'constante'] = 1

# Asignamos el 30% de cada base a testeo y por ende el 70% a entrenamiento 
x_train_2004, x_test_2004, y_train_2004, y_test_2004 = train_test_split(x_2004, y_2004, test_size = 0.3, random_state = 101)
x_train_2024, x_test_2024, y_train_2024, y_test_2024 = train_test_split(x_2024, y_2024, test_size = 0.3, random_state = 101)

# Para regularización hace falta estandarizar primero las variables, para que no pondere más por una cuestión de varianza
# Estandarizamos las variables
sc = StandardScaler()
x_train_2004 = pd.DataFrame(sc.fit_transform(x_train_2004), index= x_train_2004.index, columns= x_train_2004.columns)
x_train_2024 = pd.DataFrame(sc.fit_transform(x_train_2024), index= x_train_2024.index, columns= x_train_2024.columns)
x_test_2004 = pd.DataFrame(sc.fit_transform(x_test_2004), index= x_test_2004.index, columns= x_test_2004.columns)
x_test_2024 = pd.DataFrame(sc.fit_transform(x_test_2024), index= x_test_2024.index, columns= x_test_2024.columns)


x_train_2004.info(verbose = True)
# %% 2.4. Implementamos las penalidades de LASSo y Ridge para la regresión Logística

# Armamos una función que estima un modelo logit con distintas penalidades y devuelve las métricas para evaluarlo
def logit_penalty_eval(x_train, x_test, y_train, y_test, penalty: Literal['l1', 'l2']):
    if penalty == 'l1':
        solver = 'liblinear' # La penalidad l1 no funciona con el solver por default de logit
    else:
        solver = 'lbfgs'
    
    logit = LogisticRegression(penalty= penalty, solver=solver) 
    logit.fit(x_train, y_train) # Estimamos el modelo
    y_pred = logit.predict(x_test) # Predecimos fuera de la muestra
    y_prob = logit.predict_proba(x_test)[:, 1]
    
    # Evaluamos
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    
    return cm, auc, accuracy, fpr, tpr

# %% 2.4. Curva ROC

def plot_roc(fpr, tpr, auc, year, penalty):
    plt.figure()
    plt.plot(fpr, tpr, label= f'Curva ROC (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0]) # ajustamos escala eje x
    plt.ylim([0.0, 1.05]) # ajustamos escala eje y 
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.legend(loc='lower right')
    plt.text(0.5, -0.2, f'Penalty utilizado para el año {year}: {penalty}', 
             ha='center', va='center', fontsize=10, transform=plt.gca().transAxes)
    plt.savefig(f'./output/roc_{year}_{penalty}.png')
    plt.show()
    
# %% 2.4 Estimando los modelos y obteniendo las métrocas de rendimiento.

    
# 1) Generar la lista de resultados
results = []

# Para 2004
for p in ['l1', 'l2']:
    cm, auc, accuracy, fpr, tpr = logit_penalty_eval(
        x_train_2004, x_test_2004, y_train_2004, y_test_2004, p
    )
    # cm suele ser de la forma [[TN, FP], [FN, TP]]
    results.append({
        'year': 2004,
        'penalty': p,
        'cm': cm,
        'auc': auc,
        'acc': accuracy
    })
    
    plot_roc(fpr, tpr, auc, 2004, p)

# Para 2024
for p in ['l1', 'l2']:
    cm, auc, accuracy, fpr, tpr = logit_penalty_eval(
        x_train_2024, x_test_2024, y_train_2024, y_test_2024, p
    )
    results.append({
        'year': 2024,
        'penalty': p,
        'cm': cm,
        'auc': auc,
        'acc': accuracy
    })
    
    plot_roc(fpr, tpr, auc, 2024, p)

# 2) Imprimir el código LaTeX de la tabla en pantalla (stdout).
#    Si deseas guardarlo en un archivo, cambia los 'print' por escrituras en un archivo .tex.
print(r"\begin{table}[H]")
print(r"\centering")
print(r"\begin{tabular}{cccccccc}")
print(r"\hline")
print(r"Año & Penalidad & TN & FP & FN & TP & AUC & Accuracy \\")
print(r"\hline")

for item in results:
    year = item['year']
    penalty = item['penalty']
    cm = item['cm']   # [[TN, FP], [FN, TP]]
    auc_val = item['auc']
    acc_val = item['acc']
    
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    # Ajusta el número de decimales de AUC y Accuracy si lo deseas
    print(r"{} & {} & {} & {} & {} & {} & {:.3f} & {:.3f} \\".format(
        year, penalty, tn, fp, fn, tp, auc_val, acc_val
    ))

print(r"\hline")
print(r"\end{tabular}")
print(r"\caption{Resultados de la evaluación: matriz de confusión, AUC y Accuracy.}")
print(r"\label{tab:resultados}")
print(r"\end{table}")

    
# %% 2.5 Grilla de lambdas 

alphas = [10**i for i in range(-5, 5, 1)] # Armamos la grilla de valores posibles para lambda


# %% 2.5 Función para el boxplot

# Función para el boxplot
def box(data, year, model):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='variable', y='value', data=data)
    plt.xlabel("Alpha")
    plt.ylabel("Error cuadrático medio")
    plt.text(0.5, -0.2, f'Modelo: {model} - Año: {year}', ha='center', va='center', fontsize=10, transform=plt.gca().transAxes)
    plt.savefig(f'./output/boxplot_{year}_{model}.png')
    plt.show()

# %% 2.5 Función para el lineplot

# Función para el lineplot
def line_prop(prop, year):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=prop, x='variable', y='value', marker='o', color='b')
    plt.xscale('log')
    plt.xlabel(r'$\alpha$ (log)', fontsize=12)
    plt.ylabel('Proporción de Coeficientes = 0', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.text(0.5, -0.2, f'Año: {year}', ha='center', va='center', fontsize=10, transform=plt.gca().transAxes)
    plt.savefig(f'./output/lineplot_{year}.png')
    plt.tight_layout()
    plt.show()

# %% 2.5 Función para hacer kfold con Ridge

# Función para hacer kfold con Ridge
def ridge_cv(hyperparams, X_train, y_train):
	kf = KFold(n_splits=10, shuffle=True, random_state=101) # Definimos el objeto kfolds
	best_hyperparam = None # Inicializamos la variable que guardará el mejor hiperparámetro
	best_mse = float('inf') # Inicializamos el mejor mse con un valor muy grande
	mse_values = {param: [] for param in hyperparams} # Inicializamos un diccionario que guardará los mse para cada hiperparámetro
	
	for param in hyperparams: # Iteramos sobre los hiperparámetros
		model = LogisticRegression(penalty='l2', C=1/param, max_iter=10000) # Definimos el modelo con el hiperparámetro correspondiente
		fold_mse = [] # Inicializamos una lista que guardará los mse de cada fold
		
		for train_index, val_index in kf.split(X_train): # Iteramos sobre los kfolds
			X_train_fold, X_val_fold = X_train.iloc[train_index].values, X_train.iloc[val_index].values # Definimos los conjuntos de entrenamiento y validación
			y_train_fold, y_val_fold = y_train.iloc[train_index].values, y_train.iloc[val_index].values
			
			model.fit(X_train_fold, y_train_fold) # Ajustamos el modelo
			y_pred = model.predict(X_val_fold) # Predecimos en el conjunto de validación
			mse = mean_squared_error(y_val_fold, y_pred) # Calculamos el mse
			fold_mse.append(mse) # Guardamos el mse en la lista de mses
		
		mse_values[param] = fold_mse # Guardamos los mses de los kfolds para el hiperparámetro actual
		avg_mse = np.mean(fold_mse) # Calculamos el mse promedio para el hiperparámetro actual
		
		if avg_mse < best_mse: # Si el mse promedio es mejor que el mejor mse hasta el momento
			best_mse = avg_mse # Actualizamos el mejor mse
			best_hyperparam = param # Actualizamos el mejor hiperparámetro
	
	return best_hyperparam, mse_values


# %% 2.6 Estimando Ridge 2004
alpha, mses = ridge_cv(alphas, x_train_2004, y_train_2004)
print(f'El mejor alpha para Ridge en 2004 es: {alpha}')
box(pd.DataFrame(mses).melt(), 2004, 'Ridge')


# %% 2.6 Estimando Ridge 2024

alpha, mses = ridge_cv(alphas, x_train_2024, y_train_2024)
print(f'El mejor alpha para Ridge en 2024 es: {alpha}')
box(pd.DataFrame(mses).melt(), 2024, 'Ridge')

# %% 2.6 Función para hacer kfold con LASSO

# Función para calcular el mejor alpha y las métricas de evaluación
def lasso_logistic_cv(hyperparams, X_train, y_train):
	kf = KFold(n_splits=10, shuffle=True, random_state=101) # Definimos la partición de k-folds
	best_hyperparam = None
	best_mse = float('inf')
	mse_values = {param: [] for param in hyperparams} # Armamos un diccionario para guardar los mse de cada fold
	zero_coef_proportions = {param: [] for param in hyperparams} # Armamos un diccionario para guardar las proporciones de coeficientes nulos de cada fold
	
	for param in hyperparams: # Iteramos por cada valor de alpha
		model = LogisticRegression(penalty='l1', solver='saga', C=1/param, max_iter=7000, n_jobs= 7) # Definimos el modelo
		fold_mse = [] # Armamos una lista para guardar los mse de cada fold
		fold_zero_coef_proportions = [] # Armamos una lista para guardar las proporciones de coeficientes nulos de cada fold
		
		for train_index, val_index in kf.split(X_train): # Iteramos por cada fold
			X_train_fold, X_val_fold = X_train.iloc[train_index].values, X_train.iloc[val_index].values # Definimos los conjuntos de entrenamiento y validación
			y_train_fold, y_val_fold = y_train.iloc[train_index].values, y_train.iloc[val_index].values
			
			model.fit(X_train_fold, y_train_fold) # Estimamos el modelo
			y_pred = model.predict(X_val_fold) # Predecimos
			mse = mean_squared_error(y_val_fold, y_pred) # Calculamos el mse
			fold_mse.append(mse) # Guardamos el mse
			
			zero_coef_proportion = np.mean(model.coef_ == 0) # Calculamos la proporción de coeficientes nulos
			fold_zero_coef_proportions.append(zero_coef_proportion) # Guardamos la proporción de coeficientes nulos
		
		mse_values[param] = fold_mse # Guardamos los mse del fold para el alpha actual
		zero_coef_proportions[param] = fold_zero_coef_proportions # Guardamos las proporciones de coeficientes nulos del fold para el alpha actual
		avg_mse = np.mean(fold_mse) # Calculamos el mse promedio para comparar con el que seteamos como mejor hasta ahora
		
		if avg_mse < best_mse: # Si el mse promedio es mejor que el mejor hasta ahora
			best_mse = avg_mse # Actualizamos el mejor mse
			best_hyperparam = param # Actualizamos el mejor alpha
	
	return best_hyperparam, mse_values, zero_coef_proportions


# %% 2.6 Estimando LASSO para 2004

alpha, mses, prop = lasso_logistic_cv(alphas, x_train_2004, y_train_2004)
print(f'El mejor alpha para LASSO en 2004 es: {alpha}')
box(pd.DataFrame(mses).melt(), 2004, 'LASSO')
promedios = pd.DataFrame({key: sum(value) / len(value) for key, value in prop.items()}, index= [0]).melt()
line_prop(promedios, 2004)


# %% 2.6 Estimando LASSO para 2024

alpha, mses, prop = lasso_logistic_cv(alphas, x_train_2024, y_train_2024)
print(f'El mejor alpha para LASSO en 2024 es: {alpha}')
box(pd.DataFrame(mses).melt(), 2024, 'LASSO')
promedios = pd.DataFrame({key: sum(value) / len(value) for key, value in prop.items()}, index= [0]).melt()
line_prop(promedios, 2024)

# %% 2.6-2.7 Función de resultados LASSO 

# Ahora queremos ver cuales son las variables nulas para cada modelo LASSO con el alpha optimo
def Lasso_logit(x_train, y_train, x_test, y_test, alpha):
    model = LogisticRegression(penalty='l1', solver='saga', C=1/alpha, max_iter=7000, n_jobs=7).fit(x_train, y_train)
    var_names = x_train.columns
    coefs = model.coef_[0]
    mse = mean_squared_error(y_test, model.predict(x_test)) # También guardamos el mse para evaluar el modelo
    return dict(zip(var_names, coefs)), mse

# %% 2.6-2.7 Resultados LASSO 2004

# Para 2004
coefs, mse = Lasso_logit(x_train_2004, y_train_2004, x_test_2004, y_test_2004, 10)
lasso_coefs = pd.DataFrame(coefs, index= [0]).melt() # Guardamos los coeficientes en un df para 2004

print(f'El ECM para LASSO en 2004 es: {mse}') # Imprimimos el mse para 2004

# %%  2.6-2.7 Resultados LASSO 2024

# Para 2024
coefs, mse = Lasso_logit(x_train_2024, y_train_2024, x_test_2024, y_test_2024, 10) # Hacemos lo mismo con los coefs de 2024
lasso_coefs = pd.concat([lasso_coefs, pd.DataFrame(coefs, index= [0]).melt()['value']], axis= 1) # Juntamos los coeficientes de ambos años
lasso_coefs.columns = ['variable', '2004', '2024'] # Renombramos las columnas
lasso_coefs.to_latex('./output/lasso_coefs.tex', index= False) # Guardamos los coeficientes en un archivo latex


print(f'El ECM para LASSO en 2024 es: {mse}')

# %% 2.7 Función de resultados Ridge
  
# Queremos evaluar la capacidad predictiva de Ridge vs logit mediante el mse
def Ridge_logit(x_train, y_train, x_test, y_test, alpha):
    model = LogisticRegression(penalty='l2', C=1/alpha, max_iter=10000).fit(x_train, y_train)
    mse = mean_squared_error(y_test, model.predict(x_test)) # Guardamos el mse para evaluar el modelo
    return mse

# %% 2.7 Resultados Ridge 2004

mse = Ridge_logit(x_train_2004, y_train_2004, x_test_2004, y_test_2004, 10) # Calculamos el mse para Ridge en 2004
print(f'El ECM para Ridge en 2004 es: {mse}')


# %% 2.7 Resultados Ridge 2024

mse = Ridge_logit(x_train_2024, y_train_2024, x_test_2024, y_test_2024, 100) # Calculamos el mse para Ridge en 2024
print(f'El ECM para Ridge en 2004 es: {mse}')






















