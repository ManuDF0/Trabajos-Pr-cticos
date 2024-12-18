#________________________________________________________________________________________#
#
#                          TRABAJO PRÁCTICO N°3 Machine Learning
#
#                  Profesor: Walter Sosa Escudero. Asistente: Tomás Pacheco 
#
#           Alumnos: Manuel Díaz de la Fuente, Diego Fernández Meijide y Sofía Kastika
#________________________________________________________________________________________#


# PODRÍAMOS PONER UN ÍNDICE DEL CÓDIGO


#_______________________________________________________________________________________#
#
#                                 PARTE I: Analizando la Base
#_______________________________________________________________________________________#


# PONER MINI RESUMEN DE LO QUE VAMOS A HACER EN ESTA PARTE


#____________________________________________________________________________________________________#
# 
#                                         Inciso 1 
#____________________________________________________________________________________________________#

# La explicación se encuentra en el documento


#____________________________________________________________________________________________________#
# 
#                                        Inciso 2.A 
# Objetivos: 
#    - Quedarnos con las observaciones correspondientes a nuestro aglomerado Bahía Blanca - Cerri
#    - Unir los trimestres de la EPH 2004 y la EPH 2024 en una sola base
#____________________________________________________________________________________________________#

# Importamos librerías 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import numpy as np

# Definimos directorio 
os.chdir("C:/Users/sofia/Desktop/Maestría/Tercer trimestre/Machine Learning/Trabajos-Pr-cticos/TP3")

# Importamos ambas bases 
# EPH 2024
eph2024 = pd.read_excel("input/usu_individual_T124.xlsx") # importamos base primer trimestre 2024 
eph2024.head(2) # miramos primeras 2 observaciones 
eph2024.sample(3) # miramos 3 observaciones random
print(eph2024.columns) # miramos nombres de columnas 

# EPH 2004
eph2004 = pd.read_stata("input/Individual_t104.dta") # importamos base primer trimestre 2004 
eph2004.head(2) # miramos primeras 2 observaciones 
eph2004.sample(3) # miramos primeras 2 observaciones 
print(eph2004.columns) # miramos nombres de columnas 

# Pasamos todos los nombres de las variables a minúscula 
eph2024.columns = eph2024.columns.str.lower() 
eph2004.columns = eph2004.columns.str.lower() # en 2004 solo la variable CODUSU estaba en mayúscula, pero el código sirve igual

# Vemos qué tipo de variable es aglomerado para cada año: 
eph2024["aglomerado"].dtype # en 2024 'aglomerado' es una variable numérica (tipo int64)
eph2004["aglomerado"].dtype # en 2004 'aglomerado' es una variable categórica 

# Nos quedamos con las observaciones de Bahía Blanca - Cerri (código 3 para 2024): 
bb_2024 = eph2024[eph2024['aglomerado'] == 3] # filtramos y nos quedamos con las filas de la base que tengan un valor de aglomerado igual a 3 
bb_2004 = eph2004[eph2004['aglomerado'] == 'Bahía Blanca - Cerri'] # hacemos lo mismo para 2004. 
# Para 2004 en vez de poner el código del aglomerado, ponemos directamente el nombre del aglomerado, porque así es el nombre de la categoría en 2004

# Vemos el shape de ambas bases
print(f'Shape Bahía Blanca 2024: {bb_2024.shape}') # (1008, 177). 1008 observaciones, 177 variables
print(f'Shape Bahía Blanca 2004: {bb_2004.shape}') # (1156, 176). 1156 observaciones, 176 variables

# Nos quedamos en ambas bases con las variables (columnas) que vamos a utilizar en el trabajo: 
bb_2024 = bb_2024[['codusu', 'ano4', 'ch04', 'ch06', 'ch07', 'ch08', 'nivel_ed', 'estado', 'cat_inac', 'ipcf']]
bb_2004 = bb_2004[['codusu','ano4', 'ch04', 'ch06', 'ch07', 'ch08', 'nivel_ed', 'estado', 'cat_inac', 'ipcf']]

# Volvemos a ver el shape de ambas bases
print(f'Shape Bahía Blanca 2024: {bb_2024.shape}') # (1008, 10). 1008 observaciones, 10 variables
print(f'Shape Bahía Blanca 2004: {bb_2004.shape}') # (1156, 10). 1156 observaciones, 10 variables

"""
Comentario: 
    - Antes de unir ambas bases, nos tenemos que fijar que las categorías de las variables sean las mismas. 
    - Como se puede ver en las bases bb_2004 y bb_2024 y utilizando ambos diccionarios, si bien las categorías son las mismas, en 2004 se usa el nombre de la categoría y en 2024 el código
    - Entonces lo que hacemos es cambiar el nombre de las categorías de 2004 a su código correspondiente 
    
"""

# ch04 (sexo)

print(bb_2004['ch04'].cat.categories) # imprimimos las categorías
bb_2004['ch04'] = bb_2004['ch04'].cat.rename_categories({
    'Varón': 1,
    'Mujer': 2
}) # renombramos las categorías

print(bb_2004['ch04']) # vemos que el cambio se haya realizado
bb_2004['ch04'] = pd.to_numeric(bb_2004['ch04'], errors='coerce') # Convertimos la variable a tipo numérica 
bb_2004['ch04'].dtype # chequeamos

# ch07 (estado civil)
print(bb_2004['ch07'].cat.categories) # imprimimos las categorías

bb_2004['ch07'] = bb_2004['ch07'].cat.rename_categories({
     'Unido': '1',
     'Casado': '2',
     'Separado o divorciado': '3',
     'Viudo': '4', 
     'Soltero': '5',
     'Ns./Nr.': '9'
}) # renombramos las categorías

print(bb_2004['ch07']) # vemos que el cambio se haya realizado
bb_2004['ch07'] = pd.to_numeric(bb_2004['ch07'], errors='coerce') # convertimos la variable a tipo numérica
bb_2004['ch07'].dtype # chequeamos


# ch08 (cobertura)

print(bb_2004['ch08'].cat.categories) # imprimimos las categorías

bb_2004['ch08'] = bb_2004['ch08'].cat.rename_categories({
     'Obra social (incluye PAMI)': '1',
     'Mutual/Prepaga/Servicio de emergencia': '2',
     'Planes y seguros públicos': '3',
     'No paga ni le descuentan': '4', 
     'Ns./Nr.': '9',
     'Obra social y mutual/prepaga/servicio de emergencia': '12',
     'Obra social y planes y seguros públicos': '13',
     'Mutual/prepaga/servicio de emergencia/planes y seguros públi': '23',
     'Obra social, mutual/prepaga/servicio de emergencia y planes': '123'
     
}) # renombramos las categorías

print(bb_2004['ch08']) # vemos que el cambio se haya realizado
bb_2004['ch08'] = pd.to_numeric(bb_2004['ch08'], errors='coerce') # convertimos la variable a tipo numérica
bb_2004['ch08'].dtype # chequeamos

# Nivel_ed (nivel educativo)

print(bb_2004['nivel_ed'].cat.categories) # imprimimos las categorías

bb_2004['nivel_ed'] = bb_2004['nivel_ed'].cat.rename_categories({
     'Primaria Incompleta (incluye educación especial)': '1',
     'Primaria Completa': '2',
     'Secundaria Incompleta': '3',
     'Secundaria Completa': '4', 
     'Superior Universitaria Incompleta': '5',
     'Superior Universitaria Completa': '6',
     'Sin instrucción': '7'
}) # renombramos categorías

print(bb_2004['nivel_ed']) # vemos que el cambio se haya realizado
bb_2004['nivel_ed'] = pd.to_numeric(bb_2004['nivel_ed'], errors='coerce') # convertimos la variable a tipo numérica
bb_2004['nivel_ed'].dtype # chequeamos

# Estado 
print(bb_2004['estado'].cat.categories) # imprimimos las categorías

bb_2004['estado'] = bb_2004['estado'].cat.rename_categories({
     'Entrevista individual no realizada (no respuesta al cuestion': '0',
     'Ocupado': '1',
     'Desocupado': '2',
     'Inactivo': '3', 
     'Menor de 10 años': '4'
})

print(bb_2004['estado']) # vemos que el cambio se haya realizado
bb_2004['estado'] = pd.to_numeric(bb_2004['estado'], errors='coerce') # convertimos la variable a tipo numérica
bb_2004['estado'].dtype # chequeamos

# Cat_inac (categoría inactividad)
print(bb_2004['cat_inac'].cat.categories) # imprimimos las categorías

bb_2004['cat_inac'] = bb_2004['cat_inac'].cat.rename_categories({
     'Jubilado/pensionado': '1',
     'Rentista': '2',
     'Estudiante': '3',
     'Ama de casa': '4',
     'Menor de 6 años': '5',
     'Discapacitado': '6', 
     'Otros': '7'
})
print(bb_2004['cat_inac']) # vemos que el cambio se haya realizado
bb_2004['cat_inac'] = pd.to_numeric(bb_2004['cat_inac'], errors='coerce') # convertimos la variable a tipo numérica
bb_2004['cat_inac'].dtype # chequeamos

"""
Ya realizamos los cambios de categorías, ahora podemos unir las bases
    
"""

# Unimos las bases (concat): 
datos_bb = pd.concat([bb_2024, bb_2004]) # la función concat sirve para "pegar" una base abajo de la otra
print(datos_bb.shape) # miramos el shape de la nueva base. (2164, 10). 2164 observaciones (la suma de las observaciones de ambas bases) y 10 variables

# Reseteamos la indexación
datos_bb = datos_bb.reset_index(drop=True) 

#____________________________________________________________________________________________________#
# 
#                                        Inciso 2.B 
# Objetivo: 
#    - Descartar las observaciones que no tienen sentido (ejemplo: ingresos y edades negativos)
#____________________________________________________________________________________________________#

# Edad (ch06)
print(datos_bb['ch06'].dtype) # variable del tipo objeto
datos_bb['ch06'] = datos_bb['ch06'].astype('category') # cambio la variable a tipo categoría para que nos deje cambiar el nombre de una categoría

# Reemplazamos a los que tienen "Menos de un año" por 0 
datos_bb['ch06'] = datos_bb['ch06'].cat.rename_categories({
     'Menos de 1 año': '0',
})

# Convertimos la variable a tipo numérica 
datos_bb['ch06'] = pd.to_numeric(datos_bb['ch06'], errors='coerce')

# Nos quedamos con las observaciones que son mayores o iguales a cero (de esta manera se eliminan los valores negativos)
datos_bb = datos_bb.loc[(datos_bb['ch06'] >= 0)] # pasamos de 2164 a 2152 observaciones

# Ingreso per cápita (ipcf)

print(datos_bb['ipcf'].dtype) # variable del tipo numérica

# Nos quedamos con las observaciones que son mayores o iguales a cero (de esta manera se eliminan los valores negativos)
datos_bb = datos_bb.loc[(datos_bb['ipcf'] >= 0)] # seguimos con 2152, no hay observaciones sin sentido

#____________________________________________________________________________________________________#
# 
#                                        Inciso 2.C
# Objetivo: 
#    - Realizar un gráfico de barras mostrando la composición por sexo para 2004 y 2024
#____________________________________________________________________________________________________#

############# MEJORAR LA ESTÉTICA DE LOS GRÁFICOS!!!!

# Filtramos los datos por año
data_2024 = datos_bb[datos_bb['ano4'] == 2024]
data_2004 = datos_bb[datos_bb['ano4'] == 2004]

# Contamos la cantidad de hombres y mujeres por año
sexo_counts_2024 = data_2024['ch04'].value_counts(normalize=True) * 100  # Obtener porcentajes
sexo_counts_2004 = data_2004['ch04'].value_counts(normalize=True) * 100  # Obtener porcentajes

# Graficamos para 2024
plt.figure(figsize=(10, 5))
bars_2024 = plt.bar(sexo_counts_2024.index.astype(str), sexo_counts_2024.values, color=['blue', 'pink'])  # Azul para hombres, rosa para mujeres
plt.title('Composición por Sexo en 2024')
plt.xlabel('Sexo')
plt.ylabel('Porcentaje (%)')
plt.xticks(ticks=[0, 1], labels=['Hombres', 'Mujeres'])
plt.grid(axis='y')
plt.ylim(0, 100)  # Ajustar el límite del eje y para mostrar porcentajes

# Añadimos los porcentajes encima de las barras
for bar in bars_2024:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

plt.show()

# Graficamos para 2004
plt.figure(figsize=(10, 5))
bars_2004 = plt.bar(sexo_counts_2004.index.astype(str), sexo_counts_2004.values, color=['blue', 'pink'])  # Azul para hombres, rosa para mujeres
plt.title('Composición por Sexo en 2004')
plt.xlabel('Sexo')
plt.ylabel('Porcentaje (%)')
plt.xticks(ticks=[0, 1], labels=['Hombres', 'Mujeres'])
plt.grid(axis='y')
plt.ylim(0, 100)  # Ajustar el límite del eje y para mostrar porcentajes

# Añadimos los porcentajes encima de las barras
for bar in bars_2004:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

plt.show()

#____________________________________________________________________________________________________#
# 
#                                        Inciso 2.D
# Objetivo: 
#    - Realizar una matriz de correlación para los años 2004 y 2024 para las siguientes variables:         
#       CH04, CH06, CH07, CH08, NIVEL ED, ESTADO, CAT_INAC, IPCF
#____________________________________________________________________________________________________#

# Para hacer la matriz de correlación, necesitamos generarnos dummies de cada una de las variables categóricas, ya que sino la misma no tendría sentido

# Generamos dummies con este comando 
datos_bb_dummies = pd.get_dummies(datos_bb, columns=['ch04', 'ch07', 'nivel_ed', 'estado', 'cat_inac'], drop_first=True) 

# Eliminamos variable que no nos sirve para la matriz de correlación
datos_bb_dummies.drop('codusu', axis=1, inplace=True) 

# La función de get_dummies pone "drop_first" = True porque elimina la primera categoría de cada variable para tomarla como categoría base
# Comentario: aunque la variable ch04 (sexo) ya sea una dummy, generamos igualmente otra dummy porque preferimos que las dummies sean 0-1, y no 1-2

# Matriz de correlación 
# Sacamos esta función del link en la consigna
def heatmap(x, y, size):
    fig, ax = plt.subplots()
    
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), 
        y=y.map(y_to_num), 
        s=size * size_scale,
        marker='s'
    )
    
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=90, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

# Filtramos de la base de dummies los años
data_2024_dummies = datos_bb_dummies[datos_bb_dummies['ano4'] == 2024]
data_2004_dummies = datos_bb_dummies[datos_bb_dummies['ano4'] == 2004]

# Matriz de correlación 2004
corr = data_2024_dummies.corr() # Saco la correlación
corr = pd.melt(corr.reset_index(), id_vars='index') # Lo transforma en un df de tres columnas
corr.columns = ['x', 'y', 'value'] # Le cambio el nombre a las columnas

heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)

# Matriz de correlación 2004
corr = data_2024_dummies.corr() # Saco la correlación
corr = pd.melt(corr.reset_index(), id_vars='index') # Lo transforma en un df de tres columnas
corr.columns = ['x', 'y', 'value'] # Le cambio el nombre a las columnas

heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)

#____________________________________________________________________________________________________________#
# 
#                                        Inciso 2.E
# Objetivos: 
#    - Calcular cantidad de desocupados
#    - Calcular cantidad de inactivos
#    - Calcular la media del Ingreso Per Cápita Familiar (ipcf) según estado (ocupado, desocupado, inactivo)
#____________________________________________________________________________________________________________#

# Transformamos la variable Estado en categórica 
# 2024 
data_2024['estado'] = data_2024['estado'].replace({0: 'No responde', 1: 'Ocupado', 2: 'Desocupado', 3: 'Inactivo', 4: 'Menor de 10 años'}) # cambiamos el nombre de las categorías
print(data_2024['estado'].dtype) # vemos el tipo de la variable
data_2024['estado'] = data_2024['estado'].astype('category') # cambiamos a formato categórico

# 2004
data_2004['estado'] = data_2004['estado'].replace({0: 'No responde', 1: 'Ocupado', 2: 'Desocupado', 3: 'Inactivo', 4: 'Menor de 10 años'}) # cambiamos el nombre de las categorías
print(data_2004['estado'].dtype) # vemos el tipo de la variable
data_2004['estado'] = data_2004['estado'].astype('category') # cambiamos a formato categórico

# Contamos cantidad de desocupados e inactivos
estado2024 = data_2024['estado'].value_counts() # contamos cantidad de individuos en cada una de las categorías en el 2024
estado2004 = data_2004['estado'].value_counts() # mismo para 2004

# Media ipcf por estado
ipcf_estado_2024 = data_2024.groupby('estado')['ipcf'].mean() # agrupamos los datos por estado y calculamos la media del ipcf para cada estado
ipcf_estado_2004 = data_2004.groupby('estado')['ipcf'].mean() # mismo para 2004 

# Creamos Dataframes para 2004 y 2024
# Para 2024
tabla_2e_2024 = pd.DataFrame({
    'Categoria': estado2024.index,
    'Año': 2024,
    'Count': estado2024.values,
    'Media_IPCF': estado2024.index.map(ipcf_estado_2024).round(0).astype(int)  # Redondeamos y convertimos a int
})

# Para 2004
tabla_2e_2004 = pd.DataFrame({
    'Categoria': estado2004.index,
    'Año': 2004,
    'Count': estado2004.values,
    'Media_IPCF': estado2004.index.map(ipcf_estado_2004).round(0).astype(int)  # Redondeamos y convertimos a int
})

# Exportamos a Latex
tabla_2e_2004.to_latex("output/tabla_2e_2004.tex", index=False)
tabla_2e_2024.to_latex("output/tabla_2e_2024.tex", index=False)

#___________________________________________________________________________________________________________________#
# 
#                                                  Inciso 3
# Objetivos: 
#    - Calcular cuántas personas no respondieron cuál es su condición de actividad
#    - Crear una base que contenga únicamente a las personas que respondieron cuál es su condición de actividad
#    - Crear una base que contenga únicamente a las personas que no respondieron cuál es su condición de actividad
#___________________________________________________________________________________________________________________#

# Contamos cantidad de desocupados e inactivos
estado2024 = data_2024['estado'].value_counts() # podemos ver que en 2024 solo 1 persona no respondió cuál es su condición de actividad
estado2004 = data_2004['estado'].value_counts() # en 2004 todas las personas en Bahía Blanca respondieron sobre su condición de actividad

# Generamos una nueva base en donde solo se incluye a aquellos que respondieron cuál es su condición de actividad
respondieron = datos_bb[datos_bb['estado'] != 0] # filtramos las observaciones en donde estado es distinto de cero

# Generamos una nueva base en donde solo se incluye a aquellos que no respondieron cuál es su condición de actividad
norespondieron = datos_bb[datos_bb['estado'] == 0] # filtramos las observaciones en donde estado es gual a cero

#___________________________________________________________________________________________________________________________#
# 
#                                                  Inciso 4
# Objetivos: 
#    - Agregar a la base respondieron una columna llamada PEA (Población Económicamente Activa)
#    - Realizar un gráfico de barras mostrando la composición por PEA para 2004 y 2024    
#___________________________________________________________________________________________________________________________#

# Transformamos la variable Estado en categórica en la base respondieron
respondieron['estado'] = respondieron['estado'].replace({1: 'Ocupado', 2: 'Desocupado', 3: 'Inactivo', 4: 'Menor de 10 años'}) # cambiamos el nombre de las categorías
print(respondieron['estado'].dtype) # vemos el tipo de la variable
respondieron['estado'] = respondieron['estado'].astype('category') # cambiamos a formato categórico

# Creamos la variable PEA 
respondieron['pea'] = np.where(respondieron['estado'].isin(['Ocupado', 'Desocupado']), 1, 0) # creamos la columna. Que valga 1 si estado es Ocupado o Desocupado y que valga 0 de lo contrario
print(respondieron['pea'].dtype) # es del tipo numérica
respondieron['pea'] = respondieron['pea'].astype('category') # pasamos a tipo categórico 

# Separamos en dos bases, una para 2004 y otra para 2024
respondieron_2004 = respondieron[respondieron['ano4'] == 2004].reset_index(drop=True) # con el reset_index volvemos a setear bien los índices
respondieron_2024 = respondieron[respondieron['ano4'] == 2024].reset_index(drop=True)

# Porcentajes
respondieron_2004_counts = respondieron_2004['pea'].value_counts(normalize=True) * 100
respondieron_2024_counts = respondieron_2024['pea'].value_counts(normalize=True) * 100

# DEFINIMOS FUNCIÓN PARA GRÁFICOS DE BARRAS: 
def composicion_bar(val_2004, val_2024, x_label, label_2004_1, label_2004_2, label_2024_1, label_2024_2): 
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Crea una figura con 2 subgráficos, uno al lado del otro

    # Gráfico para 2004: 
    bars_2004 = axs[0].bar(val_2004.index, val_2004.values, color='skyblue')
    axs[0].set_title('Composición en 2004')  # Título para el gráfico de 2004
    axs[0].set_ylabel('Porcentaje')  # Etiqueta en el eje Y
    axs[0].set_ylim(0, 100)  # Limita los valores en el eje Y entre 0 y 100 (porcentaje)

    # Eliminamos las rayas y números del eje X
    axs[0].set_xticks([])  # Eliminmos los ticks del eje x
    axs[0].set_xlabel('')  # Elimina la etiqueta "Estado" en el eje X

    # Añade el porcentaje arriba de las barras en 2004
    axs[0].text(bars_2004[0].get_x() + bars_2004[0].get_width() / 2, val_2004.values[0] + 2, f'{val_2004.values[0]:.1f}%', ha='center') 
    axs[0].text(bars_2004[1].get_x() + bars_2004[1].get_width() / 2, val_2004.values[1] + 2, f'{val_2004.values[1]:.1f}%', ha='center') 

    # Añade las etiquetas debajo de las barras en 2004, sin los ":"
    axs[0].text(bars_2004[0].get_x() + bars_2004[0].get_width() / 2, -5, f'{label_2004_1}', ha='center') 
    axs[0].text(bars_2004[1].get_x() + bars_2004[1].get_width() / 2, -5, f'{label_2004_2}', ha='center') 

    # Gráfico para 2024: 
    bars_2024 = axs[1].bar(val_2024.index, val_2024.values, color='salmon')
    axs[1].set_title('Composición en 2024')  # Título para el gráfico de 2024
    axs[1].set_ylabel('Porcentaje')  # Etiqueta el eje Y
    axs[1].set_ylim(0, 100)  # Limita los valores en el eje Y entre 0 y 100 (porcentaje)

    # Eliminamos las rayas y números del eje X
    axs[1].set_xticks([])  # Eliminamos los ticks del eje X
    axs[1].set_xlabel('')  # Eliminamos etiqueta del eje x

    # Añade el porcentaje arriba de las barras en 2024
    axs[1].text(bars_2024[0].get_x() + bars_2024[0].get_width() / 2, val_2024.values[0] + 2, f'{val_2024.values[0]:.1f}%', ha='center') 
    axs[1].text(bars_2024[1].get_x() + bars_2024[1].get_width() / 2, val_2024.values[1] + 2, f'{val_2024.values[1]:.1f}%', ha='center') 

    # Agregamos las etiquetas
    axs[1].text(bars_2024[0].get_x() + bars_2024[0].get_width() / 2, -5, f'{label_2024_1}', ha='center') 
    axs[1].text(bars_2024[1].get_x() + bars_2024[1].get_width() / 2, -5, f'{label_2024_2}', ha='center') 
    
    plt.tight_layout()  # Ajustamos automáticamente el espacio entre los subgráficos

    plt.show()  
    return fig

# Aplicamos la función para ambos años: 
grafico4 = composicion_bar(respondieron_2004_counts, respondieron_2024_counts, 'Estado', 
                'PEA', 'NO PEA', 
                'PEA', 'NO PEA')


# Guardamos el gráfico como archivo PNG
grafico4.savefig('output/grafico4.png', bbox_inches='tight')

#_____________________________________________________________________________________________________________________________________________________________#
# 
#                                                  Inciso 5
# Objetivos: 
#    - Agregar a la base respondieron una columna llamada PET (Población en Edad para Trabajar) que toma 1 si el individuo tiene entre 15 y 65 años cumplidos
#    - Realizar un gráfico de barras mostrando la composición por PEA para 2004 y 2024.  
#_____________________________________________________________________________________________________________________________________________________________#

# Creamos la variable PET
respondieron['pet'] = np.where((respondieron['ch06'] > 15) & (respondieron['ch06'] < 65), 1, 0)
print(respondieron['pet'].dtype) # es del tipo numérica
respondieron['pet'] = respondieron['pet'].astype('category') # pasamos a tipo categórica 

# Separamos en dos bases, una para 2004 y otra para 2024
respondieron_2004 = respondieron[respondieron['ano4'] == 2004].reset_index(drop=True) # con el reset_index volvemos a setear bien los índices
respondieron_2024 = respondieron[respondieron['ano4'] == 2024].reset_index(drop=True)

# Porcentajes
respondieron_2004_counts = respondieron_2004['pet'].value_counts(normalize=True) * 100
respondieron_2024_counts = respondieron_2024['pet'].value_counts(normalize=True) * 100

# Volvemos a aplicar la función para ambos años: 
grafico5 = composicion_bar(respondieron_2004_counts, respondieron_2024_counts, 'Estado', 
                'PET', 'NO PET', 
                'PET', 'NO PET')

# Guardamos el gráfico como archivo PNG
grafico5.savefig('output/grafico5.png', bbox_inches='tight')

#______________________________________________________________________________________________________________________________#
# 
#                                                  Inciso 6.A
# Objetivos: 
#    - Agregar a la base respondieron una columna llamada desocupado si la persona está desocupada 
#    - Calculo la cantidad de desocupados por año 
#    - Mostrar la proporción de desocupados por nivel educativo comparando 2004 vs 2024  
#______________________________________________________________________________________________________________________________#

# Generamos una variable "desocupado" que sea una dummy con valor 1 si la persona está desocupada y 0 si no 
respondieron['desocupado'] = np.where(respondieron['estado'].isin(['Desocupado']), 1, 0) # creamos la columna. Que valga 1 si estado es Ocupado o Desocupado y que valga 0 de lo contrario

# Separamos en dos bases, una para 2004 y otra para 2024
respondieron_2004 = respondieron[respondieron['ano4'] == 2004].reset_index(drop=True) # con el reset_index volvemos a setear bien los índices
respondieron_2024 = respondieron[respondieron['ano4'] == 2024].reset_index(drop=True)

# Proporción de desocupados por nivel educativo
# 2004:

total2004 = respondieron_2004.groupby('nivel_ed')['nivel_ed'].count() # calculamos las personas totales por nivel educativo
desocupados2004 = respondieron_2004[respondieron_2004['desocupado'] == 1].groupby('nivel_ed')['desocupado'].count() # cantidad de desocupados por nivele educativo
proporcion_desocupados2004 = desocupados2004 / total2004 # hacemos la proporción de desocupados por nivel educativo

# Mostramos los resultados
print(proporcion_desocupados2004)

# 2024 
total2024 = respondieron_2024.groupby('nivel_ed')['nivel_ed'].count() # calculamos las personas totales por nivel educativo
desocupados2024 = respondieron_2024[respondieron_2024['desocupado'] == 1].groupby('nivel_ed')['desocupado'].count() # cantidad de desocupados por nivele educativo
proporcion_desocupados2024 = desocupados2024 / total2024 # hacemos la proporción de desocupados por nivel educativo
# Mostramos los resultados
print(proporcion_desocupados2024)


# Combinamos las proporciones en una tabla
tabla_combinada = pd.DataFrame({
    'Nivel Educativo': proporcion_desocupados2004.index,
    'Proporción Desocupados 2004': proporcion_desocupados2004.values,
    'Proporción Desocupados 2024': proporcion_desocupados2024.values
})

print(tabla_combinada)

# Exportamos la tabla combinada a LaTeX
tabla_combinada.to_latex("output/tabla_6a.tex", index=False)


#______________________________________________________________________________________________________________________________#
# 
#                                                  Inciso 6.B
# Objetivos: 
#    - Calcular la cantidad de personas desocupadas para cada año     
#    - Crear una variable categórica de años cumplidos agrupada de a 10 años
#    - Mostrar la proporción de desocupados por edad agrupada comparando 2004 vs 2024 
#______________________________________________________________________________________________________________________________#

# Cantidad de desocupados por año en la base respondieron: 
# 2004     
cantidad_desocupados2004 = respondieron_2004[respondieron_2004['desocupado'] == 1].shape[0]
print(cantidad_desocupados2004)

# 2024
cantidad_desocupados2024 = respondieron_2024[respondieron_2024['desocupado'] == 1].shape[0]
print(cantidad_desocupados2024)
# Igualmente, esta información ya la teníamos del inciso 2.E 

# Cantidad de desocupados por año en la base respondieron: 
# 2004     
cantidad_desocupados2004 = respondieron_2004[respondieron_2004['desocupado'] == 1].shape[0]
print(cantidad_desocupados2004)

# 2024
cantidad_desocupados2024 = respondieron_2024[respondieron_2024['desocupado'] == 1].shape[0]
print(cantidad_desocupados2024)

# Igualmente, esta información ya la teníamos del inciso 2.E 

# Creamos bins de años en 10 en 10 
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] 

# Le ponemos labels a los bins 
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']

conteo_edad_categoria = respondieron_2024['edad_categoria'].value_counts()

# 2004: 
# Creamos una variable que se llama edad_categoría
# Primero usamos la función pd.cut para categorizar la edad basada en los bins y labels que definimos
respondieron_2004['edad_categoria'] = pd.cut(respondieron_2004['ch06'], bins=bins, labels=labels, right=False)
# Calculamos la proporción de desocupados por categoría de edad
total_edad_2004 = respondieron_2004.groupby('edad_categoria')['edad_categoria'].count()  # Total de personas por categoría de edad
desocupados_edad_2004 = respondieron_2004[respondieron_2004['desocupado'] == 1].groupby('edad_categoria')['desocupado'].count()  # Desocupados por categoría de edad
# Calculamos la proporción de desocupados por categoría de edad
proporcion_desocupados2004 = desocupados_edad_2004 / total_edad_2004
# Mostramos los resultados
print(proporcion_desocupados2004)


# 2024: 
# Creamos una variable que se llama edad_categoría
respondieron_2024['edad_categoria'] = pd.cut(respondieron_2024['ch06'], bins=bins, labels=labels, right=False)
# Calculamos la proporción de desocupados por categoría de edad
total_edad_2024 = respondieron_2024.groupby('edad_categoria')['edad_categoria'].count()  # Total de personas por categoría de edad
desocupados_edad_2024 = respondieron_2024[respondieron_2024['desocupado'] == 1].groupby('edad_categoria')['desocupado'].count()  # Desocupados por categoría de edad
# Calculamos la proporción de desocupados por categoría de edad
proporcion_desocupados2024 = desocupados_edad_2024 / total_edad_2024
# Mostramos los resultados
print(proporcion_desocupados2024)


# Combinamos las proporciones en una tabla
tabla_combinada = pd.DataFrame({
    'Edad': proporcion_desocupados2004.index,
    'Proporción Desocupados 2004': proporcion_desocupados2004.values,
    'Proporción Desocupados 2024': proporcion_desocupados2024.values
})

print(tabla_combinada)

# Exportamos la tabla combinada a LaTeX
tabla_combinada.to_latex("output/tabla_6b.tex", index=False)



#_______________________________________________________________________________________#
#
#                                 PARTE II: Clasificación
#_______________________________________________________________________________________#

# Poner mini resumen de lo que vamos a hacer en esta parte

#_________________________________________________________________________________________________#
# 
#                                                  Inciso 1
# Objetivos: 
#    - Establecer a "desocupado" como variable dependiente y al resto como independientes
#    - Partir la base respondieron en una base de entrenamiento y una de test
#_________________________________________________________________________________________________#

# Importamos paquetes. Particularmente, importamos las funciones a utilizar del paquete sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score

# Partimos la base de cada año en una muestra de entrenamiento y una de test 
y_2004 = respondieron_2004['desocupado']
y_2024 = respondieron_2024['desocupado']

# Columnas: definimos un vector de variables explicativas que nos interesan
columnas =  ['ch04', 'ch07', 'nivel_ed', 'estado', 'cat_inac']

# 2004 
# Definimos el vector de variables explicativas (con las variables categóricas en formato dummy)
x_2004 = pd.get_dummies(respondieron_2004, columns = columnas, drop_first=True)

# Filtramos para quedarnos solo con las columnas que contienen las dummies generadas
x_2004 = x_2004.loc[:, x_2004.columns.str.startswith(tuple(columnas))]

# Agregamos las variables numéricas 'ch06' e 'ipcf' sin transformarlas
x_2004['ch06'] = respondieron_2004['ch06']
x_2004['ipcf'] = respondieron_2004['ipcf']

# 2024 
# Definimos el vector de variables explicativas (con las variables categóricas en formato dummy)
x_2024 = pd.get_dummies(respondieron_2024, columns = columnas, drop_first=True)

# Filtramos para quedarnos solo con las columnas que contienen las dummies generadas
x_2024 = x_2024.loc[:, x_2024.columns.str.startswith(tuple(columnas))]

# Agregamos las variables numéricas 'ch06' e 'ipcf' sin transformarlas
x_2024['ch06'] = respondieron_2024['ch06']
x_2024['ipcf'] = respondieron_2024['ipcf']

# Tenemos ya entonces el vector de variables explicativas y la variable a explicar para cada año. 

# Agregamos columnas de 1s
x_2004['constante'] = 1
x_2024['constante'] = 1

# El 30% de la base se vuelve base de testeo, por ende el 70% es de tratamiento 
x_train_2004, x_test_2004, y_train_2004, y_test_2004 = train_test_split(x_2004, y_2004, test_size = 0.3, random_state = 101)
x_train_2024, x_test_2024, y_train_2024, y_test_2024 = train_test_split(x_2024, y_2024, test_size = 0.3, random_state = 101)

#________________________________________________________________________________________________________________________#
# 
#                                                  Inciso 2
# Objetivos: 
#    - Implementar los siguientes métodos: Regresión logística, Análisis discriminante lineal, KKN con k=3 y Naive Bayes
#    - Reportar: matriz de confusión, curva ROC, valores de AUC y de Accuracy de cada uno
#_________________________________________________________________________________________________________________________#

# Definimos una función
def evaluate_model(model, X_train, X_test, y_train, y_test): # los inputs de la función son los datos de testeo y de entrenamiento
    model.fit(X_train, y_train) # ajustamos el modelo para la base de entrenamiento
    
    y_pred = model.predict(X_test) # predecimos y con los datos de la base de entrenamiento
    y_prob = model.predict_proba(X_test)[:, 1] # predecimos la probabilidad de que y esté en una clase
    
    cm = confusion_matrix(y_test, y_pred) # matriz de confusión
    
    fpr, tpr, _ = roc_curve(y_test, y_prob) # curva ROC 
    auc = roc_auc_score(y_test, y_prob) # AUC
    
    accuracy = accuracy_score(y_test, y_pred) # accuracy score
    
    return cm, fpr, tpr, auc, accuracy # que devuelva todos los valores

# Definimos los modelos a utilizar
models = {
    'Regresión Logística': LogisticRegression(max_iter= 10000),
    'Análisis Discriminante Lineal': LinearDiscriminantAnalysis(),
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'Naive Bayes': GaussianNB()
}

# 2004
# Imprimimos todos los resultados pedidos en la consigna 
results = {}
print("Resultados para 2004:")
for model_name, model in models.items():
    cm, fpr, tpr, auc, accuracy = evaluate_model(model, x_train_2004, x_test_2004, y_train_2004, y_test_2004)
    results[model_name] = {
        'Matriz de Confusión': cm,
        'AUC': auc,
        'Accuracy': accuracy,
        'FPR': fpr,
        'TPR': tpr
    }

    print(f"{model_name}:")
    print(f"Matriz de Confusión:\n{cm}")
    print(f"AUC: {auc:.2f}")
    print(f"Accuracy: {accuracy:.2f}\n") 
    
# Graficamos la Curva ROC 
plt.figure(figsize=(10, 5))
for model_name, metrics in results.items():
    plt.plot(metrics['FPR'], metrics['TPR'], label=f'{model_name} (AUC = {metrics["AUC"]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - 2004')
plt.legend(loc='lower right')
plt.show()

# 2024
# Imprimimos todos los resultados pedidos en la consigna 
results = {}
print("Resultados para 2024:")
for model_name, model in models.items():
    cm, fpr, tpr, auc, accuracy = evaluate_model(model, x_train_2024, x_test_2024, y_train_2024, y_test_2024)
    results[model_name] = {
        'Matriz de Confusión': cm,
        'AUC': auc,
        'Accuracy': accuracy,
        'FPR': fpr,
        'TPR': tpr
    }

    print(f"{model_name}:")
    print(f"Matriz de Confusión:\n{cm}")
    print(f"AUC: {auc:.2f}")
    print(f"Accuracy: {accuracy:.2f}\n")
    
    
# Graficamos la Curva ROC 
plt.figure(figsize=(10, 5))
for model_name, metrics in results.items():
    plt.plot(metrics['FPR'], metrics['TPR'], label=f'{model_name} (AUC = {metrics["AUC"]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - 2004')
plt.legend(loc='lower right')
plt.show()

#________________________________________________________________________________________________________________#
# 
#                                                  Inciso 3
# Objetivos: 
#    - Comparar los resultados de 2004 y 2024 y responder cuál de los métodos predice mejor para cada año
#________________________________________________________________________________________________________________#

# Justificación en el documento. Mejores modelos para ambos años: Modelo logístico y Naive Bayes



#________________________________________________________________________________________________________________#
# 
#                                                  Inciso 4
# Objetivos: 
#    - Con el método seleccionado en el inciso 3, predecir qué personas de la base norespondieron son desocupadas
#    - Mostrar proporción de personas de la base norespondieron que están desocupadas
#________________________________________________________________________________________________________________#













