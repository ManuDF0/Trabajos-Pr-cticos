#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ## Limpieza

# In[67]:


indiv_2004 = pd.read_stata('./input/Individual_t104.dta', convert_categoricals= False) # Leo la base individual para 2004, no quiero que me guarde el formato de stata, así que no convierto las variables categóricas
#indiv_2024 = pd.read_excel('./input/usu_individual_T124.xlsx') # Leo la base individual para 2024, acá no tengo problemas con el formato
hog_2004 = pd.read_stata('./input/Hogar_t104.dta', convert_categoricals= False) # Leo la base de hogares para 2004, devuelta cambio el formato
#hog_2024 = pd.read_excel('./input/usu_hogar_T124.xlsx') # Leo la base de hogares para 2024

# Los excel tardan bastante en cargar, así que vamos a guardarlos en otro formato


# In[68]:


#indiv_2024.to_csv('./input//usu_individual_T124.csv', index= False)
#hog_2024.to_csv('./input/usu_hogar_T124.csv', index = False)


# In[69]:


tipos_indiv = indiv_2004.dtypes.to_dict() # Vemos los dtypes de la base de 2004
# Lo guardamos como diccionario para cuando cargue el csv poder especificarle al read_csv los dtypes de la base de 2024
# Cambiamos los dtypes de las columnas que nos van a a interesar
tipos_indiv['ch04'] = 'category'
tipos_indiv['estado'] = 'category'
tipos_indiv['cat_inac'] = 'category'
tipos_indiv['nivel_ed'] = 'category'
tipos_indiv['ch07'] = 'category'
tipos_indiv['ch08'] = 'category'
tipos_indiv['ano4'] = 'category'
tipos_indiv['aglomerado'] = 'category'
tipos_indiv['trimestre'] = 'category'
tipos_indiv['nro_hogar'] = 'category'

# Las asignamos al df
indiv_2004 = indiv_2004.astype(tipos_indiv)


# In[70]:


# También guardo lod dtypes de la base de hogares para cargar bien el csv
tipos_hog = hog_2004.dtypes.to_dict()
tipos_hog['iv3'] = 'category'
tipos_hog['iv4'] = 'category'
tipos_hog['iv5'] = 'category'
tipos_hog['iv6'] = 'category'
tipos_hog['iv7'] = 'category'
tipos_hog['iv9'] = 'category'
tipos_hog['iv10'] = 'category'
tipos_hog['iv11'] = 'category'
tipos_hog['ii7'] = 'category'
tipos_hog['ii8'] = 'category'
tipos_hog['ii9'] = 'category'
tipos_hog['v1'] = 'category'
tipos_hog['v2'] = 'category'
tipos_hog['v21'] = 'category'
tipos_hog['v22'] = 'category'
tipos_hog['v3'] = 'category'
tipos_hog['v4'] = 'category'
tipos_hog['v5'] = 'category'
tipos_hog['v6'] = 'category'
tipos_hog['v7'] = 'category'
tipos_hog['v8'] = 'category'
tipos_hog['v9'] = 'category'
tipos_hog['v10'] = 'category'
tipos_hog['v11'] = 'category'
tipos_hog['v13'] = 'category'
tipos_hog['v14'] = 'category'
tipos_hog['v15'] = 'category'
tipos_hog['v16'] = 'category'
tipos_hog['v17'] = 'category'
tipos_hog['v18'] = 'category'
tipos_hog['v19_a'] = 'category'
tipos_hog['v19_b'] = 'category'

hog_2004 = hog_2004.astype(tipos_hog)


# In[71]:


# Cargo los csv que guarde antes con el dtype correcto y más rápido que un excel
indiv_2024 = pd.read_csv('./input/usu_individual_T124.csv', dtype = tipos_indiv, low_memory= False)
hog_2024 = pd.read_csv('./input/usu_hogar_T124.csv', dtype = tipos_hog, low_memory= False)


# In[72]:


cols_ind = indiv_2004.columns # columnas en la base individual
cols_hog = hog_2004.columns # columnas en la base de hogares

cols_comunes = [col for col in cols_hog if col in cols_ind] # guardo las columnas que tienen en común
data_2004 = pd.merge(indiv_2004, hog_2004, on= cols_comunes[:7], how='inner', suffixes=('_indiv', '_hog')) # hago el merge en base a las columnas que tienen en común
# Antes me duplicaba las columnas en el merge


# In[73]:


# Las bases de 2024 tienen las variables en mayúscula, lo que después puede ser un problema
indiv_2024.columns = indiv_2024.columns.str.lower() # las paso a minúscula
hog_2024.columns = hog_2024.columns.str.lower()

# De vuelta agarro las columnas en común y hago el merge en base a esas
cols_ind = indiv_2024.columns
cols_hog = hog_2024.columns

cols_comunes = [col for col in cols_hog if col in cols_ind]
data_2024 = pd.merge(indiv_2024, hog_2024, on = cols_comunes[:7], how='inner', suffixes= ('_indiv', '_hog'))

# La unica variable que quiero que tenga nombre en mayúscula es CODUSU
data_2024.rename(columns={'codusu': 'CODUSU'}, inplace=True)


# In[74]:


# Juntamos ambas bases en una única base
data = pd.concat([data_2004, data_2024], ignore_index= True)


# In[75]:


# Agarro los dtypes correctos y se los aplico a la base entera
categorical_columns = indiv_2004.select_dtypes(include= 'category').columns
# Para compatibilizar ambas bases hacemos lo mismo que para la de 2004
for col in categorical_columns:
    data[col] = data[col].astype(int) # Las paso a integer

for col in categorical_columns:
    data[col] = data[col].astype('category') # Devuelta las paso a categoría

# Hago lo mismo para los dtypes de la de hogares
categorical_columns = hog_2004.select_dtypes(include = 'category')
for col in categorical_columns:
    data[col] = data[col].astype(int) # Las paso a integer

for col in categorical_columns:
    data[col] = data[col].astype('category') # Devuelta las paso a categoría


# In[76]:


data = data.loc[data['aglomerado'] == 3] # Nos quedamos únicamente con las observaciones de Bahía Blanca

# Limpiamos la base. Nos quedamos con las observaciones que tienen un ingreso mayor o igual a 0, las edades mayores o iguales a 0
data = data.loc[(data['ipcf_indiv']>=0) & (data['ch06']>=0) & (data['ipcf_indiv'] == data['ipcf_hog'])] 
# Además, nos quedamos con las observaciones que tienen el mismo ingreso per cápita en el hogar e individual


# In[77]:


# Primera variable que armamos: cantidad de personas inactivas en el hogar
# Generamos una dummy de inactivo
data['inactiva'] = (data['estado'] == 3).astype(int)
# Agupamos por codusu y nro_hogar y contamos cuantas personas inactivas hay en el hogar
data['cantidad_inactivos'] = data.groupby(['CODUSU', 'nro_hogar'])['inactiva'].transform('sum')


# In[78]:


# Segunda variable: hacinamiento
# Contamos la cantidad de miembros por hogar
data['miembros_hogar'] = data.groupby(['CODUSU', 'nro_hogar'])['componente'].transform('count')

# Vemos si hay más de tres personas por ambiente
data['hacinamiento'] = (data['miembros_hogar'] / data['iv2'] > 3).astype(int)


# In[79]:


# Tercera variable: Ingreso no laboral per cápita
# El valor t_vi = -9 es NA, entonces puede quedarnos un t_vi negativo
data['ingreso_no_laboral_pc'] = data['t_vi'] / data['miembros_hogar']

# Filtramos el ingreso no laboral per cápita que sea mayor o igual a 0
data = data.loc[data['ingreso_no_laboral_pc']>=0]


# In[80]:


# Guardo el df en formato pickle para que me guarde los dtypes
data.to_pickle('./input/data.pkl')


# In[2]:


# Cargo la base
data = pd.read_pickle('./input/data.pkl')


# In[3]:


respondieron = data.loc[data['estado'] != 0] # creamos base con personas que respondieron su condición de actividad
norespondieron = data.loc[data['estado'] == 0] # creamos base con personas que no respondieron su condición de actividad


# In[4]:


desocupada = pd.DataFrame(
    np.where(respondieron['estado'].isin([2]), 1, 0),  # Creo la columna "desocupada"
    columns=['desocupada'],
    index=respondieron.index  # Le pongo el index del df respondieron para que haga bien el concat
)
respondieron = pd.concat([respondieron, desocupada], axis=1) # Le agrego la columna al df


# ## Clasificación

# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, roc_curve
from typing import Literal
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


# Definimos las columnas de interés
columnas = ['ch04', 'ch06', 'ch07', 'ch08', 'nivel_ed', 'ipcf_indiv', 'ingreso_no_laboral_pc', 'hacinamiento', 'cantidad_inactivos', 'v1', 'v2', 'v21', 'v22', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v17', 'v18', 'v19_a', 'v19_b']
# Definimos para cada año variable explicada
y_2004 = respondieron.loc[respondieron['ano4']==2004]['desocupada'] 
y_2024 = respondieron.loc[respondieron['ano4']==2024]['desocupada']

# Definimos para cada año vector de variables explicativas (en formato dummy) 
x_2004 = pd.get_dummies(respondieron.loc[respondieron['ano4']==2004][columnas])
x_2024 = pd.get_dummies(respondieron.loc[respondieron['ano4']==2024][columnas])

# Agregamos constantes
x_2004['constante'] = 1
x_2024.loc[:,'constante'] = 1

# Asignamos el 30% de cada base a testeo y por ende el 70% a entrenamiento 
x_train_2004, x_test_2004, y_train_2004, y_test_2004 = train_test_split(x_2004, y_2004, test_size = 0.3, random_state = 101)
x_train_2024, x_test_2024, y_train_2024, y_test_2024 = train_test_split(x_2024, y_2024, test_size = 0.3, random_state = 101)


# Para regularización hace falta estandarizar primero las variables, para que no pondere más por una cuestión de varianza

# In[7]:


# Estandarizamos las variables
sc = StandardScaler()
x_train_2004 = pd.DataFrame(sc.fit_transform(x_train_2004), index= x_train_2004, columns= x_train_2004.columns)
x_train_2024 = pd.DataFrame(sc.fit_transform(x_train_2024), index= x_train_2024, columns= x_train_2024.columns)
x_test_2004 = pd.DataFrame(sc.fit_transform(x_test_2004), index= x_test_2004, columns= x_test_2004.columns)
x_test_2024 = pd.DataFrame(sc.fit_transform(x_test_2024), index= x_test_2024, columns= x_test_2024.columns)


# In[8]:


# Armo una función que estima un modelo logit con distintas penalidades y devuelve las métricas para evaluarlo
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


# In[9]:


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


# In[10]:


# Para 2004
for p in ['l1', 'l2']:
    cm, auc, accuracy, fpr, tpr = logit_penalty_eval(x_train_2004, x_test_2004, y_train_2004, y_test_2004, p)
    print(cm)
    print(auc)
    print(accuracy)
    plot_roc(fpr, tpr, auc, 2004, p)


# In[11]:


# Para 2024
for p in ['l1', 'l2']:
    cm, auc, accuracy, fpr, tpr = logit_penalty_eval(x_train_2024, x_test_2024, y_train_2024, y_test_2024, p)
    print(cm)
    print(auc)
    print(accuracy)
    plot_roc(fpr, tpr, auc, 2024, p)


# In[12]:


alphas = [10**i for i in range(-5, 5, 1)] # Armamos la grilla de valores posibles para lambda


# In[13]:


# Función para el boxplot
def box(data, year, model):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='variable', y='value', data=data)
    plt.xlabel("Alpha")
    plt.ylabel("Error cuadrático medio")
    plt.text(0.5, -0.2, f'Modelo: {model} - Año: {year}', ha='center', va='center', fontsize=10, transform=plt.gca().transAxes)
    plt.savefig(f'./output/boxplot_{year}_{model}.png')
    plt.show()


# In[14]:


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


# In[15]:


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


# In[16]:


alpha, mses = ridge_cv(alphas, x_train_2004, y_train_2004)
print(f'El mejor alpha para Ridge en 2004 es: {alpha}')
box(pd.DataFrame(mses).melt(), 2004, 'Ridge')


# In[17]:


alpha, mses = ridge_cv(alphas, x_train_2024, y_train_2024)
print(f'El mejor alpha para Ridge en 2024 es: {alpha}')
box(pd.DataFrame(mses).melt(), 2024, 'Ridge')


# In[18]:


# Función para calcular el mejor alpha y las métricas de evaluación
def lasso_logistic_cv(hyperparams, X_train, y_train):
	kf = KFold(n_splits=10, shuffle=True, random_state=101) # Definimos la partición de k-folds
	best_hyperparam = None
	best_mse = float('inf')
	mse_values = {param: [] for param in hyperparams} # Armamos un diccionario para guardar los mse de cada fold
	zero_coef_proportions = {param: [] for param in hyperparams} # Armamos un diccionario para guardar las proporciones de coeficientes nulos de cada fold
	
	for param in hyperparams: # Iteramos por cada valor de alpha
		model = LogisticRegression(penalty='l1', solver='saga', C=1/param, max_iter=5000, n_jobs= 7) # Definimos el modelo
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


# In[19]:


alpha, mses, prop = lasso_logistic_cv(alphas, x_train_2004, y_train_2004)
print(f'El mejor alpha para LASSO en 2004 es: {alpha}')
box(pd.DataFrame(mses).melt(), 2004, 'LASSO')
promedios = pd.DataFrame({key: sum(value) / len(value) for key, value in prop.items()}, index= [0]).melt()
line_prop(promedios, 2004)


# In[20]:


alpha, mses, prop = lasso_logistic_cv(alphas, x_train_2024, y_train_2024)
print(f'El mejor alpha para LASSO en 2024 es: {alpha}')
box(pd.DataFrame(mses).melt(), 2024, 'LASSO')
promedios = pd.DataFrame({key: sum(value) / len(value) for key, value in prop.items()}, index= [0]).melt()
line_prop(promedios, 2024)


# In[16]:


# Ahora queremos ver cuales son las variables nulas para cada modelo LASSO con el alpha optimo
def Lasso_logit(x_train, y_train, x_test, y_test, alpha):
    model = LogisticRegression(penalty='l1', solver='saga', C=1/alpha, max_iter=5000, n_jobs=7).fit(x_train, y_train)
    var_names = x_train.columns
    coefs = model.coef_[0]
    mse = mean_squared_error(y_test, model.predict(x_test)) # También guardamos el mse para evaluar el modelo
    return dict(zip(var_names, coefs)), mse


# In[17]:


# Para 2004
coefs, mse = Lasso_logit(x_train_2004, y_train_2004, x_test_2004, y_test_2004, 10) 
lasso_coefs = pd.DataFrame(coefs, index= [0]).melt() # Guardamos los coeficientes en un df para 2004

print(f'El ECM para LASSO en 2004 es: {mse}') # Imprimimos el mse para 2004


# In[19]:


# Para 2024
coefs, mse = Lasso_logit(x_train_2024, y_train_2024, x_test_2024, y_test_2024, 10) # Hacemos lo mismo con los coefs de 2024
lasso_coefs = pd.concat([lasso_coefs, pd.DataFrame(coefs, index= [0]).melt()['value']], axis= 1) # Juntamos los coeficientes de ambos años
lasso_coefs.columns = ['variable', '2004', '2024'] # Renombramos las columnas

print(f'El ECM para LASSO en 2024 es: {mse}')


# In[20]:


# Queremos evaluar la capacidad predictiva de Ridge vs logit mediante el mse
def Ridge_logit(x_train, y_train, x_test, y_test, alpha):
    model = LogisticRegression(penalty='l2', C=1/alpha, max_iter=10000).fit(x_train, y_train)
    mse = mean_squared_error(y_test, model.predict(x_test)) # Guardamos el mse para evaluar el modelo
    return mse


# In[21]:


mse = Ridge_logit(x_train_2004, y_train_2004, x_test_2004, y_test_2004, 100) # Calculamos el mse para Ridge en 2004
print(f'El ECM para Ridge en 2004 es: {mse}')


# In[22]:


mse = Ridge_logit(x_train_2024, y_train_2024, x_test_2024, y_test_2024, 10) # Calculamos el mse para Ridge en 2024
print(f'El ECM para Ridge en 2004 es: {mse}')


# In[ ]:




