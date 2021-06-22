"""
Se debe seleccionar un dataset de la 1ª tarea, esta mínimamente si es tabular debe tener 
1000 registros por 7 columnas; de esta realizar el proceso de análisis de datos, es decir 
la tarea inicial de entender el problema, preprocesamiento, análisis, 
pruebas (tomar al menos 10 splits).
"""

import pandas as pd 
import numpy as np

df = pd.read_csv (r'C:\Universidad\Septimo\354\datasetVentasJuegos.csv')
pd.set_option('max_columns', None)
df = df.replace(np.nan,"0")



#LabelEncoder para convertir las columnas de plataforma(consola) y editora de cada juego
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
#df['Platform'] = encoder.fit_transform(df.Platform.values)
df['Publisher'] = encoder.fit_transform(df.Publisher.values)
df['Genre'] = encoder.fit_transform(df.Genre.values)


#LLevar los datos a un rango(0,100)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
df.iloc[:,6:11] = min_max_scaler.fit_transform(df.iloc[:,6:11].values)
#print(df.iloc[0:30,6:11], '\n', df.iloc[:,6:11].mean(), '\n', df.iloc[:,6:11].std(), '\n')


#StandardScaler para quitar varianza 
scaler = preprocessing.StandardScaler()
df.iloc[:,6:11] = scaler.fit_transform(df.iloc[:,6:11].values)
#print(df.iloc[0:30,6:11], '\n', df.iloc[:,6:11].mean(), '\n', df.iloc[:,6:11].std(), '\n')


X = df[['Genre','Publisher','Year','NA_Sales','JP_Sales','EU_Sales','Other_Sales','Global_Sales']]
y = df['Platform']

from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
#kf = model_selection.RepeatedKFold(n_splits=10, n_repeats=3)   RepeatedKfold
kf = model_selection.KFold(n_splits=10)

from sklearn.ensemble import RandomForestClassifier
clasificador = RandomForestClassifier(criterion='entropy')
clasificador = clasificador.fit(X_train, y_train)
predictions = clasificador.predict(X_test)

scores = model_selection.cross_val_score(clasificador, X_train, y_train, cv=kf, scoring="accuracy")
print("Precision K-fold:", scores)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


reporte = classification_report(y_test,predictions)
matriz = confusion_matrix(y_test,predictions)


print(reporte)
#print(matriz)