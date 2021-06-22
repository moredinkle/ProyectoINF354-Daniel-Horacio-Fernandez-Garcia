
import pandas as pd 
import numpy as np

df = pd.read_csv (r'C:\Universidad\Septimo\354\datasetVentasJuegos.csv')
pd.set_option('max_columns', None)
df = df.replace(np.nan,"0")

X = df[['Publisher', 'Genre','Year','NA_Sales','JP_Sales','EU_Sales','Other_Sales','Global_Sales']]
y = df['Platform']

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
numericas = ['Year','NA_Sales','JP_Sales','EU_Sales','Other_Sales','Global_Sales']
numeric_transformer = Pipeline(steps=[('minmax',MinMaxScaler()),('scaler', StandardScaler())])

from sklearn.preprocessing import OneHotEncoder
categoricas = ['Publisher', 'Genre']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[('cat', categorical_transformer, categoricas),('num', numeric_transformer, numericas)])

from sklearn.tree import DecisionTreeClassifier
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', DecisionTreeClassifier())])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pipe.fit(X_train, y_train)
print('Puntaje clasificador:', pipe.score(X_test, y_test))








