#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import math

from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import Normalizer
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# In[2]:


DATA_DIR = Path('.', 'data', 'final', 'public')
train_values = pd.read_csv(DATA_DIR / 'train_values.csv', index_col='building_id')
train_labels = pd.read_csv(DATA_DIR / 'train_labels.csv', index_col='building_id')
test_values = pd.read_csv(DATA_DIR / 'test_values.csv', index_col='building_id')


# In[3]:


if 'presotere' in globals():
    del presotere


# In[4]:


class presotere(TransformerMixin,BaseEstimator):
    
    def __init__(self,caso,geocode,scaler):
        self.lencoder_col=list([])
        self.object_cols=list([])
        self.number_cols = list([])
        self.lencoder_col=list([])
        self._caso=caso
        self.legeo = LabelEncoder()
        self.lencoder= list([])
        self._sns_data=None
        self._y=None
        self._geocode=geocode
        self._scaler=scaler
        
    def inicializamodelo(self,caso):

        if caso == 1:
            self.legeo = LabelEncoder()
            self.lencoder= list([])
        if caso == 2:
            self.legeo = LabelEncoder()
            self.lencoder= OneHotEncoder(handle_unknown='ignore', sparse=False)   
        
        
    def transform(self,X,y=None, **kwargs):
        print("Transformando ",self.get_params())
        if self._caso == 1:
            contador=0
            self._sns_data=X[self.number_cols].copy()
            for col in X[self.object_cols].columns:
                self.lencoder.append(LabelEncoder())
                self.lencoder[contador].fit(X[col])
                self._sns_data[col]=self.lencoder[contador].transform(X[col])  
                contador=contador+1
                
                
        if self._caso == 2:  
            nada = None
            self._sns_data=None
            nada = self.lencoder.fit_transform(X[self.object_cols])
            co1c=0
            self.lencoder_col=list([])
            for co1 in self.lencoder.categories_:
                for co2 in co1:
                    self.lencoder_col.append(self.object_cols[co1c]+"_"+co2)
                co1c=co1c+1
            objedf=pd.DataFrame(nada,columns=self.lencoder_col,
                                         index=X[self.object_cols].index.tolist())
            self._sns_data=pd.concat([X[self.number_cols].copy(),objedf],axis=1)  

                
        if self._geocode>0:        
            geo_level_1_fact=math.pow(10,int(math.log(self._sns_data['geo_level_2_id'].max(),10)+1))
            geo_level_2_fact=math.pow(10,int(math.log(self._sns_data['geo_level_3_id'].max(),10)+1))
            self._sns_data['geo_level_n']=  self._sns_data['geo_level_1_id']*geo_level_1_fact*geo_level_2_fact+self._sns_data['geo_level_2_id']*geo_level_2_fact+self._sns_data['geo_level_3_id']
            self._sns_data['geo_level']=self._sns_data['geo_level_n']#.astype(np.int64).astype(str)
            self._sns_data['geo_level_2']=self._sns_data['geo_level_n']*self._sns_data['geo_level_n']
            self._sns_data['geo_level_cod']=self._sns_data['geo_level_n'].astype(np.int64)

            self._sns_data=self._sns_data.drop(['geo_level_n','geo_level'],axis=1)

        self._sns_data=self._sns_data.drop(['count_floors_pre_eq'],axis=1)
        if self._scaler=="MinMax":
            mi_scaler = MinMaxScaler()
        if self._scaler=="Standard":
            mi_scaler = StandardScaler()            
        self._sns_data[list(self._sns_data.columns)]=mi_scaler.fit_transform(self._sns_data.values.astype(float))
        lcolum_x = list(self._sns_data.columns) 
        return self._sns_data
    
    def fit(self,X, y=None, **kwargs):
        for key, value in kwargs.items():
            if key=="caso":
                self._caso=value
            if key=="geocode":
                self._geocode=value
            if key=="scaler":
                self._scaler=value
        s = (X.dtypes == 'object')
        self.object_cols = list(s[s].index)
        s = (X.dtypes != 'object')        
        self.number_cols = list(s[s].index) 
        self._y=y
        self.inicializamodelo(self._caso)
        self.transform(X,y, **kwargs)
        return self

    def predict(self, X):
        return(self._y)

    def fit_transform(self,X, y=None, **kwargs):
            miX=X.values
            miX=miX.reshape(-1, 1)
            self.fit(X, y, **kwargs)
            nada=self.transform(X, **kwargs)
            return(nada)
        
    def set_params(self,**kwargs):
            for key, value in kwargs.items():
                if key=="caso":
                    self._caso=value
                if key=="geocode":
                    self._geocode=value
                if key=="scaler":
                    self._scaler=value                                 
            return self    
        
    def get_params(self,**kwargs):
        return({"caso":self._caso,"geocode":self._geocode,"scaler":self._scaler})
    


# In[5]:


def cargamodelo(modelcaso):
    global model
    if modelcaso==1:
        model = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=500)
    if modelcaso==2:    
        model = RandomForestRegressor(max_depth=20, random_state=0,n_estimators=100)
    if modelcaso==3:
        model = DecisionTreeClassifier(random_state=0,max_depth=20) 
    if modelcaso==4:
        model = MultinomialNB
    if modelcaso==5:
        model = DecisionTreeRegressor(random_state=0)   
    if modelcaso==6:
        model = SVC(gamma='auto',verbose=True,kernel='linear', probability=True)       
    if modelcaso==7:
        model = RandomForestClassifier(max_depth=20, random_state=0,n_estimators=100)     
    if modelcaso==9:    
        model = Sequential()
        model.add(Dense(12, input_dim=36, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
      


# In[9]:


class mimodelo(RandomForestRegressor):
    def predict(self, X):
        return(super().predict(X).round().astype("int64"))


# In[10]:


if 'steps' in globals():
    del steps
if 'pipeline' in globals():    
    del pipeline
steps = [('Preprosessor', presotere(caso=1,geocode=0,scaler="MinMax")),
         ('Modelo',mimodelo(max_depth=5, random_state=0,n_estimators=10,verbose=2,n_jobs=2))]
#steps = [('Preprosessor', presotere(caso=1))]
pipeline = Pipeline(steps)


# In[14]:


#pipeline.fit(train_values,train_labels,Preprosessor__caso=2,Preprosessor__geocode=5,Preprosessor__scaler="MinMax")
#random_grid = {'Preprosessor__caso':[1],
#               'Preprosessor__geocode':[1],
#               'Preprosessor__scaler':["Standard"]
#              }
#print(random_grid)
#grid = GridSearchCV(pipeline, param_grid=random_grid, cv=3,scoring="f1_micro", verbose=2,n_jobs=-1)
#grid.fit(train_values,train_labels )
#print("Best score",grid.best_score_)
#print(grid.best_estimator_)


# In[15]:


#pipeline.fit(train_values,train_labels)
#nada=pipeline.predict(train_values)
##test_data=pipeline.predict(test_values)
#pvalues=test_values[['geo_level_1_id']]
#pvalues['damage_grade']=test_data.round().astype(np.int64)
#pvalues=pvalues.drop(['geo_level_1_id'],axis=1)
#pvalues.to_csv(DATA_DIR / 'submission_00_05.csv')


# In[16]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'Preprosessor__caso':[1,2],
               'Preprosessor__geocode':[0,1],
               'Preprosessor__scaler':["MinMax","Standard"],
               'Modelo__n_estimators': n_estimators,
               'Modelo__max_features': max_features,
               'Modelo__max_depth': max_depth,
               'Modelo__min_samples_split': min_samples_split,
               'Modelo__min_samples_leaf': min_samples_leaf,
               'Modelo__bootstrap': bootstrap}
print(random_grid)


# In[18]:


grid = GridSearchCV(pipeline, param_grid=random_grid, cv=3,scoring="f1_micro", verbose=2,n_jobs=2)


# In[ ]:


grid.fit(train_values,train_labels )


# In[ ]:


print("Best score",grid.best_score_)
print(grid.best_estimator_)


# In[ ]:




