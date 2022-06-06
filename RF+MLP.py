#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import random
import numpy as np
import csv


# In[38]:


df = pd.read_csv("FoodNutrients.csv")


# In[39]:


df.head()


# In[43]:


df = df.drop(columns=['Public Food Key', 'Food Name'])


# In[44]:


df = df[1:]


# In[45]:


df.head()


# In[48]:


# knn imputation transform 
from numpy import isnan
from sklearn.impute import KNNImputer
# load dataset

# split into input and output elements
#data = dataframe.values
#ix = [i for i in range(data.shape[1]) if i != 9]
# X, y = data[:, ix], data[:, 9]

# print total missing
# print('Missing: %d' % sum(isnan(df).flatten()))
# define imputer
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
# fit on the dataset
imputer.fit(df)
# transform the dataset
Xtrans = imputer.transform(df)
# print total missing
# print('Missing: %d' % sum(isnan(Xtrans).flatten()))


# In[61]:


for xt in Xtrans:
    xt[0] = str(xt[0])[:2]


# In[64]:


trans_df = pd.DataFrame(Xtrans, columns=df.columns)
trans_df


# In[69]:


from sklearn.decomposition import PCA
pca = PCA(n_components=8)
df1 = trans_df.drop(['Classification'], axis=1)
df1.head()


# In[70]:


reduction = pca.fit_transform(df1)


# In[132]:


X = reduction
y = trans_df['Classification']
y -= 11


# In[133]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=0.1, random_state=42)


# In[85]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


"""
n_estimators=100, *, 
criterion='gini', 
max_depth=None, 
min_samples_split=2, 
min_samples_leaf=1, 
min_weight_fraction_leaf=0.0, 
max_features='sqrt', 
max_leaf_nodes=None, 
"""


# In[87]:


clf = RandomForestClassifier(n_estimators=100 ,max_depth=5 , random_state=7)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
accuracy_score(y_pred, y_test)


# In[98]:


from sklearn.neural_network import MLPClassifier
#  solver=lbfgs’, ‘sgd’, ‘adam’
clf = MLPClassifier(random_state=1,max_iter=300, verbose=True).fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_pred, y_test)


# In[166]:


from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras


# In[167]:


model = Sequential()
# 100 50 = 91.85
# 50 40 = 85
model.add(Dense(96, input_dim=8, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(24, activation='sigmoid'))


# In[182]:


"""
'sparse_categorical_crossentropy'
'mean_squared_error'

"""
opt = keras.optimizers.Adam(learning_rate=0.000001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# In[186]:


model.fit(X_train, y_train, epochs=80, batch_size=16, validation_split=0.3)


# In[187]:


_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))


# In[105]:





# In[106]:





# In[ ]:





# In[ ]:




