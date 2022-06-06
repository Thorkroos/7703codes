#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import random
import numpy as np
import csv


# In[188]:


df = pd.read_csv("FoodNutrients.csv")


# In[189]:


num_df = df.drop(columns=['Public Food Key', 'Food Name'])


# In[190]:


data = num_df[1:]


# In[191]:


data.head()


# # Inputation

# knn imputation transform 

# In[194]:


from numpy import isnan
from sklearn.impute import KNNImputer

# define imputer
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
# fit on the dataset
imputer.fit(data)
# transform the dataset
Xtrans = imputer.transform(data)
for xt in Xtrans:
    xt[0] = str(xt[0])[:2]


# In[196]:


trans_df = pd.DataFrame(Xtrans, columns=num_df.columns)


# In[197]:


trans_df


# PCA

# In[199]:


from sklearn.decomposition import PCA
pca = PCA(n_components=8)
df1 = trans_df.drop(['Classification'], axis=1)
df1.head()


# In[200]:


reduction = pca.fit_transform(df1)


# In[213]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

X = reduction
labels = trans_df['Classification']
le.fit(labels)
y = le.transform(labels)


# In[214]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=0.1, random_state=29)


# In[215]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


# In[240]:


clf = RandomForestClassifier(n_estimators=80 ,max_depth=7 , random_state=7)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
accuracy_score(y_pred, y_test)


# In[217]:


from sklearn.neural_network import MLPClassifier
#  solver=lbfgs’, ‘sgd’, ‘adam’
clf = MLPClassifier(random_state=1,max_iter=300, verbose=True).fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_pred, y_test)


# In[288]:


from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(42)


# In[296]:


model = Sequential()
# 100 50 = 91.85
# 50 40 = 85
model.add(Dense(66, input_dim=8, activation='relu'))
model.add(Dense(44, activation='relu'))
model.add(Dense(22, activation='sigmoid'))

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# In[297]:


model.fit(X_train, y_train, epochs=80, batch_size=32, validation_split=0.2)


# In[298]:


_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))


# In[105]:





# In[106]:





# In[ ]:





# In[ ]:




