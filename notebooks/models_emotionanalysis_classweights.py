#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle
import itertools
import os


# In[2]:


df = pd.read_pickle('../data/raw/emotions_training.pkl')


# In[3]:


df=df.reset_index()


# In[4]:


#Defining class for each emotion
df['labels'] = df['emotions'].factorize()[0]
df.head()


# In[5]:


uniquevalues = pd.unique(df[['emotions']].values.ravel())
df_unique=pd.DataFrame(uniquevalues,columns=['emotion'])


# In[6]:


df_unique


# In[7]:


df_unique.to_csv('../labels_prediction/emotions.csv',index=False)


# In[12]:


#importing libraries for models and nlp tasks
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.utils.class_weight import compute_class_weight


# ### TF-IDF Vectorization for models

# In[8]:


tfidf_vectorizer = TfidfVectorizer()


# In[9]:


y =df['labels']


# In[10]:


#Train test split of the data
Xtrain, Xtest, ytrain, ytest = train_test_split(df['text'], y, test_size=0.3,random_state=1)
Xtrain_tfidf = tfidf_vectorizer.fit_transform(Xtrain)
Xtest_tfidf = tfidf_vectorizer.transform(Xtest)


# In[15]:


#pickle.dump(tfidf_vectorizer, open('../models/tfidf_vect.pkl', 'wb'))


# ##### Calculating classweights

# In[13]:


weighting = compute_class_weight( class_weight ='balanced', classes =np.unique(y),y= y)
print(weighting)


# In[15]:


class_weights = dict(zip(np.unique(y), weighting))


# ##### Logistic Regression

# In[16]:


#Logistic Regression with One vs Rest
lr_mn = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight=class_weights)
lr_mn.fit(Xtrain_tfidf, ytrain)


# In[17]:


ypred_lr_mn=lr_mn.predict(Xtest_tfidf)


# In[18]:


tr_acc_lr_mn = lr_mn.score(Xtrain_tfidf, ytrain)*100
test_acc_lr_mn =  accuracy_score(ytest,ypred_lr_mn) * 100
print(tr_acc_lr_mn,test_acc_lr_mn)


# In[19]:


cm = confusion_matrix(ytest, ypred_lr_mn)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True,fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# In[20]:


pickle.dump(lr_mn, open('../models/lr_mn_emotion_cw.pkl', 'wb'))


# In[21]:


print(classification_report(ytest,ypred_lr_mn, digits=3))


# In[ ]:




