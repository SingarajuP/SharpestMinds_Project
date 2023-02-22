#!/usr/bin/env python
# coding: utf-8

# In[14]:


#importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle
import itertools
import os
import re
import pickle
from string import punctuation 
import nltk
import nltk.data
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords


# In[2]:


df = pd.read_pickle('../data/raw/emotions_training.pkl')


# In[3]:


df=df.reset_index()


# In[4]:


#Defining class for each emotion
df['labels'] = df['emotions'].factorize()[0]
df.head()


# In[11]:


def text_cleaning(text):
   
    text=re.sub("\(.*?\)","",text)

    text = re.sub(r"[^A-Za-z]", " ", str(text))
    
     #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])
    stopwords = nltk.corpus.stopwords.words('english')
    text = text.split()
    text = [w for w in text if not w in stopwords]
    text = " ".join(text)
        
    text = text.split()
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(lemmatized_words)
    text=text.lower()
    
    return text 


# In[15]:


df['cleaned_text'] = df['text'].apply(lambda x: text_cleaning(x))
df = df[df['cleaned_text'].map(len) > 0]


# In[16]:


df.shape


# In[17]:


uniquevalues = pd.unique(df[['emotions']].values.ravel())
df_unique=pd.DataFrame(uniquevalues,columns=['emotion'])


# In[18]:


df_unique


# In[37]:


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

# In[20]:


tfidf_vectorizer = TfidfVectorizer()


# In[21]:


y =df['labels']


# In[22]:


#Train test split of the data
Xtrain, Xtest, ytrain, ytest = train_test_split(df['text'], y, test_size=0.3,random_state=1)
Xtrain_tfidf = tfidf_vectorizer.fit_transform(Xtrain)
Xtest_tfidf = tfidf_vectorizer.transform(Xtest)


# In[23]:


pickle.dump(tfidf_vectorizer, open('../tfidfvectors/tfidf_vect_clean.pkl', 'wb'))


# ##### Logistic Regression

# In[24]:


#Logistic Regression with One vs Rest
lr_ovr = LogisticRegression(multi_class='ovr', solver='liblinear')
lr_ovr.fit(Xtrain_tfidf, ytrain)


# In[25]:


ypred_lr_ovr=lr_ovr.predict(Xtest_tfidf)


# In[26]:


len(ypred_lr_ovr)


# In[27]:


tr_acc_lr_ovr = lr_ovr.score(Xtrain_tfidf, ytrain)*100
test_acc_lr_ovr =  accuracy_score(ytest,ypred_lr_ovr) * 100
print(tr_acc_lr_ovr,test_acc_lr_ovr)


# In[28]:


cm = confusion_matrix(ytest, ypred_lr_ovr)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True,fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# In[29]:


print(classification_report(ytest,ypred_lr_ovr, digits=3))


# In[30]:


pickle.dump(lr_ovr, open('../models/lr_ovr_clean.pkl', 'wb'))


# In[31]:


#Logistic Regression with multinomial
lr_mn = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr_mn.fit(Xtrain_tfidf, ytrain)


# In[32]:


ypred_lr_mn=lr_mn.predict(Xtest_tfidf)


# In[33]:


tr_acc_lr_mn = lr_mn.score(Xtrain_tfidf, ytrain)*100
test_acc_lr_mn =  accuracy_score(ytest,ypred_lr_mn) * 100
print(tr_acc_lr_mn,test_acc_lr_mn)


# In[34]:


cm = confusion_matrix(ytest, ypred_lr_mn)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True,fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# In[35]:


pickle.dump(lr_mn, open('../models/lr_mn_clean.pkl', 'wb'))


# ##### Class weights

# In[38]:


weighting = compute_class_weight( class_weight ='balanced', classes =np.unique(y),y= y)
class_weights = dict(zip(np.unique(y), weighting))


# In[40]:


lr_mn_cw = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight=class_weights)
lr_mn_cw.fit(Xtrain_tfidf, ytrain)


# In[41]:


ypred_lr_mn_cw=lr_mn_cw.predict(Xtest_tfidf)


# In[42]:


tr_acc_lr_mn_cw = lr_mn_cw.score(Xtrain_tfidf, ytrain)*100
test_acc_lr_mn_cw =  accuracy_score(ytest,ypred_lr_mn_cw) * 100
print(tr_acc_lr_mn_cw,test_acc_lr_mn_cw)


# In[44]:


pickle.dump(lr_mn_cw, open('../models/lr_mn_clean_cw.pkl', 'wb'))


# In[ ]:




