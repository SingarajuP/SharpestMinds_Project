#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Required python libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle
import re

import nltk
import nltk.data
from string import punctuation 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords


# In[2]:


df_6emotions = pd.read_pickle('../data/raw/emotions_training.pkl')
df_senti=pd.read_csv("../data/raw/sentiments_training.csv", encoding= 'unicode_escape')


# In[3]:


df_neutral=df_senti[df_senti.sentiment=='neutral'][['text','sentiment']]
df_neutral=df_neutral.rename(columns={'sentiment':'emotions'})
df_neutral


# In[4]:


df=pd.concat([df_6emotions,df_neutral], ignore_index=True)
df


# In[5]:


def text_cleaning(text):
   
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


# In[6]:


df['cleaned_text'] = df['text'].apply(lambda x: text_cleaning(x))


# In[7]:


df.shape


# In[8]:


df=df[df['cleaned_text'].map(len) > 0]
df.shape


# In[9]:


#df.to_csv("../data/processed/allemotions_raw_cleaned_data.csv", index=False, header=False)


# In[9]:


#Defining class for each emotion
df['labels'] = df['emotions'].factorize()[0]
df.head()


# In[11]:


uniquevalues = pd.unique(df[['emotions']].values.ravel())
df_unique=pd.DataFrame(uniquevalues,columns=['emotion'])


# In[12]:


df_unique


# In[10]:


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

from sklearn import utils
from sklearn.utils.class_weight import compute_class_weight


# In[11]:


tfidf_vectorizer = TfidfVectorizer()


# In[12]:


y =df['labels']


# In[14]:


#Train test split of the data
Xtrain, Xtest, ytrain, ytest = train_test_split(df['cleaned_text'], y, test_size=0.3,random_state=1,stratify=y)
Xtrain_tfidf = tfidf_vectorizer.fit_transform(Xtrain)
Xtest_tfidf = tfidf_vectorizer.transform(Xtest)


# In[15]:


pickle.dump(tfidf_vectorizer, open('../tfidfvectors/tfidf_vect_imb.pkl', 'wb'))


# ##### Logistic Regression

# In[16]:


#Logistic Regression with multinomial
lr_mn = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr_mn.fit(Xtrain_tfidf, ytrain)


# In[17]:


ypred_lr_mn=lr_mn.predict(Xtest_tfidf)


# In[18]:


tr_acc_lr_mn = lr_mn.score(Xtrain_tfidf, ytrain)*100
test_acc_lr_mn =  accuracy_score(ytest,ypred_lr_mn) * 100
print(tr_acc_lr_mn,test_acc_lr_mn)


# In[23]:


cm = confusion_matrix(ytest, ypred_lr_mn)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True,fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# In[19]:


print(classification_report(ytest,ypred_lr_mn, digits=3))


# In[20]:


pickle.dump(lr_mn, open('../models/lr_mn_imb.pkl', 'wb'))


# In[ ]:




