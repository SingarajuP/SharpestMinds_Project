#!/usr/bin/env python
# coding: utf-8

# ##### This. notebook compares all three models with logistic regression (model taking imbalanced data, undersampling data, considering class weights) for the movie subtitles data 

# ##### Importing all necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle
import re
from string import punctuation 
import csv


# In[2]:


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


# In[3]:


import nltk
import nltk.data
from string import punctuation 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords


# ##### Loading the data, models and supporting files

# In[4]:


df_title=pd.read_csv("../data/raw/movies_titles.csv")
df_sub=pd.read_csv("../data/raw/movies_subtitles.csv")


# In[33]:


tfidf_vectorizer=pickle.load(open('../models/tfidf_vect.pkl','rb'))
tfidf_vectorizer_under=pickle.load(open('../models/tfidf_vect_undersampling.pkl','rb'))
tfidf_vectorizer_imb=pickle.load(open('../models/tfidf_vect_imb.pkl','rb'))
tfidf_vectorizer_cw=pickle.load(open('../models/tfidf_vect_classweights.pkl','rb'))


# In[34]:


test_model_lr=pickle.load(open('../models/lr_mn.pkl','rb'))
test_model_lr_under=pickle.load(open('../models/lr_mn_neutral.pkl','rb'))
test_model_lr_imb=pickle.load(open('../models/lr_mn_imb.pkl','rb'))
test_model_lr_cw=pickle.load(open('../models/lr_mn_classweights.pkl','rb'))


# In[35]:


emotion = pd.read_csv('../models/emotions.csv')
emotion_neutral = pd.read_csv('../models/emotions_neutral.csv')

dic_emotions=emotion.to_dict('series')
dic_emotions_neutral=emotion_neutral.to_dict('series')

print(dic_emotions['emotion'])
print(dic_emotions_neutral['emotion'])


# ##### Preprocessing the data

# In[12]:


df_title.head()


# In[13]:


df_sub.head()


# In[14]:


df_sub.groupby(['imdb_id']).count()


# In[15]:


imdb=df_title.loc[df_title['title'] == "Harry Potter and the Philosopher's Stone"]['imdb_id']
imdb


# In[36]:


df_harry=df_sub.loc[df_sub['imdb_id']==imdb[747]]


# In[37]:


df_harry


# In[38]:


def text_cleaning(text):
   
    text = re.sub(r"[^A-Za-z]", " ", text)
    
    
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


# In[39]:


df_harry=df_harry[~df_harry.text.str.contains('♪')]

df_harry['cleaned_text'] = df_harry['text'].apply(lambda x: text_cleaning(x))

df_harry = df_harry[df_harry['cleaned_text'].map(len) > 0]


# In[40]:


df_harry


# ##### Testing models

# In[41]:


test_tfidf = tfidf_vectorizer.transform(df_harry['cleaned_text'])
test_tfidf_under = tfidf_vectorizer_under.transform(df_harry['cleaned_text'])
test_tfidf_imb = tfidf_vectorizer_imb.transform(df_harry['cleaned_text'])
test_tfidf_cw = tfidf_vectorizer_cw.transform(df_harry['cleaned_text'])

ytest_pred=test_model_lr.predict(test_tfidf)
ytest_pred_under=test_model_lr_under.predict(test_tfidf_under)
ytest_pred_imb=test_model_lr_imb.predict(test_tfidf_imb)
ytest_pred_cw=test_model_lr_cw.predict(test_tfidf_cw)


# In[42]:


df_harry['predicted_label']=ytest_pred
df_harry['predicted_label_under']=ytest_pred_under
df_harry['predicted_label_imb']=ytest_pred_imb
df_harry['predicted_label_cw']=ytest_pred_cw


# In[43]:


df_harry


# In[46]:


df_harry['predicted_emotion'] = df_harry['predicted_label'].map(dic_emotions['emotion'])
df_harry['predicted_emotion_under'] = df_harry['predicted_label_under'].map(dic_emotions_neutral['emotion'])
df_harry['predicted_emotion_imb'] = df_harry['predicted_label_imb'].map(dic_emotions_neutral['emotion'])
df_harry['predicted_emotion_cw'] = df_harry['predicted_label_cw'].map(dic_emotions_neutral['emotion'])


# In[47]:


df_harry # Harry potter


# In[48]:


df_harry.groupby(['predicted_emotion']).count()


# In[50]:


df_harry.groupby(['predicted_emotion_under']).count() 


# In[51]:


df_harry.groupby(['predicted_emotion_imb']).count() 


# In[52]:


df_harry.groupby(['predicted_emotion_cw']).count() 


# In[54]:


df_harry['text'][0:10]


# ##### To confirm the results from the models, checking the emotion of the movie subtitle manually.

# In[56]:


df_harry.to_csv("../data/processed/manual_testing_harry.csv",header=False)


# In[64]:


df_harry.iloc[[54]] # sadness is the correct emotion


# In[65]:


df_harry.iloc[[55]] # This is also sad. all the predictions are wrong.


# In[74]:


df_harry.iloc[[150]]


# In[75]:


df_harry.iloc[[151]]


# The above two rows, if combined together should be angry emotion. When splittled the emotion is not captured correctly from any model. 

# In[ ]:




