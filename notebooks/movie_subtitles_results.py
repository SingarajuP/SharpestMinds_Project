#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import re
from string import punctuation 


# In[3]:


df=pd.read_csv("../data/raw/movies_titles.csv")


# In[4]:


df.head()


# In[5]:


df.shape


# In[3]:


df.title.to_csv('../models/titles.csv',index=False, header=False)


# In[6]:


df_sub=pd.read_csv("../data/raw/movies_subtitles.csv")


# In[7]:


df_sub.shape


# In[8]:


df_sub.head()


# In[9]:


df_sub.groupby(['imdb_id']).count()


# In[10]:


df['title'][199]


# In[11]:


imdb=df.loc[df['title'] == "Harry Potter and the Philosopher's Stone"]['imdb_id']
imdb


# In[12]:


imdb[747]


# In[13]:


df_t=df_sub.loc[df_sub['imdb_id']==imdb[747]]


# In[14]:


df_t


# In[15]:


import nltk
import nltk.data
from string import punctuation 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords


# In[16]:


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


# In[17]:


df_t['cleaned_text'] = df_t['text'].apply(lambda x: text_cleaning(x))


# In[18]:


df_t


# In[19]:


#df_t['cleaned_text'].to_csv("../data/processed/cleaned_text.csv", index=False, header=False)


# In[20]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle
import itertools
import os
import csv


# In[21]:


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


# In[22]:


tfidf_vectorizer=pickle.load(open('../models/tfidf_vect.pkl','rb'))


# In[23]:


test_tfidf = tfidf_vectorizer.transform(df_t['cleaned_text'])


# In[24]:


test_model_lr=pickle.load(open('../models/lr_mn.pkl','rb'))


# In[25]:


ytest_pred=test_model_lr.predict(test_tfidf)
ytest_pred


# In[26]:


df_t['predicted_label']=ytest_pred


# In[27]:


df_t


# In[28]:


df_t['predicted_label']


# In[29]:


emotion = pd.read_csv('../models/emotions.csv')


# In[30]:


dic_emotions=emotion.to_dict('series')


# In[31]:


dic_emotions['emotion']


# In[32]:


df_t['predicted_emotion'] = df_t['predicted_label'].map(dic_emotions['emotion'])


# In[33]:


df_t # Toy story


# In[34]:


df_t.groupby(['predicted_emotion']).count() # 83% of data is joy


# In[35]:


df_t # For Return of the Jedi


# In[36]:


df_t.groupby(['predicted_emotion']).count() #74.5% of data is joy


# In[33]:


df_t #Harry potter 1


# In[34]:


df_t.groupby(['predicted_emotion']).count() #70% of the data for joy


# In[ ]:




