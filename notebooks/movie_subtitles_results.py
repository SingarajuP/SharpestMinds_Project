#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from string import punctuation 


# In[2]:


df=pd.read_csv("../data/raw/movies_meta.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[37]:


df.title.to_csv('../models/Titles.csv',index=False, header=False)


# In[5]:


df_sub=pd.read_csv("../data/raw/movies_subtitles.csv")


# In[6]:


df_sub.shape


# In[7]:


df_sub.head()


# In[8]:


df_sub.groupby(['imdb_id']).count()


# In[14]:


df['title'][199]


# In[38]:


imdb=df.loc[df['title'] == "Harry Potter and the Philosopher's Stone"]['imdb_id']
imdb


# In[12]:


imdb[0]


# In[41]:


df_t=df_sub.loc[df_sub['imdb_id']==imdb[747]]


# In[42]:


df_t


# In[18]:


import nltk
import nltk.data
from string import punctuation 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords


# In[19]:


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


# In[43]:


df_t['cleaned_text'] = df_t['text'].apply(lambda x: text_cleaning(x))


# In[44]:


df_t


# In[55]:


df_t['cleaned_text'].to_csv("../data/processed/cleaned_text.csv", index=False, header=False)


# In[22]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle
import itertools
import os
import csv


# In[23]:


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


# In[24]:


tfidf_vectorizer=pickle.load(open('../models/tfidf_vect.pkl','rb'))


# In[45]:


test_tfidf = tfidf_vectorizer.transform(df_t['cleaned_text'])


# In[46]:


test_model_lr=pickle.load(open('../models/lr_mn.pkl','rb'))


# In[47]:


ytest_pred=test_model_lr.predict(test_tfidf)
ytest_pred


# In[48]:


df_t['Predicted_label']=ytest_pred


# In[49]:


df_t


# In[30]:


df_t['Predicted_label']


# In[31]:


emotion = pd.read_csv('../models/emotions.csv')


# In[32]:


dic_emotions=emotion.to_dict('series')


# In[33]:


dic_emotions['Emotion']


# In[50]:


df_t['Predicted_emotion'] = df_t['Predicted_label'].map(dic_emotions['Emotion'])


# In[33]:


df_t # Toy story


# In[34]:


df_t.groupby(['Predicted_emotion']).count() # 83% of data is joy


# In[35]:


df_t # For Return of the Jedi


# In[36]:


df_t.groupby(['Predicted_emotion']).count() #74.5% of data is joy


# In[51]:


df_t #Harry potter 1


# In[52]:


df_t.groupby(['Predicted_emotion']).count() #70% of the data for joy


# In[ ]:




