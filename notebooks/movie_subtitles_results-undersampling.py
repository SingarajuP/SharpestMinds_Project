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


# In[5]:


df.title.to_csv('../models/Titles.csv',index=False, header=False)


# In[6]:


df_sub=pd.read_csv("../data/raw/movies_subtitles.csv")


# In[7]:


df_sub.head()


# In[14]:


df['title'][199]


# In[91]:


imdb=df.loc[df['title'] == "Toy Story"]['imdb_id']
imdb


# In[92]:


df_t=df_sub.loc[df_sub['imdb_id']==imdb[0]]


# In[93]:


df_t


# In[62]:


import nltk
import nltk.data
from string import punctuation 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords


# ##### Removing all songs from the subtitles data 

# In[63]:


df_t.text.str.contains('♪')


# In[94]:


df_nosong=df_t[~df_t.text.str.contains('♪')]
df_nosong


# In[ ]:





# In[42]:


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


# In[95]:


df_nosong['cleaned_text'] = df_nosong['text'].apply(lambda x: text_cleaning(x))


# In[96]:


df_nosong


# In[98]:


df_nosong = df_nosong[df_nosong['cleaned_text'].map(len) > 0]
df_nosong


# In[55]:


#df_t['cleaned_text'].to_csv("../data/processed/cleaned_text.csv", index=False, header=False)


# In[44]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle
import itertools
import os
import csv


# In[45]:


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


# In[46]:


tfidf_vectorizer=pickle.load(open('../models/tfidf_vect_undersampling.pkl','rb'))


# In[99]:


test_tfidf = tfidf_vectorizer.transform(df_nosong['cleaned_text'])


# In[68]:


test_model_lr=pickle.load(open('../models/lr_neutral.pkl','rb'))


# In[100]:


ytest_pred=test_model_lr.predict(test_tfidf)
ytest_pred


# In[101]:


df_nosong['Predicted_label']=ytest_pred


# In[102]:


df_nosong


# In[30]:


df_t['Predicted_label']


# In[53]:


emotion = pd.read_csv('../models/emotions_neutral.csv')


# In[54]:


dic_emotions=emotion.to_dict('series')


# In[55]:


dic_emotions['Emotion']


# In[103]:


df_nosong['Predicted_emotion'] = df_nosong['Predicted_label'].map(dic_emotions['Emotion'])


# In[104]:


df_nosong # Toy story


# In[105]:


df_nosong.groupby(['Predicted_emotion']).count() # 98.8% of data is neutral


# In[73]:


df_nosong # For Return of the Jedi


# In[75]:


df_nosong.groupby(['Predicted_emotion']).count() #95% of data is neutral


# In[89]:


df_nosong #Harry potter 1


# In[90]:


df_nosong.groupby(['Predicted_emotion']).count() #96% of the data for neutral


# In[ ]:




