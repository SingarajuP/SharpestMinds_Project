#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import requests  
import numpy as np
import pandas as pd
from langdetect import detect
import re
import pickle
from string import punctuation 
import nltk
import nltk.data
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords


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


tfidf_vectorizer=pickle.load(open('../tfidfvectors/tfidf_vect_clean.pkl','rb'))


# In[4]:


test_model_lr_imb=pickle.load(open('../models/lr_mn_clean.pkl','rb'))
test_model_lr_cw=pickle.load(open('../models/lr_mn_clean_cw.pkl','rb'))


# In[5]:


emotion = pd.read_csv('../labels_prediction/emotions.csv')
dic_emotions=emotion.to_dict('series')

print(dic_emotions['emotion'])


# #### Webscraping goodreads website for getting reviews of a book
# ##### To get the link for the required book 

# In[6]:


data = {'q': "The Razor's Edge"}
book_url = "https://www.goodreads.com/search"
req = requests.get(book_url, params=data)

book_soup = BeautifulSoup(req.text, 'html.parser')

titles=book_soup.find_all('a', class_ = 'bookTitle')
title=[]
link=[]
for bookname in titles:
    title.append(bookname.get_text())
    link.append(bookname['href'])


# ##### From all the links first link is the most closest search 

# In[7]:


rev="http://goodreads.com"+link[0]
rev_url = requests.get(rev)
rev_soup=BeautifulSoup(rev_url.content, 'html.parser')


# ##### Getting reviews from the web page of the book

# In[8]:


rev_list=[]
for x in rev_soup.find_all("section", {"class": "ReviewText"}):
    rev_list.append(x.text)


# In[9]:


df=pd.DataFrame(rev_list, columns=['reviews'])
df


# ##### From all the languages in the reviews, selecting the english language reviews

# In[10]:


def detect_en(text):
    try:
        return detect(text) == 'en'
    except:
        return False


# In[11]:


df = df[df['reviews'].apply(detect_en)]
df=df.reset_index()
df


# In[12]:


#df.to_csv("razorsedge.csv",index=False,header=False)


# ##### Cleaning the text

# In[13]:


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


# In[14]:


df['cleaned_review'] = df['reviews'].apply(lambda x: text_cleaning(x))
df = df[df['cleaned_review'].map(len) > 0]


# In[15]:


df


# ##### Testing the reviews data for emotions using model

# In[16]:


test_tfidf = tfidf_vectorizer.transform(df['cleaned_review'])

ytest_pred_imb=test_model_lr_imb.predict(test_tfidf)
ytest_pred_cw=test_model_lr_cw.predict(test_tfidf)


# In[17]:


df['predicted_label_imb']=ytest_pred_imb
df['predicted_label_cw']=ytest_pred_cw


# In[18]:


df['predicted_emotion_imb'] = df['predicted_label_imb'].map(dic_emotions['emotion'])
df['predicted_emotion_cw'] = df['predicted_label_cw'].map(dic_emotions['emotion'])


# In[19]:


df


# In[20]:


df.groupby(['predicted_emotion_imb']).count() 


# In[21]:


df.groupby(['predicted_emotion_cw']).count() 


# In[20]:


df['cleaned_review'][0]


# In[21]:


df['reviews'][0]


# In[22]:


df['reviews'][6]


# In[23]:


df['cleaned_review'][6]


# Predict probabilities

# In[16]:


test_tfidf = tfidf_vectorizer.transform(df['cleaned_review'])


# In[17]:


ytest_pred_imb_prob=test_model_lr_imb.predict_proba(test_tfidf)
ytest_pred_cw_prob=test_model_lr_cw.predict_proba(test_tfidf)


# In[18]:


df_imb = pd.DataFrame(ytest_pred_imb_prob, columns = ['sadness','joy','love','anger','fear','surprise'])
df_cw = pd.DataFrame(ytest_pred_cw_prob, columns = ['sadness','joy','love','anger','fear','surprise'])


# In[21]:


df_comb_imb=pd.concat([df,df_imb],axis=1)
df_comb_imb


# In[22]:


df_comb_cw=pd.concat([df,df_cw],axis=1)
df_comb_cw


# In[ ]:




