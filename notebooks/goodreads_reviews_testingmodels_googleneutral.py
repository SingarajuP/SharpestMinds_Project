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


# In[5]:


tfidf_vectorizer=pickle.load(open('../tfidfvectors/tfidf_vect_emogoneutral.pkl','rb'))


# In[6]:


test_model_lr=pickle.load(open('../models/lr_mn_emogoneutral.pkl','rb'))
test_model_lr_cw=pickle.load(open('../models/lr_mn_emogoneutral_cw.pkl','rb'))


# In[7]:


emotion = pd.read_csv('../labels_prediction/emotions_googleneutral.csv')

dic_emotions=emotion.to_dict('series')

print(dic_emotions['emotion'])


# #### Webscraping goodreads website for getting reviews of a book
# ##### To get the link for the required book 

# In[25]:


data = {'q': "razor's edge"}
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

# In[26]:


rev="http://goodreads.com"+link[0]
rev_url = requests.get(rev)
rev_soup=BeautifulSoup(rev_url.content, 'html.parser')


# ##### Getting reviews from the web page of the book

# In[27]:


rev_list=[]
for x in rev_soup.find_all("section", {"class": "ReviewText"}):
    rev_list.append(x.text)


# In[28]:


df=pd.DataFrame(rev_list, columns=['reviews'])
df


# ##### From all the languages in the reviews, selecting the english language reviews

# In[29]:


def detect_en(text):
    try:
        return detect(text) == 'en'
    except:
        return False


# In[30]:


df = df[df['reviews'].apply(detect_en)]
df=df.reset_index()
df


# In[31]:


#df.to_csv("razorsedge.csv",index=False,header=False)


# ##### Cleaning the text

# In[32]:


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


# In[33]:


df['cleaned_review'] = df['reviews'].apply(lambda x: text_cleaning(x))
df = df[df['cleaned_review'].map(len) > 0]


# In[34]:


df


# ##### Testing the reviews data for emotions using model

# In[35]:


test_tfidf = tfidf_vectorizer.transform(df['cleaned_review'])

ytest_pred=test_model_lr.predict(test_tfidf)

ytest_pred_cw=test_model_lr_cw.predict(test_tfidf)


# In[36]:


df['predicted_label']=ytest_pred

df['predicted_label_cw']=ytest_pred_cw


# In[37]:


df['predicted_emotion'] = df['predicted_label'].map(dic_emotions['emotion'])
df['predicted_emotion_cw'] = df['predicted_label_cw'].map(dic_emotions['emotion'])


# In[38]:


df


# In[22]:


df['reviews'][0]


# In[58]:


df['cleaned_review'][7]


# In[24]:


df['reviews'][19] 


# In[ ]:




