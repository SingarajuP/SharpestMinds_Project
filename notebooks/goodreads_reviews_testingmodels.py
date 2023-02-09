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


# #### Webscraping goodreads website for getting reviews of a book
# ##### To get the link for the required book 

# In[2]:


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

# In[3]:


rev="http://goodreads.com"+link[0]
rev_url = requests.get(rev)
rev_soup=BeautifulSoup(rev_url.content, 'html.parser')


# ##### Getting reviews from the web page of the book

# In[4]:


rev_list=[]
for x in rev_soup.find_all("section", {"class": "ReviewText"}):
    rev_list.append(x.text)


# In[5]:


df=pd.DataFrame(rev_list, columns=['Reviews'])
df


# ##### From all the languages in the reviews, selecting the english language reviews

# In[6]:


def detect_en(text):
    try:
        return detect(text) == 'en'
    except:
        return False


# In[7]:


df = df[df['Reviews'].apply(detect_en)]
df=df.reset_index()
df


# In[41]:


#df.to_csv("razorsedge.csv",index=False,header=False)


# ##### Cleaning the text

# In[8]:


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


# In[9]:


df['cleaned_review'] = df['Reviews'].apply(lambda x: text_cleaning(x))


# In[10]:


df


# ##### Testing the reviews data for emotions using model

# In[11]:


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


# ##### Emotions model (class imbalance in training data)

# In[12]:


tfidf_vectorizer1=pickle.load(open('../models/tfidf_vect.pkl','rb'))


# In[13]:


test_tfidf1 = tfidf_vectorizer1.transform(df['cleaned_review'])


# In[14]:


test_model_lr1=pickle.load(open('../models/lr_mn.pkl','rb'))


# In[15]:


ytest_pred1=test_model_lr1.predict(test_tfidf1)
ytest_pred1


# In[16]:


df['Predicted_label_imbalance']=ytest_pred1


# In[17]:


emotion1 = pd.read_csv('../models/emotions.csv')


# In[18]:


dic_emotions1=emotion1.to_dict('series')
dic_emotions1['Emotion']


# In[19]:


df['Predicted_emotion_imbalance'] = df['Predicted_label_imbalance'].map(dic_emotions1['Emotion'])


# ##### Undersampling model

# In[20]:


tfidf_vectorizer2=pickle.load(open('../models/tfidf_vect_undersampling.pkl','rb'))


# In[21]:


test_tfidf2 = tfidf_vectorizer2.transform(df['cleaned_review'])


# In[22]:


test_model_lr2=pickle.load(open('../models/lr_neutral.pkl','rb'))


# In[23]:


ytest_pred2=test_model_lr2.predict(test_tfidf2)
ytest_pred2


# In[24]:


df['Predicted_label_undersampling']=ytest_pred2


# In[25]:


emotion2 = pd.read_csv('../models/emotions_neutral.csv')


# In[26]:


dic_emotions2=emotion2.to_dict('series')
dic_emotions2['Emotion']


# In[27]:


df['Predicted_emotion_undersampling'] = df['Predicted_label_undersampling'].map(dic_emotions2['Emotion'])


# In[28]:


df


# In[38]:


df['Reviews'][7]


# In[39]:


df['Reviews'][9]


# In[40]:


df['Reviews'][12]


# In[ ]:




