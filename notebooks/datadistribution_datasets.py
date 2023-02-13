#!/usr/bin/env python
# coding: utf-8

# In[12]:


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


#Reading dataset into dataframe. 
df_emotions = pd.read_pickle('../data/raw/emotions_training.pkl')
df_senti=pd.read_csv("../data/raw/sentiments_training.csv", encoding= 'unicode_escape')


# In[3]:


#The dataset is already cleaned and preprocessed
df_emotions.head()


# In[4]:


df_senti.head()


# In[5]:


df_emotions['emotions'].value_counts()


# In[7]:


df_senti['sentiment'].value_counts()


# In[8]:


df_neutral=df_senti[df_senti.sentiment=='neutral'][['text','sentiment']]
df_neutral=df_neutral.rename(columns={'sentiment':'emotions'})


# In[9]:


df_neutral['emotions'].value_counts()


# In[10]:


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


# In[13]:


df_emotions['cleaned_text'] = df_emotions['text'].apply(lambda x: text_cleaning(x))
df_neutral['cleaned_text'] = df_neutral['text'].apply(lambda x: text_cleaning(x))


# In[19]:


#Getting the count of words in each row. 
df_emotions['number_words']=df_emotions['cleaned_text'].str.split().apply(len)


# In[20]:


df_neutral['number_words']=df_neutral['cleaned_text'].str.split().apply(len)


# In[21]:


df_emotions['number_words'].min(),df_emotions['number_words'].max(), df_emotions['number_words'].median()


# In[22]:


df_neutral['number_words'].min(),df_neutral['number_words'].max(), df_neutral['number_words'].median()


# In[23]:


df_emotions = df_emotions[df_emotions['cleaned_text'].map(len) > 0]
df_neutral = df_neutral[df_neutral['cleaned_text'].map(len) > 0]


# In[24]:


df_emotions.head()


# ##### Visualization of data

# In[25]:


df_emotions['number_words'].hist()


# In[26]:


df_neutral['number_words'].hist()


# In[27]:


#Check the distribution of number of words 
text_3std_emotions = df_emotions['number_words'][~((df_emotions['number_words'] - df_emotions['number_words'].mean()).abs() > 3*df_emotions['number_words'].std())]


# In[28]:


print(text_3std_emotions.skew()) 
print(text_3std_emotions.mean()) 


# In[29]:


text_3std_emotions.hist()


# In[30]:


text_3std_neutral = df_neutral['number_words'][~((df_neutral['number_words'] - df_neutral['number_words'].mean()).abs() > 3*df_neutral['number_words'].std())]


# In[31]:


print(text_3std_neutral.skew()) 
print(text_3std_neutral.mean()) 


# In[32]:


text_3std_neutral.hist()


# In[ ]:




