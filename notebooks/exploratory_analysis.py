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
import torch
import itertools
import os


# In[6]:


#Reading dataset into dataframe. 
df = pd.read_pickle('../data/raw/merged_training.pkl')


# In[7]:


#The dataset is already cleaned and preprocessed
df.head()


# ##### Exploring the statistics of the data

# In[8]:


#Number of rows in the dataframe
df.shape


# In[5]:


#Count of each emotion in the dataset
df['emotions'].value_counts()


# In[6]:


#Percentage of data for each emotion
df['emotions'].value_counts(normalize=True)*100


# In[7]:


df=df.reset_index()


# In[8]:


df['text'][0]


# In[9]:


#Getting the count of words in each row. 
df['Number_words']=df['text'].str.split().apply(len)


# In[10]:


df.head()


# In[11]:


df['Number_words'].min(),df['Number_words'].max(), df['Number_words'].median()


# In[12]:


df[df['Number_words'] == 1].count()


# In[13]:


df.loc[df['Number_words'] == 1, ['text','emotions']]


# The minimum words in each sample is 1 and is further explored to know whether the single word is directly the emotion itself. But surprisingly, the words and the corresponding emotion doesn't relate in most of the words among 26 samples. Consider, the rows where the word is when, the emotion is different each time and also meaningless. But as 26 rows among the total data of >400K is negligible. 

# In[14]:


df[df['Number_words'] == 178].count()


# In[15]:


df.loc[df['Number_words'] == 178, ['text','emotions']]


# In[16]:


df['text'][369479]


# ##### Visualization of data

# In[12]:



sns.catplot(data=df, x="emotions", kind="count", palette="ch:.25")


# In[17]:


df['Number_words'].hist()


# In[18]:


#Check the distribution of number of words 
text_3std = df['Number_words'][~((df['Number_words'] - df['Number_words'].mean()).abs() > 3*df['Number_words'].std())]


# In[19]:


print(text_3std.skew()) 
print(text_3std.mean()) 


# In[20]:


text_3std.hist()


# In[ ]:




