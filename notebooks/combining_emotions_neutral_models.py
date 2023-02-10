#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


df_6emotions = pd.read_pickle('../data/raw/emotions_training.pkl')
df_senti=pd.read_csv("../data/raw/sentiments_training.csv", encoding= 'unicode_escape')


# ##### There are three sentiments like positive,negative and neutral in the data for sentiment analysis.  As we need only the data having neutral label we need to separate it. 

# In[5]:


df_senti.head()


# In[6]:


df_neutral=df_senti[df_senti.sentiment=='neutral'][['text','sentiment']]
df_neutral


# In[7]:


df_6emotions.head()


# In[8]:


df_neutral=df_neutral.rename(columns={'sentiment':'emotions'})


# In[9]:


df_neutral.head()


# In[10]:


df=pd.concat([df_6emotions,df_neutral], ignore_index=True)
df


# In[11]:


df=df.reset_index()


# In[12]:


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


df['cleaned_text'] = df['text'].apply(lambda x: text_cleaning(x))


# In[22]:


df['cleaned_text'].to_csv("../data/processed/cleaned_text_neutral.csv", index=False, header=False)


# In[14]:


#Defining class for each emotion
df['labels'] = df['emotions'].factorize()[0]
df.head()


# In[15]:


uniquevalues = pd.unique(df[['emotions']].values.ravel())
df_unique=pd.DataFrame(uniquevalues,columns=['emotion'])


# In[16]:


df_unique


# In[17]:


df_unique.to_csv('../models/emotions_neutral.csv',index=False)


# #### Undersampling the data

# In[18]:


import imblearn
from imblearn.under_sampling import RandomUnderSampler


# In[31]:


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


# In[20]:


tfidf_vectorizer = TfidfVectorizer()


# In[21]:


y =df['labels']


# In[22]:


#Train test split of the data
Xtrain, Xtest, ytrain, ytest = train_test_split(df['cleaned_text'], y, test_size=0.3,random_state=1)
Xtrain_tfidf = tfidf_vectorizer.fit_transform(Xtrain)
Xtest_tfidf = tfidf_vectorizer.transform(Xtest)


# In[35]:


with open('../models/tfidf_vect_neutral.pkl', 'wb') as file:  
    pickle.dump(tfidf_vectorizer, file) 


# ##### Taking equal number of samples in test data also

# In[23]:


#Train test split of the data
Xtrain, Xtest, ytrain, ytest = train_test_split(df['cleaned_text'], y, test_size=0.3,random_state=1)
Xtrain_tfidf = tfidf_vectorizer.fit_transform(Xtrain)


# In[27]:


#ytest.value_counts()
type(ytest)


# In[28]:


df_test= pd.concat([Xtest, ytest], axis=1)


# In[34]:


df_test.head()


# In[30]:


df_test.labels.value_counts()


# In[32]:


df_test=utils.shuffle(df_test.groupby("labels").head(3305))


# In[33]:


df_test.labels.value_counts()


# In[35]:


Xtest_bal=df_test['cleaned_text']
ytest_bal=df_test['labels']


# In[36]:


Xtest_bal_tfidf = tfidf_vectorizer.transform(Xtest_bal)


# In[ ]:





# ##### For undersampling the data, the text data has to be vectorized, otherwise getting an error. Hence, the data has been split into train and test and applied tfidf vectorization.

# In[37]:


undersample = RandomUnderSampler()
X_under, y_under = undersample.fit_resample(Xtrain_tfidf, ytrain)


# #### Models
# ##### Logistic Regression

# In[38]:


#Logistic Regression with multinomial
lr_mn = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr_mn.fit(X_under, y_under)


# In[39]:


ypred_lr_mn=lr_mn.predict(Xtest_tfidf)


# In[40]:


tr_acc_lr_mn = lr_mn.score(X_under, y_under)*100
test_acc_lr_mn =  accuracy_score(ytest,ypred_lr_mn) * 100
print(tr_acc_lr_mn,test_acc_lr_mn)


# In[51]:


pickle.dump(lr_mn, open('../models/lr_neutral.pkl', 'wb'))


# In[43]:


#Logistic Regression with One vs Rest
lr_ovr = LogisticRegression(multi_class='ovr', solver='liblinear')
lr_ovr.fit(X_under, y_under)


# In[44]:


ypred_lr_ovr=lr_ovr.predict(Xtest_tfidf)


# In[45]:


tr_acc_lr_ovr = lr_ovr.score(X_under, y_under)*100
test_acc_lr_ovr =  accuracy_score(ytest,ypred_lr_ovr) * 100
print(tr_acc_lr_ovr,test_acc_lr_ovr)


# In[52]:


pickle.dump(lr_ovr, open('../models/lr_ovr_neutral.pkl', 'wb'))


# For balanced test data:

# In[41]:


ypred_lr_mn=lr_mn.predict(Xtest_bal_tfidf)


# In[42]:


test_acc_lr_mn =  accuracy_score(ytest_bal,ypred_lr_mn) * 100
print(tr_acc_lr_mn,test_acc_lr_mn)


# In[ ]:





# ##### SVM

# In[46]:


svm = SVC( kernel ='linear',C = 1, decision_function_shape='ovo')
svm.fit(X_under, y_under)


# In[47]:


ypred_svm=svm.predict(Xtest_tfidf)


# In[48]:


tr_acc_svm = svm.score(X_under, y_under)*100
test_acc_svm =  accuracy_score(ytest,ypred_svm) * 100
print(tr_acc_svm,test_acc_svm)


# In[49]:


pickle.dump(svm, open('../models/svm_neutral.pkl', 'wb'))


# In[ ]:





# In[ ]:




