#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
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


dfn=pd.read_csv('../data/raw/go_emotions_dataset.csv')
dfe = pd.read_pickle('../data/raw/emotions_training.pkl')


# In[3]:


dfn=dfn[['text','neutral']]


# In[4]:


dfn=dfn[dfn.neutral==1]


# In[5]:


mapp={1:'neutral'}
dfn['neutral']=dfn['neutral'].map(mapp)
dfn=dfn.rename(columns={'neutral':'emotions'})

dfn


# In[42]:


dfn['text'][12]


# In[6]:


dfe


# In[7]:


df=pd.concat([dfe,dfn], ignore_index=True)
df=df.reset_index()
df


# In[8]:


df['cleaned_text'] = df['text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)


# In[9]:


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


# In[10]:


df['cleaned_text'] = df['cleaned_text'].apply(lambda x: text_cleaning(x))


# In[11]:


df.to_csv("../data/processed/emotions_neutral_raw_cleaned_data.csv", index=False, header=False)


# In[12]:


df.shape


# In[13]:


df=df[df['cleaned_text'].map(len) > 0]
df.shape


# In[15]:


df['labels'] = df['emotions'].factorize()[0]
df.head()


# In[16]:


uniquevalues = pd.unique(df[['emotions']].values.ravel())
df_unique=pd.DataFrame(uniquevalues,columns=['emotion'])
df_unique


# In[17]:


df_unique.to_csv('../labels_prediction/emotions_googleneutral.csv',index=False)


# In[18]:


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
from sklearn.utils.class_weight import compute_class_weight


# In[19]:


tfidf_vectorizer = TfidfVectorizer()


# In[20]:


y=df['labels']


# In[21]:


Xtrain, Xtest, ytrain, ytest = train_test_split(df['cleaned_text'], y, test_size=0.3,random_state=1,stratify=y)
Xtrain_tfidf = tfidf_vectorizer.fit_transform(Xtrain)
Xtest_tfidf = tfidf_vectorizer.transform(Xtest)


# In[35]:


with open('../tfidfvectors/tfidf_vect_emogoneutral.pkl', 'wb') as file:  
    pickle.dump(tfidf_vectorizer, file) 


# In[22]:


lr_mn = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr_mn.fit(Xtrain_tfidf, ytrain)


# In[23]:


ypred_lr_mn=lr_mn.predict(Xtest_tfidf)


# In[24]:


tr_acc_lr_mn = lr_mn.score(Xtrain_tfidf, ytrain)*100
test_acc_lr_mn =  accuracy_score(ytest,ypred_lr_mn) * 100
print(tr_acc_lr_mn,test_acc_lr_mn)


# In[25]:


cm = confusion_matrix(ytest, ypred_lr_mn)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True,fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# In[26]:


print(classification_report(ytest,ypred_lr_mn, digits=3))


# In[27]:


pickle.dump(lr_mn, open('../models/lr_mn_emogoneutral.pkl', 'wb'))


# In[28]:


weighting = compute_class_weight( class_weight ='balanced', classes =np.unique(y),y= y)
class_weights = dict(zip(np.unique(y), weighting))


# In[29]:


lr_mn_cw = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight=class_weights)
lr_mn_cw.fit(Xtrain_tfidf, ytrain)


# In[30]:


ypred_lr_mn_cw=lr_mn_cw.predict(Xtest_tfidf)


# In[31]:


tr_acc_lr_mn_cw = lr_mn_cw.score(Xtrain_tfidf, ytrain)*100
test_acc_lr_mn_cw =  accuracy_score(ytest,ypred_lr_mn_cw) * 100
print(tr_acc_lr_mn_cw,test_acc_lr_mn_cw)


# In[32]:


print(classification_report(ytest,ypred_lr_mn_cw, digits=3))


# In[33]:


pickle.dump(lr_mn_cw, open('../models/lr_mn_emogoneutral_cw.pkl', 'wb'))


# In[ ]:




