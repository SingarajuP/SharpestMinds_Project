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
import re
import torch
import nltk
import nltk.data
from string import punctuation 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords


# In[3]:


df_senti=pd.read_csv("../data/raw/sentiments_training.csv", encoding= 'unicode_escape')
df_test=pd.read_csv("../data/raw/sentiments_test.csv", encoding= 'unicode_escape')


# In[4]:


df_senti.head()


# In[5]:


df_test.head()


# In[6]:


df_senti.shape,df_test.shape


# In[7]:


print("Training set:",df_senti['sentiment'].value_counts())
print("Test set:",df_test['sentiment'].value_counts())


# In[8]:


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


# In[47]:


df_senti['cleaned_text'][5]


# In[9]:


df_senti['cleaned_text'] = df_senti['text'].apply(lambda x: text_cleaning(x))
df_test['cleaned_text'] = df_test['text'].apply(lambda x: text_cleaning(x))


# In[10]:


df_senti['number_words']=df_senti['cleaned_text'].str.split().apply(len)
df_test['number_words']=df_test['cleaned_text'].str.split().apply(len)


# In[11]:


df_senti['number_words'].min(),df_senti['number_words'].max(), df_senti['number_words'].median()


# In[12]:


df_test['number_words'].min(),df_test['number_words'].max(), df_test['number_words'].median()


# In[13]:


df=df_senti[df_senti['number_words']>1]
df_test=df_test[df_test['number_words']>0]

print(df.shape)
print(df_test.shape)


# In[14]:


print(df['sentiment'].value_counts())


# In[15]:


df=df[['cleaned_text','sentiment']]


# In[16]:


df=df.reset_index()
df


# In[17]:


df.to_csv("../data/processed/cleaned_text_forbert_sentiment.csv", index=False, header=False)


# In[17]:


df['labels'] = df['sentiment'].factorize()[0]
df.head()


# In[18]:


uniquevalues = pd.unique(df[['sentiment']].values.ravel())
df_unique=pd.DataFrame(uniquevalues,columns=['sentiment'])
df_unique


# In[19]:


df_test=df_test.dropna()
df_test


# In[20]:


df_test=df_test.reset_index()
df_test=df_test[['cleaned_text','sentiment']].copy()


# In[21]:


mapp={'neutral':0,'negative':1,'positive':2}
df_test['labels']=df_test['sentiment'].map(mapp)


# In[22]:


df_test


# In[23]:


#importing libraries for models and nlp tasks
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

from sentence_transformers import SentenceTransformer


# In[24]:


ytrain =df['labels']
ytest=df_test['labels']


# In[28]:


Xtrain=df['cleaned_text']
Xtest=df_test['cleaned_text']


# In[25]:


model = SentenceTransformer('all-MiniLM-L6-v2')


# In[30]:


def sentence(text):
    return model.encode(text)
    


# In[36]:


def bertembeddings(Xtrain,Xtest):
    Xtrain=Xtrain
    Xtest=Xtest
    
    with torch.no_grad():    


        train = Xtrain.apply(lambda x: sentence(x))
        test = Xtest.apply(lambda x: sentence(x))


    tf= [x for x in train.transpose()]
    train_features = np.asarray(tf)

    t= [x for x in test.transpose()]
    test_features = np.asarray(t)
    
    return train_features,test_features


# In[37]:


train_embed,test_embed=bertembeddings(Xtrain,Xtest)


# In[34]:


print(train_embed[0])


# In[ ]:





# In[ ]:





# In[ ]:





# ##### Logistic Regression model

# In[38]:


#Logistic Regression with multinomial
lr_mn = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr_mn.fit(train_embed, ytrain)


# In[39]:


ypred_lr_mn=lr_mn.predict(test_embed)


# In[40]:


tr_acc_lr_mn = lr_mn.score(train_embed, ytrain)*100
test_acc_lr_mn =  accuracy_score(ytest,ypred_lr_mn) * 100
print(tr_acc_lr_mn,test_acc_lr_mn)


# In[41]:


cm = confusion_matrix(ytest, ypred_lr_mn)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True,fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


# In[42]:


pickle.dump(lr_mn, open('../models/lr_mn_sentiment_bert.pkl', 'wb'))


# ##### SVM Classifier

# In[43]:


get_ipython().run_cell_magic('time', '', "svm = SVC( kernel ='linear',C = 1, decision_function_shape='ovo')\nsvm.fit(train_embed, ytrain)")


# In[44]:


ypred_svm=svm.predict(test_embed)


# In[45]:


tr_acc_svm= svm.score(train_embed, ytrain)*100
test_acc_svm =  accuracy_score(ytest,ypred_svm) * 100
print(tr_acc_svm,test_acc_svm)


# In[46]:


pickle.dump(svm, open('../models/svm_sentiment_bert.pkl', 'wb'))


# In[ ]:




