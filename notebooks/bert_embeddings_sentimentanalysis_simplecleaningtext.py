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


# In[2]:


df_senti=pd.read_csv("../data/raw/sentiments_training.csv", encoding= 'unicode_escape')
df_test=pd.read_csv("../data/raw/sentiments_test.csv", encoding= 'unicode_escape')


# In[3]:


df_senti.head()


# In[4]:


df_test.head()


# In[5]:


df_senti.shape,df_test.shape


# In[6]:


print("Training set:",df_senti['sentiment'].value_counts())
print("Test set:",df_test['sentiment'].value_counts())


# In[7]:


df_senti['cleaned_text'] = df_senti['text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)


# In[8]:


df_senti['cleaned_text'][5]


# In[9]:


print(df_senti['sentiment'].value_counts())


# In[11]:


df=df_senti[['cleaned_text','sentiment']]


# In[12]:


df=df.reset_index()
df


# In[13]:


df.to_csv("../data/processed/cleaned_text_forbert_sentiment.csv", index=False, header=False)


# In[14]:


df['labels'] = df['sentiment'].factorize()[0]
df.head()


# In[15]:


uniquevalues = pd.unique(df[['sentiment']].values.ravel())
df_unique=pd.DataFrame(uniquevalues,columns=['sentiment'])
df_unique


# In[31]:


df['cleaned_text'] = df['cleaned_text'].astype(str)


# In[16]:


df_test=df_test.dropna()
df_test


# In[18]:


df_test['cleaned_text'] = df_test['text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)


# In[19]:


df_test=df_test.reset_index()
df_test=df_test[['cleaned_text','sentiment']].copy()


# In[20]:


mapp={'neutral':0,'negative':1,'positive':2}
df_test['labels']=df_test['sentiment'].map(mapp)


# In[21]:


df_test


# In[22]:


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


# In[23]:


ytrain =df['labels']
ytest=df_test['labels']


# In[32]:


Xtrain=df['cleaned_text']
Xtest=df_test['cleaned_text']


# In[25]:


model = SentenceTransformer('all-MiniLM-L6-v2')


# In[26]:


def sentence(text):
    return model.encode(text)
    


# In[27]:


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


# In[33]:


train_embed,test_embed=bertembeddings(Xtrain,Xtest)


# In[ ]:





# In[ ]:





# In[ ]:





# ##### Logistic Regression model

# In[34]:


#Logistic Regression with multinomial
lr_mn = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr_mn.fit(train_embed, ytrain)


# In[35]:


ypred_lr_mn=lr_mn.predict(test_embed)


# In[36]:


tr_acc_lr_mn = lr_mn.score(train_embed, ytrain)*100
test_acc_lr_mn =  accuracy_score(ytest,ypred_lr_mn) * 100
print(tr_acc_lr_mn,test_acc_lr_mn)


# In[37]:


cm = confusion_matrix(ytest, ypred_lr_mn)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True,fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


# In[38]:


pickle.dump(lr_mn, open('../models/lr_mn_sentiment_bert.pkl', 'wb'))


# ##### SVM Classifier

# In[39]:


get_ipython().run_cell_magic('time', '', "svm = SVC( kernel ='linear',C = 1, decision_function_shape='ovo')\nsvm.fit(train_embed, ytrain)")


# In[40]:


ypred_svm=svm.predict(test_embed)


# In[41]:


tr_acc_svm= svm.score(train_embed, ytrain)*100
test_acc_svm =  accuracy_score(ytest,ypred_svm) * 100
print(tr_acc_svm,test_acc_svm)


# In[42]:


pickle.dump(svm, open('../models/svm_sentiment_bert.pkl', 'wb'))


# ##### KNN classifier

# In[45]:


get_ipython().run_cell_magic('time', '', 'k_range = range(10,200,10)\ntrain_scores = []\ntest_scores = []\nfor k in k_range:\n    neigh = KNeighborsClassifier(n_neighbors=k)\n    knn=neigh.fit(train_embed,ytrain)\n    tr_acc_knn = knn.score(train_embed, ytrain)*100\n    ypred_knn = knn.predict(test_embed)\n    accuracy_knn = accuracy_score(ytest,ypred_knn)\n    test_acc_knn = accuracy_knn * 100\n    train_scores.append(tr_acc_knn)\n    test_scores.append(test_acc_knn)')


# In[46]:


K=pd.Series(k_range)


# In[47]:


plt.plot(K, train_scores,color='r',label='Training Accuracy')
plt.plot(K, test_scores,color='b',label='Test accuracy')


plt.xlabel('Value of K for KNN')
plt.ylabel('Training and Testing Accuracy')
plt.legend()
plt.show()


# for small range of k values

# In[48]:


get_ipython().run_cell_magic('time', '', 'k_range = range(70,90)\ntrain_scores = []\ntest_scores = []\nfor k in k_range:\n    neigh = KNeighborsClassifier(n_neighbors=k)\n    knn=neigh.fit(train_embed,ytrain)\n    tr_acc_knn = knn.score(train_embed, ytrain)*100\n    ypred_knn = knn.predict(test_embed)\n    accuracy_knn = accuracy_score(ytest,ypred_knn)\n    test_acc_knn = accuracy_knn * 100\n    train_scores.append(tr_acc_knn)\n    test_scores.append(test_acc_knn)')


# In[49]:


K=pd.Series(k_range)
plt.plot(K, train_scores,color='r',label='Training Accuracy')
plt.plot(K, test_scores,color='b',label='Test accuracy')
plt.xlabel('Value of K for KNN')
plt.ylabel('Training and Testing Accuracy')
plt.legend()
plt.show()


# In[50]:


neigh = KNeighborsClassifier(n_neighbors=73)
knn=neigh.fit(train_embed,ytrain)
tr_acc_knn = knn.score(train_embed, ytrain)*100
ypred_knn = knn.predict(test_embed)
accuracy_knn = accuracy_score(ytest,ypred_knn)
test_acc_knn = accuracy_knn * 100

print(tr_acc_knn,test_acc_knn)


# In[51]:


pickle.dump(knn, open('../models/knn_sentiment_bert.pkl', 'wb'))


# ##### XGB Classifier

# In[57]:


xgb = XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth=5,
                      min_child_weight=1, objective= 'multi:softmax', num_class= 3,
                      gamma=0,  subsample=0.8, colsample_bytree=0.8, seed=27)


# In[58]:


eval_set = [(train_embed, ytrain),(test_embed, ytest)]


# In[59]:


xgb.fit(train_embed, ytrain, eval_metric='auc', eval_set=eval_set, verbose=True)


# In[60]:


results = xgb.evals_result()
epochs = len(results['validation_0']['auc'])
x_axis = range(0, epochs)


# In[61]:


fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['auc'], label='Train')
ax.plot(x_axis, results['validation_1']['auc'], label='Test')
ax.legend()
plt.ylabel('AUC')
plt.title('XGBoost')
plt.show()


# In[67]:


xgb = XGBClassifier(learning_rate =0.1, n_estimators=20, max_depth=5,
                      min_child_weight=1, objective= 'multi:softmax', num_class= 3,
                      gamma=0,  subsample=0.8, colsample_bytree=0.8, seed=27)


# In[68]:


get_ipython().run_cell_magic('time', '', 'xgb.fit(train_embed, ytrain)')


# In[69]:


ypred_xgb=xgb.predict(test_embed)


# In[70]:


tr_acc_xgb = xgb.score(train_embed, ytrain)*100
test_acc_xgb =  accuracy_score(ytest,ypred_xgb) * 100
print(tr_acc_xgb,test_acc_xgb)


# In[71]:


pickle.dump(xgb, open('../models/xgb_basic_sentiment_bert.pkl', 'wb'))


# In[ ]:




