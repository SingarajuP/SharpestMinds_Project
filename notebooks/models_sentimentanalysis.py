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


# In[8]:


df_senti['cleaned_text'] = df_senti['text'].apply(lambda x: text_cleaning(x))
df_test['cleaned_text'] = df_test['text'].apply(lambda x: text_cleaning(x))


# In[9]:


df_senti['number_words']=df_senti['cleaned_text'].str.split().apply(len)
df_test['number_words']=df_test['cleaned_text'].str.split().apply(len)


# In[10]:


df_senti['number_words'].min(),df_senti['number_words'].max(), df_senti['number_words'].median()


# In[11]:


df_test['number_words'].min(),df_test['number_words'].max(), df_test['number_words'].median()


# In[12]:


df_senti[df_senti['number_words'] == 0]


# In[13]:


len(df_senti[df_senti['number_words'] == 0])


# In[17]:


df_senti[df_senti['number_words'] == 1]


# While observing the one word sentences, many of the words doesn't make sense. As these are 2% of the whole data, need to check whether to remove them or not.

# In[16]:


df0=df_senti[df_senti['number_words']>0]
df1=df_senti[df_senti['number_words']>1]

df_test=df_test[df_test['number_words']>0]
print(df_senti.shape)
print(df0.shape)
print(df1.shape)

print(df_test.shape)


# In[17]:


print(df0['sentiment'].value_counts())
print(df1['sentiment'].value_counts())


# In[18]:


print(df0['number_words'].min(),df0['number_words'].max(), df0['number_words'].median())
print(df1['number_words'].min(),df1['number_words'].max(), df1['number_words'].median())


# In[19]:


df_test['number_words'].min(),df_test['number_words'].max(), df_test['number_words'].median()


# In[20]:


df0=df0[['cleaned_text','sentiment']]


# In[21]:


df0=df0.reset_index()
df0


# In[38]:


df1=df1[['cleaned_text','sentiment']]
df1=df1.reset_index()
df1


# In[22]:


df0.to_csv("../data/processed/cleaned_text_sentiment0.csv", index=False, header=False)


# In[39]:


df1.to_csv("../data/processed/cleaned_text_sentiment1.csv", index=False, header=False)


# In[23]:


df0['labels'] = df0['sentiment'].factorize()[0]
df0.head()


# In[40]:


df1['labels'] = df1['sentiment'].factorize()[0]
df1.head()


# In[24]:


uniquevalues = pd.unique(df0[['sentiment']].values.ravel())
df_unique=pd.DataFrame(uniquevalues,columns=['sentiment'])
df_unique


# In[29]:


df_unique.to_csv('../labels_prediction/sentiments.csv',index=False)


# In[25]:


df_test=df_test.dropna()
df_test


# In[26]:


df_test=df_test.reset_index()
df_test=df_test[['cleaned_text','sentiment']].copy()


# In[27]:


mapp={'neutral':0,'negative':1,'positive':2}
df_test['labels']=df_test['sentiment'].map(mapp)


# In[28]:


df_test


# In[29]:


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


# In[30]:


tfidf_vectorizer = TfidfVectorizer()


# In[31]:


ytrain =df0['labels']
ytest=df_test['labels']


# In[32]:


Xtrain_tfidf = tfidf_vectorizer.fit_transform(df0['cleaned_text'])
Xtest_tfidf = tfidf_vectorizer.transform(df_test['cleaned_text'])


# In[33]:


with open('../tfidfvectors/tfidf_vect_sentiment0.pkl', 'wb') as file:  
    pickle.dump(tfidf_vectorizer, file) 


# ##### Logistic Regression model

# In[34]:


#Logistic Regression with multinomial
lr_mn = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr_mn.fit(Xtrain_tfidf, ytrain)


# In[35]:


ypred_lr_mn=lr_mn.predict(Xtest_tfidf)


# In[36]:


tr_acc_lr_mn = lr_mn.score(Xtrain_tfidf, ytrain)*100
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


# Removing 1 word data which might be noise and see whether there is any improvement in the result

# In[41]:


ytrain =df1['labels']


# In[42]:


Xtrain_tfidf = tfidf_vectorizer.fit_transform(df1['cleaned_text'])
Xtest_tfidf = tfidf_vectorizer.transform(df_test['cleaned_text'])


# In[44]:


with open('../tfidfvectors/tfidf_vect_sentiment1.pkl', 'wb') as file:  
    pickle.dump(tfidf_vectorizer, file) 


# In[45]:


lr_mn = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr_mn.fit(Xtrain_tfidf, ytrain)


# In[46]:


ypred_lr_mn=lr_mn.predict(Xtest_tfidf)


# In[47]:


tr_acc_lr_mn = lr_mn.score(Xtrain_tfidf, ytrain)*100
test_acc_lr_mn =  accuracy_score(ytest,ypred_lr_mn) * 100
print(tr_acc_lr_mn,test_acc_lr_mn)


# In[49]:


pickle.dump(lr_mn, open('../models/lr_mn_sentiment.pkl', 'wb'))


# ##### SVM Classifier

# In[48]:


get_ipython().run_cell_magic('time', '', "svm = SVC( kernel ='linear',C = 1, decision_function_shape='ovo')\nsvm.fit(Xtrain_tfidf, ytrain)")


# In[50]:


ypred_svm=svm.predict(Xtest_tfidf)


# In[51]:


tr_acc_svm= svm.score(Xtrain_tfidf, ytrain)*100
test_acc_svm =  accuracy_score(ytest,ypred_svm) * 100
print(tr_acc_svm,test_acc_svm)


# In[52]:


pickle.dump(svm, open('../models/svm_sentiment.pkl', 'wb'))


# ##### KNN classifier

# In[57]:


get_ipython().run_cell_magic('time', '', 'k_range = range(10,200,10)\ntrain_scores = []\ntest_scores = []\nfor k in k_range:\n    neigh = KNeighborsClassifier(n_neighbors=k)\n    knn=neigh.fit(Xtrain_tfidf,ytrain)\n    tr_acc_knn = knn.score(Xtrain_tfidf, ytrain)*100\n    ypred_knn = knn.predict(Xtest_tfidf)\n    accuracy_knn = accuracy_score(ytest,ypred_knn)\n    test_acc_knn = accuracy_knn * 100\n    train_scores.append(tr_acc_knn)\n    test_scores.append(test_acc_knn)')


# In[58]:


K=pd.Series(k_range)


# In[59]:


plt.plot(K, train_scores,color='r',label='Training Accuracy')
plt.plot(K, test_scores,color='b',label='Test accuracy')


plt.xlabel('Value of K for KNN')
plt.ylabel('Training and Testing Accuracy')
plt.legend()
plt.show()


# for small range of k values

# In[60]:


get_ipython().run_cell_magic('time', '', 'k_range = range(100,125)\ntrain_scores = []\ntest_scores = []\nfor k in k_range:\n    neigh = KNeighborsClassifier(n_neighbors=k)\n    knn=neigh.fit(Xtrain_tfidf,ytrain)\n    tr_acc_knn = knn.score(Xtrain_tfidf, ytrain)*100\n    ypred_knn = knn.predict(Xtest_tfidf)\n    accuracy_knn = accuracy_score(ytest,ypred_knn)\n    test_acc_knn = accuracy_knn * 100\n    train_scores.append(tr_acc_knn)\n    test_scores.append(test_acc_knn)')


# In[61]:


K=pd.Series(k_range)
plt.plot(K, train_scores,color='r',label='Training Accuracy')
plt.plot(K, test_scores,color='b',label='Test accuracy')
plt.xlabel('Value of K for KNN')
plt.ylabel('Training and Testing Accuracy')
plt.legend()
plt.show()


# In[65]:


neigh = KNeighborsClassifier(n_neighbors=100)
knn=neigh.fit(Xtrain_tfidf,ytrain)
tr_acc_knn = knn.score(Xtrain_tfidf, ytrain)*100
ypred_knn = knn.predict(Xtest_tfidf)
accuracy_knn = accuracy_score(ytest,ypred_knn)
test_acc_knn = accuracy_knn * 100

print(tr_acc_knn,test_acc_knn)


# In[66]:


pickle.dump(knn, open('../models/knn_sentiment.pkl', 'wb'))


# ##### XGB Classifier

# In[67]:


xgb = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5,
                      min_child_weight=1, objective= 'multi:softmax', num_class= 6,
                      gamma=0,  subsample=0.8, colsample_bytree=0.8, seed=27)


# In[68]:


eval_set = [(Xtrain_tfidf, ytrain),(Xtest_tfidf, ytest)]


# In[69]:


xgb.fit(Xtrain_tfidf, ytrain, eval_metric='auc', eval_set=eval_set, verbose=True)


# In[70]:


results = xgb.evals_result()
epochs = len(results['validation_0']['auc'])
x_axis = range(0, epochs)


# In[71]:


fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['auc'], label='Train')
ax.plot(x_axis, results['validation_1']['auc'], label='Test')
ax.legend()
plt.ylabel('AUC')
plt.title('XGBoost')
plt.show()


# In[77]:


xgb = XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth=5,
                      min_child_weight=1, objective= 'multi:softmax', num_class= 3,
                      gamma=0,  subsample=0.8, colsample_bytree=0.8, seed=27)


# In[78]:


get_ipython().run_cell_magic('time', '', 'xgb.fit(Xtrain_tfidf, ytrain)')


# In[79]:


ypred_xgb=xgb.predict(Xtest_tfidf)


# In[80]:


tr_acc_xgb = xgb.score(Xtrain_tfidf, ytrain)*100
test_acc_xgb =  accuracy_score(ytest,ypred_xgb) * 100
print(tr_acc_xgb,test_acc_xgb)


# In[81]:


pickle.dump(xgb, open('../models/xgb_basic_sentiment.pkl', 'wb'))


# Tuning max_depth and min_child_weight

# In[86]:


get_ipython().run_cell_magic('time', '', 'xgb_hp1=XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth=5,\n                      min_child_weight=1, objective= \'multi:softmax\', num_class= 3,\n                      gamma=0,  subsample=0.8, colsample_bytree=0.8, seed=27)\n\n\n\nparam_grid1={ \'max_depth\':range(3,10,2),\n              \'min_child_weight\':[1,2,3,4,5]}\n\ngrid_search1 = GridSearchCV(xgb_hp1, param_grid1, scoring="f1_macro", n_jobs=-1, cv=5)\n\n\ngrid_result1 = grid_search1.fit(Xtrain_tfidf, ytrain)')


# In[87]:


print("Best: %f using %s" % (grid_result1.best_score_, grid_result1.best_params_))


# In[89]:


xgb_hp2=XGBClassifier(learning_rate =0.1, n_estimators=100,
                      min_child_weight=1, objective= 'multi:softmax', num_class= 3,
                      gamma=0,  subsample=0.8, colsample_bytree=0.8, seed=27)

param_grid2={
            'max_depth':[8,9,10]
            

}

grid_search2 = GridSearchCV(xgb_hp2, param_grid2, scoring="f1_macro", n_jobs=-1, cv=5)


# In[90]:


get_ipython().run_cell_magic('time', '', 'grid_result2 = grid_search2.fit(Xtrain_tfidf, ytrain)')


# In[91]:


print("Best: %f using %s" % (grid_result2.best_score_, grid_result2.best_params_))
print(grid_result2.cv_results_['mean_test_score'])


# Tuning gamma

# In[92]:


xgb_hp3=XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth=10,
                      min_child_weight=1, objective= 'multi:softmax', num_class= 3,
                      subsample=0.8, colsample_bytree=0.8, seed=27)

param_grid3={
 'gamma':[i/10.0 for i in range(0,5)]
}
grid_search3 = GridSearchCV(xgb_hp3, param_grid3, scoring="f1_macro", n_jobs=-1, cv=5)


# In[93]:


get_ipython().run_cell_magic('time', '', 'grid_result3 = grid_search3.fit(Xtrain_tfidf, ytrain)')


# In[94]:


print("Best: %f using %s" % (grid_result3.best_score_, grid_result3.best_params_))
print(grid_result3.cv_results_['mean_test_score'])


# Tuning subsample and colsample_bytree

# In[95]:


xgb_hp4=XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth=10,
                      min_child_weight=1, objective= 'multi:softmax', num_class= 3,
                      gamma=0, seed=27)

param_grid4={
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
grid_search4 = GridSearchCV(xgb_hp4, param_grid4, scoring="f1_macro", n_jobs=-1, cv=5)


# In[96]:


get_ipython().run_cell_magic('time', '', 'grid_result4 = grid_search4.fit(Xtrain_tfidf, ytrain)')


# In[97]:


print("Best: %f using %s" % (grid_result4.best_score_, grid_result4.best_params_))
print(grid_result4.cv_results_['mean_test_score'])


# In[98]:


param_grid4_small={
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}
grid_search4_small = GridSearchCV(xgb_hp4, param_grid4_small, scoring="f1_macro", n_jobs=-1, cv=5)


# In[99]:


get_ipython().run_cell_magic('time', '', 'grid_result4_small = grid_search4_small.fit(Xtrain_tfidf, ytrain)')


# In[100]:


print("Best: %f using %s" % (grid_result4_small.best_score_, grid_result4_small.best_params_))
print(grid_result4_small.cv_results_['mean_test_score'])


# In[101]:


xgb_clf= XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth=10,
                      min_child_weight=1, objective= 'multi:softmax', num_class= 3,
                      gamma=0,  subsample=0.75, colsample_bytree=0.75, seed=27)


# In[102]:


xgb_clf.fit(Xtrain_tfidf, ytrain)


# In[103]:


ypred_xgb_clf=xgb_clf.predict(Xtest_tfidf)


# In[104]:


tr_acc_xgb_clf = xgb_clf.score(Xtrain_tfidf, ytrain)*100
test_acc_xgb_clf =  accuracy_score(ytest,ypred_xgb_clf) * 100
print(tr_acc_xgb_clf,test_acc_xgb_clf)


# In[105]:


pickle.dump(xgb_clf, open('../models/xgb_hp_sentiment.pkl', 'wb'))


# In[ ]:




