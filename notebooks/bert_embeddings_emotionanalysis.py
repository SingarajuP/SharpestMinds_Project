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


df = pd.read_pickle('../data/raw/emotions_training.pkl')
df=df.reset_index()


# In[3]:


df.head()


# In[5]:


df.shape


# In[6]:


df['labels'] = df['emotions'].factorize()[0]
df.head()


# In[7]:


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


# In[8]:


Xtrain, Xtest, ytrain, ytest = train_test_split(df['text'], df['labels'], test_size=0.3,random_state=1)


# In[9]:


model = SentenceTransformer('all-MiniLM-L6-v2')


# In[10]:


def sentence(text):
    return model.encode(text)
    


# In[11]:


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


# In[12]:


train_embed,test_embed=bertembeddings(Xtrain,Xtest)


# In[13]:


print(train_embed[0])


# In[ ]:





# In[ ]:





# In[ ]:





# ##### Logistic Regression model

# In[14]:


#Logistic Regression with multinomial
lr_mn = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr_mn.fit(train_embed, ytrain)


# In[15]:


ypred_lr_mn=lr_mn.predict(test_embed)


# In[16]:


tr_acc_lr_mn = lr_mn.score(train_embed, ytrain)*100
test_acc_lr_mn =  accuracy_score(ytest,ypred_lr_mn) * 100
print(tr_acc_lr_mn,test_acc_lr_mn)


# In[17]:


cm = confusion_matrix(ytest, ypred_lr_mn)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True,fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


# In[42]:


pickle.dump(lr_mn, open('../models/lr_mn_emotion_bert.pkl', 'wb'))


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




