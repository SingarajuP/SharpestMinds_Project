#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle
import itertools
import os


# In[2]:


df = pd.read_pickle('../data/raw/emotions_training.pkl')


# In[3]:


df=df.reset_index()


# In[4]:


#Defining class for each emotion
df['labels'] = df['emotions'].factorize()[0]
df.head()


# In[5]:


uniquevalues = pd.unique(df[['emotions']].values.ravel())
df_unique=pd.DataFrame(uniquevalues,columns=['emotion'])


# In[6]:


df_unique


# In[7]:


df_unique.to_csv('../models/emotions.csv',index=False)


# In[1]:


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


# ### TF-IDF Vectorization for models

# In[13]:


tfidf_vectorizer = TfidfVectorizer()


# In[10]:


y =df['labels']


# In[14]:


#Train test split of the data
Xtrain, Xtest, ytrain, ytest = train_test_split(df['text'], y, test_size=0.3,random_state=1)
Xtrain_tfidf = tfidf_vectorizer.fit_transform(Xtrain)
Xtest_tfidf = tfidf_vectorizer.transform(Xtest)


# In[15]:


pickle.dump(tfidf_vectorizer, open('../models/tfidf_vect.pkl', 'wb'))


# ### Different models 

#   
#  The models I am exploring to find a suitable classification algorithm for the dataset are:
#  1. Logistic Regression
#  2. Random Forest Classifier
#  3. K- Nearest Neighbours
#  4. Support Vector machine
#  5. XGB
#  
# The models like KNN, Random forest and XGB are suitable for multiclass classification. Some models which are used for binary classification like logistic regression, SVM can be expanded to the multiclass classification by breaking down the multiclass into small sets of binary classifications. 
#   There are two common approaches to use them for multi-class classification: one-vs-rest and one-vs-one.
#  In one-vs-rest, each classifier (binary) is trained to determine whether or not an example is part of class  or not. To predict the class for a new example , we run all  classifiers on and choose the class with the highest score. One main drawback is that when there are lots of classes, each binary classifier sees a highly imbalanced dataset, which may degrade performance.
#  In one-vs-one, we train separate binary classification models, one for each possible pair of classes. To predict the class for a new example , we run all classifiers on  and choose the class with the most “votes.” A major drawback is that there can exist fairly large regions in the decision space with ties for the class with the most number of votes.
#  

# ##### Logistic Regression

# In[27]:


#Logistic Regression with One vs Rest
lr_ovr = LogisticRegression(multi_class='ovr', solver='liblinear')
lr_ovr.fit(Xtrain_tfidf, ytrain)


# In[28]:


ypred_lr_ovr=lr_ovr.predict(Xtest_tfidf)


# In[29]:


len(ypred_lr_ovr)


# In[26]:


tr_acc_lr_ovr = lr_ovr.score(Xtrain_tfidf, ytrain)*100
test_acc_lr_ovr =  accuracy_score(ytest,ypred_lr_ovr) * 100
print(tr_acc_lr_ovr,test_acc_lr_ovr)


# In[27]:


cm = confusion_matrix(ytest, ypred_lr_ovr)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True,fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# In[28]:


print(classification_report(ytest,ypred_lr_ovr, digits=3))


# In[31]:


pickle.dump(lr_ovr, open(working_directory+'/models/lr_ovr.pkl', 'wb'))


# In[16]:


#Logistic Regression with multinomial
lr_mn = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr_mn.fit(Xtrain_tfidf, ytrain)


# In[17]:


ypred_lr_mn=lr_mn.predict(Xtest_tfidf)


# In[19]:


tr_acc_lr_mn = lr_mn.score(Xtrain_tfidf, ytrain)*100
test_acc_lr_mn =  accuracy_score(ytest,ypred_lr_mn) * 100
print(tr_acc_lr_mn,test_acc_lr_mn)


# In[33]:


cm = confusion_matrix(ytest, ypred_lr_mn)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True,fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# In[20]:


pickle.dump(lr_mn, open('../models/lr_mn.pkl', 'wb'))


# ##### Random Forest

# In[151]:


rf = RandomForestClassifier()
rf.fit(Xtrain_tfidf, ytrain)


# In[152]:


rf.get_params()
#rf.n_classes_


# In[153]:


ypred_rf=rf.predict(Xtest_tfidf)


# In[154]:


tr_acc_rf = rf.score(Xtrain_tfidf, ytrain)*100
test_acc_rf =  accuracy_score(ytest,ypred_rf) * 100
print(tr_acc_rf,test_acc_rf)


# In[155]:


pickle.dump(rf, open('../models/rf_basic.pkl', 'wb'))


# In[170]:


# tuning max_depth in the tree

param1= { 
         'max_depth':[10,20,30,40,50,60,70,80,90,100],
        # 'min_samples_split':[2,4,8],
         'criterion' : ['entropy']
    
}

gridcv1 = GridSearchCV(rf, param1, cv = 2, verbose = 1, 
                      n_jobs = -1)


# In[171]:


get_ipython().run_cell_magic('time', '', '\ngrid_param1=gridcv1.fit(Xtrain_tfidf, ytrain)')


# In[178]:


res= grid_param1.cv_results_['mean_test_score']

#epochs = len(results['validation_0']['auc'])
#x_axis = range(0, epochs)


# In[179]:


res


# In[172]:


print(grid_param1.best_score_, grid_param1.best_params_)


# In[180]:


ypred_rf_param1=grid_param1.predict(Xtest_tfidf)


# In[181]:


tr_acc_rf_param1 = grid_param1.score(Xtrain_tfidf, ytrain)*100
test_acc_rf_param1 =  accuracy_score(ytest,ypred_rf_param1) * 100
print(tr_acc_rf_param1,test_acc_rf_param1)


# In[182]:


param2= { 
         'max_depth':[100,110,120],
         'min_samples_split':[2,4,8],
         'criterion' : ['entropy'],
         'min_samples_leaf' : [1, 3, 4],
         'n_estimators':[50,100,200,300]
    
}

gridcv2 = GridSearchCV(rf, param2, cv = 2, verbose = 1, scoring="f1_macro",
                      n_jobs = -1)


# In[183]:


get_ipython().run_cell_magic('time', '', '\ngrid_param2=gridcv2.fit(Xtrain_tfidf, ytrain)')


# From stackoverflow: Removing n_jobs=-1 in GridSearchCV solves the issue  of above warning

# In[185]:


print(grid_param2.best_score_, grid_param2.best_params_)


# In[186]:


ypred_rf_param2=grid_param2.predict(Xtest_tfidf)


# In[187]:


tr_acc_rf_param2 = grid_param2.score(Xtrain_tfidf, ytrain)*100
test_acc_rf_param2 =  accuracy_score(ytest,ypred_rf_param2) * 100
print(tr_acc_rf_param2,test_acc_rf_param2)


# In[189]:


pickle.dump(grid_param2, open('../models/rf_hp.pkl', 'wb'))


# ##### SVM Classifier

# In[91]:


get_ipython().run_cell_magic('time', '', "svm = SVC( kernel ='linear',C = 1, decision_function_shape='ovo')\nsvm.fit(Xtrain_tfidf, ytrain)\n#Run time approximately 4hrs")


# In[ ]:


pickle.dump(svm, open('../models/svm.pkl', 'wb'))


# In[15]:


ypred_svm=svm.predict(Xtest_tfidf)


# In[16]:


tr_acc_svm = svm.score(Xtrain_tfidf, ytrain)*100
test_acc_svm =  accuracy_score(ytest,ypred_svm) * 100
print(tr_acc_svm,test_acc_svm)


# In[32]:


cm = confusion_matrix(ytest, ypred_svm)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True,fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# In[33]:


print(classification_report(ytest,ypred_svm, digits=3))


# ##### KNN Classifier

# In[34]:


get_ipython().run_cell_magic('time', '', 'k_range = [25,50,100,150,200,300,400]\ntrain_scores = []\ntest_scores = []\nfor k in k_range:\n    neigh = KNeighborsClassifier(n_neighbors=k)\n    knn=neigh.fit(Xtrain_tfidf,ytrain)\n    tr_acc_knn = knn.score(Xtrain_tfidf, ytrain)*100\n    ypred_knn = knn.predict(Xtest_tfidf)\n    accuracy_knn = accuracy_score(ytest,ypred_knn)\n    test_acc_knn = accuracy_knn * 100\n    train_scores.append(tr_acc_knn)\n    test_scores.append(test_acc_knn)\n')


# In[22]:


dif=[train - test for train,test in zip(train_scores, test_score)]
print(dif)


# In[24]:


plt.plot(k, train_scores,color='r',label='Training Accuracy')
plt.plot(k, test_scores,color='b',label='Test accuracy')


plt.xlabel('Value of K for KNN')
plt.ylabel('Training and Testing Accuracy')
plt.legend()
plt.show()


# In[28]:


plt.savefig('knn_elbow.png')


# In[25]:


neigh = KNeighborsClassifier(n_neighbors=100)
knn=neigh.fit(Xtrain_tfidf,ytrain)
tr_acc_knn = knn.score(Xtrain_tfidf, ytrain)*100
ypred_knn = knn.predict(Xtest_tfidf)
accuracy_knn = accuracy_score(ytest,ypred_knn)
test_acc_knn = accuracy_knn * 100

print(tr_acc_knn,test_acc_knn)


# In[26]:


pickle.dump(knn, open('../models/knn.pkl', 'wb'))


# ##### XGB Classifier

# Hyper parameter tuning 

# Tuning n_estimators:

# In[35]:


xgb = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5,
                      min_child_weight=1, objective= 'multi:softmax', num_class= 6,
                      gamma=0,  subsample=0.8, colsample_bytree=0.8, seed=27)


# In[36]:


eval_set = [(Xtrain_tfidf, ytrain),(Xtest_tfidf, ytest)]


# In[37]:


xgb.fit(Xtrain_tfidf, ytrain, eval_metric='auc', eval_set=eval_set, verbose=True)


# In[38]:


results = xgb.evals_result()
epochs = len(results['validation_0']['auc'])
x_axis = range(0, epochs)


# In[39]:


fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['auc'], label='Train')
ax.plot(x_axis, results['validation_1']['auc'], label='Test')
ax.legend()
plt.ylabel('AUC')
plt.title('XGBoost')
plt.show()


# In[52]:


xgb = XGBClassifier(learning_rate =0.1, n_estimators=200, max_depth=5,
                      min_child_weight=1, objective= 'multi:softmax', num_class= 6,
                      gamma=0,  subsample=0.8, colsample_bytree=0.8, seed=27)


# In[53]:


get_ipython().run_cell_magic('time', '', 'xgb.fit(Xtrain_tfidf, ytrain)')


# In[54]:


ypred_xgb=xgb.predict(Xtest_tfidf)


# In[55]:


tr_acc_xgb = xgb.score(Xtrain_tfidf, ytrain)*100
test_acc_xgb =  accuracy_score(ytest,ypred_xgb) * 100
print(tr_acc_xgb,test_acc_xgb)


# In[73]:


print(classification_report(ytest,ypred_xgb, digits=3))


# In[74]:


pickle.dump(xgb, open('../models/xgb_basic.pkl', 'wb'))


# Tuning max_depth and min_child_weight

# In[76]:


xgb_hp1=XGBClassifier(learning_rate =0.1, n_estimators=200, max_depth=5,
                      min_child_weight=1, objective= 'multi:softmax', num_class= 6,
                      gamma=0,  subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1, seed=27)


# In[77]:


param_grid1={ 'max_depth':range(3,10,2),
              'min_child_weight':[1,2,3,4,5]}


# In[86]:


grid_search1 = GridSearchCV(xgb_hp1, param_grid1, scoring="f1_macro", n_jobs=-1, cv=2)


# In[87]:


get_ipython().run_cell_magic('time', '', 'grid_result1 = grid_search1.fit(Xtrain_tfidf, ytrain)')


# In[88]:


print("Best: %f using %s" % (grid_result1.best_score_, grid_result1.best_params_))


# In[89]:


grid_result1.cv_results_['mean_test_score']


# In[90]:


xgb_hp2=XGBClassifier(learning_rate =0.1, n_estimators=200,
                      min_child_weight=4, objective= 'multi:softmax', num_class= 6,
                      gamma=0,  subsample=0.8, colsample_bytree=0.8, seed=27)


# In[91]:


param_grid2={
            'max_depth':[6,7,8]
            

}

grid_search2 = GridSearchCV(xgb_hp2, param_grid2, scoring="f1_macro", n_jobs=-1, cv=2)


# In[92]:


get_ipython().run_cell_magic('time', '', 'grid_result2 = grid_search2.fit(Xtrain_tfidf, ytrain)')


# In[93]:


print("Best: %f using %s" % (grid_result2.best_score_, grid_result2.best_params_))
print(grid_result2.cv_results_['mean_test_score'])


# In[94]:


xgb_hp3=XGBClassifier(learning_rate =0.1, n_estimators=200, max_depth=7,
                      min_child_weight=4, objective= 'multi:softmax', num_class= 6,
                      subsample=0.8, colsample_bytree=0.8, seed=27)


# In[95]:


param_grid3={
 'gamma':[i/10.0 for i in range(0,5)]
}
grid_search3 = GridSearchCV(xgb_hp3, param_grid3, scoring="f1_macro", n_jobs=-1, cv=2)


# In[96]:


get_ipython().run_cell_magic('time', '', 'grid_result3 = grid_search3.fit(Xtrain_tfidf, ytrain)')


# In[97]:


print("Best: %f using %s" % (grid_result3.best_score_, grid_result3.best_params_))
print(grid_result3.cv_results_['mean_test_score'])


# Tuning subsample and colsample_bytree

# In[98]:


xgb_hp4=XGBClassifier(learning_rate =0.1, n_estimators=200, max_depth=7,
                      min_child_weight=4, objective= 'multi:softmax', num_class= 6,
                      gamma=0.2, seed=27)


# In[99]:


param_grid4={
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
grid_search4 = GridSearchCV(xgb_hp4, param_grid4, scoring="f1_macro", n_jobs=-1, cv=2)


# In[101]:


get_ipython().run_cell_magic('time', '', 'grid_result4 = grid_search4.fit(Xtrain_tfidf, ytrain)')


# In[102]:


print("Best: %f using %s" % (grid_result4.best_score_, grid_result4.best_params_))
print(grid_result4.cv_results_['mean_test_score'])


# In[105]:


param_grid4_small={
 'subsample':[i/100.0 for i in range(85,100,5)],
 'colsample_bytree':[i/100.0 for i in range(85,100,5)]
}
grid_search4_small = GridSearchCV(xgb_hp4, param_grid4_small, scoring="f1_macro", n_jobs=-1, cv=2)


# In[106]:


get_ipython().run_cell_magic('time', '', 'grid_result4_small = grid_search4_small.fit(Xtrain_tfidf, ytrain)')


# In[107]:


print("Best: %f using %s" % (grid_result4_small.best_score_, grid_result4_small.best_params_))
print(grid_result4_small.cv_results_['mean_test_score'])


# Applying the tuned hyper parameters to the model

# In[108]:


xgb_clf= XGBClassifier(learning_rate =0.1, n_estimators=200, max_depth=7,
                      min_child_weight=4, objective= 'multi:softmax', num_class= 6,
                      gamma=0.2,  subsample=0.95, colsample_bytree=0.9, seed=27)


# In[109]:


xgb_clf.fit(Xtrain_tfidf, ytrain)


# In[117]:


ypred_xgb_clf=xgb_clf.predict(Xtest_tfidf)


# In[118]:


tr_acc_xgb_clf = xgb_clf.score(Xtrain_tfidf, ytrain)*100
test_acc_xgb_clf =  accuracy_score(ytest,ypred_xgb_clf) * 100
print(tr_acc_xgb_clf,test_acc_xgb_clf)


# In[188]:


pickle.dump(xgb_clf, open('../models/xgb_hp.pkl', 'wb'))


# Seems like it is overfitting compared to initial results. Regularizing the hyper parameters.

# In[112]:


xgb_hp5= XGBClassifier(learning_rate =0.1, n_estimators=200, max_depth=7,
                      min_child_weight=4, objective= 'multi:softmax', num_class= 6,
                      gamma=0.2,  subsample=0.95, colsample_bytree=0.9, seed=27)

param_grid5={
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
grid_search5 = GridSearchCV(xgb_hp5, param_grid5, scoring="f1_macro", n_jobs=-1, cv=2)


# In[113]:


get_ipython().run_cell_magic('time', '', 'grid_result5 = grid_search5.fit(Xtrain_tfidf, ytrain)')


# In[114]:


print("Best: %f using %s" % (grid_result5.best_score_, grid_result5.best_params_))
print(grid_result5.cv_results_['mean_test_score'])


# In[115]:


xgb_clf1= XGBClassifier(learning_rate =0.1, n_estimators=200, max_depth=7,
                      min_child_weight=4, objective= 'multi:softmax', num_class= 6,
                      gamma=0.2,  subsample=0.95, colsample_bytree=0.9,reg_alpha= 1e-05, seed=27)


# In[116]:


xgb_clf1.fit(Xtrain_tfidf, ytrain)


# In[119]:


ypred_xgb_clf1=xgb_clf1.predict(Xtest_tfidf)


# In[120]:


tr_acc_xgb_clf1 = xgb_clf1.score(Xtrain_tfidf, ytrain)*100
test_acc_xgb_clf1 =  accuracy_score(ytest,ypred_xgb_clf1) * 100
print(tr_acc_xgb_clf1,test_acc_xgb_clf1)


# In[ ]:





# Reducing learning rate and adding more trees

# In[121]:


xgb_lr=XGBClassifier(learning_rate =0.01, n_estimators=2000, max_depth=7,
                      min_child_weight=4, objective= 'multi:softmax', num_class= 6,
                      gamma=0.2,  subsample=0.95, colsample_bytree=0.9, seed=27)


# In[122]:


eval_set = [(Xtrain_tfidf, ytrain),(Xtest_tfidf, ytest)]


# In[123]:


xgb_lr.fit(Xtrain_tfidf, ytrain, eval_metric='auc', eval_set=eval_set,early_stopping_rounds=50, verbose=True)


# In[124]:


results = xgb_lr.evals_result()
epochs = len(results['validation_0']['auc'])
x_axis = range(0, epochs)


# In[125]:


fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['auc'], label='Train')
ax.plot(x_axis, results['validation_1']['auc'], label='Test')
ax.legend()
plt.ylabel('AUC')
plt.title('XGBoost')
plt.show()


# In[143]:


xgb_lr.fit(Xtrain_tfidf, ytrain, eval_metric='merror', eval_set=eval_set,early_stopping_rounds=50, verbose=True)


# In[145]:


results = xgb_lr.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)


# In[146]:


fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Test')
ax.legend()
plt.ylabel('merror')
plt.title('XGBoost')
plt.show()


# In[ ]:





# In[60]:


xgb_clf2= XGBClassifier(learning_rate =0.01, n_estimators=750, max_depth=7,
                      min_child_weight=4, objective= 'multi:softmax', num_class= 6,
                      gamma=0.2,  subsample=0.95, colsample_bytree=0.9, seed=27)


# In[61]:


xgb_clf2.fit(Xtrain_tfidf, ytrain)


# In[62]:


ypred_xgb_clf2=xgb_clf2.predict(Xtest_tfidf)


# In[63]:


tr_acc_xgb_clf2 = xgb_clf2.score(Xtrain_tfidf, ytrain)*100
test_acc_xgb_clf2 =  accuracy_score(ytest,ypred_xgb_clf2) * 100
print(tr_acc_xgb_clf2,test_acc_xgb_clf2)


# In[ ]:




