#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import Counter
import sklearn.preprocessing as preprocessing


# In[7]:


pip install imblearn


# In[8]:


from imblearn.over_sampling import SMOTE
from imblearn import over_sampling


# In[9]:


df = pd.read_csv(r'C:\Users\itp\Downloads\heart fail.csv')


# In[10]:


df.head()


# In[11]:


df.DEATH_EVENT.value_counts()


# In[12]:


#create independent & dependent variable vectors
x=df
y=df['DEATH_EVENT']
print(x)
print(y)


# In[13]:


x.drop('DEATH_EVENT',axis=1,inplace=True)


# In[14]:


x.columns


# In[15]:


df.shape


# In[16]:


#Count 0 and 1 in dependent or target column
print(sorted(Counter(y).items()))


# In[17]:


#Oversampling
from imblearn.over_sampling import RandomOverSampler


# In[18]:


#oversampling
r = RandomOverSampler(random_state = 0)
X,Y= r.fit_resample(x,y)
print(sorted(Counter(Y).items()),Y.shape)


# In[19]:


print(X.shape)
print(Y.shape)


# In[20]:


#missing value finding
X.isnull().sum()


# In[21]:


Y.isnull().sum()


# In[22]:



#Feature selection
#unarative
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# In[23]:


FIT_FEATURES = SelectKBest(score_func=f_classif)


# In[24]:


FIT_FEATURES.fit(X,Y)


# In[25]:


pd.DataFrame(FIT_FEATURES.scores_)
SCORE_COL = pd.DataFrame(FIT_FEATURES.scores_,columns =['Score'])
SCORE_COL


# In[26]:


SCORE_COL.nlargest(8,'Score')


# In[27]:



df = df.drop(['diabetes','smoking','time'],axis=1)


# In[28]:


df.head()


# In[29]:


#Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[30]:


X.shape


# In[31]:



# Importing modules
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn import svm


# In[32]:


#simplify the accuracy test 
def get_score(model,X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train)
    return model.score(X_test, Y_test)


# In[33]:



#cross valodation with stratifiedKFold cross validation
from sklearn.model_selection import KFold
skf= KFold(n_splits=7,random_state=None,shuffle=False)


# In[34]:



score_logistic = []
score_svm = []
score_randomforest = []
score_knn = []
score_decisiontree = []
score_extratree = []
score_Naivebayes = []

for train_index, test_index in skf.split(X,Y):
    print("Train:", train_index, "Test:",test_index)
    X_train,X_test = X[train_index], X[test_index]
    Y_train,Y_test = Y[train_index], Y[test_index]
    score_logistic.append(get_score(LogisticRegression(), X_train, X_test, Y_train, Y_test)*100)
    score_svm.append(get_score(SVC(),X_train, X_test, Y_train, Y_test)*100)
    score_randomforest.append(get_score(RandomForestClassifier(),X_train, X_test, Y_train, Y_test)*100)
    score_knn.append(get_score(KNeighborsClassifier(),X_train, X_test, Y_train, Y_test)*100)
    score_decisiontree.append(get_score(DecisionTreeClassifier(),X_train, X_test, Y_train, Y_test)*100)
    score_extratree.append(get_score(ExtraTreeClassifier(),X_train, X_test, Y_train, Y_test)*100)
    score_Naivebayes.append(get_score(GaussianNB(),X_train, X_test, Y_train, Y_test)*100)
    


# In[35]:


#Accuracy measure
score_logistic


# In[37]:


import statistics
Mean_logistic = statistics.mean(score_logistic)
print(Mean_logistic)


# In[38]:


score_svm


# In[40]:


Mean_svm = statistics.mean(score_svm)
print(Mean_svm)


# In[41]:


score_randomforest


# In[43]:


Mean_randomforest = statistics.mean(score_randomforest)
print(Mean_randomforest)


# In[44]:


score_decisiontree


# In[48]:


Mean_decisiontree = statistics.mean(score_decisiontree)
print(Mean_decisiontree)


# In[49]:


score_knn


# In[50]:


Mean_knn = statistics.mean(score_knn)
print(Mean_knn)


# In[46]:


score_extratree


# In[51]:


Mean_extratree = statistics.mean(score_extratree)
print(Mean_extratree)


# In[47]:


score_Naivebayes


# In[52]:


Mean_Naivebayes = statistics.mean(score_Naivebayes)


# In[53]:


print(Mean_Naivebayes)


# In[ ]:




