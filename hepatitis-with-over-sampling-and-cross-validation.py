#!/usr/bin/env python
# coding: utf-8

# In[53]:


#libraries
import sklearn.preprocessing as preprocessing
import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import Counter


# In[54]:


#for oversampling
from imblearn import over_sampling
from imblearn.over_sampling import SMOTE


# In[55]:


df = pd.read_csv(r'C:\Users\SWARNA\Downloads\hepatitis_csv.csv')


# In[56]:


df.head()


# In[57]:


df.tail()


# In[58]:


df.isnull().sum()


# In[59]:


df.shape


# In[60]:


#Filling null values with mean for integer and float values

df2 = df.fillna(df.mean()["bilirubin":"protime"])


# In[61]:


df2


# In[62]:


df2.isnull().sum()


# In[63]:


#Filling Null values with previous values for categorical values

df3 = df2.fillna(method = 'pad')


# In[64]:


df3


# In[65]:


df3.isnull().sum()


# In[66]:


type(df3)


# In[67]:


from sklearn.preprocessing import LabelEncoder


# In[68]:


labelencoder = LabelEncoder()


# In[69]:


df3['g'] = labelencoder.fit_transform(df3['sex'])
df3.head()

df3['st'] = labelencoder.fit_transform(df3['steroid'])
df3.head()

df3['an'] = labelencoder.fit_transform(df3['antivirals'])
df3.head()

df3['f'] = labelencoder.fit_transform(df3['fatigue'])
df3.head()

df3['m'] = labelencoder.fit_transform(df3['malaise'])
df3.head()

df3['aa'] = labelencoder.fit_transform(df3['anorexia'])
df3.head()

df3['lb'] = labelencoder.fit_transform(df3['liver_big'])
df3.head()

df3['lf'] = labelencoder.fit_transform(df3['liver_firm'])
df3.head()

df3['sp'] = labelencoder.fit_transform(df3['spleen_palpable'])
df3.head()

df3['s'] = labelencoder.fit_transform(df3['spiders'])
df3.head()

df3['as'] = labelencoder.fit_transform(df3['ascites'])
df3.head()

df3['v'] = labelencoder.fit_transform(df3['varices'])
df3.head()

df3['h'] = labelencoder.fit_transform(df3['histology'])
df3.head()

df3['c'] = labelencoder.fit_transform(df3['class'])
df3.head()


# In[70]:


df3 = df3.drop(["sex","steroid","antivirals","fatigue","malaise","anorexia","liver_big","liver_firm","spleen_palpable","spiders","ascites","varices","histology","class"] ,axis='columns')


# In[71]:


df3


# In[72]:


df3 = df3.rename(columns = {"g": "sex","st": "steroid","an": "antivirals","f": "fatigue","m": "malaise","aa": "anorexia","lb": "liver_big","lf": "liver_firm","sp": "spleen_palpable","s": "spiders","as": "ascites","v": "varices","h": "histology","c": "class"})


# In[73]:


df3.head()


# In[74]:


#create independent & dependent variable vectors
x=df3
y=df3['class']
print(x)
print(y)


# In[75]:


#drop target column from indpendent variable
x.drop('class',axis=1,inplace=True)
x.columns


# In[76]:


#Count 0 and 1 in dependent or target column
print(sorted(Counter(y).items()))


# In[77]:


#Oversampling
from imblearn.over_sampling import RandomOverSampler


# In[78]:


#oversampling
r = RandomOverSampler(random_state = 0)
X,Y= r.fit_resample(x,y)
print(sorted(Counter(Y).items()),Y.shape)


# In[79]:


print(X.shape)
print(Y.shape)


# In[80]:


print(X)
print(Y)


# In[81]:


#unarative
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

FIT_FEATURES = SelectKBest(score_func=f_classif)
FIT_FEATURES.fit(X,Y)


# In[82]:


pd.DataFrame(FIT_FEATURES.scores_)
SCORE_COL = pd.DataFrame(FIT_FEATURES.scores_,columns =['Score'])
SCORE_COL


# In[83]:


SCORE_COL.nlargest(19,'Score')


# In[84]:


X


# In[85]:


#drop the columns which score below 100
X = X.drop(["age","alk_phosphate","sgot","sex","steroid","antivirals","anorexia","liver_big","liver_firm"] ,axis='columns')
X


# In[86]:


print (X.shape)
print (Y.shape)


# In[87]:


#Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[88]:


X


# In[89]:


# Importing modules
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn import svm


# In[90]:


#simplify the accuracy test 
def get_score(model,X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train)
    return model.score(X_test, Y_test)


# In[91]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=7,random_state=None,shuffle=False)


# In[92]:


score_logistic = []
score_svm = []
score_randomforest = []
score_knn = []
score_decisiontree = []
score_Naivebayes = []
for train_index,test_index in kf.split(X,Y):
    print("Train:", train_index,"Test:",test_index)
    X_train,X_test = X[train_index],X[test_index]
    Y_train,Y_test = Y[train_index], Y[test_index]
    score_logistic.append(get_score(LogisticRegression(), X_train, X_test, Y_train, Y_test)*100)
    score_svm.append(get_score(SVC(),X_train, X_test, Y_train, Y_test)*100)
    score_randomforest.append(get_score(RandomForestClassifier(),X_train, X_test, Y_train, Y_test)*100)
    score_knn.append(get_score(KNeighborsClassifier(),X_train, X_test, Y_train, Y_test)*100)
    score_decisiontree.append(get_score(DecisionTreeClassifier(),X_train, X_test, Y_train, Y_test)*100)
    score_Naivebayes.append(get_score(GaussianNB(),X_train, X_test, Y_train, Y_test)*100)


# In[93]:


#Accuracy measure
score_logistic


# In[94]:


score_svm


# In[95]:


score_randomforest


# In[96]:


score_decisiontree


# In[97]:


score_knn


# In[98]:


score_Naivebayes


# In[99]:


import statistics


# In[100]:


Mean_logistic = statistics.mean(score_logistic)
Mean_svm = statistics.mean(score_svm)
Mean_randomforest = statistics.mean(score_randomforest)
Mean_decisiontree = statistics.mean(score_decisiontree)
Mean_knn = statistics.mean(score_knn)
Mean_Naivebayes = statistics.mean(score_Naivebayes)


# In[101]:


print(Mean_logistic)


# In[102]:


print(Mean_svm)


# In[103]:


print(Mean_randomforest)


# In[104]:


print(Mean_decisiontree)


# In[105]:


print(Mean_knn)


# In[106]:


print(Mean_Naivebayes)


# In[ ]:




