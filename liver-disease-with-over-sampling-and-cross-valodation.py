#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries
import sklearn.preprocessing as preprocessing
import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import Counter


# In[2]:


df = pd.read_csv(r'C:\Users\SWARNA\Desktop\patient.csv')


# In[3]:


df.head()


# In[4]:


#Encoding with replace
df["Gender"].replace({"Female": "1", "Male": "0"}, inplace=True)
print(df)


# In[5]:


#missing value finding
df.isnull().sum()


# In[6]:


#Filling null values with mean for integer and float values

df = df.fillna(df.mean()["Albumin_and_Globulin_Ratio"])
df


# In[7]:


#missing value finding
df.isnull().sum()


# In[8]:


#create independent & dependent variable vectors
x=df
y=df['Dataset']
print(x)
print(y)


# In[9]:


#drop target column from indpendent variable
x.drop('Dataset',axis=1,inplace=True)
x.columns


# In[10]:


df.shape


# In[11]:


#Count 0 and 1 in dependent or target column
print(sorted(Counter(y).items()))


# In[13]:


#for oversampling
from imblearn import over_sampling


# In[14]:


#Oversampling
from imblearn.over_sampling import RandomOverSampler


# In[15]:


#oversampling
r = RandomOverSampler(random_state = 0)
X,Y= r.fit_resample(x,y)
print(sorted(Counter(Y).items()),Y.shape)


# In[16]:


print(X.shape)
print(Y.shape)


# In[17]:


#Feature selection
#unarative
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

FIT_FEATURES = SelectKBest(score_func=f_classif)

FIT_FEATURES.fit(X,Y)


# In[18]:


pd.DataFrame(FIT_FEATURES.scores_)
SCORE_COL = pd.DataFrame(FIT_FEATURES.scores_,columns =['Score'])
SCORE_COL


# In[19]:


SCORE_COL.nlargest(10,'Score')


# In[20]:


#drop the columns which score below 100
X = X.drop(['Gender','Total_Protiens'],axis=1)
X


# In[21]:


print (X.shape)
print (Y.shape)


# In[22]:


#Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
X


# In[23]:


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


# In[24]:


#simplify the accuracy test 
def get_score(model,X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train)
    return model.score(X_test, Y_test)


# In[25]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=7,random_state=None,shuffle=False)


# In[26]:


score_logistic = []
score_svm = []
score_randomforest = []
score_knn = []
score_decisiontree = []
score_extratree = []
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


# In[27]:


#Accuracy measure
score_logistic


# In[28]:


score_svm


# In[29]:


score_randomforest


# In[30]:


score_decisiontree


# In[31]:


score_knn


# In[32]:


score_Naivebayes


# In[33]:


import statistics


# In[34]:


Mean_logistic = statistics.mean(score_logistic)
Mean_svm = statistics.mean(score_svm)
Mean_randomforest = statistics.mean(score_randomforest)
Mean_decisiontree = statistics.mean(score_decisiontree)
Mean_knn = statistics.mean(score_knn)
Mean_Naivebayes = statistics.mean(score_Naivebayes)


# In[35]:


print(Mean_logistic)
print(Mean_svm)
print(Mean_randomforest)
print(Mean_decisiontree)
print(Mean_knn)
print(Mean_Naivebayes)


# In[ ]:




