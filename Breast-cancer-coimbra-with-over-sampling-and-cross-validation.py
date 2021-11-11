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


df = pd.read_csv(r'C:\Users\SWARNA\Downloads\dataR2.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.isnull().sum()


# In[6]:


df.shape


# In[7]:


#create independent & dependent variable vectors
x=df
y=df['Classification']
print(x)
print(y)


# In[8]:


#drop target column from indpendent variable
x.drop('Classification',axis=1,inplace=True)
x.columns


# In[9]:


#Count 0 and 1 in dependent or target column
print(sorted(Counter(y).items()))


# In[10]:


#for oversampling
from imblearn import over_sampling


# In[11]:


#Oversampling
from imblearn.over_sampling import RandomOverSampler


# In[12]:


#oversampling
r = RandomOverSampler(random_state = 0)
X,Y= r.fit_resample(x,y)
print(sorted(Counter(Y).items()),Y.shape)


# In[13]:


print(X.shape)
print(Y.shape)


# In[14]:


#missing value finding
X.isnull().sum()


# In[15]:


#missing value finding
Y.isnull().sum()


# In[17]:


#unarative
from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2

#FIT_FEATURES = SelectKBest(score_func=f_classif)

#FIT_FEATURES.fit(x,y)

#bestfeatures = SelectKBest(score_func=chi2, k=9)
#fit = bestfeatures.fit(x,y)
#dfscores = pd.DataFrame(fit.scores_)
#dfcolumns = pd.DataFrame(x.columns)
#concat two dataframes for better visualization 
#featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#featureScores.columns = ['Specs','Score']  #naming the dataframe columns
#print(featureScores.nlargest(10,'Score'))  #print 10 best features


# Feature extraction
test = SelectKBest(score_func=chi2, k=6)
fit = test.fit(X,Y)

# Summarize scores
#np.set_printoptions(precision=3)
#print(fit.scores_)

#features = fit.transform(x)
# Summarize selected features
#print(features[0:8,:])


# In[18]:


pd.DataFrame(fit.scores_)
SCORE_COL = pd.DataFrame(fit.scores_,columns =['Score'])
SCORE_COL


# In[19]:


SCORE_COL.nlargest(9,'Score')


# In[20]:


X = X.drop(["Age","BMI","Leptin","Adiponectin"] ,axis='columns')


# In[21]:


X


# In[22]:


print (X.shape)
print (Y.shape)


# In[23]:


#Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[24]:


X


# In[25]:


X.shape


# In[26]:


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


# In[27]:


#simplify the accuracy test 
def get_score(model,X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train)
    return model.score(X_test, Y_test)


# In[28]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=7,random_state=None,shuffle=False)


# In[29]:


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
    


# In[30]:


#Accuracy measure
score_logistic


# In[31]:


score_svm


# In[32]:


score_randomforest


# In[33]:


score_decisiontree


# In[34]:


score_knn


# In[35]:


score_Naivebayes


# In[36]:


import statistics


# In[37]:


Mean_logistic = statistics.mean(score_logistic)
Mean_svm = statistics.mean(score_svm)
Mean_randomforest = statistics.mean(score_randomforest)
Mean_decisiontree = statistics.mean(score_decisiontree)
Mean_knn = statistics.mean(score_knn)
Mean_Naivebayes = statistics.mean(score_Naivebayes)


# In[38]:


print(Mean_logistic)


# In[39]:


print(Mean_svm)


# In[40]:


print(Mean_randomforest)


# In[41]:


print(Mean_decisiontree)


# In[42]:


print(Mean_knn)


# In[43]:


print(Mean_Naivebayes)


# In[ ]:




