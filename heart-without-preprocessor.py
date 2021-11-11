#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np


# In[33]:


hdf = pd.read_csv(r'C:\Users\SWARNA\Desktop\HeartAcc.csv')


# In[34]:


hdf.head()


# In[35]:


#create independent & dependent variable vectors
x=hdf.iloc[:,:-1].values
y=hdf.iloc[:,-1].values
print(x)
print(y)


# In[36]:


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


# In[37]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=10)
print (x_train.shape)
print (y_train.shape)
print (x_test.shape)
print (y_test.shape)


# In[39]:


# logistic regression 

logreg = LogisticRegression()
# Train the model using the training sets and check score
logreg.fit(x_train, y_train)
#Predict Output
log_predicted= logreg.predict(x_test)

logreg_score = round(logreg.score(x_train, y_train) * 100, 2)
logreg_score_test = round(logreg.score(x_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Logistic Regression Training Score: \n', logreg_score)
print('Logistic Regression Test Score: \n', logreg_score_test)
print('Coefficient: \n', logreg.coef_)
print('Accuracy: \n', accuracy_score(y_test,log_predicted)*100, )
print('Confusion Matrix: \n', confusion_matrix(y_test,log_predicted))
print('Classification Report: \n', classification_report(y_test,log_predicted))


# In[40]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
#Predict Output
gauss_predicted = gaussian.predict(x_test)

gauss_score = round(gaussian.score(x_train, y_train) * 100, 2)
gauss_score_test = round(gaussian.score(x_test, y_test) * 100, 2)
print('Gaussian Score: \n', gauss_score)
print('Gaussian Test Score: \n', gauss_score_test)
print('Accuracy: \n', accuracy_score(y_test, gauss_predicted)*100)
print('Confusion Matrix: \n',confusion_matrix(y_test,gauss_predicted))
print('Classification Report: \n',classification_report(y_test,gauss_predicted))


# In[41]:


# KNN

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
#Predict Output
knn_predicted = knn.predict(x_test)

knn_score = round(knn.score(x_train, y_train) * 100, 2)
knn_score_test = round(knn.score(x_test, y_test) * 100, 2)
print('KNN Score: \n', knn_score)
print('KNN Test Score: \n', knn_score_test)
print('Accuracy: \n', accuracy_score(y_test, knn_predicted)*100)
print('Confusion Matrix: \n',confusion_matrix(y_test,knn_predicted))
print('Classification Report: \n',classification_report(y_test,knn_predicted))


# In[42]:


from sklearn import svm


# In[43]:


# Support Vector Machine

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
clf_score = round(clf.score(x_train, y_train) * 100, 2)
clf_score_test = round(clf.score(x_test, y_test) * 100, 2)
print('Support Vector Machine Score: \n', clf_score)
print('Support Vector Machine Test Score: \n', clf_score_test)
print('Accuracy:\n',metrics.accuracy_score(y_test, y_pred)*100)
print('Confusion Matrix: \n',confusion_matrix(y_test,y_pred))
print('Classification Report: \n',classification_report(y_test,y_pred))


# In[44]:


# Decision Tree Classifier

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
#Predict Output
dt_predicted = dt.predict(x_test)

dt_score = round(dt.score(x_train, y_train) * 100, 2)
dt_score_test = round(dt.score(x_test, y_test) * 100, 2)
print('Decision Tree Classifier Score: \n', dt_score)
print('Decision Tree Classifier Test Score: \n', dt_score_test)
print('Accuracy: \n', accuracy_score(y_test, dt_predicted)*100)
print('Confusion Matrix: \n',confusion_matrix(y_test,dt_predicted))
print('Classification Report: \n',classification_report(y_test,dt_predicted))


# In[45]:


from sklearn.tree import ExtraTreeClassifier


# In[46]:


# Extra Tree Classifier

et = ExtraTreeClassifier()
et.fit(x_train, y_train)
#Predict Output
et_predicted = et.predict(x_test)

et_score = round(et.score(x_train, y_train) * 100, 2)
et_score_test = round(et.score(x_test, y_test) * 100, 2)
print('Extra Tree Classifier Score: \n', et_score)
print('Extra Tree Classifier Test Score: \n', et_score_test)
print('Accuracy: \n', accuracy_score(y_test, et_predicted)*100)
print('Confusion Matrix: \n',confusion_matrix(y_test,et_predicted))
print('Classification Report: \n',classification_report(y_test,et_predicted))


# In[47]:



# Random Forest Classifier

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
#Predict Output
rf_predicted = rf.predict(x_test)

rf_score = round(rf.score(x_train, y_train) * 100, 2)
rf_score_test = round(rf.score(x_test, y_test) * 100, 2)
print('Random Forest Classifier Score: \n', rf_score)
print('Random Forest Classifier Test Score: \n', rf_score_test)
print('Accuracy: \n', accuracy_score(y_test, rf_predicted)*100)
print('Confusion Matrix: \n',confusion_matrix(y_test,rf_predicted))
print('Classification Report: \n',classification_report(y_test,rf_predicted))


# In[ ]:





# In[ ]:




