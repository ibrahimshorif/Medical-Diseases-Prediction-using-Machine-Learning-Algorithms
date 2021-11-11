#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd


# In[26]:


df = pd.read_csv("E:\level 4 term 1\CSE 441\patient.csv")


# In[27]:


df.head()


# In[28]:


#Lets Encode the nominal features

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Gender'].unique()


# In[29]:


df.head()


# In[30]:


#count missing values
print(df.isnull().sum())


# In[31]:


#drop missing value record
df.dropna(inplace=True)


# In[32]:


df.head()


# In[33]:


df["Albumin_and_Globulin_Ratio"] = df.Albumin_and_Globulin_Ratio.fillna(df['Albumin_and_Globulin_Ratio'].mean())


# In[34]:


df.head()


# In[35]:


#count missing values
print(df.isnull().sum())


# In[39]:


#dependent and indepandent variable
#independent 
x= df.iloc[:,:-1].values
#dependent 
y= df.iloc[:,-1].values


# In[40]:


print(x)
print(y)


# In[41]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[42]:


x_test.shape


# In[43]:


y_test.shape


# In[44]:


x_train


# In[45]:


x_test


# In[46]:


y_train 


# In[47]:


y_test


# # Algorithm

# # LogisticRegression

# In[48]:


#LogisticRegression
from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(x_train,y_train)


# In[49]:


model.predict(x_test)


# In[50]:


model.score(x_test,y_test)*100


# In[51]:


pred_log = model.predict(x_test)


# In[52]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[53]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# # DecisionTree

# In[56]:


from sklearn.tree import DecisionTreeClassifier 
model= DecisionTreeClassifier()
model.fit(x_train,y_train)


# In[57]:


model.predict(x_test)


# In[58]:


model.score(x_test,y_test)*100


# In[59]:


pred_log = model.predict(x_test)


# In[60]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[61]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# # KNN

# In[62]:


from sklearn.neighbors import KNeighborsClassifier
model= KNeighborsClassifier()
model.fit(x_train,y_train)


# In[63]:


model.predict(x_test)


# In[64]:


model.score(x_test,y_test)*100


# In[65]:


pred_log = model.predict(x_test)


# In[66]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[67]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# # RandomForest

# In[68]:


from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier()
model.fit(x_train,y_train)


# In[69]:


model.predict(x_test)


# In[70]:


model.score(x_test,y_test)*100


# In[71]:


pred_log = model.predict(x_test)


# In[72]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[73]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# # NB

# In[74]:


from sklearn.naive_bayes import GaussianNB
model= GaussianNB()
model.fit(x_train,y_train)


# In[75]:


model.predict(x_test)


# In[76]:


model.score(x_test,y_test)*100


# In[77]:


pred_log = model.predict(x_test)


# In[78]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[79]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# # SVM

# In[80]:


from sklearn.svm import SVC
model= SVC()
model.fit(x_train,y_train)


# In[81]:


model.predict(x_test)


# In[82]:


model.score(x_test,y_test)*100


# In[83]:


pred_log = model.predict(x_test)


# In[84]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[85]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# In[ ]:




