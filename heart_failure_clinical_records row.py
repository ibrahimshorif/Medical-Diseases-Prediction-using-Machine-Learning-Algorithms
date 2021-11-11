#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("E:\level 4 term 1\CSE 441\heart_failure_clinical_records.csv")


# In[3]:


df.head()


# In[4]:


#count missing values
print(df.isnull().sum())


# In[5]:


#dependent and indepandent variable
#independent 
x= df.iloc[:,:-1].values
#dependent 
y= df.iloc[:,-1].values


# In[6]:


print(x)
print(y)


# In[7]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[8]:


x_test.shape


# In[9]:


y_test


# In[10]:


x_train.shape


# In[11]:


y_train.shape


# In[12]:


#LogisticRegression
from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(x_train,y_train)


# In[13]:


model.predict(x_test)


# In[14]:


model.score(x_test,y_test)*100


# In[15]:


pred_log = model.predict(x_test)


# In[16]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[17]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# In[18]:


#decision_tree
from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[19]:


from sklearn.tree import DecisionTreeClassifier 
model01= DecisionTreeClassifier()
model01.fit(x_train,y_train)


# In[20]:


model01.predict(x_test)


# In[21]:


model01.score(x_test,y_test)*100


# In[22]:


pred_log = model01.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[23]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# In[24]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[25]:


from sklearn.neighbors import KNeighborsClassifier
model02= KNeighborsClassifier()
model02.fit(x_train,y_train)


# In[26]:


model02.predict(x_test)


# In[27]:


model02.score(x_test,y_test)*100


# In[28]:


pred_log = model02.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[29]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# In[ ]:




