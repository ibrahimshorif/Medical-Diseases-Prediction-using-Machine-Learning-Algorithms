#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("E:\level 4 term 1\CSE 441\diabetes.csv")


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


print(x.shape)
print(y.shape)


# In[7]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[8]:


x_test


# In[10]:


y_test


# In[11]:


#LogisticRegression
from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(x_train,y_train)


# In[12]:


model.predict(x_test)


# In[13]:


model.score(x_test,y_test)*100


# In[14]:


pred_log = model.predict(x_test)


# In[15]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[16]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# In[17]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[18]:


from sklearn.tree import DecisionTreeClassifier 
model01= DecisionTreeClassifier()
model01.fit(x_train,y_train)


# In[19]:


model01.predict(x_test)


# In[20]:


model01.score(x_test,y_test)*100


# In[21]:


pred_log = model01.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[22]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# In[23]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
model02= KNeighborsClassifier()
model02.fit(x_train,y_train)


# In[25]:


model02.predict(x_test)


# In[26]:


model02.score(x_test,y_test)*100


# In[27]:


pred_log = model02.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[28]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# In[ ]:





# In[ ]:





# In[ ]:




