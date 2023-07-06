#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


credit=pd.read_csv("D:\ML\dataset\creditcard.csv")


# In[3]:


credit


# In[4]:


credit.info()


# In[5]:


credit.isnull().sum()


# In[6]:


credit.describe()


# In[7]:


credit['Class'].value_counts()


# In[8]:


credit.shape


# In[9]:


legit=credit[credit.Class==0]
fraud=credit[credit.Class==1]


# In[10]:


legit


# In[11]:


fraud


# In[12]:


legit.Amount.describe()


# In[13]:


fraud.Amount.describe()


# In[14]:


credit.groupby('Class').mean()


# In[15]:


legit_sample=legit.sample(n=492)


# In[16]:


new_credit=pd.concat([legit_sample,fraud],axis=0)


# In[17]:


new_credit


# In[18]:


new_credit.shape


# In[19]:


new_credit['Class'].value_counts()


# In[21]:


new_credit.groupby('Class').mean()


# In[22]:


X=new_credit.drop('Class',axis=1)


# In[23]:


X


# In[24]:


Y=new_credit['Class']


# In[25]:


Y


# In[28]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[30]:


model=LogisticRegression()


# In[31]:


model.fit(X_train,Y_train)


# In[32]:


X_train_prediction=model.predict(X_train)
training_data=accuracy_score(X_train_prediction,Y_train)


# In[33]:


print("The accuracy score:",training_data)


# In[34]:


X_test_prediction=model.predict(X_test)
testing_data=accuracy_score(X_test_prediction,Y_test)


# In[35]:


print("The accuracy score:",testing_data)


# In[ ]:




