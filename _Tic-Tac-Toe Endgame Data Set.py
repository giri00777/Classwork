#!/usr/bin/env python
# coding: utf-8

# In[124]:


import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[125]:


data=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data")


# In[126]:


data.head()


# In[127]:


data.shape


# In[128]:


data.isnull().sum()


# In[129]:


data=data.replace('negative',0)
data=data.replace('positive',1)
data=data.replace('x',2)
data=data.replace('o',3)
data=data.replace('b',4)


# In[130]:


data.head()


# In[131]:


x=data.drop(['positive'],axis=1)


# In[132]:


y=data['positive']


# In[133]:


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=42)


# In[134]:


model=LogisticRegression()
model.fit(train_x,train_y)


# In[135]:


perdiction=model.predict(test_x)


# In[136]:


prob=model.predict_proba(test_x)


# In[138]:


score=model.score(test_x,test_y)
score


# In[139]:


data.hist()


# In[ ]:




