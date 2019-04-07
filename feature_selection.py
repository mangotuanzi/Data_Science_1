#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


data =pd.read_table(r'C:\Users\10377\Scripts\AwA2-features.txt',)


# In[4]:


data_np=np.loadtxt(r'C:\Users\10377\Scripts\AwA2-features.txt')


# In[5]:


label_np=np.loadtxt(r'C:\Users\10377\Scripts\AwA2-labels.txt')


# In[6]:


label=pd.DataFrame(label_np)


# In[7]:


from feature_selector import FeatureSelector


# In[8]:


data=pd.DataFrame(data_np)


# In[27]:


fc = FeatureSelector(data,labels=label)


# In[10]:


fc.identify_missing(missing_threshold=0.95)


# In[11]:


fc.missing_stats.head()


# In[12]:


fc.identify_collinear(correlation_threshold=0.98)


# In[17]:


fc.identify_zero_importance(task='classification',eval_metric='auc',n_iterations=10,early_stopping=True)


# In[18]:


fc.identify_low_importance(cumulative_importance=0.95)


# In[19]:


fc.identify_single_unique()


# In[30]:


train_removed=fc.remove(methods = 'all')


# In[31]:


np.save('train_removed2',train_removed.values)


# In[29]:


fc.identify_all(selection_params={'missing_threshold':0.7,
                                 'correlation_threshold':0.9,
                                 'task':'classification',
                                 'eval_metric':'auc',
                                 'cumulative_importance':0.9})


# In[ ]:




