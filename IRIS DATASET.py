#!/usr/bin/env python
# coding: utf-8

# # IMPORT THE LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report


# # importing the data

# In[2]:


from sklearn.datasets import load_iris
iris=load_iris()


# In[3]:


iris.keys()


# In[4]:


iris.data.shape


# In[5]:


iris.target.shape


# In[6]:


df=pd.DataFrame(np.c_[iris['data'],(iris['target'])],columns=np.append(iris['feature_names'],'target'))


# In[7]:


df.head()


# In[8]:


df.target.tail()


# # DATA EXPLORATION AND VISUALISATION

# In[9]:


plt.figure(figsize=[8,4])
sns.scatterplot(x='sepal length (cm)',y='sepal width (cm)',hue='target',data=df)
plt.title('SCATTER PLOT')


# In[10]:


sns.pairplot(data=df,vars=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)'],hue='target')


# In[11]:


sns.heatmap(df.corr(),cmap='plasma',annot=True)


# In[12]:


features=df.drop('target',axis=1)
target=df.target


# In[13]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=42)


# # MODEL FITTING

# In[20]:


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=2,metric='minkowski',p=2)


# In[21]:


classifier.fit(x_train,y_train)
y_predicted=classifier.predict(x_test)


# # MODEL TESTING

# In[22]:


cm=confusion_matrix(y_test,y_predicted)
sns.heatmap(cm,annot=True,fmt="d",cmap='YlGnBu')
print(classification_report(y_test,y_predicted))


# In[ ]:




