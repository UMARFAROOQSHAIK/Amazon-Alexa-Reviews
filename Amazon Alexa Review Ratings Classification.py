#!/usr/bin/env python
# coding: utf-8

# # IMPORT THE LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.metrics import confusion_matrix,classification_report


# # IMPORT THE DATA

# In[2]:


df=pd.read_csv(r"C:\Users\Password\Desktop\RealLife Projects\DATA\ML Classification Package\6. Decision Trees and Random Forest\amazon_alexa.tsv",sep='\t')
df.head()


# # EXPLORATION AND DATA ANALYSIS

# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


sns.heatmap(df.isnull(),yticklabels='Blues',cmap='plasma')


# In[6]:


positive=df[df['feedback']==1]


# In[7]:


negative=df[df['feedback']==0]


# In[8]:


plt.figure(figsize=[5,5])
sns.countplot(x='feedback',data=df)


# In[9]:


reviews=df['verified_reviews']


# In[10]:


reviews.head()


# In[11]:


df['variation'].unique()


# In[12]:


df=df.drop('date',axis=1)


# In[13]:


df.head()


# In[14]:


variation_dummies=pd.get_dummies(df['variation'],drop_first=True)


# In[15]:


variation_dummies.head()


# In[16]:


type(variation_dummies)


# In[17]:


type(df)


# In[18]:


alexa=pd.concat([df,variation_dummies],axis=1)


# In[19]:


alexa.head()


# In[20]:


alexa.drop('variation',axis=1)


# ## Encoding the verified_reviews

# In[21]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
alexa_vectoriser=cv.fit_transform(alexa['verified_reviews'])
print(cv.get_feature_names())


# In[22]:


alexa_vectoriser.toarray()


# In[23]:


type(alexa_vectoriser.toarray())


# In[24]:


amazon=pd.DataFrame(alexa_vectoriser.toarray())


# In[25]:


amazon.head()


# In[26]:


type(amazon)


# In[27]:


type(alexa)


# In[28]:


alexa_final=pd.concat([alexa,amazon],axis=1)


# In[29]:


alexa_final.drop(['variation','verified_reviews'],axis='columns',inplace=True)


# In[30]:


alexa_final.head()


# In[31]:


alexa_final.shape


# In[32]:


features=alexa_final.drop('feedback',axis=1)
target=alexa_final.feedback


# In[33]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2)


# # Model fitting

# In[34]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,criterion='entropy')
model.fit(x_train,y_train)
y_predicted=model.predict(x_test)


# In[35]:


cm=confusion_matrix(y_test,y_predicted)
sns.heatmap(cm,annot=True,fmt='d',cmap='plasma')
print(classification_report(y_test,y_predicted))


# # IMPROVING THE MODEL

# In[36]:


alexa_final.shape


# In[37]:


alexa_final['length']=alexa['verified_reviews'].apply(len)


# In[38]:


alexa_final.head()


# In[39]:


features1=alexa_final.drop('feedback',axis=1)
target1=alexa_final.feedback


# In[40]:


from sklearn.model_selection import train_test_split
x_train1,x_test1,y_train1,y_test1=train_test_split(features1,target1,test_size=0.2)


# In[41]:


model.fit(x_train1,y_train1)
y_predicted1=model.predict(x_test1)


# In[42]:


cm=confusion_matrix(y_test1,y_predicted1)
sns.heatmap(cm,annot=True,fmt='d',cmap='plasma')
print(classification_report(y_test1,y_predicted1))


# In[ ]:




