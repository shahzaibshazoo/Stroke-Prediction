#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[61]:


df=pd.read_csv('healthcare-dataset-stroke-data.csv')
df


# In[62]:


df.describe()


# In[66]:


df.isna().sum()


# In[64]:


df['bmi'].fillna(df['bmi'].mean(),inplace=True)
df['bmi']


# In[65]:


categorical= df.select_dtypes("object").columns
categorical


# In[67]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in categorical:
    df[i]=le.fit_transform(df[i])
df


# In[68]:


from featurewiz import featurewiz as ftwz


# In[69]:


target=['stroke']
features,train=ftwz(df,target,corr_score=0.7)


# In[70]:


features


# In[71]:


X=df[features]
y=df['stroke']


# In[72]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=43)


# In[73]:


X_train.shape


# In[74]:


X_test.shape


# In[89]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)
pred=model.predict(X_test)


# In[93]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)


# In[94]:


model.score(X_test,y_test)


# In[88]:


import joblib
joblib.dump(model,'Stroke Prediciton.pkl')

