#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Created by Vaibhav Mehta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[15]:


dataset= pd.read_csv('weather.csv')
print(dataset.shape)


# In[16]:


print(dataset.describe())


# In[17]:


dataset.plot(x='MinTemp', y='MaxTemp' , style='o')
plt.title('min vs max')
plt.xlabel('mintemp')
plt.ylabel('maxtemp')
plt.show()


# In[18]:


plt.figure(figsize=(15,10))
plt.tight_layout()
seaborn.distplot(dataset['MaxTemp'])
plt.show()


# In[19]:


X= dataset['MinTemp'].values.reshape(-1,1)
y= dataset['MaxTemp'].values.reshape(-1,1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)


# In[20]:


model =LinearRegression()
model.fit(X_train,y_train)


# In[21]:


print('Intercept is :',model.intercept_)


# In[22]:


print('Coefficient is :' ,model.coef_)


# In[23]:


y_pred= model.predict(X_test)


# In[24]:


df=  pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)


# In[25]:


df1= df.head(25)
df1.plot(kind='bar', figsize=(16,10))
plt.grid(which='major', linestyle='-',linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':',linewidth='0.5', color='black')
plt.show()


# In[26]:


plt.scatter(X_test,y_test,color='gray')
plt.plot(X_test,y_pred,color='red',linewidth=2)
plt.show()


# In[27]:


print('Mean abolute error is:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean squared error is:', metrics.mean_squared_error(y_test,y_pred))
print('Root mean squared error is:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[ ]:





# In[ ]:





# In[ ]:




