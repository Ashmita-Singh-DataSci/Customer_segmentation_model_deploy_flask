#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
PROBLEM STATEMENT:

Create a K-means clustering algorithm to group customers of a retail store based on their purchase history."
'''


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib


# In[3]:


#Load the data
data = pd.read_csv('Mall_Customers.csv')


# In[4]:


data


# In[5]:


data.columns


# In[6]:


data.describe()


# In[7]:


# Check for null values
data.isnull().sum().sum()


# In[8]:


# Visualize 
plt.style.use('dark_background')
plt.scatter(data['Age'],data['Annual Income (k$)'],color='purple')
plt.title('age v/s annual income')
plt.xlabel('age')
plt.ylabel('annual income')


# In[9]:


plt.style.use('dark_background')
plt.scatter(data['Spending Score (1-100)'],data['Annual Income (k$)'])
plt.title('age v/s annual income')
plt.xlabel('age')
plt.ylabel('annual income')


# In[10]:


plt.style.use('dark_background')

plt.figure(1, figsize=(15, 7))
for gender in ['Male', 'Female']:
    plt.scatter(x='Annual Income (k$)', y='Spending Score (1-100)',
                data=data[data['Gender'] == gender], s=200, alpha=0.7, label=gender)
    
plt.xlabel('Annual Income (k$)', fontsize=14)
plt.ylabel('Spending Score (1-100)', fontsize=14)
plt.title('Annual Income vs Spending Score w.r.t Gender', fontsize=16)
plt.legend(title='Gender', title_fontsize='13', fontsize='12')

plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# In[11]:


# Divide and Assign
X = data.iloc[:,3:5].values
X


# In[12]:


n_clusters = range(2, 10)
wcss = []


# Add `for` loop to train model and calculate WCSS 
for k in n_clusters:
    model = KMeans(n_clusters=k, random_state=42)
    # Train the model
    model.fit(X)
    # Calculate WCSS
    wcss.append(model.inertia_)
    # Calculate Silhouette Score
 

print('WCSS:', wcss[:6])
print()



# In[13]:


plt.plot(range(2,10),wcss,marker='o')
plt.xlabel('numbers of clusters')
plt.ylabel('WCSS')
plt.show()


# In[14]:


# Train the final model
Fmodel = KMeans(n_clusters=5,random_state=101)
Fmodel.fit(X)


# In[15]:


ypred = Fmodel.predict(X)


# In[16]:


ypred


# In[17]:


plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'],c=ypred,cmap='viridis')
plt.title('Customers Segmented territories')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()


# In[18]:


# Save the model 
joblib.dump(Fmodel, 'customer_segkmeans_model.pkl')

