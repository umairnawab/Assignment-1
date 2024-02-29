#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pandas.plotting import scatter_matrix
df = pd.read_csv('car_prices.csv')
df


# In[83]:


print("Dataset Information:")
print(df.info())


# In[84]:


missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)



# In[39]:


# Plot histogram
plt.hist(df["price"], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Car Prices')
plt.xlabel('price in pound')
plt.ylabel('Frequency')
plt.show()


# In[85]:


plt.scatter(df['horsepower'], df['peak-rpm'], color='green')
plt.title('horsepower vs. peak-rpm')
plt.xlabel('horse power')
plt.ylabel('peak-rpm')
plt.show()


# In[86]:


cm = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[100]:


plt.figure(figsize=(6, 4))
sns.boxplot(x='body-style', y='price', data=df)
plt.title('Box Plot of Car Price vs Body style')
plt.show()


# In[ ]:




