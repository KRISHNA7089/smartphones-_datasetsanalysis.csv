#!/usr/bin/env python
# coding: utf-8

# # importing some important library

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


# # Loading data

# In[2]:


df = pd.read_csv('smartphones .csv')
# df


# In[3]:


# df.head() is used to display the first few rows of the dataframe.
df.head()


# In[4]:


# df.tail() is used to display the last few rows of the dataframe.
df.tail()


# In[5]:


print("Number of rows = ",df.shape[0])#returns the number of rows in the DataFrame.
print("Number of columns = ",df.shape[1])#returns the number of columns in the DataFrame.


# In[6]:


# df.info() provides a concise summary of the DataFrame 
df.info()


# In[7]:


#df.dtype is use to check datatype of our dataframe
df.dtypes


# In[8]:


# df.isnull().sum() is used to count the number of missing values (null values) in each column of the DataFrame
print(df.isnull().sum())


# In[9]:


# Visualize the missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(),cmap='viridis')
plt.show()


# In[10]:


#we are try to drop null values of our datasets 
df.dropna(inplace=True)


# In[11]:


#Than we check the number null value are remove or not in our dataframe
df.isnull().sum()


# In[12]:


# after drop missing value then visualizstion of our data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap='viridis')
plt.show()


# In[13]:


#  The describe() method provides summary statistics of the DataFrame's numeric columns aiding in 
# understanding the data's distribution and characteristics.
df.describe()


# In[14]:


# df.columns are retrieves the column names of the DataFrame. df using the columns attribute. 
df.columns


# In[15]:


#df.rename() method are use to change name of the columns in our dataframe
df.rename(columns={'processor':'processors'})


# In[16]:


# univariate analysis on rating columns
df['rating'].describe()


# In[17]:


#little skewd graph
sns.displot(kind='hist',data=df,x='rating',kde=True)


# In[18]:


# This code are plots a bar chart of the first five entries of the 'rating' column from the DataFrame 
df['rating'].head(5).plot(kind='bar')


# In[19]:


# closer to zero means normal distribution
df['rating'].skew()


# In[20]:


# there is no outliers present
sns.boxplot(x=df['rating']) 


# In[21]:


# Grouping the DataFrame by 'price', calculating the sum of 'rating' for each price category,
# and sorting the results by the total ratings in ascending order.
df.groupby(['price'],as_index=False)['rating'].sum().sort_values(by='rating',ascending=True)


# # Observation

# 1. Price Range: The prices range from ₹3,990 to ₹19,999, showing a variety of price points within the dataset.
# 2. Rating Distribution: Ratings vary from 60.0 to 1533.0. There is a wide range of ratings, indicating diverse feedback or evaluation of the products.
# 3. Price and Rating Relationship: While there isn't a clear pattern visible from this snippet alone, it seems that even within a similar price range, ratings can vary widely. For example, items priced around ₹7,000 have ratings of 60.0, while another item priced at ₹11,999 has a much higher rating of 1269.0.
# 4. High-Rated Items: Towards the higher end of the price spectrum (around ₹15,000 to ₹20,000), there are some items with notably high ratings (e.g., 1385.0 and 1533.0). This suggests that some higher-priced items may be perceived positively by customers.

# In[22]:


# Creating a scatter plot to explore the relationship between 'rating' and 'price' columns.
fig=px.scatter(df,x='rating',y='price')
fig.show()


# In[23]:


# Grouping the DataFrame by multiple columns ('model', 'price', 'rating', 'processor', 'display', 'camera'),
# summing the 'ram' column for each group, and sorting the results by the total RAM in ascending order.
df.groupby(['model','price','rating','processor','display','camera'],as_index=False)['ram'].sum().sort_values(by='ram',ascending=True)


# # Observation

# here are some observations based on the insights provided:
# 
# 1. Price Range:-The prices of the mobile phones in the dataset vary significantly, ranging from ₹22,999 for Xiaomi Redmi Note 11 Pro Plus 5G to ₹1,62,990 for Huawei Mate Xs 
# 
# 2. Processor Information:-The dataset includes a variety of processors such as Google Tensor, Snapdragon 8+ Gen1, Dimensity 9000 Plus, Snapdragon 865, and Snapdragon 888, among others, with varying clock speeds.
# 
# 3. Display Specifications:-Display sizes range from 6.1 inches to 7.8 inches with different resolutions and refresh rates like 1080 x 2340 px, 1440 x 3120 px, and 2200 x 2480 px.
# 
# 4. Camera Configurations:-Mobile phones feature various camera setups, including 108 MP + 8 MP + 2 MP Triple Rear, 50 MP + 12 MP + 10 MP Triple Rear, and 64 MP + 2 MP Dual Rear cameras.
# 
# 5. RAM and Storage:-Most of the phones in the dataset come with 12 GB RAM and 256 GB inbuilt storage, but there are variations such as 8 GB RAM and 512 GB inbuilt storage in Huawei Mate Xs 2.
# 
# 6. Ratings:-The ratings in the dataset range from 83.0 to 89.0, indicating the perceived quality or performance of the devices.

# In[24]:


# Grouping the DataFrame by multiple columns ('model', 'price', 'rating', 'processor', 'ram', 'battery'),
# counting the occurrences of each unique combination, and resetting the index with the counts renamed as 'Total'.
df.groupby(['model', 'price', 'rating','processor','ram','battery']).size().reset_index().rename(columns={0:'Total'})


# # Observation

#  here are some observations based on the insights provided:
# 
# 1. Apple iPhone Series:-The dataset includes various models from the Apple iPhone series, such as iPhone 11, iPhone 11 Pro Max, iPhone 12, and their variants with different RAM configurations and storage capacities.
# 
# 2. Processor and RAM:-Different processors are used in the devices, such as A13 Bionic, Bionic A14, Snapdragon 695, and Snapdragon 7 Gen2, paired with varying RAM sizes like 4 GB, 6 GB, and 8 GB.
# 
# 3. Battery and Charging:-The devices come with diverse battery capacities ranging from 3110 mAh to 5000 mAh, with varying fast charging capabilities like 18W, 45W, 55W, and 67W.
# 
# 4. Price Range and Ratings:-The prices of the phones vary significantly, with the Apple iPhone 11 Pro Max being the most expensive at ₹1,09,900 and the itel Vision 3 being the most affordable at ₹6,699. Ratings range from 61.0 to 80.0, indicating the perceived quality of the devices.
# 
# 5. Display and Camera:-Some devices feature specific display details like screen size and resolution, while others may have additional camera specifications not explicitly mentioned in the provided excerpt.
# 
# 6. Model Variants:-Variants of certain models with different storage capacities are present, such as the Apple iPhone 11 and iPhone 12 series.

# In[25]:


# Grouping the DataFrame by multiple columns ('model', 'os', 'price', 'battery', 'ram', 'card', 'sim'),
# counting the occurrences of each unique combination, and resetting the index with the count renamed as 'total'.
df[['model','os','price','battery','ram','card','sim']].groupby(['model','os','price','battery','ram','card','sim']).size().reset_index().rename(columns={0:'total'})


# # Observation

# here are some observations based on the insights provided:
# 1. Operating System and Version:-The dataset includes mobile phones running on various operating systems like iOS v13, iOS v14, and Android v12, with each phone having a specific OS version.
# 
# 2. Price Range and Features:-The prices of the phones vary, ranging from ₹6,699 for itel Vision 3 to ₹1,09,900 for Apple iPhone 11 Pro Max, with different features and specifications offered at different price points.
# 
# 3. Battery and Charging Capabilities:-Phones come with diverse battery capacities and charging technologies, such as 3110 mAh Battery, 3500 mAh Battery with 18W Fast Charging, 4200 mAh Battery with 55W Fast Charging, and 5000 mAh Battery with various fast charging speeds.
# 
# 4. RAM and Storage Options:-The devices offer different RAM configurations ranging from 3 GB to 8 GB and varying inbuilt storage capacities like 64 GB and 128 GB. Some devices also support memory card expansion.
# 
# 5. SIM and Network Support:-The phones feature dual SIM support along with compatibility for 3G, 4G, 5G, VoLTE, Wi-Fi, and NFC connectivity options.
# 
# 6. Memory Card Support:-Some devices support memory card expansion, while others do not, and the supported memory card capacities vary from not supported to up to 1 TB.

# In[26]:


# Computing the frequency count of each unique value in the 'model' column.
df['model'].value_counts()


# # Observation

# Based on the provided output, the observation is as follows:
# 
# The dataset consists of 879 unique mobile phone models with each model appearing exactly once, indicating that the dataset 
# contains a comprehensive list of distinct mobile phone models without any duplicate entries. 
# This level of accuracy in data representation suggests a precise and complete record of individual mobile phone models.

# In[27]:


# Select the top 10 processor data
top_processors = df['model'].value_counts().head(10)
sns.set(rc={'figure.figsize':(45,29)})
# Create a countplot using Seaborn
sns.countplot(x='model', data=df[df['model'].isin(top_processors.index)], palette=['white', 'red', 'orange', 'green', 'blue'])
plt.show()


# In[28]:


# Count the occurrences of each unique value in the 'processors' column
temp = df['processor'].value_counts().reset_index()
# Rename the columns for clarity
temp = temp.rename(columns={'index': 'processor', 'processor': 'count'})
# Display the resulting DataFrame
temp


# # Observation

# Based on the provided insight, the observation is as follows:
# 
# The dataset shows a distribution of various mobile phone processors along with their respective counts.It indicates that
# there is a mix of popular and less common processors listed. The count values beside each processor suggest the frequency 
# of occurrence of each processor type in the dataset.

# In[29]:


# Select the top 10 processor data
top_processors = df['processor'].value_counts().head(10)
sns.set(rc={'figure.figsize':(45,29)})
# Create a countplot using Seaborn
sns.countplot(x='processor', data=df[df['processor'].isin(top_processors.index)], palette=['white', 'red', 'orange', 'green', 'blue'])
plt.show()


# In[30]:


# Calculating the frequency count of each unique RAM value in the 'ram' column,
# resetting the index, renaming the columns, and selecting the top 10 most frequent RAM values.
temp=df['ram'].value_counts().reset_index().rename(columns={'index':'ram','ram':'total_count'}).head(10)
temp


# # Observation

# The dataset showcases various combinations of RAM and inbuilt storage capacities found in mobile phones. The most prevalent configuration is 8 GB RAM with 128 GB inbuilt storage, followed by other configurations such as 6 GB RAM with 128 GB inbuilt storage and 4 GB RAM with 64 GB inbuilt storage. This insight provides valuable information about the popularity and frequency of different RAM and storage combinations among mobile devices in the dataset.

# In[31]:


# Select the top 10 processor data
top_processors = df['ram'].value_counts().head(10)
sns.set(rc={'figure.figsize':(45,29)})
# Create a countplot using Seaborn
sns.countplot(x='ram', data=df[df['ram'].isin(top_processors.index)], palette=['white', 'red', 'orange', 'green', 'blue'])
plt.show()


# In[32]:


df['camera'].value_counts()


# # Observation

# 1. the observation is as follows:-
#     
# The dataset contains a variety of camera setups for mobile phones, with the most common configurations being triple rear cameras
# with varying megapixel specifications along with front camera details. These configurations include combinations
# like 50 MP + 8 MP + 2 MP Triple Rear & 16 MP Front Camera, 64 MP + 8 MP + 2 MP Triple Rear & 16 MP Front Camera,
# and 50 MP + 2 MP + 2 MP Triple Rear & 16 MP Front Camera. This insight highlights the diverse camera setups available
# in mobile devices, showcasing the popularity of specific camera configurations within the dataset.

# In[33]:


temp=df['camera'].value_counts().reset_index().rename(columns={'index':'camera','camera':'total_count'})
temp


# In[34]:


# Assuming 'camera' is a column in your DataFrame `temp`
fig = px.pie(temp, values='count', names='total_count', title='Pie chart of Number of rear camera', width=800, height=600)
fig.show()


# In[35]:


temp=df['os'].value_counts().reset_index().rename(columns={'index':'os', 'os':'os_count'}).head(20)
temp


# # Observation

# Based on the provided insight on different operating systems (OS) and their respective counts, the observation is as follows:-
# 
# The dataset showcases a variety of operating systems, with Android versions like v12, v11, v10, v13, and older versions
# such as v9.0 (Pie), v8.1 (Oreo), and v4.4.2 (KitKat) being represented. Additionally, it includes OS mentions like
# No FM Radio and Bluetooth, along with iOS versions such as v16, v15, v13, and v17. 
# The insight provides a comprehensive overview of the distribution of different operating systems and 
# their frequencies within the dataset, offering insights into the prevalence of various OS versions in mobile devices.

# In[36]:


# Setting the figure size for seaborn plots and creating a count plot of the 'os' column from the DataFrame 'df',
# with a custom color palette for the bars.
sns.set(rc={"figure.figsize":(51,43)})
sns.countplot(x='os',data=df,palette=['white','red','orange','green','green'])


# In[37]:


# Creating a new DataFrame 'temp_df' with only the 'price' and 'battery' columns from the original DataFrame 'df'.
temp_df=df[['price','battery']]
temp_df


# # Observation

# Based on the provided insight regarding mobile phone prices and battery specifications, the observation is as follows:
# 
# The dataset includes mobile phones with various price points and battery configurations. The phones have different battery 
# capacities ranging from 5000 mAh to 5080 mAh, paired with fast charging technologies with different power outputs like
# 15W, 22.5W, 33W, 67W, 68.2W, and 100W. This insight highlights the diversity in battery capacities and fast charging
# capabilities offered in mobile devices across different price ranges.

# In[38]:


plt.figure(figsize=(10, 6))  # Adjust size as needed
df.plot(kind='line', figsize=(10,6))
plt.title('Price and Battery Over Time')
plt.xlabel('Index')
plt.ylabel('Values')
plt.grid(True)
plt.show()


# In[42]:


# Creating a new DataFrame 'temp_df' with the first five rows of the 'camera' and 'price' columns from the original DataFrame 'df'.
temp_df=df[['camera','price']]
temp_df


# # Observation

# Based on the provided insight regarding camera configurations and their corresponding prices, the observation is as follows:
# 
# The dataset includes mobile phones with diverse camera setups and price points. Camera configurations range from triple
# rear cameras with various megapixel combinations to front camera specifications.
# The prices associated with these devices vary, indicating that there is a range of pricing based on the camera features
# offered in each phone. This insight highlights the correlation between camera specifications and pricing in the mobile phone
# market.

# In[40]:


# Convert 'price' column to numeric (assuming it contains strings with non-numeric characters)
temp_df['price'] = temp_df['price'].apply(lambda x: float(x.replace('₹', '').replace(',', ''))if isinstance(x, str) else x)
sns.set(rc={'figure.figsize':(31,29)})
# Plot the barplot
sns.barplot(x="camera", y="price", data=temp_df)
plt.show()


# temp_df['price'] = temp_df['price'].apply(lambda x: float(x.replace('₹', '').replace(',', '')) if isinstance(x, str) else x): This line converts the 'price' column to numeric format. It uses the apply method to apply a lambda function to each element in the 'price' column. The lambda function checks if the element x is a string (isinstance(x, str)), and if so, it removes non-numeric characters ('₹' and ',') using the replace method and then converts the result to a float using the float function. If the element is already a float, it leaves it unchanged. This ensures that all values in the 'price' column are numeric.

# # Conclusion

# 1. Preference for Higher RAM and Storage Configurations: There is a clear preference among consumers for smartphones with higher RAM capacities and ample inbuilt storage options, as evidenced by the higher counts for configurations like 8GB RAM with 128GB inbuilt storage.
# 
# 2. Android Dominance: Android continues to dominate the smartphone market, with a wide range of Android versions represented in the data. Versions like Android v12 and v11 are the most prevalent, reflecting the ongoing adoption of newer Android releases by users.
# 
# 3. Continued Usage of Older Android Versions: Despite the availability of newer Android releases, there are still significant counts for older versions like Android v10 and v9.0 (Pie), indicating that some users may be holding onto devices with older operating systems for various reasons.
# 
# 4. Limited iOS Representation: While iOS versions are included in the data, their counts are lower compared to Android versions, reflecting the smaller market share of iOS devices globally.
# 
# 5. Feature Preferences: Features like FM Radio and memory card support are still relevant to some consumers, although their counts are lower compared to operating system versions. This suggests that while these features may still hold importance for certain users, they may not be significant factors driving smartphone purchasing decisions for the majority of consumers.
# 
# 6. Trend Towards Updated Android Versions: There is a trend among users towards migrating to more recent Android releases, as evidenced by the relatively lower counts for older versions like Android v8.1 (Oreo) and v8.0 (Oreo) compared to newer releases.

# Overall, the analysis underscores the importance of offering a diverse range of configurations and features to cater to the varied preferences of smartphone consumers. Additionally, it highlights the ongoing dominance of Android in the market and the importance of staying updated with the latest operating system versions to meet evolving user expectations.

# In[ ]:




