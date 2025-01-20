#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

raw_df = pd.read_csv("high_popularity_spotify_data.csv", encoding = "UTF-8")
raw_df


# In[2]:


#checking missing data
raw_df.isnull().sum()


# In[3]:


#dropping the one missing value row as it is not crucial for my work
raw_df.dropna(subset=["track_album_name"], inplace = True)


# In[4]:


# List of numerical features to visualize
features = ['valence', 'tempo', 'danceability', 'track_popularity']

# Create histograms for each feature
for feature in features:
    plt.figure(figsize=(8, 4))
    sns.histplot(raw_df[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()


# In[5]:


feature_pairs = [
    ('valence', 'energy'),
    ('danceability', 'track_popularity'),
    ('tempo', 'track_popularity'),
    ('valence', 'track_popularity')
]

for x, y in feature_pairs:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=raw_df, x=x, y=y, alpha=0.6)
    plt.title(f'{y} vs. {x}')
    plt.xlabel(x.capitalize())
    plt.ylabel(y.capitalize())
    plt.show()


# In[6]:


numeric_df = raw_df.select_dtypes(include = ["float64","int64"])


# In[7]:


# Compute the correlation matrix
correlation_matrix = numeric_df.corr()

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Spotify Features")
plt.show()


# In[8]:


# Define the mood classification function
def mood_category(row):
    if row['valence'] > 0.5 and row['energy'] > 0.5:
        return "Happy Energetic"
    elif row['valence'] > 0.5 and row['energy'] <= 0.5:
        return "Calm Happy"
    elif row['valence'] <= 0.5 and row['energy'] > 0.5:
        return "Intense"
    else:
        return "Sad"

# Apply the function to the dataset
raw_df['mood'] = raw_df.apply(mood_category, axis=1)

raw_df[['valence', 'energy', 'mood']].head()


# In[9]:


raw_df["mood"].value_counts()


# In[10]:


# Create a feature that combines acousticness and loudness
raw_df['acoustic_loudness'] = raw_df['acousticness'] * (-raw_df['loudness'])

raw_df[['acousticness', 'loudness', 'acoustic_loudness']].head()


# In[11]:


#normalising song duration column/feature
raw_df["duration_min"] = raw_df["duration_ms"]/60000
raw_df[['duration_ms', 'duration_min']].head()


# In[12]:


# Define the scaler
scaler = MinMaxScaler()

# List of features to normalize
features_to_normalize = ['tempo', 'energy', 'danceability']

# Apply scaling
raw_df[features_to_normalize] = scaler.fit_transform(raw_df[features_to_normalize])

raw_df[features_to_normalize].head()


# In[13]:


# Define the features and target
X = raw_df[['valence', 'energy', 'danceability', 'tempo', 'acoustic_loudness']]  # Feature columns
y = raw_df['mood']  # Target column

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Testing set shape: {X_test.shape}, {y_test.shape}")


# In[16]:


# Function to recommend songs based on user preferences
def recommend_songs(user_preferences, data, top_n=5):
    # Compute cosine similarity between user preferences and dataset features
    similarity_scores = cosine_similarity([user_preferences], data)[0]
    
    # Rank songs by similarity scores
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    
    # Return the top N recommendations
    return raw_df.iloc[top_indices][['track_name', 'track_artist', 'mood','track_href']]

# Recommend songs
recommendations = recommend_songs(user_preferences, X_train.values, top_n=5)


# Streamlit app starts here
st.title("Spotify Music Recommendation System")

# Streamlit sidebar inputs
st.sidebar.header("User Preferences")
valence = st.sidebar.slider("Valence (Happiness)", 0.0, 1.0, 0.5)
energy = st.sidebar.slider("Energy (Liveliness)", 0.0, 1.0, 0.5)
danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.5)
tempo = st.sidebar.slider("Tempo", 0.0, 1.0, 0.5)
acoustic_loudness = st.sidebar.slider("Acoustic-Loudness Interaction", 0.0, 1.0, 0.5)

user_preferences = [valence, energy, danceability, tempo, acoustic_loudness]

# Generate recommendations
recommendations = recommend_songs(
    user_preferences, 
    raw_df[['valence', 'energy', 'danceability', 'tempo', 'acoustic_loudness']].values
)

# Display recommendations
st.subheader("Recommended Songs")
for index, row in recommendations.iterrows():
    st.write(f"🎵 **{row['track_name']}** by {row['track_artist']} ({row['mood']})")
    st.write(f"[Listen on Spotify]({row['track_href']})")
if __name__ == "__main__":
    import streamlit as st


# In[ ]:




