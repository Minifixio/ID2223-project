#!/usr/bin/env python
# coding: utf-8

# In[51]:


from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
import hopsworks


# In[52]:


with open('../secrets/hopsworks_api_key.txt', 'r') as file:
    HOPSWORKS_API_KEY = file.readline().strip()


# In[53]:


project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store() 


# In[54]:


user_embeddings_fg = fs.get_feature_group(
    name='spotify_user_embeddings',
    version=2,
)

user_embeddings_df = user_embeddings_fg.read()
user_embeddings_df.head()


# In[55]:


user_embeddings_df['full_embedding'] = user_embeddings_df.apply(
    lambda row: np.concatenate(
        [row['genre_embedding'], row['artist_embedding'], row['playlist_embedding'], row['release_year_embedding']]
    ),
    axis=1
)
normalized_embeddings = normalize(np.array(user_embeddings_df['full_embedding'].tolist()))
user_embeddings_df['normalized_embedding'] = normalized_embeddings.tolist()
user_embeddings_df


# In[56]:


def build_user_tower(input_dim, embedding_dim=128):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    user_embedding = layers.Dense(embedding_dim, activation=None)(x)  # Final user embedding
    return Model(inputs, user_embedding, name="UserTower")

def build_candidate_tower(input_dim, embedding_dim=128):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    candidate_embedding = layers.Dense(embedding_dim, activation=None)(x)  # Final candidate embedding
    return Model(inputs, candidate_embedding, name="CandidateTower")


# In[57]:


# Instantiate towers
input_dim = len(normalized_embeddings[0])  # Dimensionality of the concatenated embedding
embedding_dim = 128

user_tower = build_user_tower(input_dim, embedding_dim)
candidate_tower = build_candidate_tower(input_dim, embedding_dim)

# Compute cosine similarity
user_embedding = user_tower.output
candidate_embedding = candidate_tower.output
cosine_similarity_model = tf.keras.layers.Dot(axes=1, normalize=True)([user_embedding, candidate_embedding])

# Final model
model = tf.keras.Model(inputs=[user_tower.input, candidate_tower.input], outputs=cosine_similarity_model)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# In[58]:


def generate_pairs(embeddings, similarity_threshold=0.8, negative_ratio=1):
    pairs = []
    labels = []

    # Compute cosine similarity for all pairs
    similarity_matrix = cosine_similarity(embeddings)  # This is a valid pairwise similarity matrix

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if similarity_matrix[i, j] > similarity_threshold:
                # Positive pair
                pairs.append((embeddings[i], embeddings[j]))
                labels.append(1)

                # Generate negative pairs
                for _ in range(negative_ratio):
                    negative_index = np.random.choice(len(embeddings))
                    while negative_index == i or negative_index == j:
                        negative_index = np.random.choice(len(embeddings))
                    pairs.append((embeddings[i], embeddings[negative_index]))
                    labels.append(0)
    
    return np.array(pairs), np.array(labels)


# In[59]:


# Generate training data
pairs, labels = generate_pairs(normalized_embeddings)
user_1 = np.array([pair[0] for pair in pairs])
user_2 = np.array([pair[1] for pair in pairs])

history = model.fit(
    [user_1, user_2],  # Input: pairs of user embeddings
    labels,            # Output: similarity labels
    batch_size=32,
    epochs=10,
    validation_split=0.2
)


# In[60]:


mr = project.get_model_registry()
model.save("two_tower_model.keras")

# Create a new model version
model_dir = "two_tower_model.keras"
model_name = "two_tower_recommender"

model_registry = mr.python.create_model(
    name=model_name,
    metrics={"accuracy": history.history["accuracy"][-1]},  # Log the final accuracy
    description="Two-Tower Recommender Model for User Similarity",
)

model_registry.save(model_dir)
print(f"Model '{model_name}' uploaded to Hopsworks!")

