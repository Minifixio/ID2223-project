#!/usr/bin/env python
# coding: utf-8

# In[38]:

import numpy as np
import hopsworks
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


# In[39]:


if os.getenv('HOPSWORKS_API_KEY') is not None:
    HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
else:
    with open('../secrets/hopsworks_api_key.txt', 'r') as file:
        HOPSWORKS_API_KEY = file.readline().strip()


# In[40]:


project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store() 


# In[41]:


user_embeddings_fg = fs.get_feature_group(
    name='spotify_user_embeddings',
    version=2,
)

user_embeddings_df = user_embeddings_fg.read()
print(f"A total of {len(user_embeddings_df)} user embeddings are available.")
user_embeddings_df.head()


# In[43]:


user_embeddings_df['genre_embedding'] = user_embeddings_df['genre_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float)
)
user_embeddings_df['artist_embedding'] = user_embeddings_df['artist_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float)
)
user_embeddings_df['playlist_embedding'] = user_embeddings_df['playlist_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float)
)
user_embeddings_df['release_year_embedding'] = user_embeddings_df['release_year_embedding'].apply(lambda x: torch.tensor([x], dtype=torch.float))


# In[44]:


class EmbeddingDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id = self.data.iloc[idx]['user_id']
        genre_embedding = self.data.iloc[idx]['genre_embedding']
        artist_embedding = self.data.iloc[idx]['artist_embedding']
        playlist_embedding = self.data.iloc[idx]['playlist_embedding']
        return user_id, genre_embedding, artist_embedding, playlist_embedding


class Tower(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Tower, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class TwoTowerModel(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(TwoTowerModel, self).__init__()
        # Separate processing for each embedding type
        self.genre_fc = Tower(input_dim=embedding_dim, output_dim=output_dim)
        self.artist_fc = Tower(input_dim=embedding_dim, output_dim=output_dim)
        self.playlist_fc = Tower(input_dim=embedding_dim, output_dim=output_dim)
        
        # Joint Tower for final embedding
        self.fc_merge = nn.Sequential(
            nn.Linear(output_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, genre, artist, playlist):
        genre_embed = self.genre_fc(genre)
        artist_embed = self.artist_fc(artist)
        playlist_embed = self.playlist_fc(playlist)
        
        # Concatenate embeddings and pass through final layers
        combined = torch.cat([genre_embed, artist_embed, playlist_embed], dim=-1)
        final_embed = self.fc_merge(combined)
        return final_embed

    def compute_similarity(self, query_embedding, database_embedding):
        # Cosine similarity for comparison
        return torch.nn.functional.cosine_similarity(query_embedding, database_embedding)



# In[45]:


# Initialize model
df = user_embeddings_df
print(df['genre_embedding'][0].shape)
embedding_dim = len(df['genre_embedding'][0])  # Assuming all embeddings have the same dimension
output_dim = 64
margin = 0.5

model = TwoTowerModel(embedding_dim=embedding_dim, output_dim=output_dim)
criterion = nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1/10e9, weight_decay=1e-5)

# Prepare dataset and dataloader
train_dataset = EmbeddingDataset(df)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Dummy negative sampling (replace with actual negatives)
def negative_sample(batch_size, df):
    # Randomly sample other embeddings as negatives
    sampled = df.sample(batch_size)  # Ensure the sample size matches the batch size
    return torch.stack(sampled['genre_embedding'].tolist()), \
           torch.stack(sampled['artist_embedding'].tolist()), \
           torch.stack(sampled['playlist_embedding'].tolist())


# Training loop
for epoch in range(100):
    total_loss = 0
    for user_ids, genres, artists, playlists in train_loader:
        # Ensure embeddings are converted to tensors
        genres = torch.tensor(genres.tolist(), dtype=torch.float32)
        artists = torch.tensor(artists.tolist(), dtype=torch.float32)
        playlists = torch.tensor(playlists.tolist(), dtype=torch.float32)
        
        # Generate negative samples
        neg_genres, neg_artists, neg_playlists = negative_sample(len(genres), df)
        
        # Forward pass for positives and negatives
        positive_embed = model(genres, artists, playlists)
        negative_embed = model(neg_genres, neg_artists, neg_playlists)
        
        # Create labels for the current batch size
        labels = torch.ones(positive_embed.size(0))
        
        # Calculate loss - note we're using just one pair of embeddings and their labels
        loss = criterion(positive_embed, negative_embed, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss}")


model_dir = "torch_model"
os.makedirs(model_dir, exist_ok=True)

# Save the model and metadata
torch.save({
    'model_state_dict': model.state_dict(),
    'embedding_dim': embedding_dim,
    'output_dim': output_dim,
}, os.path.join(model_dir, 'two_tower_model_torch.pth'))


# In[48]:


# Get the model registry handle
mr = project.get_model_registry()
model_registry = mr.get_model("two_tower_model_torch", version=1) 
model_registry.delete()

# Create the model metadata object
torch_model = mr.torch.create_model(
    name="two_tower_model_torch",
    metrics={'final_loss': total_loss},  # You can add your training metrics here
    description="Two-tower model for music recommendations",
    version=1,
    input_example={
        'genre_embedding': genres[0].numpy().tolist(),
        'artist_embedding': artists[0].numpy().tolist(),
        'playlist_embedding': playlists[0].numpy().tolist()
    }
)

# Save the model to the registry
torch_model.save(model_dir)

