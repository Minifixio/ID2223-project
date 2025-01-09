#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import hopsworks
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import normalize
from collections import Counter
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import re
import gradio as gr


# In[34]:


# Load credentials
with open('../secrets/hopsworks_api_key.txt', 'r') as file:
    HOPSWORKS_API_KEY = file.readline().strip()

with open('../secrets/spotify_client_id.txt', 'r') as file:
    SPOTIFY_CLIENT_ID = file.readline().strip()

with open('../secrets/spotify_client_secret.txt', 'r') as file:
    SPOTIFY_CLIENT_SECRET = file.readline().strip()


# In[35]:


client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# In[36]:


# Connect to Hopsworks
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()


# In[37]:


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


# In[38]:


mr = project.get_model_registry()

# Retrieve the PyTorch model from the model registry
model_registry = mr.get_model("two_tower_model_torch", version=1)  # Adjust version as needed
model_file_path = model_registry.download()

# Load the model
checkpoint = torch.load(os.path.join(model_file_path, 'two_tower_model_torch.pth'))

# Recreate the model architecture
model = TwoTowerModel(
    embedding_dim=checkpoint['embedding_dim'],
    output_dim=checkpoint['output_dim']
)

# Load the state dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode

print("Model loaded successfully!")


# In[39]:


transformer_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # This can be replaced if needed


# In[41]:


def generate_user_embedding(user_playlists, transformer_model, top_artist_count, playlists_count):
    print("Generating user embedding...")
    all_genres = []
    all_artists = []
    all_release_years = []
    playlist_features = []

    per_playlist_genre_embeddings = []  # Collect genre embeddings for each playlist

    for playlist in user_playlists[:playlists_count]:  # Limit to first playlists_count playlists
        # print(f"Processing playlist: {playlist['name']}")
        playlist_name = playlist.get("name", "Unknown")
        playlist_id = playlist["id"]

        # Fetch tracks in the playlist
        tracks = sp.playlist_tracks(playlist_id)["items"]
        # print(f"Number of tracks: {len(tracks)}")

        genres = []
        popularity = []
        release_years = []
        explicit_flags = []
        artist_names = []
        artist_ids = []

        # Collect all artist IDs for batch processing
        for item in tracks:
            track = item["track"]
            if not track or track["is_local"]:
                continue
            artist_ids.append(track["artists"][0]["id"])  # Only taking the first artist for simplicity
            release_date = track["album"]["release_date"]

            # Extract year from release date
            release_year = release_date.split('-')[0]
            release_years.append(int(release_year))

            popularity.append(track.get("popularity", 0))
            explicit_flags.append(track.get("explicit", False))

        # Batch the artist IDs for the Get Several Artists API call
        batch_size = 50
        artist_info = []
        for i in range(0, len(artist_ids), batch_size):
            batch = artist_ids[i:i + batch_size]
            response = sp.artists(batch)
            artist_info.extend(response["artists"])

        # Process artist information
        for artist in artist_info:
            artist_name = artist.get("name", "Unknown")
            track_genres = artist.get("genres", [])

            artist_names.append(artist_name)
            genres.extend(track_genres)

        # Generate per-playlist genre embedding
        if genres:
            genre_embeddings = transformer_model.encode(genres, show_progress_bar=False)
            playlist_genre_embedding = np.mean(genre_embeddings, axis=0)  # Average embedding for this playlist
        else:
            playlist_genre_embedding = np.zeros(384)

        per_playlist_genre_embeddings.append(playlist_genre_embedding)

        # Playlist-level features
        playlist_features.append({
            "playlist_name": playlist_name,
            "num_tracks": len(tracks),
            "avg_popularity": np.mean(popularity) if popularity else 0,
            "explicit_ratio": np.mean(explicit_flags) if explicit_flags else 0
        })

        all_genres.extend(genres)
        all_artists.extend(artist_names)
        all_release_years.extend(release_years)

    # Combine per-playlist genre embeddings using playlist sizes as weights
    if per_playlist_genre_embeddings:
        playlist_sizes = [p["num_tracks"] for p in playlist_features]
        playlist_weights = normalize(np.array(playlist_sizes).reshape(1, -1))[0]
        playlist_embedding = np.sum(
            [playlist_weights[i] * per_playlist_genre_embeddings[i] for i in range(len(per_playlist_genre_embeddings))],
            axis=0
        )
    else:
        playlist_embedding = np.zeros(384)

    # Generate overall artist and genre embeddings
    print("Generating contextual embeddings...")

    # Genre Embeddings
    genre_embeddings = transformer_model.encode(all_genres, show_progress_bar=False) if all_genres else np.zeros((1, 384))
    genre_embedding = np.mean(genre_embeddings, axis=0) if len(genre_embeddings) > 0 else np.zeros(384)

    # Artist Embeddings
    artist_counter = Counter(all_artists)
    top_artists = [artist for artist, _ in artist_counter.most_common(top_artist_count)]
    artist_embeddings = transformer_model.encode(top_artists, show_progress_bar=False) if top_artists else np.zeros((1, 384))
    artist_embedding = np.mean(artist_embeddings, axis=0) if len(artist_embeddings) > 0 else np.zeros(384)

    # Release year embedding
    release_year_embedding = np.array([np.mean(all_release_years)]) if all_release_years else np.zeros(1)

    return genre_embedding, artist_embedding, playlist_embedding, release_year_embedding


# In[42]:


def extract_user_id(spotify_url):
    """Extract the Spotify user ID from the URL."""
    match = re.search(r"user/([^?]+)", spotify_url)
    if match:
        return match.group(1)
    return None


# In[43]:


def get_best_matching_users(user_id, user_embeddings_df, transformer_model, top_artist_count, playlists_count, top_k):
    # Fetch user playlists
    playlists = sp.user_playlists(user_id)["items"]
    if not playlists:
        print(f"No playlists found for user {user_id}")
        return None

    # Generate the user's embedding
    genre_embedding, artist_embedding, playlist_embedding, release_year_embedding = generate_user_embedding(
        playlists, transformer_model, top_artist_count, playlists_count
    )

    # Convert to PyTorch tensors
    genre_embedding = torch.tensor(genre_embedding, dtype=torch.float)
    artist_embedding = torch.tensor(artist_embedding, dtype=torch.float)
    playlist_embedding = torch.tensor(playlist_embedding, dtype=torch.float)

    print("User embeddings generated successfully!")

    model.eval()
    top_k_indices = []
    with torch.no_grad():
        query_embedding = model(genre_embedding.unsqueeze(0), artist_embedding.unsqueeze(0), playlist_embedding.unsqueeze(0))
        
        # Compute embeddings for all database entries
        db_genres = torch.stack(user_embeddings_df['genre_embedding'].tolist())
        db_artists = torch.stack(user_embeddings_df['artist_embedding'].tolist())
        db_playlists = torch.stack(user_embeddings_df['playlist_embedding'].tolist())
        db_embeddings = model(db_genres, db_artists, db_playlists)
        
        # Compute similarities
        similarities = torch.nn.functional.cosine_similarity(query_embedding, db_embeddings)
        top_k_indices = torch.topk(similarities, k=top_k+1).indices
        top_k_indices, similarities[top_k_indices]
        scores = similarities[top_k_indices]

        # Find user_id ID in the dataset and remove it from the top_k_indices
        user_index = user_embeddings_df[user_embeddings_df['user_id'] == user_id].index[0]
        index_to_delete = np.where(top_k_indices == user_index)[0][0]
        top_k_indices = np.delete(top_k_indices, index_to_delete)
        scores = np.delete(scores, index_to_delete)

    return user_embeddings_df.iloc[top_k_indices], scores


# In[47]:


def recommend_users(spotify_url, top_artist_count, playlists_count, progress=gr.Progress(track_tqdm=True)):
    user_id = spotify_url
    if not user_id:
        return "Invalid Spotify profile URL."

    try:
        # Step 1: Gathering profile data
        progress(0, desc="Gathering profile data")
        print("Gathering profile data...")
        playlists = sp.user_playlists(user_id)["items"]
        
        if not playlists:
            return f"No playlists found for user {user_id}."

        # Step 2: Computing the best candidates
        progress(0.2, desc="Generating user embeddings")
        print("Generating user embeddings...")

        genre_embedding, artist_embedding, playlist_embedding, release_year_embedding = generate_user_embedding(
            playlists, transformer_model, top_artist_count, playlists_count
        )

        # Add the user's embedding to Hopsworks
        print("Adding the user's embedding to Hopsworks...")
        user_embedding_dict = {
            "user_id": user_id,
            "genre_embedding": genre_embedding.tolist(),
            "artist_embedding": artist_embedding.tolist(),
            "playlist_embedding": playlist_embedding.tolist(),
            "release_year_embedding": release_year_embedding.tolist()
        }
        user_embedding_df = pd.DataFrame([user_embedding_dict])  # Create a DataFrame with a single row

        # Insert into the feature store
        feature_store = project.get_feature_store()
        feature_group = feature_store.get_feature_group(name="spotify_user_embeddings", version=2)
        feature_group.insert(user_embedding_df)
        print(f"User embedding for {user_id} added to Hopsworks successfully.")

        # Get all user embeddings from the database (assuming these are already stored in the feature store)
        user_embeddings_fg = fs.get_feature_group(name="spotify_user_embeddings", version=2)
        user_embeddings_df = user_embeddings_fg.read()

        # Exclude the current user from the dataset
        user_embeddings_df = user_embeddings_df[user_embeddings_df["user_id"] != user_id]
        user_embeddings_df = user_embeddings_fg.read()

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

        # Compute cosine similarity for all users
        top_k = 5
        similar_users, similarity_scores = get_best_matching_users(user_id, user_embeddings_df, transformer_model, top_artist_count, playlists_count, top_k)

        # Finding the top matches
        progress(0.9, desc="Finding matches")
        results = []
        
        j = 0
        for i, row in similar_users.iterrows():
            matched_user_id = row.user_id
            matched_similarity = similarity_scores[j].item()
            print(f"User ID: {matched_user_id}, Similarity: {matched_similarity:.5f}")

            try:
                matched_profile = sp.user(matched_user_id)                
                # Check for profile picture; if none, use default
                profile_pic = matched_profile.get("images", [{}])[0].get("url", "https://upload.wikimedia.org/wikipedia/commons/a/ac/Default_pfp.jpg")  # Default image URL
                display_name = matched_profile.get("display_name", "Unknown User")
                profile_url = matched_profile.get("external_urls", {}).get("spotify", "#")
                
                results.append({
                    "profile_pic": profile_pic,
                    "display_name": display_name,
                    "profile_url": profile_url,
                    "similarity": matched_similarity
                })
            except Exception as e:
                print(f"Error fetching profile: {e}")
            
            j += 1

        # Formatting results
        progress(0.95, desc="Formatting results")
        output = []
        for result in results:
            if result["profile_pic"]:
                img_tag = f'<img src="{result["profile_pic"]}" alt="Profile Picture" width="100" height="100">'
            else:
                img_tag = "<img src='https://upload.wikimedia.org/wikipedia/commons/a/ac/Default_pfp.jpg' alt='Default Profile Picture' width='100' height='100'>"  # Default image
            link_tag = f'<a href="{result["profile_url"]}" target="_blank">{result["display_name"]}</a>'
            similarity_tag = f" (Similarity: {result['similarity']})"
            output.append(f"{img_tag} {link_tag} {similarity_tag}")

        progress(1.0, desc="Complete")
        return "<br>".join(output)

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return f"Error processing request: {str(e)}"


# In[48]:


# Create the Gradio UI
playlists_count = 5
top_artist_count = 10

interface = gr.Interface(
    fn=recommend_users,
    inputs=[
        gr.Textbox(label="Spotify Profile URL"),  # For the user's Spotify URL
        gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Top Artist Count"),  # For top artist count
        gr.Slider(minimum=1, maximum=20, step=1, value=5, label="Playlists Count")  # For playlists count
    ],
    outputs=gr.HTML(label="Recommended Spotify Profiles"),
    title="Spotify Profile Recommender",
    description="Enter your Spotify profile URL to find the most similar users based on your playlists!"
)


# In[49]:


# Launch the app
interface.launch()

