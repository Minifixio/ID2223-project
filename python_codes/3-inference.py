#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import tensorflow as tf
import hopsworks
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pickle
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import time
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import Model


# In[64]:


# Load credentials
with open('../secrets/hopsworks_api_key.txt', 'r') as file:
    HOPSWORKS_API_KEY = file.readline().strip()

with open('../secrets/spotify_client_id.txt', 'r') as file:
    SPOTIFY_CLIENT_ID = file.readline().strip()

with open('../secrets/spotify_client_secret.txt', 'r') as file:
    SPOTIFY_CLIENT_SECRET = file.readline().strip()


# In[65]:


client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# In[66]:


# Connect to the project and feature store
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()


# In[67]:


# Get the model registry
mr = project.get_model_registry()

# Retrieve the Keras model from the model registry
model_registry = mr.get_model("two_tower_recommender", version=1)
model_file_path = model_registry.download()
model = tf.keras.models.load_model(
    model_file_path + '/two_tower_model.keras',
    custom_objects={"Model": Model}
)

print("Model loaded successfully!")


# In[68]:


def get_embeddings(genres, artists, model):
    """
    Generate embeddings for genres and artists using a SentenceTransformer model.
    """
    # Combine genres and artists into a single list for embedding
    inputs = genres + artists
    
    # Generate embeddings
    embeddings = model.encode(inputs, show_progress_bar=True)
    
    # Split the embeddings back into genres and artists
    genre_embeddings = embeddings[:len(genres)]
    artist_embeddings = embeddings[len(genres):]
    
    return genre_embeddings, artist_embeddings


# In[69]:


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

    print("User embedding generated successfully!")
    print("Genre embedding shape:", genre_embedding.shape)
    print("Artist embedding shape:", artist_embedding.shape)
    print("Playlist embedding shape:", playlist_embedding.shape)
    print("Release year embedding shape:", release_year_embedding.shape)

    # Return individual embeddings
    return genre_embedding, artist_embedding, playlist_embedding, release_year_embedding


# In[ ]:


# Inference function
def get_best_matching_user(user_id, transformer_model, top_artist_count, playlists_count):
    # Fetch user playlists
    playlists = sp.user_playlists(user_id)["items"]
    if not playlists:
        print(f"No playlists found for user {user_id}")
        return None

    # Generate the user's embedding
    genre_embedding, artist_embedding, playlist_embedding, release_year_embedding = generate_user_embedding(
        playlists, transformer_model, top_artist_count, playlists_count
    )

    print("User embeddings generated successfully!")

    # Concatenate all embeddings into a single vector
    user_embedding = np.concatenate([genre_embedding, artist_embedding, playlist_embedding, release_year_embedding])

    # Get all user embeddings from the database (assuming these are already stored in the feature store)
    user_embeddings_fg = fs.get_feature_group(name="spotify_user_embeddings", version=2)
    all_user_embeddings = user_embeddings_fg.read()

    # Exclude the current user from the dataset
    all_user_embeddings = all_user_embeddings[all_user_embeddings["user_id"] != user_id]

    all_user_embeddings['full_embedding'] = all_user_embeddings.apply(
        lambda row: np.concatenate(
            [row['genre_embedding'], row['artist_embedding'], row['playlist_embedding'], row['release_year_embedding']]
        ),
        axis=1
    )
    normalized_embeddings = normalize(np.array(all_user_embeddings['full_embedding'].tolist()))
    all_user_embeddings['normalized_embedding'] = normalized_embeddings.tolist()

    # Normalize all embeddings
    normalized_user_embeddings = np.array(all_user_embeddings["normalized_embedding"].tolist())
    user_embedding_normalized = normalize(user_embedding.reshape(1, -1))

    # Compute cosine similarity for all users
    similarities = cosine_similarity(user_embedding_normalized, normalized_user_embeddings).flatten()

    # Get the index of the most similar user
    best_match_index = np.argmax(similarities)
    best_match_user_id = all_user_embeddings.iloc[best_match_index]["user_id"]

    return best_match_user_id, similarities[best_match_index]


# In[53]:


user_id = "minifixiowow"
top_artist_count = 5
playlists_count = 5

transformer_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # You can replace this with another model if needed

best_match_user_id, similarity_score = get_best_matching_user(user_id, transformer_model, top_artist_count, playlists_count)

print(f"The best match for user {user_id} is user {best_match_user_id} with a similarity score of {similarity_score}")

