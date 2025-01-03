#!/usr/bin/env python
# coding: utf-8

# In[5]:


import gradio as gr
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import hopsworks
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle


# In[6]:


# Load credentials
with open('../secrets/hopsworks_api_key.txt', 'r') as file:
    HOPSWORKS_API_KEY = file.readline().strip()

with open('../secrets/spotify_client_id.txt', 'r') as file:
    SPOTIFY_CLIENT_ID = file.readline().strip()

with open('../secrets/spotify_client_secret.txt', 'r') as file:
    SPOTIFY_CLIENT_SECRET = file.readline().strip()

client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Connect to Hopsworks
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()
mr = project.get_model_registry()

# Retrieve the genre embedding model
genre_model_registry = mr.get_model("genre_embedding_model", version=1)
genre_model_file_path = genre_model_registry.download()
with open(genre_model_file_path + '/genre_embedding_model.pkl', "rb") as f:
    genre_embedding_model = pickle.load(f)

# Retrieve the artist embedding model
artist_model_registry = mr.get_model("artist_embedding_model", version=1)
artist_model_file_path = artist_model_registry.download()
with open(artist_model_file_path + '/artist_embedding_model.pkl', "rb") as f:
    artist_embedding_model = pickle.load(f)

# Retrieve the Keras model from the model registry
model_registry = mr.get_model("two_tower_recommender", version=1)
model_file_path = model_registry.download()
model = tf.keras.models.load_model(model_file_path + '/two_tower_model.keras')


# In[7]:


def extract_user_id(spotify_url):
    """Extract the Spotify user ID from the URL."""
    match = re.search(r"user/([^?]+)", spotify_url)
    if match:
        return match.group(1)
    return None

def generate_user_embedding(user_playlists, genre_embedding_model, artist_embedding_model, top_artist_count=5, playlists_count=10):
    """Generate the embedding for a Spotify user."""
    all_genres, all_artists = [], []
    for playlist in user_playlists[:playlists_count]:
        tracks = sp.playlist_tracks(playlist["id"])["items"]
        for item in tracks:
            track = item["track"]
            if track:
                artist_id = track["artists"][0]["id"]
                artist = sp.artist(artist_id)
                genres = artist.get("genres", [])
                all_genres.extend(genres)
                all_artists.append(artist.get("name", "Unknown"))

    artist_embedding = np.mean(
        [artist_embedding_model.wv[artist] for artist in all_artists if artist in artist_embedding_model.wv],
        axis=0
    ) if all_artists else np.zeros(100)

    genre_embedding = np.mean(
        [genre_embedding_model.wv[genre] for genre in all_genres if genre in genre_embedding_model.wv],
        axis=0
    ) if all_genres else np.zeros(100)

    return np.concatenate([genre_embedding, artist_embedding])

def recommend_users(spotify_url):
    user_id = extract_user_id(spotify_url)
    if not user_id:
        return "Invalid Spotify profile URL."

    # Fetch user playlists
    try:
        playlists = sp.user_playlists(user_id)["items"]
        if not playlists:
            return f"No playlists found for user {user_id}."
    except Exception as e:
        return f"Error fetching playlists for user {user_id}: {e}"

    # Generate user embedding
    user_embedding = generate_user_embedding(playlists, genre_embedding_model, artist_embedding_model)
    user_embedding_normalized = normalize(user_embedding.reshape(1, -1))

    # Insert into the feature store
    user_embeddings_fg = fs.get_or_create_feature_group(
        name="spotify_user_embeddings",
        version=1,
        primary_key=["user_id"],
        description="Spotify user embeddings"
    )
    user_embeddings_fg.insert(pd.DataFrame([{
        "user_id": user_id,
        "normalized_embedding": user_embedding.tolist()
    }]))

    # Retrieve all user embeddings
    all_user_embeddings = user_embeddings_fg.read()
    all_embeddings = np.array(all_user_embeddings["normalized_embedding"].tolist())
    all_user_ids = all_user_embeddings["user_id"].tolist()

    # Compute similarity scores
    similarities = cosine_similarity(user_embedding_normalized, normalize(all_embeddings)).flatten()

    # Find the top matches
    top_indices = np.argsort(similarities)[::-1][:5]
    results = []
    for idx in top_indices:
        matched_user_id = all_user_ids[idx]
        matched_similarity = similarities[idx]
        try:
            matched_profile = sp.user(matched_user_id)
            profile_pic = matched_profile.get("images", [{}])[0].get("url", None)
            display_name = matched_profile.get("display_name", "Unknown User")
            profile_url = matched_profile.get("external_urls", {}).get("spotify", "#")
            results.append({
                "profile_pic": profile_pic,
                "display_name": display_name,
                "profile_url": profile_url,
                "similarity": matched_similarity
            })
        except Exception as e:
            results.append({
                "profile_pic": None,
                "display_name": f"Error fetching profile: {e}",
                "profile_url": "#",
                "similarity": matched_similarity
            })

    # Format results
    output = []
    for result in results:
        if result["profile_pic"]:
            img_tag = f'<img src="{result["profile_pic"]}" alt="Profile Picture" width="100" height="100">'
        else:
            img_tag = ""
        link_tag = f'<a href="{result["profile_url"]}" target="_blank">{result["display_name"]}</a>'
        similarity_tag = f" (Similarity: {result['similarity']:.2f})"
        output.append(f"{img_tag} {link_tag} {similarity_tag}")

    return "<br>".join(output)


# In[8]:


# Create the Gradio UI
interface = gr.Interface(
    fn=recommend_users,
    inputs=gr.Textbox(label="Spotify Profile URL"),
    outputs=gr.HTML(label="Recommended Spotify Profiles"),
    title="Spotify Profile Recommender",
    description="Enter your Spotify profile URL to find the most similar users based on your playlists!"
)

# Launch the app
interface.launch()

