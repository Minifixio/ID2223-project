#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from collections import Counter
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import Model


# In[2]:


# Load credentials
with open('../secrets/hopsworks_api_key.txt', 'r') as file:
    HOPSWORKS_API_KEY = file.readline().strip()

with open('../secrets/spotify_client_id.txt', 'r') as file:
    SPOTIFY_CLIENT_ID = file.readline().strip()

with open('../secrets/spotify_client_secret.txt', 'r') as file:
    SPOTIFY_CLIENT_SECRET = file.readline().strip()


# In[3]:


client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# In[4]:


# Connect to Hopsworks
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()


# In[5]:


# Retrieve the Keras model from the model registry
mr = project.get_model_registry()
model_registry = mr.get_model("two_tower_recommender", version=1)
model_file_path = model_registry.download()
model = tf.keras.models.load_model(
    model_file_path + '/two_tower_model.keras',
    custom_objects={"Model": Model}
)


# In[6]:


transformer_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # This can be replaced if needed


# In[7]:


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

    # print("User embedding generated successfully!")
    # print("Genre embedding shape:", genre_embedding.shape)
    # print("Artist embedding shape:", artist_embedding.shape)
    # print("Playlist embedding shape:", playlist_embedding.shape)
    # print("Release year embedding shape:", release_year_embedding.shape)

    return genre_embedding, artist_embedding, playlist_embedding, release_year_embedding


# In[8]:


def extract_user_id(spotify_url):
    """Extract the Spotify user ID from the URL."""
    match = re.search(r"user/([^?]+)", spotify_url)
    if match:
        return match.group(1)
    return None


# In[ ]:


def recommend_users(spotify_url, top_artist_count, playlists_count, progress=gr.Progress(track_tqdm=True)):
    user_id = spotify_url
    if not user_id:
        return "Invalid Spotify profile URL."

    try:
        # Step 1: Gathering profile data
        progress(0, desc="Gathering profile data")
        print("Gathering profile data...")
        playlists = sp.user_playlists(user_id)["items"]
        print(playlists)
        if not playlists:
            return f"No playlists found for user {user_id}."

        # Step 2: Computing the best candidates
        progress(0.2, desc="Generating user embeddings")
        print("Generating user embeddings...")

        genre_embedding, artist_embedding, playlist_embedding, release_year_embedding = generate_user_embedding(
            playlists, transformer_model, top_artist_count, playlists_count
        )

        user_embedding_concat = np.concatenate([genre_embedding, artist_embedding, playlist_embedding, release_year_embedding])
        user_embedding_normalized = normalize(user_embedding_concat.reshape(1, -1))

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
        all_user_ids = all_user_embeddings["user_id"].tolist()

        # Normalize all embeddings
        normalized_user_embeddings = np.array(all_user_embeddings["normalized_embedding"].tolist())

        # Compute cosine similarity for all users
        similarities = cosine_similarity(user_embedding_normalized, normalized_user_embeddings).flatten()
        print(f"Similarities shape: {similarities.shape}")
        print(similarities)

        # Finding the top matches
        progress(0.9, desc="Finding matches")
        print("Finding the top matches...")
        top_indices = np.argsort(similarities)[::-1][:5]
        results = []

        print("Top matches and matched similarities:")
        print(similarities[top_indices])
        
        for idx in top_indices:
            matched_user_id = all_user_ids[idx]
            matched_similarity = similarities[idx]
            print(f"User ID: {matched_user_id}, Similarity: {matched_similarity:.2f}")

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


# In[128]:


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


# In[ ]:


# Launch the app
interface.launch()

