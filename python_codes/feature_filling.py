import os
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec
from datasets import load_dataset
import hopsworks
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

SPOTIFY_CLIENT_ID = "b810c6ca8b2743daa8d3a3e355b7299b"
SPOTIFY_CLIENT_SECRET = "b569748b5ff041f097402a603428fb3d"
SPOTIFY_BASE_URL = "https://api.spotify.com/v1/"
HOPSWORKS_API_KEY = "Tm6d6Kium30I2GOm.KmRBKltxfV2ETd9rgONdjvDCsnMQ8ENTyWTCUghbILYycU4fcKhYTC7VB9nx9PQ0"

client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Normalize release dates to a range [0, 1]
def normalize_dates(release_dates):
    if not release_dates:
        return 0.0
    dates = [datetime.strptime(date, "%Y-%m-%d").timestamp() for date in release_dates if date]
    return np.mean(dates) / datetime.now().timestamp()

# Generate embedding for a user's profile
def generate_user_embedding(user_playlists, genre_embedding_model, artist_embedding_model, top_artist_count):
    all_genres = []
    all_artists = []
    all_release_years = []
    playlist_features = []

    for playlist in user_playlists[:10]:  # Limit to first 10 playlists
        playlist_name = playlist.get("name", "Unknown")
        playlist_id = playlist["id"]

        # Fetch tracks in the playlist
        tracks = sp.playlist_tracks(playlist_id)["items"]

        genres = []
        popularity = []
        release_years = []
        explicit_flags = []
        artist_names = []

        for item in tracks:
            track = item["track"]
            if not track:
                continue

            # Extract artist information
            artist_id = track["artists"][0]["id"]
            artist = sp.artist(artist_id)
            track_genres = artist.get("genres", [])
            artist_name = artist.get("name", "Unknown")

            # Append track features
            genres.extend(track_genres)
            artist_names.append(artist_name)
            release_date = track["album"]["release_date"]
            
            # Extract year from release date
            release_year = release_date.split('-')[0]
            release_years.append(int(release_year))

            popularity.append(track.get("popularity", 0))
            explicit_flags.append(track.get("explicit", False))

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

    # Top artist embedding
    artist_counter = Counter(all_artists)
    top_artists = [artist for artist, _ in artist_counter.most_common(top_artist_count)]
    artist_embedding = np.mean(
        [artist_embedding_model.wv[artist] for artist in top_artists if artist in artist_embedding_model.wv],
        axis=0
    ) if top_artists else np.zeros(100)

    # Genre embedding
    genre_vectors = [
        genre_embedding_model.wv[genre] for genre in all_genres if genre in genre_embedding_model.wv
    ]
    genre_embedding = np.mean(genre_vectors, axis=0) if genre_vectors else np.zeros(100)

    # Aggregated playlist embedding
    playlist_sizes = [p["num_tracks"] for p in playlist_features]
    playlist_weights = normalize(np.array(playlist_sizes).reshape(1, -1))[0]
    playlist_embedding = np.sum([playlist_weights[i] * genre_embedding for i in range(len(playlist_features))], axis=0)

    # Release year embedding
    release_year_embedding = np.array([np.mean(all_release_years)])

    # Return individual embeddings
    return genre_embedding, artist_embedding, playlist_embedding, release_year_embedding

profiles_count = 10  # Number of profiles to process
top_artist_count = 5  # Number of top artists to embed

# Load dataset
dataset = load_dataset("erenfazlioglu/spotifyuserids")
rows = dataset["train"][:profiles_count]

# Train custom Word2Vec model for genres and artists
corpus = []
for spotify_id in rows["spotify_id"]:
    playlists = sp.user_playlists(spotify_id)["items"]
    for playlist in playlists:
        tracks = sp.playlist_tracks(playlist["id"])["items"]
        for item in tracks:
            track = item["track"]
            if not track:
                continue
            artist_id = track["artists"][0]["id"]
            artist = sp.artist(artist_id)
            track_genres = artist.get("genres", [])
            artist_name = artist.get("name", "Unknown")
            corpus.append(track_genres + [artist_name])

# Train Word2Vec model
genre_embedding_model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
artist_embedding_model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

# Collect embeddings in a DataFrame
embeddings = []
for spotify_id in rows["spotify_id"]:
    try:
        # Fetch user playlists
        playlists = sp.user_playlists(spotify_id)["items"]
        if not playlists:
            print(f"No playlists found for user {spotify_id}")
            continue

        # Generate individual embeddings
        genre_embedding, artist_embedding, playlist_embedding, release_year_embedding = generate_user_embedding(
            playlists, genre_embedding_model, artist_embedding_model, top_artist_count
        )

        # Append embeddings to list as a dictionary
        embeddings.append({
            "user_id": spotify_id,
            "genre_embedding": genre_embedding.tolist(),
            "artist_embedding": artist_embedding.tolist(),
            "playlist_embedding": playlist_embedding.tolist(),
            "release_year_embedding": release_year_embedding.tolist()
        })

    except Exception as e:
        print(f"Error processing user {spotify_id}: {e}")

# Create a DataFrame from the embeddings
df_embeddings = pd.DataFrame(embeddings)
print(f"Embeddings shape: {df_embeddings.shape}")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

feature_store = project.get_feature_store()
feature_group = feature_store.get_or_create_feature_group(
    name="spotify_user_embeddings",
    version=1,
    primary_key=["user_id"],
    description="Spotify user embeddings based on playlists"
)

feature_group.insert(df_embeddings)
