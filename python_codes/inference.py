import numpy as np
import tensorflow as tf
import hopsworks
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pickle
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Load credentials
with open('../secrets/hopsworks_api_key.txt', 'r') as file:
    HOPSWORKS_API_KEY = file.readline().strip()

with open('../secrets/spotify_client_id.txt', 'r') as file:
    SPOTIFY_CLIENT_ID = file.readline().strip()

with open('../secrets/spotify_client_secret.txt', 'r') as file:
    SPOTIFY_CLIENT_SECRET = file.readline().strip()

client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Connect to the project and feature store
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# Get the model registry
mr = project.get_model_registry()

# Retrieve the genre embedding model
genre_model_registry = mr.get("genre_embedding_model")
genre_model_file_path = genre_model_registry.download()
with open(genre_model_file_path, "rb") as f:
    genre_embedding_model = pickle.load(f)

# Retrieve the artist embedding model
artist_model_registry = mr.get("artist_embedding_model")
artist_model_file_path = artist_model_registry.download()
with open(artist_model_file_path, "rb") as f:
    artist_embedding_model = pickle.load(f)

# Retrieve the Keras model from the model registry
model_registry = mr.get("two_tower_recommender")
model_file_path = model_registry.download()
model = tf.keras.models.load_model(model_file_path)

print("Models loaded successfully!")

# Load the trained two-tower model
# model = tf.keras.models.load_model('two_tower_model.keras')

# Define function to generate user embedding (same logic as in training)
def generate_user_embedding(user_playlists, genre_embedding_model, artist_embedding_model, top_artist_count, playlists_count):
    print("Generating user embedding...")
    all_genres = []
    all_artists = []
    all_release_years = []
    playlist_features = []

    for playlist in user_playlists[:playlists_count]:  # Limit to first playlists_count playlists
        print(f"Processing playlist: {playlist['name']}")
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
            print(f"Processing track: {item['track']['name']}")
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

# Inference function
def get_best_matching_user(user_id, genre_embedding_model, artist_embedding_model, top_artist_count, playlists_count):
    # Fetch user playlists
    playlists = sp.user_playlists(user_id)["items"]
    if not playlists:
        print(f"No playlists found for user {user_id}")
        return None

    # Generate the user's embedding
    genre_embedding, artist_embedding, playlist_embedding, release_year_embedding = generate_user_embedding(
        playlists, genre_embedding_model, artist_embedding_model, top_artist_count, playlists_count
    )

    print("User embeddings generated successfully!")

    # Concatenate all embeddings into a single vector
    user_embedding = np.concatenate([genre_embedding, artist_embedding, playlist_embedding, release_year_embedding])

    # Get all user embeddings from the database (assuming these are already stored in the feature store)
    user_embeddings_fg = fs.get_feature_group(name="spotify_user_embeddings", version=1)
    all_user_embeddings = user_embeddings_fg.read()

    # Normalize all embeddings
    normalized_user_embeddings = np.array(all_user_embeddings["normalized_embedding"].tolist())
    user_embedding_normalized = normalize(user_embedding.reshape(1, -1))

    # Compute cosine similarity for all users
    similarities = cosine_similarity(user_embedding_normalized, normalized_user_embeddings).flatten()

    # Get the index of the most similar user
    best_match_index = np.argmax(similarities)
    best_match_user_id = all_user_embeddings.iloc[best_match_index]["user_id"]

    return best_match_user_id, similarities[best_match_index]

# Example usage
user_id = "minifixiowow"
top_artist_count = 5
playlists_count = 10

best_match_user_id, similarity_score = get_best_matching_user(user_id, genre_embedding_model, artist_embedding_model, top_artist_count, playlists_count)

print(f"The best match for user {user_id} is user {best_match_user_id} with a similarity score of {similarity_score}")