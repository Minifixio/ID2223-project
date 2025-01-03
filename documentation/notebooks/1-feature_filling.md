## 1. Spotify User Embeddings Generation

#### Overview
This script generates embeddings for Spotify users based on their playlists. These embeddings are numerical representations of user preferences derived from playlist genres, artists, track popularity, and release years. The embeddings are stored in a feature store for further use in machine learning models or recommendation systems.

---

#### Key Features:
1. **Playlist Data Extraction**:
   - Fetches playlists for multiple Spotify users using the Spotify API.
   - Extracts detailed track-level and artist-level information.

2. **Embedding Generation**:
   - **Genre Embeddings**: Encodes playlist genres into numerical vectors using the `SentenceTransformer` model (`paraphrase-MiniLM-L6-v2`).
   - **Artist Embeddings**: Identifies top artists and encodes their names similarly.
   - **Playlist Embeddings**: Aggregates per-playlist embeddings by averaging genre embeddings.
   - **Release Year Embedding**: Encodes average release year of tracks.

3. **Feature Storage**:
   - Stores user embeddings in a Hopsworks Feature Store for easy access and integration with machine learning pipelines.

---

#### Dependencies:
- **Python Libraries**:
  - `spotipy`: For interacting with the Spotify API.
  - `sentence-transformers`: For embedding generation.
  - `hopsworks`: For storing features.
  - `pandas` and `numpy`: For data manipulation.
  - `datasets`: For loading Spotify user IDs dataset.

---

#### Setup:
1. **API Keys**:
   - Place your Spotify and Hopsworks API keys in `../secrets/spotify_client_id.txt`, `../secrets/spotify_client_secret.txt`, and `../secrets/hopsworks_api_key.txt`.

2. **Python Environment**:
   Install required libraries:
   ```bash
   pip install spotipy sentence-transformers hopsworks datasets pandas numpy scikit-learn gensim
   ```

3. **Dataset**:
   - Uses the `erenfazlioglu/spotifyuserids` dataset from the Hugging Face `datasets` library.

---

#### Usage:
1. Set the parameters:
   - `profiles_count`: Number of user profiles to process.
   - `top_artist_count`: Number of top artists to consider for embeddings.
   - `playlists_count`: Number of playlists to include per user.

2. Run the script:
   ```bash
   python script_name.py
   ```

3. Embeddings will be stored in the Hopsworks Feature Store under the feature group `spotify_user_embeddings`.

---

#### Outputs:
- A DataFrame of user embeddings with columns:
  - `user_id`: Spotify user ID.
  - `genre_embedding`: Averaged embedding of playlist genres.
  - `artist_embedding`: Averaged embedding of top artists.
  - `playlist_embedding`: Weighted embedding based on playlist sizes.
  - `release_year_embedding`: Encoded release year of tracks.

- These embeddings can be accessed and used for tasks like recommendation systems, clustering, or user preference analysis.

