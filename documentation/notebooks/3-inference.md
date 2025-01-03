### README: Spotify User Recommendation System Using Two-Tower Model

#### Overview
This notebook implements a personalized Spotify user recommendation system. It uses a trained two-tower model and Spotify user embeddings to identify the most similar user for a given Spotify user based on their listening preferences.

---

### Key Features:

1. **Integration with Spotify**:
   - Uses Spotify's Web API to fetch playlists, track metadata, and artist information.
   - Extracts features like genres, artists, and release years from user playlists.

2. **Embedding Generation**:
   - Generates contextual embeddings for genres, artists, playlists, and release years using a SentenceTransformer model (`paraphrase-MiniLM-L6-v2`).
   - Combines individual embeddings into a unified user embedding.

3. **Two-Tower Model**:
   - Loads a pre-trained two-tower model from the Hopsworks Model Registry.
   - Uses the model to compute similarity between the target user and other users in the feature store.

4. **Recommendation**:
   - Retrieves user embeddings from the Hopsworks Feature Store.
   - Finds the most similar user to the target user by computing cosine similarity between embeddings.

---

### Dependencies:

#### Libraries:
- `tensorflow`: For loading the trained two-tower model.
- `spotipy`: For interacting with the Spotify Web API.
- `sentence-transformers`: For generating embeddings.
- `hopsworks`: For accessing the feature store and model registry.
- `scikit-learn`: For cosine similarity computation and embedding normalization.
- `numpy`: For numerical operations.
- `collections`: For counting artists and genres.

#### External Requirements:
- **Spotify API Credentials**:
  - Client ID and Client Secret are required to access Spotify's Web API.
  - Place these credentials in `../secrets/spotify_client_id.txt` and `../secrets/spotify_client_secret.txt`.
- **Hopsworks API Key**:
  - Required for accessing the Hopsworks Feature Store and Model Registry.
  - Store the API key in `../secrets/hopsworks_api_key.txt`.

---

### Setup:

1. **Install Required Libraries**:
   ```bash
   pip install tensorflow spotipy sentence-transformers hopsworks scikit-learn numpy
   ```

2. **Set Up API Keys**:
   - Save your Spotify Client ID and Secret in `../secrets/spotify_client_id.txt` and `../secrets/spotify_client_secret.txt`.
   - Save your Hopsworks API key in `../secrets/hopsworks_api_key.txt`.

3. **Ensure Feature Store Availability**:
   - The Spotify user embeddings feature group (`spotify_user_embeddings`, version 2) must exist in the Hopsworks Feature Store.

4. **Pre-trained Two-Tower Model**:
   - The notebook retrieves the two-tower model (`two_tower_recommender`, version 1) from the Hopsworks Model Registry.

---

### How It Works:

1. **User Embedding Generation**:
   - Fetches playlists for a target user using Spotify API.
   - Processes playlist metadata to extract genres, artists, and release years.
   - Generates embeddings using the SentenceTransformer model.
   - Combines embeddings (genre, artist, playlist, release year) into a single user embedding.

2. **Similarity Computation**:
   - Retrieves all user embeddings from the feature store.
   - Normalizes all embeddings for cosine similarity computation.
   - Computes similarity between the target user and all other users.

3. **Best Match Identification**:
   - Finds the user with the highest cosine similarity to the target user.
   - Returns the most similar user's ID and the similarity score.

---

### Example Usage:

#### Inputs:
- **User ID**: The target Spotify user's ID (e.g., `minifixiowow`).
- **Top Artist Count**: Number of top artists to consider for embedding (default: `5`).
- **Playlists Count**: Number of playlists to process for the target user (default: `5`).

#### Outputs:
- The most similar user's Spotify ID.
- The similarity score.

```python
user_id = "minifixiowow"
top_artist_count = 5
playlists_count = 5

transformer_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

best_match_user_id, similarity_score = get_best_matching_user(
    user_id, transformer_model, top_artist_count, playlists_count
)

print(f"The best match for user {user_id} is user {best_match_user_id} with a similarity score of {similarity_score}")
```