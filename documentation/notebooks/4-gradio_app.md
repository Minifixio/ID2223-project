Here's a README for your Spotify Profile Recommender project in the requested format:
## Spotify Profile Recommender System Gradio UI

#### Overview
This project implements a Spotify Profile Recommender system that identifies users with similar music tastes based on their playlists. It leverages embeddings generated from user playlists and finds the most similar users using cosine similarity.

The system is deployed as an interactive Gradio application, allowing users to input their Spotify profile URL and view recommendations.

---

#### Key Features:
1. **Data Retrieval**:
   - Fetches user playlist data, including track details, genres, artists, and release dates, using the Spotify API.
   - Accesses precomputed Spotify user embeddings from the Hopsworks Feature Store (`spotify_user_embeddings` feature group).

2. **Embedding Generation**:
   - **Per-Playlist Features**: Extracts genre embeddings, average popularity, and explicit content ratios for each playlist.
   - **Overall User Embedding**:
     - Combines genre, artist, playlist, and release year embeddings.
     - Normalizes these embeddings for consistency.
   - Uses the `SentenceTransformer` model (`paraphrase-MiniLM-L6-v2`) to generate textual embeddings.

3. **Recommendation System**:
   - **Cosine Similarity**: Compares user embeddings to all stored embeddings in the feature store.
   - Identifies the top 5 most similar Spotify users.

4. **Interactive User Interface**:
   - Built using Gradio.
   - Accepts Spotify profile URL as input and displays recommended profiles with pictures, names, similarity scores, and links.

---

#### Dependencies:
- **Python Libraries**:
  - `gradio`: For building the interactive web interface.
  - `spotipy`: For interacting with the Spotify API.
  - `tensorflow`: For loading the pre-trained two-tower model.
  - `sentence-transformers`: For generating text embeddings.
  - `hopsworks`: For accessing the feature store and model registry.
  - `pandas` and `numpy`: For data manipulation.
  - `scikit-learn`: For normalization and similarity calculations.

---

#### Setup:
1. **API Keys**:
   - Place your API keys in the `../secrets/` directory:
     - `hopsworks_api_key.txt`: Hopsworks API key.
     - `spotify_client_id.txt`: Spotify Client ID.
     - `spotify_client_secret.txt`: Spotify Client Secret.

2. **Python Environment**:
   Install required libraries:
   ```bash
   pip install gradio spotipy tensorflow sentence-transformers pandas numpy scikit-learn hopsworks
   ```

3. **Hopsworks Feature Store**:
   Ensure that Spotify user embeddings are stored in the `spotify_user_embeddings` feature group.

4. **Spotify API**:
   Ensure your Spotify API credentials have sufficient access to user playlists.

---

#### Usage:
1. **Running the Application**:
   - Execute the script to launch the Gradio interface:
     ```bash
     python app.py
     ```
   - Alternatively, deploy the application on HuggingFace Spaces for easy access.

2. **User Interaction**:
   - Enter a Spotify profile URL.
   - Adjust the sliders to configure:
     - **Top Artist Count**: Number of top artists to consider.
     - **Playlists Count**: Number of playlists to analyze.
   - View the top 5 recommended Spotify profiles with links and similarity scores.

---

#### Outputs:
**Recommended Profiles**:
   - A ranked list of Spotify profiles, including:
     - Display names.
     - Profile pictures.
     - Links to their Spotify profiles.
     - Similarity scores.

