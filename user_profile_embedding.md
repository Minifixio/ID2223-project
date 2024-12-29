# User Profile Embedding

Gathering all the tracks of all the playlists of the user and analysing them according to audio tracks features & song's artist genre to create the embedding.

### **Steps for Creating User Embeddings**
1. **Gather Data from User Playlists**:
    - Retrieve all public playlists of the user using the Spotify API.
    - Extract the list of tracks from all playlists (ensure there are no duplicates).

2. **Extract Audio Features**:
    - Use the Spotify API's `audio-features` endpoint to fetch the audio features for all tracks.
    - Features to extract (as provided by Spotify):
        - **Danceability**
        - **Energy**
        - **Key**
        - **Loudness**
        - **Speechiness**
        - **Acousticness**
        - **Instrumentalness**
        - **Valence**
        - **Tempo**
        - **Duration**
        
3. **Extract Genre Information**:
    - Use the Spotify API's `artists` endpoint to get artist metadata for each track.
    - Retrieve the **genres** associated with each artist.
    - Note that a track may have multiple genres through its associated artist(s).
    
4. **Feature Aggregation for Tracks**:
    - Aggregate audio features for all tracks. Common aggregation methods include:
        - **Mean**: Average values of features like danceability, energy, valence, etc., to represent the user's overall musical taste.
        - **Standard Deviation**: Adds insight into the diversity of preferences (e.g., whether they listen to both high- and low-energy tracks).
    - For genre data:
        - Create a **genre frequency vector**: Count the frequency of each genre across all tracks, normalized to sum to 1 (to account for users with different numbers of tracks).
        - Use a predefined genre vocabulary (from Spotify or external data) to ensure uniformity across users.
5. **Combine Audio Features and Genre Data**:
    - Concatenate the aggregated audio features (vector of fixed length, e.g., 10 dimensions for Spotify's audio features) with the genre frequency vector (size based on the genre vocabulary, e.g., 200 genres).
    - Resulting user embedding might look like this:
        
6. **Normalize and Store Embeddings**:
    - Normalize the embedding vector (e.g., using MinMax scaling or Z-score normalization).
    - Store the embedding in your feature store (e.g., Hopsworks) for retrieval and comparison.