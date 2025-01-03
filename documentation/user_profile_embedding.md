# User Profile Embedding

This process involves creating a user embedding by analyzing their playlists, focusing on track-level features and the artist's genre. Due to recent deprecation of access to Spotify's audio features, alternative features are used to generate a comprehensive user profile.

### **Steps for Creating User Embeddings**
1. **Gather Data from User Playlists**:
    - Use the Spotify API to retrieve all public playlists of the user.
    - Extract the tracks from each playlist, ensuring no duplicates.

2. **Extract Features from Track Artists**:
    - Since Spotify has deprecated audio feature access, we rely on alternative features:
        - **Genres**: Extracted from the artist of each track, encoded using a SentenceTransformer, and averaged across all playlists.
        - **Artists**: Identify the top `N` most frequent artists and encode them, then average their embeddings.
        - **Release Years**: Calculate the average release year of tracks in the playlists.

3. **Playlist-Level Features**:
    - **Number of Tracks**: Total number of tracks in the user's playlists.
    - **Average Popularity**: Average popularity score for tracks.
    - **Explicitness Ratio**: Ratio of explicit tracks in the playlists.

4. **Playlist Embedding**:
    - Aggregate the genre embeddings for each playlist, weighted by the number of tracks in each playlist.

5. **Combine All Features**:
    - **Final Embedding**: Combine the following features into a single user embedding:
        - **Genre Embedding**
        - **Artist Embedding**
        - **Playlist Embedding**
        - **Release Year**

6. **Normalize and Store Embedding**:
    - Normalize the final embedding (using techniques like MinMax scaling or Z-score normalization).
    - Store the resulting embedding in the feature store (e.g., Hopsworks) for future retrieval and comparison.

By using these features, we create a user embedding that represents their musical preferences and habits without relying on deprecated audio features.