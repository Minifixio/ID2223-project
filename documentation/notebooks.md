
# Notebooks

### Notebook 1 - Feature Filling
- **Purpose**: Populate the feature store with embeddings from Spotify user profiles.
- **Steps**:
    1. **Retrieve Spotify Data**: Gather user IDs, fetch playlists, tracks, audio features, and artist genres.
    2. **Create Embeddings**: Compute user embeddings using audio features and genre data.
    3. **Store in Feature Store**: Save embeddings in Hopsworks, ensuring metadata and primary keys.

### Notebook 2 - Training Retrieval Model
- **Purpose**: Train a two-tower architecture to retrieve similar user profiles.
- **Steps**:
    1. **Data Preprocessing**: Load embeddings, split into train/validation sets, and create user pairs (positive/negative).
    2. **Model Training**: Train a two-tower model using contrastive/triplet loss to optimize similarity.
    3. **Save Model**: Save the trained model for inference.

### Notebook 3 - Inference Pipeline
- **Purpose**: Recommend similar users for a given Spotify profile.
- **Steps**:
    1. **Retrieve User Data**: Fetch playlists, tracks, audio features, and genres for the input user.
    2. **Compute Embedding**: Generate the user embedding using the same method as Notebook 1.
    3. **Retrieve Matches**: Use the two-tower model to find and rank similar users from the feature store.
    4. **Update Feature Store**: Add the new user embedding for future use.