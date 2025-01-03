
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

### Notebook 4 - Gradio UI for User Recommendations
- **Purpose**: Create an interactive Gradio UI to recommend similar Spotify user profiles based on user input.
- **Steps**:
    1. **User Input**: Allow users to input their Spotify profile URL or ID.
    2. **Embedding Generation**: Generate the user's embedding by applying the same method from Notebook 1.
    3. **Model Inference**: Use the two-tower model (from Notebook 2) to retrieve the most similar users from the feature store.
    4. **Display Results**: Show the top similar users, along with their genres, playlists, and other relevant details.
    5. **Deploy to HuggingFace**: Package the Gradio interface and deploy it to HuggingFace for easy access and use.
