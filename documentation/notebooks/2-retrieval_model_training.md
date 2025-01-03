## 2. Two-Tower Recommender System for Spotify User Similarity

#### Overview
This notebook implements a two-tower neural network architecture to compute similarity between Spotify users based on their embeddings. It trains a model to predict whether two users are similar by using embeddings extracted from their playlist preferences.

---

#### Key Features:
1. **Data Retrieval**:
   - Loads Spotify user embeddings from the Hopsworks Feature Store (`spotify_user_embeddings` feature group).
   - Normalizes the embeddings for improved similarity calculations.

2. **Embedding Concatenation**:
   - Combines individual user embeddings (genre, artist, playlist, and release year) into a unified vector for each user.
   - Normalizes these concatenated embeddings for further processing.

3. **Two-Tower Architecture**:
   - **User Tower**: Maps a user's embedding to a latent space.
   - **Candidate Tower**: Maps another user's embedding to the same latent space.
   - **Cosine Similarity**: Computes similarity between the outputs of the two towers.

4. **Training**:
   - Generates training pairs:
     - Positive pairs: Users with high cosine similarity.
     - Negative pairs: Users with low cosine similarity.
   - Trains the model using these pairs to predict whether two users are similar.

5. **Model Storage**:
   - Saves the trained model in the Hopsworks Model Registry for versioning and deployment.

---

#### Dependencies:
- **Python Libraries**:
  - `tensorflow`: For building and training the two-tower architecture.
  - `hopsworks`: For accessing the feature store and model registry.
  - `scikit-learn`: For cosine similarity computation.
  - `numpy`: For array manipulation.

---

#### Setup:
1. **API Key**:
   - Place your Hopsworks API key in `../secrets/hopsworks_api_key.txt`.

2. **Python Environment**:
   Install required libraries:
   ```bash
   pip install tensorflow hopsworks scikit-learn numpy
   ```

3. **Feature Store**:
   Ensure that user embeddings are available in the `spotify_user_embeddings` feature group within the Hopsworks Feature Store.

---

#### Usage:
1. **Model Training**:
   - Adjust parameters such as `similarity_threshold` and `negative_ratio` in the `generate_pairs` function to control pair generation.
   - Run the notebook to train the model with the default configuration:
     - Batch size: `32`
     - Epochs: `10`
     - Validation split: `0.2`

2. **Model Saving**:
   - The trained model is saved as `two_tower_model.keras` and uploaded to the Hopsworks Model Registry.

---

#### Outputs:
1. **Trained Model**:
   - A TensorFlow two-tower model saved locally and in the Hopsworks Model Registry under the name `two_tower_recommender`.

2. **Metrics**:
   - Logs training and validation accuracy during training.
   - Stores the final accuracy in the Model Registry for future reference.

