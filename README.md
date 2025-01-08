# Project Overview: Dynamic Spotify User Recommender System

This project demonstrates a **dynamic recommendation system** for Spotify users, built using a **two-tower deep learning architecture**, **Hopsworks Feature Store**, and an interactive **Gradio web application**. It incorporates various advanced machine learning and data engineering techniques to deliver personalized user recommendations, while continuously updating the recommendation model and database with new data. Below is an overview of the project's key components, workflows, and code snippets.

## Project architecture diagram
[![Project Architecture](./documentation/project-architecture-v2.png)](./documentation/project-architecture-v2.png)

## Project parts

### 1. **Dataset and Feature Store Integration**

The system is powered by a **dataset of Spotify profile IDs**, sourced from **Hugging Face**. This dataset contains many different users' profile data, which is processed to generate user embeddings. These embeddings are stored in the **Feature Store** of **Hopsworks**, a centralized data storage and feature management system that serves as the foundation of our recommendation engine. The user embeddings include data such as user preferences across genres, artists, playlists, and release years.

**Code Example:** In the model retrieval notebook, the following code fetches the user embeddings from the feature store:

```python
user_embeddings_fg = fs.get_feature_group(
    name='spotify_user_embeddings',
    version=2,
)

user_embeddings_df = user_embeddings_fg.read()
print(f"A total of {len(user_embeddings_df)} user embeddings are available.")
user_embeddings_df.head()
```

This step retrieves the pre-processed embeddings, which are later used for training and recommendations.


### 2. **Two-Tower Architecture for Model Training and Retrieval**

The heart of this project is the **two-tower architecture**, a widely-used model for recommendation systems. The architecture involves two separate towers (models):

- One tower represents the **user profile**.
- The other tower represents the **candidate users** to recommend.

The output of both towers is compared using **cosine similarity** to find the most similar users. The two towers are connected via a **cosine similarity layer** that computes the similarity score between user embeddings.

**Code Example:** The user and candidate towers are built using the Keras library in TensorFlow. Below is an example of how the two towers are constructed:

```python
def build_user_tower(input_dim, embedding_dim=128):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    user_embedding = layers.Dense(embedding_dim, activation=None)(x)  # Final user embedding
    return Model(inputs, user_embedding, name="UserTower")

def build_candidate_tower(input_dim, embedding_dim=128):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    candidate_embedding = layers.Dense(embedding_dim, activation=None)(x)  # Final candidate embedding
    return Model(inputs, candidate_embedding, name="CandidateTower")
```

This structure allows the model to process embeddings for users and candidates independently before comparing them through similarity metrics.


### 3. **Automated Retraining and Continuous Model Updates**

To ensure the model adapts to new user data, it is designed to be **retrained automatically every week** using **GitHub Actions**. This automated retraining pipeline ensures that the system remains up-to-date with the latest user embeddings and improves over time based on new interactions.

For this, the **model training notebook** generates pairs of embeddings and labels, which are then used to train the two-tower model. The training is done periodically to keep the model updated with the latest data.

**Code Example:** Here’s a portion of the code where we generate pairs of user embeddings to train the model:

```python
def generate_pairs(embeddings, similarity_threshold=0.8, negative_ratio=1):
    pairs = []
    labels = []

    # Compute cosine similarity for all pairs
    similarity_matrix = cosine_similarity(embeddings)  # This is a valid pairwise similarity matrix

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if similarity_matrix[i, j] > similarity_threshold:
                # Positive pair
                pairs.append((embeddings[i], embeddings[j]))
                labels.append(1)

                # Generate negative pairs
                for _ in range(negative_ratio):
                    negative_index = np.random.choice(len(embeddings))
                    while negative_index == i or negative_index == j:
                        negative_index = np.random.choice(len(embeddings))
                    pairs.append((embeddings[i], embeddings[negative_index]))
                    labels.append(0)
    
    return np.array(pairs), np.array(labels)
```

These pairs are used for **training the model**, where the goal is to minimize the loss based on the cosine similarity between pairs of embeddings.


### 4. **Dynamic User Integration into the Feature Store**

One of the most crucial features of this project is its **dynamic nature**. When a user performs a recommendation search via the **Gradio app**, their profile is not only used for generating recommendations, but it is also **added to the Feature Store**. This ensures that the system continuously improves as more data is gathered from user interactions, making the database dynamic and allowing for more accurate and personalized recommendations.

When a new user profile is added, the system performs the following steps:

1. It generates embeddings for the user's profile using their playlist data.
2. It normalizes and stores the user's embedding in the Feature Store.

**Code Example:** Here’s how the user embedding is generated and added to the Feature Store:

```python
def generate_user_embedding(user_playlists, transformer_model, top_artist_count, playlists_count):
    # Embedding generation code here...
    user_embedding_dict = {
        "user_id": user_id,
        "genre_embedding": genre_embedding.tolist(),
        "artist_embedding": artist_embedding.tolist(),
        "playlist_embedding": playlist_embedding.tolist(),
        "release_year_embedding": release_year_embedding.tolist()
    }
    user_embedding_df = pd.DataFrame([user_embedding_dict])  # Create a DataFrame with a single row

    # Insert into the feature store
    feature_store = project.get_feature_store()
    feature_group = feature_store.get_feature_group(name="spotify_user_embeddings", version=2)
    feature_group.insert(user_embedding_df)
    print(f"User embedding for {user_id} added to Hopsworks successfully.")
```

This functionality allows the system to grow and adapt as users continue interacting with the platform.


### 5. **Gradio Web App for User Interaction**

The **Gradio web interface** is where users interact with the recommendation system. Users input their **Spotify profile URL**, and the system computes recommendations based on their **playlists, genres, artists**, and other features. The recommendations are then displayed on the web interface, showing the **most similar users** along with their profile pictures, names, and similarity scores.

The app is deployed on **Hugging Face Spaces**, and users can access it directly via the following link:  
[Spotify Profile Recommender - Gradio App](https://huggingface.co/spaces/minifixio/ID2223-final-project)

**Code Example:** Here’s the Gradio interface setup that allows users to input their data and get recommendations:

```python
interface = gr.Interface(
    fn=recommend_users,
    inputs=[
        gr.Textbox(label="Spotify Profile URL"),  # For the user's Spotify URL
        gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Top Artist Count"),  # For top artist count
        gr.Slider(minimum=1, maximum=20, step=1, value=5, label="Playlists Count")  # For playlists count
    ],
    outputs=gr.HTML(label="Recommended Spotify Profiles"),
    title="Spotify Profile Recommender",
    description="Enter your Spotify profile URL to find the most similar users based on your playlists!"
)
interface.launch()
```

This allows users to easily enter their profile URL, and in return, they receive a list of recommended profiles with detailed information.


## Key Features and Workflow:

- **Dynamic Feature Store**: New user embeddings are continuously added as users interact with the system, making the database dynamic and always up to date.
- **Two-Tower Model**: Uses a deep learning architecture to compute similarities between users and recommend the most relevant profiles based on cosine similarity.
- **Automated Retraining**: Model is retrained automatically every week using GitHub Actions to incorporate the latest user data into the model.
- **Interactive Web Interface**: Gradio web app provides an intuitive, user-friendly interface for making personalized Spotify recommendations.

# References
- https://slides.com/kirillkasjanov/recommender-systems#/3/6
- https://www.youtube.com/watch?v=9vBRjGgdyTY&t=834s
- https://www.youtube.com/watch?v=o-pZk5R0TZg
- https://www.youtube.com/watch?v=7_E4wnZGJKo
- https://medium.com/codex/similarity-search-of-spotify-songs-using-gcp-vector-search-vertex-ai-python-sdk-in-15-minutes-621573cd7b19
- https://www.hopsworks.ai/dictionary/two-tower-embedding-model
- https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture
- https://github.com/decodingml/personalized-recommender-course
- https://github.com/kirajano/two_tower_recommenders