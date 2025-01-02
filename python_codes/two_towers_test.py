import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, List
from tensorflow.keras.layers import StringLookup, Normalization
import numpy as np

class TrackTower(tf.keras.Model):
    def __init__(
        self,
        track_ids: list,
        artist_ids: list,
        genres: list,
        emb_dim: int
    ):
        super().__init__()
        
        # Track embeddings
        self.track_embedding = tf.keras.Sequential([
            StringLookup(vocabulary=track_ids, mask_token=None),
            tf.keras.layers.Embedding(
                len(track_ids) + 1,
                emb_dim
            )
        ])
        
        # Artist embeddings
        self.artist_embedding = tf.keras.Sequential([
            StringLookup(vocabulary=artist_ids, mask_token=None),
            tf.keras.layers.Embedding(
                len(artist_ids) + 1,
                emb_dim
            )
        ])
        
        # Genre tokenizer
        self.genre_tokenizer = StringLookup(
            vocabulary=genres,
            mask_token=None
        )
        
        # Audio features normalization
        self.normalized_audio_features = {
            feature: Normalization(axis=None)
            for feature in [
                'danceability', 'energy', 'instrumentalness',
                'acousticness', 'valence', 'speechiness',
                'loudness', 'liveness'
            ]
        }
        
        # Neural network for final embedding
        self.fnn = tf.keras.Sequential([
            tf.keras.layers.Dense(emb_dim * 2, activation="relu"),
            tf.keras.layers.Dense(emb_dim)
        ])

    def call(self, inputs):
        # Process track ID
        track_emb = self.track_embedding(inputs["track_id"])
        
        # Process artist ID
        artist_emb = self.artist_embedding(inputs["artist_id"])
        
        # Process genres (multi-hot encoding)
        genre_indices = self.genre_tokenizer(inputs["genres"])
        genre_embedding = tf.one_hot(
            genre_indices,
            len(self.genre_tokenizer.get_vocabulary())
        )
        
        # Process audio features
        audio_features = tf.stack([
            self.normalized_audio_features[feature](inputs[feature])
            for feature in self.normalized_audio_features.keys()
        ], axis=1)
        
        # Concatenate all features
        concatenated = tf.concat([
            track_emb,
            artist_emb,
            genre_embedding,
            audio_features
        ], axis=1)
        
        # Final embedding
        return self.fnn(concatenated)

class UserTower(tf.keras.Model):
    def __init__(
        self,
        user_ids: list,
        emb_dim: int
    ):
        super().__init__()
        
        # User embeddings
        self.user_embedding = tf.keras.Sequential([
            StringLookup(vocabulary=user_ids, mask_token=None),
            tf.keras.layers.Embedding(
                len(user_ids) + 1,
                emb_dim
            )
        ])
        
        # Neural network for final embedding
        self.fnn = tf.keras.Sequential([
            tf.keras.layers.Dense(emb_dim, activation="relu"),
            tf.keras.layers.Dense(emb_dim)
        ])

    def call(self, inputs):
        # Process user ID
        user_emb = self.user_embedding(inputs["user_id"])
        return self.fnn(user_emb)

class SpotifyTwoTower(tf.keras.Model):
    def __init__(
        self,
        user_tower: UserTower,
        track_tower: TrackTower,
        track_dataset: tf.data.Dataset,
        batch_size: int
    ):
        super().__init__()
        self.user_tower = user_tower
        self.track_tower = track_tower
        
        # Define retrieval task
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=track_dataset.batch(batch_size).map(self.track_tower)
            )
        )

    def compute_loss(self, batch, training=True):
        user_embeddings = self.user_tower(batch)
        track_embeddings = self.track_tower(batch)
        
        return self.task(user_embeddings, track_embeddings, compute_metrics=not training)

class SpotifyDataset:
    def __init__(self, feature_view, batch_size: int):
        self._feature_view = feature_view
        self._batch_size = batch_size
        self._properties = None
        
    @property
    def user_features(self) -> List[str]:
        return ["user_id"]
    
    @property
    def track_features(self) -> List[str]:
        return [
            "track_id",
            "artist_id",
            "genres",
            "danceability",
            "energy",
            "instrumentalness",
            "acousticness",
            "valence",
            "speechiness",
            "loudness",
            "liveness"
        ]
    
    def prepare_data(self, profiles_count: int) -> Dict:
        """
        Prepare dataset from user profiles
        """
        # Get data from feature store
        train_df, val_df = self._feature_view.train_test_split(
            description="Spotify user-track dataset"
        )
        
        # Limit to specified number of profiles
        train_df = train_df.head(profiles_count)
        
        # Create TensorFlow datasets
        train_ds = tf.data.Dataset.from_tensor_slices({
            col: train_df[col].values for col in train_df.columns
        }).batch(self._batch_size)
        
        val_ds = tf.data.Dataset.from_tensor_slices({
            col: val_df[col].values for col in val_df.columns
        }).batch(self._batch_size)
        
        # Store properties
        self._properties = {
            "user_ids": train_df["user_id"].unique().tolist(),
            "track_ids": train_df["track_id"].unique().tolist(),
            "artist_ids": train_df["artist_id"].unique().tolist(),
            "genres": train_df["genres"].explode().unique().tolist()
        }
        
        return {
            "train_ds": train_ds,
            "val_ds": val_ds,
            "properties": self._properties
        }

def train_model(
    feature_view,
    profiles_count: int,
    batch_size: int = 128,
    emb_dim: int = 32,
    epochs: int = 10
):
    """
    Train the two-tower model
    """
    # Prepare dataset
    dataset = SpotifyDataset(feature_view, batch_size)
    data = dataset.prepare_data(profiles_count)
    
    # Create towers
    user_tower = UserTower(
        user_ids=data["properties"]["user_ids"],
        emb_dim=emb_dim
    )
    
    track_tower = TrackTower(
        track_ids=data["properties"]["track_ids"],
        artist_ids=data["properties"]["artist_ids"],
        genres=data["properties"]["genres"],
        emb_dim=emb_dim
    )
    
    # Create and compile model
    model = SpotifyTwoTower(
        user_tower=user_tower,
        track_tower=track_tower,
        track_dataset=data["train_ds"],
        batch_size=batch_size
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001)
    )
    
    # Train model
    history = model.fit(
        data["train_ds"],
        validation_data=data["val_ds"],
        epochs=epochs
    )
    
    return model, history