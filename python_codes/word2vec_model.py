from gensim.models import Word2Vec

# Prepare a list of artist genres (or artist names)
# This is just a simplified example. You'll want to extract these from the playlists and artists.
genres = [
    "pop", "rock", "hip-hop", "jazz", "classical", "electronic", "indie", "metal", "blues", "reggae"
]

# Training a Word2Vec model on this list (for demonstration purposes)
# Ideally, you would extract genres and artist names from your data here.
sentences = [[genre] for genre in genres]  # Each genre is treated as a sentence

# Initialize and train the Word2Vec model
genre_embedding_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Save the model
genre_embedding_model.save("genre_embeddings.model")
