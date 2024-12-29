# Project Roadmap

- **Task 1:** Gather a lot of Spotify user IDs in order to scrap a maximum of existing Spotify profiles in order to fill our database.

- **Task 2:** Create a pipeline which, taking a list of Spotify user IDs or profiles URL, scrap their playlist ([see this](https://developer.spotify.com/documentation/web-api/reference/get-list-users-playlists)), perform embedding according to *User Profile Embedding* ([see this](./user_profile_embedding.md)) instructions and store it in the Hopsworks feature store (or Hopsworks Vector DB ?).
	- ***Sub-task 1:*** Write a function using Spotify API ([doc](https://developer.spotify.com/documentation/web-api)) that takes a list of Spotify profile IDs or URL and gather all the tracks from the public playlists and for each track extract the necessary audio features ([see this](https://developer.spotify.com/documentation/web-api/reference/get-audio-features)) and genres (using artist genre from the [track's artist](https://developer.spotify.com/documentation/web-api/reference/get-an-artist))
	- ***Sub-task 2:*** See how to make a great embedding for all those informations using the two-towers architecture.
	- ***Sub-task 3:*** Write a function that takes the tracks and their features from previous function and perform the embedding and store the feature to Hopsworks.

- **Task 3:** Create a pipeline that train the two-tower architecture retrieval model using the features on Hopsworks and save this model.

- **Task 4:** Create a pipeline that takes a user Spotify profile URL or profile ID and perform embedding (using Task 2 method) and then recommend the closest exiting user already stored in Hopsworks using the two-tower architecture recommender.

- **Task 5:** Develop an UI where a user can enter its profile URL, trigger the recommendation (Task 3 pipeline), and get the results of the most similar profiles (EXTRA: as well as some information like the music genre they match the most, etc...).

