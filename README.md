Written by David Habboosh

**Acknowledgements**
* Valentina Paredes (songwriter) for writing detailed analysis of songs for model fine-tuning.
* [Audio features and lyrics of Spotify songs](https://www.kaggle.com/datasets/imuhammad/audio-features-and-lyrics-of-spotify-songs?resource=download) -- dataset used to represent all of Spotify.
* [GetSongBPM API](https://getsongbpm.com/api) for accessing song key and mode.

**How to run**
* Replace all API keys in the example file with your own, then remove ".example" from the filename.
* Run process_csv.py to create the initial database.
* Run fetchspotifydata.py to gather listening history. Run any time to update.
* Run respondtoquery.py to talk to the chatbot!
