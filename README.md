Written by David Habboosh

**Acknowledgements**
* Valentina Paredes (songwriter) for writing detailed analysis of songs for model fine-tuning.
* [Audio features and lyrics of Spotify songs](https://www.kaggle.com/datasets/imuhammad/audio-features-and-lyrics-of-spotify-songs?resource=download) -- dataset used to represent all of Spotify.
* [GetSongBPM API](https://getsongbpm.com/api) for accessing song key and mode.

**How to run**
* Download [Capybara](https://huggingface.co/TheBloke/Nous-Capybara-34B-GGUF/resolve/main/nous-capybara-34b.Q6_K.gguf?download=true) and [Mistral](https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q6_K.gguf?download=true) models and put them in the *models.example/* folder, then remove *.example* from the filename.
* Replace all API keys in *spotvars.env.example* with your own, then remove *.example* from the filename.
* Run main.py to talk to the chatbot! Initial run will take some time to create database and gather entire listening history.