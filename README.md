Written by David Habboosh.

**Acknowledgements**
* Valentina Paredes (songwriter) for writing detailed analysis of songs for model fine-tuning.
* [GetSongBPM API](https://getsongbpm.com/api) for accessing song key and mode.

**How to run**
* Download [Capybara](https://huggingface.co/TheBloke/Nous-Capybara-34B-GGUF/resolve/main/nous-capybara-34b.Q6_K.gguf?download=true) and [Mistral](https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q6_K.gguf?download=true) models and put them in the "models/" folder
* Replace all API keys in the example file with your own, then remove ".example" from the filename.
* Run process_csv.py to create the initial database.
* Run fetchspotifydata.py to gather listening history. Run any time to update.
* Run respondtoquery.py to talk to the chatbot!
