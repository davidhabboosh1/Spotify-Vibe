import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np
from dotenv import load_dotenv
import os
os.environ["CHROMA_TELEMETRY"] = "False"
import pylast
import time
import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
import datetime
import chromadb
import json
from lyricsgenius import Genius
from process_csv import format_single
import sentence_transformers
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
from tqdm import tqdm
import warnings
import logging
from langdetect import detect

warnings.filterwarnings("ignore")
logging.getLogger("tenacity").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)

load_dotenv('spotvars.env')

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

SESSION_KEY_FILE = 'session_key'
network = pylast.LastFMNetwork(API_KEY, API_SECRET)
if not os.path.exists(SESSION_KEY_FILE):
    skg = pylast.SessionKeyGenerator(network)
    url = skg.get_web_auth_url()

    print(f"Please authorize this script to access your account: {url}\n")
    import time
    import webbrowser

    webbrowser.open(url)

    while True:
        try:
            session_key = skg.get_web_auth_session_key(url)
            test_network = pylast.LastFMNetwork(API_KEY, API_SECRET, session_key)
            test_user = test_network.get_authenticated_user()
            
            with open(SESSION_KEY_FILE, "w") as f:
                f.write(session_key)
            break
        except pylast.WSError:
            time.sleep(1)
else:
    session_key = open(SESSION_KEY_FILE).read()

# print(session_key)
network.session_key = session_key

def serialize(track):
    row = {}
    retry_delay = 2
    
    dt = datetime.datetime.strptime(track.playback_date, '%d %b %Y, %H:%M')
    date_only = datetime.datetime(dt.year, dt.month, dt.day)
    row['played_date'] = str(int(date_only.timestamp()))
    
    track = track.track
    
    while True:
        try:
            tags = track.get_top_tags()
            row['mbid'] = track.get_mbid()
            break
        except pylast.WSError as e:
            print(f"Rate limit exceeded at {datetime.datetime.now()}. Retrying in {retry_delay} seconds...", flush=True)
            time.sleep(retry_delay)
            retry_delay *= 2
    
    tags = [tag.item.get_name() for tag in tags]  # Convert pylast.Tag objects to strings
    row['orig_artist'] = track.artist.name
    row['orig_title'] = track.title
    
    row['playlist_genre'] = tags[0] if tags and len(tags) > 0 else 'NA'
    row['playlist_subgenre'] = tags[1] if tags and len(tags) > 1 else 'NA'
    
    return row

if not os.path.exists('last_processed_date.txt'):
    with open('last_processed_date.txt', 'w') as f:
        f.write('0')
        
lock = Lock()
def serialize_and_write(track, f):
    row = serialize(track)
    line = json.dumps(row) + '\n'
    with lock:
        f.write(line)
    return True

with open('last_processed_date.txt', 'r') as f:
    last_processed_date = f.read().strip()
    
print('Fetching data since ', last_processed_date)
data = network.get_user('DavidH3022').get_recent_tracks(limit=None, time_from=int(float(last_processed_date) // 1))
if len(data) == 0:
    print('No data found since this timestamp')
    exit()

# split into chunks of 1000 to avoid memory issues with the last being the leftover
data = [data[i:i + 1000] for i in range(0, len(data), 1000)]

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope="user-library-read"))

def get_audio_features(spotify_id, track, artist):
    key, mode = get_songdata_key(track, artist)
    key = key if key is not None else 'NA'
    mode = mode if mode is not None else 'NA'
    
    url = f"https://api.reccobeats.com/v1/track?ids={spotify_id}"

    headers = {
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers)
    
    features = {'acousticness': 'NA', 'danceability': 'NA', 'energy': 'NA', 'instrumentalness': 'NA', 'liveness': 'NA', 'loudness': 'NA', 'speechiness': 'NA', 'tempo': 'NA', 'valence': 'NA', 'key': key, 'mode': mode}
    if len(response.json()['content']) == 0:
        return features
    id = response.json()['content'][0]['id']
    
    features_url = f"https://api.reccobeats.com/v1/track/{id}/audio-features"
    features_response = requests.get(features_url, headers=headers)
    features_response = features_response.json()
    
    # set all elements in features to those in features_response
    for key in features.keys():
        if key in features_response:
            features[key] = features_response[key]
    
    return features

key_map = {
    'C': 0, 'C♯': 1, 'D♭': 1, 'D': 2, 'D♯': 3, 'E♭': 3, 'E': 4, 'F': 5, 'F♯': 6, 'G♭': 6, 'G': 7, 'G♯': 8, 'A♭': 8, 'A': 9, 'A♯': 10, 'B♭': 10, 'B': 11
}

def get_songdata_key(track_name, artist_name):
    retry_delay = 2
    
    query = urllib.parse.quote_plus(f"{track_name} {artist_name}")
    search_url = f'https://songdata.io/search?query={query}'

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "DNT": "1",  # Do Not Track
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    session = requests.Session()

    while True:
        res = session.get(search_url, headers=headers)
        if res.status_code == 403:
            print(f"Blocked by Songdata search page at {datetime.datetime.now()}.")
            return None, None

        soup = BeautifulSoup(res.text, "html.parser")    
        element = soup.find(class_="table_key")
        try:
            key_mode = element.text
            break
        except:
            # print(f"Hit rate limit at {datetime.datetime.now()}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2
    key, mode = key_mode.split(' ')
    
    key = key_map.get(key, 'NA')
    if mode == 'Major':
        mode = 1
    elif mode == 'Minor':
        mode = 0
    else:
        mode = -1
    
    return key, mode

GENIUS_API_TOKEN = "8Rih_2HsWMaufDyf80Xo3IJVDxmKHHi1VW9nvA6Wt9y61pHg-1KwBX_NXQX3L2hW"
genius = Genius(GENIUS_API_TOKEN, verbose=False)
genius.response_format = 'plain'

max_tries = 3
def search_genius_lyrics(song_title, artist_name):
    song = None
    
    cur_try = 0
    while cur_try < max_tries:
        try:
            song = genius.search_song(song_title, artist_name)
            if song:
                return song.lyrics
            else:
                cur_try += 1
                time.sleep(2)
        except Exception as e:
            cur_try += 1
            time.sleep(2)
    return song.lyrics if song else 'NA'

client = chromadb.PersistentClient(path='chromadb_data')
collection = client.get_collection(name='spotify_songs')
model = sentence_transformers.SentenceTransformer('BAAI/bge-small-en')

spotify_lock = Semaphore(1)
def retry_spotify_call(fn):
    retry_after = 2
    while True:
        try:
            return fn()
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                print(f"[Spotipy] Rate limit hit. Sleeping {retry_after}s...", flush=True)
                time.sleep(retry_after)
                retry_after *= 2
            else:
                raise

def add_to_collection(item):
    row = serialize(item)
    
    # check if track already exists in collection
    if row['mbid'] and row['mbid'] != 'NA':
        existing = collection.get(where={"mbid": row['mbid']})
        if existing['ids']:
            # add the played date to the metadata
            existing_metadata = existing['metadatas'][0]
            
            updated_metadata = existing_metadata.copy()
            cur_listened = updated_metadata.get(row['played_date'], 0) + 1
            updated_metadata[row['played_date']] = cur_listened
            updated_metadata['times_listened'] += 1
            collection.update(
                ids=existing['ids'],
                metadatas=[updated_metadata]
            )
            
            asstring = datetime.datetime.fromtimestamp(float(row['played_date'])).strftime('%Y-%m-%d')
            print('Set existing track:', row['orig_title'], 'to', cur_listened, 'on', asstring)
            return
    
    # search spotify for the track
    with spotify_lock:
        results = retry_spotify_call(lambda: sp.search(q=f"track:{row['orig_title']} artist:{row['orig_artist']}", type='track', limit=1))
    # results = sp.search(q=f"track:{row['orig_title']} artist:{row['orig_artist']}", type='track', limit=1)
    if results['tracks']['items']:
        sp_track = results['tracks']['items'][0]
        row['track_id'] = sp_track['id']
        
        # check again for duplication by spotify track id
        existing = collection.get(where={'track_id': row['track_id']})
        if existing['ids']:
            # add the played date to the metadata
            existing_metadata = existing['metadatas'][0]
            
            updated_metadata = existing_metadata.copy()
            cur_listened = updated_metadata.get(row['played_date'], 0) + 1
            updated_metadata[row['played_date']] = cur_listened
            updated_metadata['times_listened'] += 1
            updated_metadata['mbid'] = row['mbid']
            collection.update(
                ids=existing['ids'],
                metadatas=[updated_metadata]
            )
            
            asstring = datetime.datetime.fromtimestamp(float(row['played_date'])).strftime('%Y-%m-%d')
            print('Set existing track:', row['orig_title'], 'to', cur_listened, 'on', asstring)
            return
        
        row['track_name'] = sp_track['name']
        track_artists = sp_track['artists'] # join with comma if multiple artists with & before the last
        if len(track_artists) > 1:
            row['track_artist'] = ', '.join([artist['name'] for artist in track_artists[:-1]]) + ' & ' + track_artists[-1]['name']
        else:
            row['track_artist'] = track_artists[0]['name']
        row['track_popularity'] = sp_track['popularity']
        row['track_album_id'] = sp_track['album']['id']
        row['track_album_name'] = sp_track['album']['name']
        row['track_album_release_date'] = sp_track['album']['release_date']
        row['duration_ms'] = sp_track['duration_ms']
        
        audio_desc = get_audio_features(row['track_id'], row['track_name'], row['track_artist'])
        
        row = {**row, **audio_desc}
            
        row['lyrics'] = search_genius_lyrics(row['track_name'], row['track_artist'])
        row['language'] = detect(row['lyrics']) if row['lyrics'] != 'NA' else 'NA'
        
        format_single(row, model, collection)
        
# data = [data[i:i + 1000] for i in range(0, len(data), 1000)]

for chunk_num, chunk in enumerate(data):
    print(f"Processing chunk {chunk_num + 1}/{len(data)}...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(add_to_collection, track) for track in chunk]

        for _ in tqdm(as_completed(futures), total=len(chunk), desc="Updating database"):
            _.result()
            
    print('Finished, sleep for 10 seconds...')
    time.sleep(10)
        
# create file to store the most recent timestamp processed
with open('last_processed_date.txt', 'w') as f:
    f.write(str(time.time()))