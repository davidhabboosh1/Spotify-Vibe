import pandas as pd
import sentence_transformers
import chromadb
import json
from tqdm import tqdm
from concurrent.futures import as_completed, ThreadPoolExecutor
from threading import Lock
import os
from datetime import datetime

data = pd.read_csv('spotify_songs.csv')

"""metadata:
track_id
track_name
track_artist
track_popularity
# track_album_id
track_album_name
track_album_release_date
duration_ms
language
"""

"""embedding:
lyrics
playlist_genre
playlist_subgenre
danceability
energy
key
loudness
mode
speechiness
acousticness
instrumentalness
liveness
valence
tempo
"""

keys = ['C', 'C#/D♭', 'D', 'D#/E♭', 'E', 'F', 'F#/G♭', 'G', 'G#/A♭', 'A', 'A#/B♭', 'B']
modes = {-1: 'NA', 0: 'Minor', 1: 'Major'}

def format_for_embedding(row):
    audio_desc = (
        f"Key: {'NA' if row['key'] == -1 else keys[row['key']]}, Mode: {modes[row['mode']]}, "
        f"Tempo: {row['tempo']} BPM, Loudness: {row['loudness']} dB, "
        f"Danceability: {row['danceability']}, Energy: {row['energy']}, "
        f"Valence: {row['valence']}, Speechiness: {row['speechiness']}, "
        f"Acousticness: {row['acousticness']}, Instrumentalness: {row['instrumentalness']}, Liveness: {row['liveness']}"
    )
    
    text = (
        f"Track: {row['track_name']} by {row['track_artist']}\n"
        f"Genre & subgenre: {row['playlist_genre']} & {row['playlist_subgenre']})\n"
        f"Lyrics: {row['lyrics']}\n"
        f"Audio Profile: {audio_desc}"
    )
    return text

def format_metadata(row):
    metadata = {
        'track_id': row['track_id'],
        'track_name': row['track_name'],
        'track_artist': row['track_artist'],
        'track_popularity': row['track_popularity'],
        # 'track_album_id': row['track_album_id'],
        'track_album_name': row['track_album_name'],
        'track_album_release_date': row['track_album_release_date'],
        # 'listened': {row['played_date']: True} if 'played_date' in row.keys() else {},
        'mbid': row['mbid'] if 'mbid' in row.keys() and row['mbid'] else 'NA',
        'duration_ms': row['duration_ms'],
        'language': row['language'] if 'language' in row.keys() else 'NA',
    }
    
    times_listened = 0
    for key in row.keys():
        if key.isdigit():
            metadata[key] = 1
            times_listened += 1
    
    metadata['times_listened'] = times_listened
    
    return metadata

lock = Lock()
def format_single(row, model, collection):
    text = format_for_embedding(row)
    metadata = format_metadata(row)
    
    encoding = model.encode(text, convert_to_numpy=True)
    
    with lock:
        try:
            collection.add(
                ids=[row['track_id']],
                embeddings=[encoding],
                metadatas=[metadata],
                documents=[text]
            )
        except Exception as e:
            print(e, flush=True)
    
def create_client():
    client = chromadb.PersistentClient(path='chromadb_data')
    collection = client.create_collection(name='spotify_songs')
    model = sentence_transformers.SentenceTransformer('BAAI/bge-small-en')
    
    return collection, model

def format_all(data):
    collection, model = create_client()
    
    for i, row in enumerate(data.itertuples()):
        row = row._asdict()
        format_single(row, model, collection)
        
        if i % 100 == 0:
            print(f"Processed {i} rows...")
            
def format_all_threaded(data):
    collection, model = create_client()
    
    MAX_WORKERS = 10 #TODO: Change to 10
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(format_single, row._asdict(), model, collection) for row in data.itertuples()]
        
        for _ in tqdm(as_completed(futures), total=len(data), desc="Serializing"):
            _.result()
            
if __name__ == '__main__':
    format_all_threaded(data)