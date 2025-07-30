from llama_cpp import Llama
import datetime
import chromadb
import json
import sentence_transformers
import os

JSON_CTX_WINDOW = 32768
CHAT_CTX_WINDOW = 200000

print('Loading json llm...')
json_llm = Llama(
    model_path="models/openhermes-2.5-mistral-7b.Q6_K.gguf",
    n_ctx=JSON_CTX_WINDOW,
    n_threads=8,
    n_gpu_layers=35,  # Set to 0 for CPU only, or tune based on GPU VRAM
    verbose=False,
)

print('\nLoading chat LLM...')
chat_llm = Llama(
    model_path="models/nous-capybara-34b.Q6_K.gguf",
    n_ctx=CHAT_CTX_WINDOW,
    n_threads=8,
    n_gpu_layers=35,  # Set to 0 for CPU only, or tune based on GPU VRAM
    verbose=False,
)
print('Done.')

def get_response(user_prompt, system_prompt, llm, ctx_window):
    full_prompt = f"<|system|>\n{system_prompt.strip()}\n<|user|>\n{user_prompt.strip()}\n<|assistant|>\n"
    tokens = ctx_window - len(llm.tokenize(full_prompt.encode("utf-8")))
    
    response = llm(full_prompt, stop=["</s>", "<|user|>"], temperature=0.2, max_tokens=tokens)
    return response['choices'][0]['text'].strip()

query_txt = input('What would you like to know? ')

system_prompt = f"""
You are an AI that extracts comparison traits from user music search prompts. Your job is to **strictly isolate surface-level traits** (like date, duration, artist, etc.) and ignore all **semantic or emotional content**, no matter how strongly implied.

⚠️ WARNING: If you include anything related to emotion, energy, meaning, genre, vibe, recovery, sadness, love, heartbreak, themes, or any kind of subjective or inferred idea, you have FAILED.

- Only use the following traits:
  - song name
  - song artist
  - song popularity
  - album name
  - release date
  - song duration
  - song language
  - when the user listened to the song
  - number of times the user has listened to the song

- Respond in list[str] format as shown in the examples below.
- If a phrase contains both valid and invalid traits, ONLY extract the valid part.
- If a phrase contains no valid traits, discard it. Do NOT include it at all.

Examples:
❌ Prompt: "Songs about heartbreak I listened to last week"  
✅ Output: ["songs listened to last week"]

❌ Prompt: "Very unknown songs that help me cope with death"  
✅ Output: ["very unknown songs"]

❌ Prompt: "High-energy songs shorter than 3 minutes"  
✅ Output: ["songs shorter than 3 minutes"]

❌ Prompt: "Compare the Spanish songs I listened to this month to the ones from last month"  
✅ Output: ["Spanish songs listened to this month", "Spanish songs listened to last month"]

Do NOT try to be helpful. Do NOT reword. Do NOT summarize meaning. Only extract valid literal traits.

Your output should be a single list of plain English descriptions, with one entry per valid trait set.
"""

print('\nGetting text filters...')
text_filters = get_response(query_txt, system_prompt, json_llm, JSON_CTX_WINDOW)
print('Done.')
# print(text_filters) # debugging

system_prompt = f"""
You are an AI that converts user queries into JSON filters for music search.
The current date is {datetime.datetime.now().isoformat()}. All mentions of time should be interpreted as relative to this date.

- Input is a list of plain English phrases, each describing one subset of music listening behavior.
- Output a JSON list where each object has:
  - "description": the original phrase, unmodified
  - "filter": a structured filter object using only allowed keys

- Only return a single valid JSON object, or a list of such objects if multiple filters are required.
- Use only the following keys:
  - track_name (string): the song title
  - track_artist (string): the artist's name
  - track_popularity (int): popularity score from 0 to 100
  - track_album_name (string): the name of the album
  - track_album_release_date (int): track/album release date in seconds since epoch
  - duration_ms (int): duration of the track in milliseconds
  - language (string): two-letter ISO 639-1 code (e.g., "en" for English)
  - times_listened (int): number of times the song has been listened to
  - *date* (int) - if the song has been listened to on a certain date, there will be a key for that date as seconds since epoch with a value being the number of times the song was listened to on that date

- When filtering by dates (e.g. "this month", "last week", "before 2020"), check if **any timestamp** in the `listened` list falls within the date range.
- Do this by creating a list of each date within the date range then checking if any of the listened dates are in that list. Ignore times.
- Do not assume only the first or most recent timestamp matters.

- Only include keys that are relevant to the query.
- Do not include inferred genre, mood, or style descriptions (like "rock", "chill", "romantic") in the JSON. These will be handled by a separate semantic search model.
- Use only the following comparison operators: $eq, $ne, $lt, $gt, $gte, $lte, $in, $nin.
- Use $and and $or to join multiple key filters. Do not use them if only one key is being filtered.
- If the prompt implies multiple filters (e.g., “Compare this month vs. last”), return a list of JSON objects with a "description" field explaining each segment.
- Otherwise return a single JSON object in a list with the "description" field set to an empty string.
- Do not include any explanations, natural language output, or formatting outside the raw JSON.
- Do not infer traits not explicitly mentioned in the prompt. If a constraint is not explicitly mentioned, DO NOT include it in the JSON.

EXAMPLE:

prompt: ["Tracks the user listened to this month", "Tracks shorter than 4 minutes that the user has not heard"]

filter:
[
  {{
    "description": "Tracks the user listened to this month",
    "filter": {{
      "$or": [
        {{ "2025-07-01": true }},
        {{ "2025-07-02": true }},
        {{ "2025-07-03": true }},
        {{ "2025-07-04": true }},
        {{ "2025-07-05": true }},
        {{ "2025-07-06": true }},
        {{ "2025-07-07": true }},
        {{ "2025-07-08": true }},
        {{ "2025-07-09": true }},
        {{ "2025-07-10": true }},
        {{ "2025-07-11": true }},
        {{ "2025-07-12": true }},
        {{ "2025-07-13": true }},
        {{ "2025-07-14": true }},
        {{ "2025-07-15": true }},
        {{ "2025-07-16": true }},
        {{ "2025-07-17": true }},
        {{ "2025-07-18": true }},
        {{ "2025-07-19": true }},
        {{ "2025-07-20": true }},
        {{ "2025-07-21": true }},
        {{ "2025-07-22": true }},
        {{ "2025-07-23": true }},
        {{ "2025-07-24": true }},
        {{ "2025-07-25": true }},
        {{ "2025-07-26": true }},
        {{ "2025-07-27": true }}
      ]
    }}
  }},
  {{
    "description": "Tracks shorter than 4 minutes that the user has not heard",
    "filter": {{
      "$and": [
        {{ "duration_ms": {{ "$lt": 240000 }} }},
        {{ "times_listened": {{ "$eq": 0 }} }}
      ]
    }}
  }}
]

IMPORTANT REMINDER: Do not infer traits including popularity, language, etc. unless explicitly mentioned in the prompt. If a constraint is not explicitly mentioned, DO NOT include it in the JSON.
IMPORTANT REMINDER: Do not use any filters that are not in the keys listed above. If any item in the given list has nothing to do with those filters, ignore it.
"""
  
print('\nCreating filter...')
filter_json = get_response(text_filters, system_prompt, json_llm, JSON_CTX_WINDOW)
print('Done.')
# print(filter_json) # debugging

print('\nEmbedding query...')
model = sentence_transformers.SentenceTransformer('BAAI/bge-small-en')
query_embedding = model.encode(query_txt, convert_to_numpy=True)
print('Done.')

client = chromadb.PersistentClient(path='chromadb_data')
collection = client.get_collection(name='spotify_songs')

# convert filter_json string to json
filter_json = json.loads(filter_json)

def filter_query(filters):
    description = filters['description']
    
    response = collection.query(
        query_embeddings=[query_embedding],
        where=filters['filter'],
        n_results=100,
        include=['embeddings', 'metadatas', 'documents', 'distances']
    )
    
    section_str = f'{description}:\n'
    
    distances = response['distances'][0]
    if len(distances) == 0:
      section_str += f"\t* None\n\n"
      return section_str
    
    try:
      THRESHOLD = max(0.2, distances[min(9, len(distances) - 1)])
    except Exception as e:
      print(filters['filter'])
      raise e
    
    for i in range(len(response['ids'][0])):
        distance = distances[i]
        if distance <= THRESHOLD:
            section_str += f"\t* {response['documents'][0][i]}\n\n"
    
    return section_str

print('\nApplying filters...')
document_prompt = ''
for filters in filter_json:
    document_prompt += filter_query(filters)
print('Done.')
# print(document_prompt) # debugging

# combine user input, document prompt, and system prompt to make the final prompt!

system_prompt = f"""
You are a music assistant AI that analyzes, summarizes, and recommends music based on a user's Spotify listening history.

Below is the user's relevant listening history and potentially relevant songs outside of it:

{document_prompt}

Guidelines:
- DO NOT simply list or reprint all the songs above.

INSTEAD:
- Analyze patterns in audio features and lyrics.
- Reference specific songs **only** when they illustrate an interesting observation, comparison, or recommendation.
- If the user asks for recommendations, suggest new songs that are **not already in the listening history**.
- Be creative and insightful in your analysis. Answer naturally, like a smart music curator.

Answer the user’s query using only the information above and your reasoning abilities. Do not invent listening data not shown.
"""

print('Getting response...')
print(get_response(query_txt, system_prompt, chat_llm, CHAT_CTX_WINDOW))