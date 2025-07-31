from llama_cpp import Llama
import datetime
import chromadb
import json
import sentence_transformers
import fetchspotifydata
import os
import subprocess
import psutil

def get_model_size_gb(path):
  return os.path.getsize(path) / (1024 ** 3)

def get_gpu_with_most_free_memory():
    try:
        result = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,nounits,noheader"
        ], stderr=subprocess.DEVNULL)

        free_list_mb = [int(x) for x in result.decode("utf-8").strip().split('\n')]
        best_gpu_index = max(range(len(free_list_mb)), key=lambda i: free_list_mb[i])
        return best_gpu_index, free_list_mb[best_gpu_index] / 1024
    except Exception:
        return None, 0

def get_n_threads():
  threads =  max(1, psutil.cpu_count(logical=False) - 2)
  print('Threads: ', threads)
  return threads

def get_n_gpu_layers(model_path, layers, free_gb):
    print(free_gb, 'GB free')
  
    buffer_gb = 1.0
    usable_gb = max(0, free_gb - buffer_gb)
    layer_size = get_model_size_gb(model_path) / layers
    print('Size per layer: ', layer_size, 'GB')
    max_layers = int(usable_gb / layer_size)
    
    if max_layers >= layers:
      max_layers = -1
    print('Layers on GPU: ', max_layers)
    
    return max_layers

def create_llms(JSON_CTX_WINDOW, CHAT_CTX_WINDOW):
  print('Loading json llm...')
  json_model_path = "models/openhermes-2.5-mistral-7b.Q6_K.gguf"
  gpu_index, free_gb = get_gpu_with_most_free_memory()
  if gpu_index is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
  json_llm = Llama(
      model_path=json_model_path,
      n_ctx=JSON_CTX_WINDOW,
      n_threads=psutil.cpu_count(logical=False),
      n_gpu_layers=get_n_gpu_layers(json_model_path, 32, free_gb),
      verbose=False,
  )

  print('\nLoading chat LLM...')
  chat_model_path = "models/nous-capybara-34b.Q6_K.gguf"
  gpu_index, free_gb = get_gpu_with_most_free_memory()
  if gpu_index is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
  chat_llm = Llama(
      model_path=chat_model_path,
      n_ctx=CHAT_CTX_WINDOW,
      n_threads=psutil.cpu_count(logical=False),
      n_gpu_layers=get_n_gpu_layers(chat_model_path, 60, free_gb),
      verbose=False,
  )
  print('Done.')
  
  return json_llm, chat_llm

def initialize_collection():
  client = chromadb.PersistentClient(path='chromadb_data')
  return client.get_collection(name='spotify_songs') 

def get_response(user_prompt, system_prompt, llm, ctx_window):
    full_prompt = f"<|system|>\n{system_prompt.strip()}\n<|user|>\n{user_prompt.strip()}\n<|assistant|>\n"
    tokens = ctx_window - len(llm.tokenize(full_prompt.encode("utf-8")))
    
    response = llm(full_prompt, stop=["</s>", "<|user|>"], temperature=0.2, max_tokens=tokens)
    return response['choices'][0]['text'].strip()

def embed_prompt(prompt, model):
  prompt_embedding = model.encode(prompt, convert_to_numpy=True)
  
  return prompt_embedding

def filter_query(filters, collection, prompt_embedding):
    description = filters['description']
    
    response = collection.query(
        query_embeddings=[prompt_embedding],
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

def main():
  # get user input
  user_txt = input('What would you like to know? ')
  
  # get the database
  collection = initialize_collection()
  
  # update spotify data
  print('\nUpdating Spotify data...')
  fetchspotifydata.main()
  print('Done.\n')
  
  # variables
  JSON_CTX_WINDOW = 32768
  CHAT_CTX_WINDOW = 200000
  
  # create LLMs
  json_llm, chat_llm = create_llms(JSON_CTX_WINDOW, CHAT_CTX_WINDOW)
  
  # get concrete filters as a list
  print('\nGetting text filter list...')
  with open('prompt_to_list_prompt.txt', encoding='utf-8') as f:
    prompt_to_list_prompt = f.read()
  text_filters = get_response(user_txt, prompt_to_list_prompt, json_llm, JSON_CTX_WINDOW)
  print('Done.')
  
  # convert filter list to json
  print('\nCreating JSON filter...')
  with open('list_to_json_prompt.txt', encoding='utf-8') as f:
    list_to_json_prompt = f.read().format(time=datetime.datetime.now().isoformat())
  filter_json = get_response(text_filters, list_to_json_prompt, json_llm, JSON_CTX_WINDOW)
  filter_json = json.loads(filter_json)
  print('Done.')
  
  # embed prompt & get relevant documents
  print('\nApplying filter...')
  embed_model = sentence_transformers.SentenceTransformer('BAAI/bge-small-en')
  prompt_embedding = embed_prompt(user_txt, embed_model)
  documents = ''
  for filters in filter_json:
      documents += filter_query(filters, collection, prompt_embedding)
  print('Done.')
  
  # ask the question!
  print('\nGetting response...')
  with open('final_prompt.txt', encoding='utf-8') as f:
    final_prompt = f.read().format(documents=documents)
  print(get_response(user_txt, final_prompt, chat_llm, CHAT_CTX_WINDOW))
  
if __name__ == '__main__':
  main()