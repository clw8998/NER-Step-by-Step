# python .\tf_idf_revised.py ../02_程式集/Coupang_Scraping-main/results -k 5 -i
# %%
# Import required libraries
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import jieba
import os
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


items_folder = 'random_samples_1M'
pickle_file = 'random_samples_1M.pkl'
# Path to save/load the models
model_path = 'tf_idf_checkpoint.pkl'

timer_start = time.time()
path_to_item_file = [file for file in os.listdir(items_folder) if file.endswith('.csv')]
# merge all csv files into one dataframe
items_df = pd.concat([pd.read_csv(os.path.join(items_folder, file)) for file in path_to_item_file], ignore_index=True)

# Ensure all product_name entries are strings
items_df['product_name'] = items_df['product_name'].astype(str)
items_df['product_name'] = items_df['product_name'].map(html.unescape)
items_df['product_name'] = items_df['product_name'].fillna('')
items_df = items_df.drop_duplicates(subset='product_name')

print(f'Processed {len(items_df)} items in {time.time() - timer_start:.2f} seconds.')

# 保存為 pickle 檔案
with open(pickle_file, 'wb') as f:
    pickle.dump(items_df, f)
print(f'Data saved to {pickle_file}.')


# %%
# Initialize tokenizer
timer_start = time.time()


def jieba_tokenizer(text):
    tokens = jieba.lcut(text, cut_all=False)
    stop_words = ['【','】','/','~','＊','、','（','）','+','‧',' ','']
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

tokenizer = jieba_tokenizer



# Function to save the models
def save_models_and_matrices(tfidf, items_tfidf_matrix, path):
    with open(path, 'wb') as file:
        pickle.dump({
            'tfidf': tfidf,
            'items_tfidf_matrix': items_tfidf_matrix,
        }, file)

# Function to load the models
def load_models_and_matrices(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data['tfidf'], data['items_tfidf_matrix']


# Check if the models are already saveda
if os.path.exists(model_path):
    # If saved, load the models
    tfidf, items_tfidf_matrix = load_models_and_matrices(model_path)
else:
    # If not saved, create the models
    print('TF-IDF models not found. Creating them...')

    tfidf = TfidfVectorizer(token_pattern=None, tokenizer=tokenizer, ngram_range=(1,2))
    items_tfidf_matrix = tfidf.fit_transform(tqdm(items_df['product_name']))
    
    save_models_and_matrices(tfidf, items_tfidf_matrix, model_path)

print(f'TF-IDF models loaded in {time.time() - timer_start:.2f} seconds.')


# %%

# Function to search for the top k items
def tf_idf(query, top_k=10):
    query_tfidf = tfidf.transform([query]) # sparse array
    scores = cosine_similarity(query_tfidf, items_tfidf_matrix)
    top_k_indices = np.argsort(-scores[0])[:top_k]
    
    top_k_names = items_df['product_name'].values[top_k_indices]
    top_k_scores = scores[0][top_k_indices]

    return top_k_names, top_k_scores



