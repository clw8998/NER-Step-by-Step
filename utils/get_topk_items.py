import os
import pickle
import numpy as np
import pandas as pd
import jieba
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

class TFIDFModel:
    def __init__(self, corpus_name):
        self.corpus_name = corpus_name
        self.tokenizer = self.jieba_tokenizer  # Fixed tokenizer
        self.model_path = f'{self.corpus_name}_tf_idf_checkpoint.pkl'
        self.pickle_file = f'{self.corpus_name}.pkl'
        self.items_df = None
        self.tfidf = None
        self.items_tfidf_matrix = None
        self._initialized = False

    @staticmethod
    def jieba_tokenizer(text):

        tokens = jieba.lcut(text, cut_all=False)
        stop_words = ['【', '】', '/', '~', '＊', '、', '（', '）', '+', '‧', ' ', '']
        tokens = [t for t in tokens if t not in stop_words]
        return tokens

    def _load_data(self):

        if os.path.exists(self.pickle_file):
            with open(self.pickle_file, 'rb') as f:
                self.items_df = pickle.load(f)
            print(f'Data loaded from {self.pickle_file}.')
        else:
            # If the pickle file does not exist, read the corresponding CSV file
            dataset = load_dataset("clw8998/NER-step-by-step-dataset", data_files={"coupang": "coupang.csv", "pchome": "pchome.csv"})

            self.items_df = dataset[self.corpus_name].to_pandas()
            # Data cleaning
            self.items_df['product_name'] = self.items_df['product_name'].astype(str)
            self.items_df['product_name'] = self.items_df['product_name'].fillna('')
            self.items_df = self.items_df.drop_duplicates(subset='product_name')
            # Save as a pickle file
            with open(self.pickle_file, 'wb') as f:
                pickle.dump(self.items_df, f)
            print(f'Data processed and saved to {self.pickle_file}.')
        
    def _init_tf_idf(self):

        if os.path.exists(self.model_path):
            # Load the saved model
            with open(self.model_path, 'rb') as file:
                data = pickle.load(file)
                self.tfidf = data['tfidf']
                self.items_tfidf_matrix = data['items_tfidf_matrix']
            print(f'TF-IDF model loaded from {self.model_path}.')
        else:
            # Train a new model if no saved model is found
            print(f'TF-IDF model for {self.corpus_name} not found. Creating a new model...')
            self.tfidf = TfidfVectorizer(token_pattern=None, tokenizer=self.tokenizer, ngram_range=(1, 2))
            self.items_tfidf_matrix = self.tfidf.fit_transform(tqdm(self.items_df['product_name']))
            # Save the trained model
            with open(self.model_path, 'wb') as file:
                pickle.dump({
                    'tfidf': self.tfidf,
                    'items_tfidf_matrix': self.items_tfidf_matrix,
                }, file)
            print(f'TF-IDF model saved to {self.model_path}.')

    def initialize(self):

        if not self._initialized:
            self._load_data()
            self._init_tf_idf()
            self._initialized = True

    def query(self, query_str, top_k=5):

        if not self._initialized:
            self.initialize()
        query_tfidf = self.tfidf.transform([query_str])
        scores = cosine_similarity(query_tfidf, self.items_tfidf_matrix)
        top_k_indices = np.argsort(-scores[0])[:top_k]
        top_k_names = self.items_df['product_name'].values[top_k_indices]
        top_k_scores = scores[0][top_k_indices]
        return top_k_names