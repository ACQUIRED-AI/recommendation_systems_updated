# content_based.py
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import pandas as pd
import numpy as np

class ContentBasedRecommender:
    def __init__(self):
        # text vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        # scaler for numerical features
        self.numerical_scaler = StandardScaler()
        # learned matrices
        self.item_features = None
        self.similarity_matrix = None
        # mappings between product_id ↔ index
        self.product_to_idx = {}
        self.idx_to_product = {}

    def fit(self, products_df: pd.DataFrame):
        """
        Fit the content-based model on product metadata.
        Combines text (name, desc, brand, category) + numerical features (price, ratio).
        """
        products_df = products_df.copy()
        products_df["product_id"] = products_df["product_id"].astype(str)  # <-- force string IDs


        # ensure required text columns exist, even if empty
        for col in ['name', 'description', 'category', 'brand']:
            if col not in products_df.columns:
                products_df[col] = ""

        # combine text columns into one field
        products_df['combined_features'] = (
            products_df['name'].fillna('') + ' ' +
            products_df['description'].fillna('') + ' ' +
            products_df['category'].fillna('') + ' ' +
            products_df['brand'].fillna('')
        )

        # TF-IDF on combined text
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(products_df['combined_features'])

        # numerical features
        num_cols = ['price_current', 'price_ratio']
        for c in num_cols:
            if c not in products_df.columns:
                products_df[c] = 0.0
        numerical = products_df[num_cols].fillna(0).values
        numerical_scaled = self.numerical_scaler.fit_transform(numerical)

        # combine text + numerical
        self.item_features = hstack([tfidf_matrix, numerical_scaled])

        # precompute similarity between all items
        self.similarity_matrix = cosine_similarity(self.item_features)

        # build mappings
        self.product_to_idx = {pid: idx for idx, pid in enumerate(products_df['product_id'])}
        self.idx_to_product = {idx: pid for idx, pid in enumerate(products_df['product_id'])}

    def get_similar_items(self, product_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top_k most similar items for a given product_id.
        Returns a list of (other_product_id, similarity_score).
        """
        if self.similarity_matrix is None:
            return []

        pid = str(product_id)  # <-- normalize to string
        if pid not in self.product_to_idx:
            return []

        i = self.product_to_idx[pid]

        sims = self.similarity_matrix[i]

        # sort by similarity, skip itself
        idxs = sims.argsort()[::-1]
        idxs = [j for j in idxs if j != i][:top_k]

        return [(self.idx_to_product[j], float(sims[j])) for j in idxs]
