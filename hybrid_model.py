# hybrid_model.py
import numpy as np
from typing import Dict, List, Tuple
import joblib

class HybridRecommender:
    def __init__(self, cf_model, content_model, user_features_model):
        self.cf_model = cf_model
        self.content_model = content_model
        self.user_features_model = user_features_model
        # weight each branch: CF, content, user-features
        self.weights = {'cf': 0.5, 'content': 0.3, 'user_features': 0.2}

    def _content_based_prediction(self, user_id: str, product_id: str, user_features: Dict) -> float:
        """
        Map content similarity to a 0–5 “rating”.
        Uses average similarity to the user's recent products (if any).
        Falls back to 3.0 when we can't compute a signal.
        """
        # must have a fitted content model and a known product
        if (self.content_model is None or
            getattr(self.content_model, "similarity_matrix", None) is None or
            product_id not in self.content_model.product_to_idx):
            return 3.0

        recent_ids = user_features.get("recent_product_ids", []) or []
        if not recent_ids:
            return 3.0

        i = self.content_model.product_to_idx[product_id]
        sims = []
        for rid in recent_ids:
            j = self.content_model.product_to_idx.get(rid)
            if j is not None:
                sims.append(float(self.content_model.similarity_matrix[i, j]))

        if not sims:
            return 3.0

        # similarity ~[0,1], rating ~[0,5]
        sim_avg = sum(sims) / len(sims)
        return max(0.0, min(5.0, 5.0 * sim_avg))

    def predict_rating(self, user_id: str, product_id: str, user_features: Dict) -> float:
        """
        Blend CF + content + user-features model into a single 0–5 score.
        Each branch is try/except-safe and falls back to 3.0 (“neutral”) if unavailable.
        """
        predictions = {}

        # CF branch
        try:
            predictions['cf'] = float(self.cf_model.predict(user_id, product_id))
        except Exception:
            predictions['cf'] = 3.0

        # Content branch
        try:
            predictions['content'] = float(self._content_based_prediction(user_id, product_id, user_features))
        except Exception:
            predictions['content'] = 3.0

        # User-features branch (later)
        try:
            predictions['user_features'] = float(self.user_features_model.predict([user_features])[0])
        except Exception:
            predictions['user_features'] = 3.0

        # Weighted blend (normalize weights just in case)
        wsum = sum(self.weights.values()) or 1.0
        final = sum(self.weights.get(k, 0.0) * v for k, v in predictions.items()) / wsum

        # clip to valid rating range
        return max(0.0, min(5.0, final))

    def get_recommendations(
        self,
        user_id: str,
        user_features: Dict,
        candidate_products: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Score candidate products for a user and return the top_k.
        """
        scores = []
        for pid in candidate_products:
            s = self.predict_rating(user_id, pid, user_features)
            scores.append((pid, float(s)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
