# training_pipeline.py
import os
import joblib
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import numpy as np
import pandas as pd
import sqlite3

from content_based import ContentBasedRecommender
from hybrid_model import HybridRecommender
from matrix_factorization import MatrixFactorization

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        mlflow.set_tracking_uri(config['mlflow_uri'])

    def train_collaborative_filtering(self, interactions_df):
        """Train collaborative filtering model with MSE loss on ratings"""
        with mlflow.start_run(run_name="collaborative_filtering"):
            # Encode user/item as categorical indices
            user_ids = interactions_df['user_id'].astype('category').cat.codes
            item_ids = interactions_df['product_id'].astype('category').cat.codes
            ratings = interactions_df['rating'].values

            self.user_map = dict(enumerate(interactions_df['user_id'].astype('category').cat.categories))
            self.item_map = dict(enumerate(interactions_df['product_id'].astype('category').cat.categories))

            # Train/test split
            X = np.column_stack([user_ids, item_ids])
            X_train, X_test, y_train, y_test = train_test_split(
                X, ratings, test_size=0.2, random_state=42
            )

            n_users = len(self.user_map)
            n_items = len(self.item_map)

            model = MatrixFactorization(
            n_users, n_items, self.config['n_factors'],
            user_map=self.user_map,
            item_map=self.item_map
            )


            optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
            criterion = torch.nn.MSELoss()

            for epoch in range(self.config['epochs']):
                model.train()
                train_loss = 0
                for batch_start in range(0, len(X_train), self.config['batch_size']):
                    batch_end = min(batch_start + self.config['batch_size'], len(X_train))
                    user_batch = torch.LongTensor(X_train[batch_start:batch_end, 0])
                    item_batch = torch.LongTensor(X_train[batch_start:batch_end, 1])
                    rating_batch = torch.FloatTensor(y_train[batch_start:batch_end])

                    optimizer.zero_grad()
                    preds = model(user_batch, item_batch)
                    loss = criterion(preds, rating_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # Validation
                model.eval()
                with torch.no_grad():
                    val_user = torch.LongTensor(X_test[:, 0])
                    val_item = torch.LongTensor(X_test[:, 1])
                    val_rating = torch.FloatTensor(y_test)
                    val_preds = model(val_user, val_item)
                    val_loss = criterion(val_preds, val_rating).item()

                mlflow.log_metric("train_loss", train_loss / len(X_train), step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss={train_loss/len(X_train):.4f}, Val Loss={val_loss:.4f}")

            # Final RMSE/MAE
            rmse = np.sqrt(val_loss)
            mae = float(torch.mean(torch.abs(val_preds - val_rating)).item())
            mlflow.log_metric("final_rmse", rmse)
            mlflow.log_metric("final_mae", mae)

            mlflow.pytorch.log_model(model, "cf_model")

            return model, self.user_map, self.item_map

    def train_content_based(self, products_df):
        """Train content-based model"""
        with mlflow.start_run(run_name="content_based"):
            cb = ContentBasedRecommender()
            cb.fit(products_df)
            joblib.dump(cb, "models/content_model.pkl")
            mlflow.log_artifact("models/content_model.pkl")
            return cb

    def evaluate_recommendations(self, model, test_interactions, user_map, item_map, K=10):
        """Ranking evaluation based on predicted ratings"""
        precisions, recalls, ndcgs = [], [], []
        all_recs = set()
        n_items = len(item_map)

        test_by_user = test_interactions.groupby("user_id")["product_id"].apply(list).to_dict()

        for u, gt_items in test_by_user.items():
            if u not in user_map.values():
                continue
            u_idx = [k for k, v in user_map.items() if v == u][0]

            items = torch.arange(n_items, dtype=torch.long)
            users = torch.full((n_items,), u_idx, dtype=torch.long)
            scores = model(users, items).detach().numpy()
            ranked = np.argsort(-scores)

            ranked_items = [item_map[i] for i in ranked]
            topk = ranked_items[:K]
            hits = [1 if pid in gt_items else 0 for pid in topk]

            p = sum(hits) / K
            r = sum(hits) / len(gt_items)
            dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_items), K)))
            ndcg = dcg / idcg if idcg > 0 else 0

            precisions.append(p); recalls.append(r); ndcgs.append(ndcg)
            all_recs.update(topk)

        metrics = {
            f"precision_at_{K}": float(np.mean(precisions)),
            f"recall_at_{K}": float(np.mean(recalls)),
            f"ndcg_at_{K}": float(np.mean(ndcgs)),
            "coverage": len(all_recs) / n_items if n_items > 0 else 0.0,
            "diversity": 1.0  # placeholder
        }
        print("Ranking Metrics:", metrics)
        return metrics


def main():
    config = {
        'mlflow_uri': 'file:./mlruns',
        'n_factors': 100,
        'learning_rate': 0.001,
        'epochs': 50,
        'batch_size': 512,
    }
    trainer = ModelTrainer(config)

    # Load from SQLite instead of CSV
    conn = sqlite3.connect("asos.db")
    interactions_df = pd.read_sql("SELECT * FROM user_interactions", conn)
    products_df = pd.read_sql("SELECT * FROM products", conn)
    conn.close()
    print("Loaded data from SQLite.", interactions_df.shape, products_df.shape)

    # Train
    cf_model, user_map, item_map = trainer.train_collaborative_filtering(interactions_df)
    content_model = trainer.train_content_based(products_df)

    # Hybrid
    hybrid = HybridRecommender(cf_model, content_model, None)
    joblib.dump(hybrid, "models/hybrid_model.pkl")

    # Save supporting models
    joblib.dump(content_model, "models/content_model.pkl")
    joblib.dump(content_model.tfidf_vectorizer, "models/vectorizer.pkl")
    joblib.dump(content_model.numerical_scaler, "models/numerical_scaler.pkl")
    joblib.dump(products_df, "models/product_catalog.pkl")

    # Evaluate
    trainer.evaluate_recommendations(cf_model, interactions_df, user_map, item_map, K=10)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
