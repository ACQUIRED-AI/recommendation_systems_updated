# make_interactions_easy.py
import os, random
import pandas as pd
from collections import defaultdict
import sqlite3

PRODUCTS_PATH = "data/products.csv"
OUT_CSV = "data/interactions.csv"

NUM_USERS = 500
EVENTS_PER_USER = 100
SEED = 42
random.seed(SEED)

if not os.path.exists(PRODUCTS_PATH):
    raise FileNotFoundError("Run fetch_asos_products.py first to create data/products.csv")

p = pd.read_csv(PRODUCTS_PATH)
p["product_id"] = p["product_id"].astype(str)
p["category"] = p["category"].fillna("").astype(str)
p["name"] = p["name"].fillna("").astype(str)

def bucket(row):
    text = f"{row.get('category','')} {row.get('name','')}".lower()
    if any(k in text for k in ["shoe", "sneaker", "trainer", "boot"]): return "shoes"
    if "dress" in text: return "dresses"
    if any(k in text for k in ["jacket", "coat", "puffer", "parka"]): return "jackets"
    return "other"

p["bucket"] = p.apply(bucket, axis=1)

by_bucket = defaultdict(list)
for _, r in p.iterrows():
    by_bucket[r["bucket"]].append(r["product_id"])

catalog_ids = p["product_id"].tolist()
for b in ["shoes","dresses","jackets","other"]:
    if not by_bucket[b]:
        by_bucket[b] = catalog_ids

users = [f"u{i}" for i in range(1, NUM_USERS+1)]
bucket_cycle = ["shoes","dresses","jackets","other"]
user_bucket = {u: bucket_cycle[(i-1) % 4] for i, u in enumerate(users, start=1)}

rows = []
all_buckets = ["shoes","dresses","jackets","other"]

for u in users:
    liked = user_bucket[u]
    pos_pool = list(set(by_bucket[liked]))  # unique liked items
    neg_pool = [pid for b in all_buckets if b != liked for pid in by_bucket[b]]

    n_history = EVENTS_PER_USER - 1
    used_pos = set()

    # History: mix of ratings 1–5
    for _ in range(n_history):
        r = random.random()
        if r < 0.5:  # strong like (50% chance)
            pid = random.choice(pos_pool)
            rating = 5
            used_pos.add(pid)
        elif r < 0.65:  # mild like (15%)
            pid = random.choice(pos_pool)
            rating = 4
            used_pos.add(pid)
        elif r < 0.75:  # neutral (10%)
            pid = random.choice(catalog_ids)
            rating = 3
        elif r < 0.9:  # mild dislike (15%)
            pid = random.choice(neg_pool)
            rating = 2
        else:          # strong dislike (10%)
            pid = random.choice(neg_pool)
            rating = 1
        rows.append([u, pid, rating])

    # Test item = unseen positive (always 5)
    unused_pos = list(set(pos_pool) - used_pos)
    if not unused_pos:
        unused_pos = pos_pool
    test_pid = random.choice(unused_pos)
    rows.append([u, test_pid, 5])


df = pd.DataFrame(rows, columns=["user_id","product_id","rating"])

os.makedirs("data", exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df)} interactions to {OUT_CSV}")

conn = sqlite3.connect("asos.db")
df.to_sql("user_interactions", conn, if_exists="replace", index=False)
conn.close()
print(f"Saved {len(df)} interactions into 'user_interactions' table in asos.db")

