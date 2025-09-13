import os
import math
import requests
import pandas as pd
import sqlite3

API_KEY = os.getenv("RAPIDAPI_KEY", "PASTE_YOUR_KEY_HERE")   # or set RAPIDAPI_KEY in your environment
STORE = "US"
CURRENCY = "USD"
COUNTRY = "US"
LANG = "en-US"

# One or more category IDs to fetch
CATEGORY_IDS = ["4209", "2641", "2640", "8799", "2238"]

CATEGORY_NAME_MAP = {
    "4209": "Men's Shoes",
    "2641": "Women's Coats and Jackets",
    "2640": "Women's Pants",
    "8799": "Women's Dresses",
    "2238": "Women's Swimwear and Beachwear",
}

PAGE_LIMIT = 10
MAX_ITEMS_PER_CATEGORY = 40   # 5 categories × 40 = ~200 products

# Paths
OUTPUT_PATH = "data/products.csv" 
DB_PATH = "asos.db"  

API_KEY = "XXXXXXXXXXXXXXXXXXX"

LIST_URL = "https://asos2.p.rapidapi.com/products/v2/list"
HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "asos2.p.rapidapi.com",
}

def clean_image_url(img: str | None) -> str | None:
    if not img:
        return None
    if img.startswith("//"):
        return "https:" + img
    if img.startswith("http"):
        return img
    return "https://" + img.lstrip("/")

def extract_price_fields(price_obj: dict | None) -> tuple[float | None, float | None]:
    if not isinstance(price_obj, dict):
        return None, None
    cur = (price_obj.get("current") or {}).get("value")
    prev = (price_obj.get("previous") or {}).get("value")
    ratio = (cur / prev) if (cur and prev and prev > 0) else None
    return cur, ratio

def to_full_url(url_path: str | None, pid: int | str) -> str | None:
    if not url_path:
        return f"https://www.asos.com/prd/{pid}"
    if url_path.startswith("http"):
        return url_path
    return "https://www.asos.com" + url_path

def fetch_page(category_id: str, offset: int, limit: int) -> list[dict]:
    params = {
        "store": STORE,
        "offset": str(offset),
        "categoryId": str(category_id),
        "limit": str(limit),
        "currency": CURRENCY,
        "country": COUNTRY,
        "lang": LANG,
    }
    r = requests.get(LIST_URL, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    data = r.json() or {}
    return data.get("products", [])

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    all_rows: list[dict] = []
    seen_ids: set[str] = set()

    for cat in CATEGORY_IDS:
        total_fetched = 0
        offset = 0
        pages = math.ceil(MAX_ITEMS_PER_CATEGORY / PAGE_LIMIT)

        for _ in range(pages):
            products = fetch_page(cat, offset, PAGE_LIMIT)
            if not products:
                break

            for prod in products:
                pid = prod.get("id")
                if pid is None:
                    continue
                pid_str = str(pid)
                if pid_str in seen_ids:
                    continue

                name = prod.get("name")
                brand = prod.get("brandName")
                category = CATEGORY_NAME_MAP.get(str(cat), "Unknown")

                price_current, price_ratio = extract_price_fields(prod.get("price"))
                image_url = clean_image_url(prod.get("imageUrl"))
                url_full = to_full_url(prod.get("url"), pid_str)

                all_rows.append({
                    "product_id": pid_str,
                    "name": name,
                    "brand": brand,
                    "category": category,
                    "category_id": str(cat),
                    "price_current": price_current,
                    "price_ratio": price_ratio,
                    "image_url": image_url,
                    "url": url_full,
                })
                seen_ids.add(pid_str)
                total_fetched += 1

                if total_fetched >= MAX_ITEMS_PER_CATEGORY:
                    break

            if total_fetched >= MAX_ITEMS_PER_CATEGORY:
                break
            offset += PAGE_LIMIT

    if not all_rows:
        print("No products returned. Check your API key/plan and category IDs.")
        return

    df = pd.DataFrame(all_rows)

    # Ensure columns exist even if API missed some fields
    for col in ["product_id","name","brand","category_id","category",
                "price_current","price_ratio","image_url","url"]:
        if col not in df.columns:
            df[col] = None

    # Shuffle the rows so categories are mixed
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # CSV (for easier viewing and testing)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} products to {OUTPUT_PATH}")

    # SQL
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("products", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved {len(df)} products into 'products' table in {DB_PATH}")

if __name__ == "__main__":
    main()
