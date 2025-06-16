# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "sentence_transformers",
#   "faiss-cpu"
# ]
# ///

import json
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from pathlib import Path

def clean_text(text):
    return " ".join(text.strip().split())

# === Load and combine data ===
topics = {}

for json_file in Path("discourse_posts_updated").glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        topic_data = json.load(f)
        topic_id = topic_data["topic_id"]
        topics[topic_id] = {
            "topic_title": topic_data.get("topic_title", ""),
            "posts": topic_data["posts"]
        }

print(f"Loaded {len(topics)} topics from {len(list(Path('discourse_posts_updated').glob('*.json')))} JSON files.")

# === Sort posts within each topic ===
for topic_id in topics:
    topics[topic_id]["posts"].sort(key=lambda p: p["post_number"])

# === Prepare embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

embedding_data = []
embeddings = []

def build_reply_map(posts):
    reply_map = defaultdict(list)
    posts_by_number = {}
    for post in posts:
        posts_by_number[post["post_number"]] = post
        parent = post.get("reply_to_post_number")
        reply_map[parent].append(post)
    return reply_map, posts_by_number

def extract_subthread(root_post_number, reply_map, posts_by_number):
    collected = []
    def dfs(post_num):
        post = posts_by_number[post_num]
        collected.append(post)
        for child in reply_map.get(post_num, []):
            dfs(child["post_number"])
    dfs(root_post_number)
    return collected

# === Build embeddings ===
print("Building embeddings and FAISS index...")

for topic_id, topic_data in tqdm(topics.items()):
    posts = topic_data["posts"]
    topic_title = topic_data["topic_title"]

    reply_map, posts_by_number = build_reply_map(posts)
    root_posts = reply_map[None]

    for root_post in root_posts:
        root_num = root_post["post_number"]
        subthread_posts = extract_subthread(root_num, reply_map, posts_by_number)

        combined_text = f"Topic title: {topic_title}\n\n"
        combined_text += "\n\n---\n\n".join(clean_text(p["content"]) for p in subthread_posts)
        combined_text += "\n\n---\n\n".join(clean_text(p["image_description"]) for p in subthread_posts if p["image_description"] is not None)

        emb = model.encode(combined_text, convert_to_numpy=True, normalize_embeddings=True)
        embedding_data.append({
            "topic_id": topic_id,
            "topic_title": topic_title,
            "root_post_number": root_num,
            "post_numbers": [p["post_number"] for p in subthread_posts],
            "combined_text": combined_text,
            "url": [p["post_url"] for p in subthread_posts]
        })
        embeddings.append(emb)

# === Save results ===
with open("embedding_data.json", "w", encoding="utf-8") as f:
    json.dump(embedding_data, f, indent=2)

embeddings_np = np.vstack(embeddings).astype("float32")
index = faiss.IndexFlatIP(embeddings_np.shape[1])
index.add(embeddings_np)
faiss.write_index(index, "faiss_index.idx")

print("✅ Embeddings and FAISS index saved.")

