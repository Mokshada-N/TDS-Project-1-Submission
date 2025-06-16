# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "sentence_transformers",
#   "faiss-cpu"
# ]
# ///

import faiss
import json

# === Load indexes ===
print("🔹 Loading Discourse FAISS index...")
disc_index = faiss.read_index("faiss_index.idx")

print("🔹 Loading Markdown FAISS index...")
md_index = faiss.read_index("faiss_md_index.idx")

# === Merge indexes ===
print(f"🔹 Discourse index vectors: {disc_index.ntotal}")
print(f"🔹 Markdown index vectors: {md_index.ntotal}")

# Simply add markdown vectors into discourse index
disc_index.add(md_index.reconstruct_n(0, md_index.ntotal))

print(f"✅ Combined index vectors: {disc_index.ntotal}")

# === Save combined index ===
faiss.write_index(disc_index, "faiss_combined_index.idx")
print("✅ Combined FAISS index saved as faiss_combined_index.idx")

# === Combine metadata ===
print("🔹 Loading metadata...")
with open("embedding_data.json", "r", encoding="utf-8") as f:
    disc_meta = json.load(f)

with open("embedding_md_data.json", "r", encoding="utf-8") as f:
    md_meta = json.load(f)

combined_meta = disc_meta + md_meta

# === Save combined metadata ===
with open("embedding_combined_data.json", "w", encoding="utf-8") as f:
    json.dump(combined_meta, f, indent=2)

print("✅ Combined metadata saved as embedding_combined_data.json")
