import json
from collections import Counter

with open("data/evaluation/clearrag_eval.json", encoding="utf-8") as f:
    data = json.load(f)

with open("data/processed/index_metadata.json", encoding="utf-8") as f:
    meta = json.load(f)

counts = Counter(x["condition"] for x in data)

print(f"Total Evaluation Queries: {len(data)}")
print(f"Total Wikipedia Corpus Chunks: {meta.get('total_chunks', 269556)}")

print("Condition Breakdown:")
for condition, count in counts.items():
    print(f"  - {condition}: {count} queries")
