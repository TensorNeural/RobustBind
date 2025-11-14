import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Config ===
embedding_dir = "/data/output/embeddings"
output_dir = "/data/output/diagnostics"
os.makedirs(output_dir, exist_ok=True)

# === Load UNIBIND_CLEAN modality embeddings ===
embedding_files = sorted([
    f for f in os.listdir(embedding_dir)
    if f.startswith("UNIBIND_CLEAN_") and f.endswith(".npy")
])

print(f"Found {len(embedding_files)} UNIBIND_CLEAN_* embeddings")

# === Compute center of each modality and normalize ===
embeddings = {}
for fname in embedding_files:
    path = os.path.join(embedding_dir, fname)
    arr = np.load(path)  # shape: [N, D]
    center = arr.mean(axis=0, keepdims=True)  # [1, D]
    center = F.normalize(torch.tensor(center), dim=-1)  # unit vector
    label = fname.replace(".npy", "")  # e.g., UNIBIND_CLEAN_IMAGE
    embeddings[label] = center

# === Compute cosine similarity matrix ===
labels = list(embeddings.keys())
matrix = np.zeros((len(labels), len(labels)))

for i, k1 in enumerate(labels):
    for j, k2 in enumerate(labels):
        sim = torch.matmul(embeddings[k1], embeddings[k2].T).item()
        matrix[i, j] = sim

# === Save as CSV ===
df = pd.DataFrame(matrix, index=labels, columns=labels)
csv_path = os.path.join(output_dir, "unibind_clean_center_similarity.csv")
df.to_csv(csv_path)
print(f"Saved cosine similarity CSV: {csv_path}")

# === Save heatmap ===
plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Cosine Similarity Between Modality Centers (UNIBIND_CLEAN_*)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

heatmap_path = os.path.join(output_dir, "unibind_clean_center_similarity.png")
plt.savefig(heatmap_path, dpi=300)
print(f"Saved heatmap: {heatmap_path}")
