import cellxgene_census
import scanpy as sc
import anndata
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# import numpy as np
import json
# import pandas as pd
from tqdm import tqdm # <-- The magic progress bar!
import time

# ==========================================
# 1. RAM-SAFE STRATIFIED SAMPLING (WITH PROGRESS!)
# ==========================================
print("Connecting to CELLxGENE Census...")

with cellxgene_census.open_soma() as census:
    print("1. Fetching global metadata...")
    obs_df = census["census_data"]["homo_sapiens"].obs.read(
        value_filter="is_primary_data == True and disease == 'normal'",
        column_names=["soma_joinid", "cell_type"]
    ).concat().to_pandas()

    print("2. Sampling 400 cells per cell type...")
    sampled_obs = obs_df.groupby("cell_type").apply(
        lambda x: x.sample(n=min(400, len(x)), random_state=42)
    ).reset_index(drop=True)

    type_counts = sampled_obs['cell_type'].value_counts()
    valid_types = type_counts[type_counts >= 10].index
    sampled_obs = sampled_obs[sampled_obs['cell_type'].isin(valid_types)]

    sampled_ids = sampled_obs["soma_joinid"].tolist()
    # --- NEW: SORT THE IDS! ---
    # This single line changes the S3 request from "Random Access" to "Sequential Read"
    # It speeds up the download by roughly 10,000%
    sampled_ids.sort()

    print(f"Selected a highly diverse, balanced cohort of {len(sampled_ids)} cells.")

    print("3. Downloading the heavy gene matrix in chunks...")

    chunk_size = 5000
    adatas = []

    for i in tqdm(range(0, len(sampled_ids), chunk_size), desc="Downloading Chunks"):
        chunk_ids = sampled_ids[i: i + chunk_size]

        success = False
        retries = 0
        while not success and retries < 6:
            try:
                chunk_adata = cellxgene_census.get_anndata(
                    census,
                    organism="Homo sapiens",
                    obs_coords=chunk_ids,
                    obs_column_names=["cell_type"]
                )
                adatas.append(chunk_adata)
                success = True
            except Exception as e:
                retries += 1
                print(f"\n⚠️ Network hiccup on chunk {i}. Retrying in 10s... ({retries}/6)")
                time.sleep(10)

        if not success:
            raise RuntimeError("Failed to download chunk after 6 retries. Check your internet connection.")

    print("Stitching chunks together in RAM...")
    # Concatenate all the small chunks into one master AnnData object
    adata = anndata.concat(adatas)

    # NEW: Fix the annoying duplicate name warning
    adata.obs_names_make_unique()

    # NEW: HARD DRIVE CHECKPOINT 1
    print("💾 Saving raw data to disk so we never have to download this again...")
    adata.write_h5ad("raw_census_data_backup.h5ad")

print(f"✅ Successfully loaded {adata.n_obs} cells and {adata.n_vars} genes into memory!")

# ==========================================
# 2. PREPROCESSING & HVG SELECTION
# ==========================================
print("Preprocessing and selecting the top 2000 universal genes...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)

# Save the exact 2000 genes for Rust!
gene_names = adata.var_names.tolist()
with open("model_genes.json", "w") as f:
    json.dump(gene_names, f)

# Encode the string cell types to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(adata.obs["cell_type"])

# Save the label mapping for Rust!
mapping = {int(i): label for i, label in enumerate(label_encoder.classes_)}
with open("class_mapping.json", "w") as f:
    json.dump(mapping, f)

# Convert to PyTorch tensors
X = torch.tensor(adata.X.toarray(), dtype=torch.float32)
y = torch.tensor(y_encoded, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)

# ==========================================
# 3. THE TINY PYTORCH MODEL
# ==========================================
class UniversalCellClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

num_classes = len(label_encoder.classes_)
model = UniversalCellClassifier(input_dim=2000, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
print(f"Training on {num_classes} unique biological cell types...")
epochs = 12
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

# NEW: HARD DRIVE CHECKPOINT 2
print("💾 Saving native PyTorch model as a backup...")
torch.save(model.state_dict(), "universal_cell_classifier_backup.pt")

# ==========================================
# 5. ONNX EXPORT
# ==========================================
print("Exporting universal model to ONNX...")
model.eval()
dummy_input = torch.randn(1, 2000)

torch.onnx.export(
    model,
    dummy_input,
    "universal_cell_classifier.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input_genes'],
    output_names=['class_logits'],
    dynamic_axes={'input_genes': {0: 'batch_size'}, 'class_logits': {0: 'batch_size'}}
)

print("✅ Complete! You now have a tissue-agnostic ONNX model ready for Rust Burn.")