#Mechanical (Skin, Lung, Esophagus, Colon, Duodenum, Rectum, Small intestine, Cervix)
#Metobolic (Pancreas, Liver, Salivary gland, Gallbladder, Prostate)
#Endocrine/Reproductive (Adrenal, Parathyroid, Ovary, Endometrium, Placenta)

import scvi
import json
import numpy as np
import torch
from safetensors.torch import save_file

# ==========================================
# 1. TRAINING PHASE
# ==========================================

# Setup AnnData with the Batch Key (Crucial for Pan-Tissue)
scvi.model.SCVI.setup_anndata(
    adata_pan,
    labels_key='universal_cell_type',
    batch_key='batch_tissue' # e.g., "Skin", "Lung", "Gut"
)

# Train the Unsupervised Latent Space (scVI)
print("Training scVI Latent Space...")
pan_scvi_model = scvi.model.SCVI(adata_pan, n_latent=20)
pan_scvi_model.train(accelerator='mps')

# Train the Supervised Classifier (scANVI)
print("Training scANVI Classifier...")
pan_scanvi_model = scvi.model.SCANVI.from_scvi_model(
    pan_scvi_model,
    unlabeled_category="Unknown"
)
pan_scanvi_model.train(accelerator='mps')

print("✅ Training Complete. Starting Wasm Export Pipeline...")

# ==========================================
# 2. THE WASM EXPORT PIPELINE
# ==========================================

# --- Artifact 1: model_genes.json ---
# The exact ordered list of 768 genes the tensor expects.
model_genes = adata_pan.var_names.tolist()

with open("model_genes.json", "w") as f:
    json.dump(model_genes, f)
print(f"📦 Exported model_genes.json ({len(model_genes)} features)")


# --- Artifact 2: model_means.json ---
# The biological averages used by Rust to pad missing hardware genes.
# We must handle both sparse (CSR) and dense matrices safely.
if type(adata_pan.X).__name__ in ["csr_matrix", "csc_matrix"]:
    # .mean(axis=0) returns a 2D matrix, .A1 flattens it to a 1D numpy array
    model_means = adata_pan.X.mean(axis=0).A1.tolist()
else:
    model_means = adata_pan.X.mean(axis=0).tolist()

with open("model_means.json", "w") as f:
    json.dump(model_means, f)
print("📦 Exported model_means.json (Mean-Padding values)")


# --- Artifact 3: scanvi_weights.safetensors ---
# Extracting the raw PyTorch weights for Burn.
# We grab the underlying PyTorch module from the scvi-tools wrapper.
pytorch_module = pan_scanvi_model.module

# Safetensors requires all tensors to be contiguous and on the CPU.
state_dict = pytorch_module.state_dict()
clean_state_dict = {
    key: tensor.cpu().contiguous()
    for key, tensor in state_dict.items()
}

# Save strictly the weights to the highly efficient Safetensors format
save_file(clean_state_dict, "scanvi_pan_barrier.safetensors")
print("📦 Exported scanvi_pan_barrier.safetensors")

print("🚀 Export complete! Ready to compile into Rust Wasm.")