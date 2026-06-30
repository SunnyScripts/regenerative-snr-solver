import tifffile

print(tifffile.imread('../../cellular_data/Spatial/Manchester/back/morphology_focus/ch0001_atp1a1_cd45_e-cadherin.ome.tif').shape)


import scvi
import scanpy as sc
import anndata as ad
import numpy as np

# Assuming you have:
# adata_spatial (Your 479-gene Xenium data)
# adata_seq (Your 20,000-gene Reference data, labeled in 'granular_cell_type')

# ==========================================
# PHASE 0: Imputation (gimVI)
# ==========================================
print("Training gimVI for Spatial Imputation...")

scvi.model.GIMVI.setup_anndata(adata_spatial)
scvi.model.GIMVI.setup_anndata(adata_seq, labels_key='granular_cell_type')

gimvi_model = scvi.model.GIMVI(adata_seq, adata_spatial)
gimvi_model.train(max_epochs=200, accelerator="mps", devices=1)

# Get the dense 20,000 gene profiles for your spatial cells
# normalized=True means these are now continuous floats, not raw integers
imputed_spatial = gimvi_model.get_imputed_values(normalized=True)

# Overwrite the sparse 479-gene matrix with the dense 20k matrix
adata_spatial.X = imputed_spatial

# ==========================================
# THE DATA MERGE
# ==========================================
print("Merging Datasets...")
# THE TRAP: adata_spatial.X is now continuous floats. adata_seq.X is still raw integers.
# If we merge them now, scVI will crash because the math distributions don't match.
# THE FIX: We must normalize the reference data to match the imputed data's format.

sc.pp.normalize_total(adata_seq, target_sum=1e4)
sc.pp.log1p(adata_seq)

# Give the spatial data a dummy label so the columns align
adata_spatial.obs['granular_cell_type'] = "Unknown"

# Stack them on top of each other into one giant AnnData object
# 'batch_key' will remember which rows came from which dataset
adata_joint = ad.concat(
    [adata_seq, adata_spatial],
    label="dataset_origin",
    keys=["reference_seq", "spatial_xenium"]
)

# ==========================================
# PHASE 1: scVI (The Sharpener)
# ==========================================
print("Training Joint scVI Latent Space...")

# Tell scVI to correct for the batch effect between Seq and Xenium
scvi.model.SCVI.setup_anndata(adata_joint, batch_key="dataset_origin")

# CRITICAL: Because we are using normalized/imputed floats, gene_likelihood MUST be "normal"
joint_scvi_model = scvi.model.SCVI(
    adata_joint,
    n_layers=2,
    n_latent=30,
    gene_likelihood="normal"
)

joint_scvi_model.train(
    max_epochs=100,
    accelerator="mps",
    devices=1,
    early_stopping=True
)

# ==========================================
# PHASE 2 & 3: scANVI (The Granular Classifier)
# ==========================================
print("Training Granular scANVI Classifier...")

# Re-register the joint dataset, this time telling it where the labels are
scvi.model.SCANVI.setup_anndata(
    adata_joint,
    batch_key="dataset_origin",
    labels_key="granular_cell_type",
    unlabeled_category="Unknown"
)

granular_scanvi_model = scvi.model.SCANVI.from_scvi_model(
    joint_scvi_model,
    labels_key="granular_cell_type",
    unlabeled_category="Unknown"
)

# Train the classifier to separate the 70 types
granular_scanvi_model.train(
    max_epochs=50,
    batch_size=2048,
    accelerator="mps",
    devices=1,
    early_stopping=True
)

# ==========================================
# PHASE 4: Extract the Results for Rust
# ==========================================
print("Extracting final predictions for Rust...")

# Ask the model to predict the labels for EVERY cell in the joint object
adata_joint.obs["predicted_granular_type"] = granular_scanvi_model.predict()

# Filter the giant object to just get your Xenium spatial cells back
final_spatial_data = adata_joint[adata_joint.obs["dataset_origin"] == "spatial_xenium"].copy()

# final_spatial_data.obs["predicted_granular_type"] now contains your 70 exact strings!
print(final_spatial_data.obs["predicted_granular_type"].value_counts())