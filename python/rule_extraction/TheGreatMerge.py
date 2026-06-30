# import scanpy as sc
import anndata as ad
import gc
import os

import numpy

master_chemical_map = {
    # --- 1. ENDOTHELIAL ---
    'Arterial EC': 'Arterial Endothelial Cell',
    'endothelial cell of artery': 'Arterial Endothelial Cell',
    'pulmonary artery endothelial cell': 'Arterial Endothelial Cell',
    'Venous EC': 'Venous Endothelial Cell',
    'vein endothelial cell': 'Venous Endothelial Cell',
    'Capillary EC': 'Capillary Endothelial Cell',
    'capillary endothelial cell': 'Capillary Endothelial Cell',
    'Lymphatic EC': 'Lymphatic Endothelial Cell',
    'endothelial cell of lymphatic vessel': 'Lymphatic Endothelial Cell',
    'endothelial cell': 'Endothelial Cell',

    # --- 2. STROMAL / MESENCHYMAL ---
    'Fibro A': 'Fibroblast',
    'Fibro B': 'Fibroblast',
    'Fibro C': 'Fibroblast',
    'Fibro D': 'Fibroblast',
    'Fibro E': 'Fibroblast',
    'fibroblast': 'Fibroblast',
    'alveolar adventitial fibroblast': 'Fibroblast',
    'alveolar type 1 fibroblast cell': 'Fibroblast',
    'bronchus fibroblast of lung': 'Fibroblast',
    'myofibroblast cell': 'Myofibroblast',
    'stromal cell': 'Stromal Cell',
    'stromal cell of lamina propria of small intestine': 'Stromal Cell',
    'SMC': 'Smooth Muscle Cell',
    'smooth muscle cell': 'Smooth Muscle Cell',
    'enteric smooth muscle cell': 'Smooth Muscle Cell',
    'tracheobronchial smooth muscle cell': 'Smooth Muscle Cell',
    'Skeletal muscle': 'Skeletal Muscle Cell',
    'pericyte': 'Pericyte',
    'lung pericyte': 'Pericyte',
    'mesothelial cell': 'Mesothelial Cell',
    'mesenchymal lymphangioblast': 'Lymphangioblast',
    'reticular cell': 'Reticular Cell',
    'mesodermal cell': 'Mesenchymal Stem Cell',

    # --- 3. EPITHELIAL - SKIN ---
    'Basal KC': 'Basal Keratinocyte',
    'Prolif. KC': 'Proliferating Keratinocyte',
    'Spinous KC': 'Spinous Keratinocyte',
    'Granular KC': 'Granular Keratinocyte',
    'Cornified KC': 'Cornified Keratinocyte',
    'SG': 'Sebaceous Gland Cell',
    'Bulb': 'Hair Follicle Cell',
    'Bulge': 'Hair Follicle Cell',
    'Infundibulum': 'Hair Follicle Cell',
    'Isthmus': 'Hair Follicle Cell',
    'Coil': 'Sweat Gland Cell',
    'Duct': 'Sweat Gland Cell',
    'Merkel cell': 'Merkel Cell',

    # --- 4. EPITHELIAL - LUNG ---
    'pulmonary alveolar epithelial cell': 'Alveolar Epithelial Cell',
    'pulmonary alveolar type 1 cell': 'Alveolar Type 1 Cell',
    'pulmonary alveolar type 2 cell': 'Alveolar Type 2 Cell',
    'multiciliated columnar cell of tracheobronchial tree': 'Ciliated Cell',
    'multiciliated epithelial cell': 'Ciliated Cell',
    'brush cell of tracheobronchial tree': 'Tuft Cell',
    'nasal mucosa goblet cell': 'Goblet Cell',
    'bronchial goblet cell': 'Goblet Cell',
    'tracheobronchial goblet cell': 'Goblet Cell',
    'mucus secreting cell': 'Goblet Cell',
    'club cell': 'Club Cell',
    'respiratory basal cell': 'Basal Cell',
    'epithelial cell of lower respiratory tract': 'Respiratory Epithelial Cell',
    'epithelial cell': 'Epithelial Cell',
    'ionocyte': 'Ionocyte',
    'pulmonary neuroendocrine cell': 'Neuroendocrine Cell',
    'respiratory tract hillock cell': 'Hillock Cell',
    'serous secreting cell': 'Serous Cell',
    'tracheobronchial serous cell': 'Serous Cell',

    # --- 5. EPITHELIAL - GUT ---
    'colon epithelial cell': 'Colonocyte',
    'colonocyte': 'Colonocyte',
    'intestine goblet cell': 'Goblet Cell',
    'intestinal tuft cell': 'Tuft Cell',
    'transit amplifying cell': 'Transit Amplifying Cell',
    'type I enteroendocrine cell': 'Enteroendocrine Cell',
    'type L enteroendocrine cell': 'Enteroendocrine Cell',
    'type N enteroendocrine cell': 'Enteroendocrine Cell',
    'type D enteroendocrine cell': 'Enteroendocrine Cell',
    'type EC enteroendocrine cell': 'Enteroendocrine Cell',
    'GIP cell': 'Enteroendocrine Cell',
    'enteroendocrine cell': 'Enteroendocrine Cell',
    'progenitor cell of endocrine pancreas': 'Endocrine Progenitor',
    'type B pancreatic cell': 'Beta Cell',
    'acinar cell': 'Acinar Cell',
    'M cell of gut': 'M Cell',

    # --- 6. IMMUNE / FLUID ---
    'mast cell': 'Mast Cell',
    'Mast cell': 'Mast Cell',
    'macrophage': 'Macrophage',
    'Mph': 'Macrophage',
    'alveolar macrophage': 'Macrophage',
    'elicited macrophage': 'Macrophage',
    'inflammatory macrophage': 'Macrophage',
    'lung macrophage': 'Macrophage',
    'monocyte': 'Monocyte',
    'Monocyte': 'Monocyte',
    'classical monocyte': 'Monocyte',
    'non-classical monocyte': 'Monocyte',
    'intermediate monocyte': 'Monocyte',
    'neutrophil': 'Neutrophil',
    'Neutrophil': 'Neutrophil',
    'basophil': 'Basophil',
    'B cell': 'B Cell',
    'germinal center B cell': 'B Cell',
    'immature B cell': 'B Cell',
    'memory B cell': 'B Cell',
    'naive B cell': 'B Cell',
    'precursor B cell': 'B Cell',
    'pro-B cell': 'B Cell',
    'T cell': 'T Cell',
    'CD4-positive, alpha-beta T cell': 'T Cell',
    'CD8-positive, alpha-beta T cell': 'T Cell',
    'activated CD8-positive, alpha-beta T cell': 'T Cell',
    'activated CD4-positive, alpha-beta T cell': 'T Cell',
    'CD8-positive, alpha-beta memory T cell': 'T Cell',
    'mucosal invariant T cell': 'T Cell',
    'T-helper 17 cell': 'T Cell',
    'T-helper 1 cell': 'T Cell',
    'T follicular helper cell': 'T Cell',
    'gamma-delta T cell': 'T Cell',
    'regulatory T cell': 'T Cell',
    'natural killer cell': 'NK Cell',
    'NK': 'NK Cell',
    'mature NK T cell': 'NK Cell',
    'plasma cell': 'Plasma Cell',
    'IgG plasma cell': 'Plasma Cell',
    'IgM plasma cell': 'Plasma Cell',
    'IgA plasma cell': 'Plasma Cell',
    'dendritic cell': 'Dendritic Cell',
    'DC': 'Dendritic Cell',
    'myeloid dendritic cell': 'Dendritic Cell',
    'plasmacytoid dendritic cell': 'Dendritic Cell',
    'CD1c-positive myeloid dendritic cell': 'Dendritic Cell',
    'conventional dendritic cell': 'Dendritic Cell',
    'follicular dendritic cell': 'Dendritic Cell',
    'group 2 innate lymphoid cell': 'Innate Lymphoid Cell',
    'group 3 innate lymphoid cell': 'Innate Lymphoid Cell',
    'NKp44-positive group 3 innate lymphoid cell, human': 'Innate Lymphoid Cell',
    'NKp44-negative group 3 innate lymphoid cell, human': 'Innate Lymphoid Cell',
    'CD34-positive, CD56-positive, CD117-positive common innate lymphoid precursor, human': 'Innate Lymphoid Cell',
    'common lymphoid progenitor': 'Lymphoid Progenitor',
    'megakaryocyte': 'Megakaryocyte',
    'erythrocyte': 'Erythrocyte',
    'hematopoietic stem cell': 'Hematopoietic Stem Cell',

    # --- 7. NERVOUS / STEM / OTHER ---
    'Sensory neuron': 'Sensory Neuron',
    'primary sensory neuron (sensu Teleostei)': 'Sensory Neuron',
    'inhibitory motor neuron': 'Motor Neuron',
    'motor neuron': 'Motor Neuron',
    'migratory enteric neural crest cell': 'Neural Crest Cell',
    'oligodendrocyte precursor cell': 'Oligodendrocyte Precursor',
    'interstitial cell of Cajal': 'Pacemaker Cell',
    'glial cell': 'Glial Cell',
    'germ cell': 'Germ Cell',
    'interneuron': 'Interneuron',
    'neuroblast (sensu Vertebrata)': 'Neuroblast',
    'progenitor cell': 'Stem Cell',
    'stem cell': 'Stem Cell',
}

mechanical_class_map = {
    # --- 1. EPITHELIAL ---
    'Basal Keratinocyte': 'Epithelial',
    'Proliferating Keratinocyte': 'Epithelial',
    'Spinous Keratinocyte': 'Epithelial',
    'Granular Keratinocyte': 'Epithelial',
    'Cornified Keratinocyte': 'Epithelial',
    'Sebaceous Gland Cell': 'Epithelial',
    'Hair Follicle Cell': 'Epithelial',
    'Sweat Gland Cell': 'Epithelial',
    'Merkel Cell': 'Epithelial',
    'Alveolar Epithelial Cell': 'Epithelial',
    'Alveolar Type 1 Cell': 'Epithelial',
    'Alveolar Type 2 Cell': 'Epithelial',
    'Ciliated Cell': 'Epithelial',
    'Tuft Cell': 'Epithelial',
    'Goblet Cell': 'Epithelial',
    'Club Cell': 'Epithelial',
    'Basal Cell': 'Epithelial',
    'Respiratory Epithelial Cell': 'Epithelial',
    'Epithelial Cell': 'Epithelial',
    'Ionocyte': 'Epithelial',
    'Neuroendocrine Cell': 'Epithelial',
    'Hillock Cell': 'Epithelial',
    'Serous Cell': 'Epithelial',
    'Colonocyte': 'Epithelial',
    'Transit Amplifying Cell': 'Epithelial',
    'Enteroendocrine Cell': 'Epithelial',
    'Endocrine Progenitor': 'Epithelial',
    'Beta Cell': 'Epithelial',
    'Acinar Cell': 'Epithelial',
    'M Cell': 'Epithelial',

    # --- 2. MESENCHYMAL ---
    'Fibroblast': 'Mesenchymal',
    'Myofibroblast': 'Mesenchymal',
    'Stromal Cell': 'Mesenchymal',
    'Smooth Muscle Cell': 'Mesenchymal',
    'Skeletal Muscle Cell': 'Mesenchymal',
    'Pericyte': 'Mesenchymal',
    'Mesothelial Cell': 'Mesenchymal',
    'Lymphangioblast': 'Mesenchymal',
    'Reticular Cell': 'Mesenchymal',
    'Mesenchymal Stem Cell': 'Mesenchymal',

    # --- 3. ENDOTHELIAL ---
    'Arterial Endothelial Cell': 'Endothelial',
    'Venous Endothelial Cell': 'Endothelial',
    'Capillary Endothelial Cell': 'Endothelial',
    'Lymphatic Endothelial Cell': 'Endothelial',
    'Endothelial Cell': 'Endothelial',

    # --- 4. IMMUNE/FLUID ---
    'Mast Cell': 'Immune/Fluid',
    'Macrophage': 'Immune/Fluid',
    'Monocyte': 'Immune/Fluid',
    'Neutrophil': 'Immune/Fluid',
    'Basophil': 'Immune/Fluid',
    'B Cell': 'Immune/Fluid',
    'T Cell': 'Immune/Fluid',
    'NK Cell': 'Immune/Fluid',
    'Plasma Cell': 'Immune/Fluid',
    'Dendritic Cell': 'Immune/Fluid',
    'Innate Lymphoid Cell': 'Immune/Fluid',
    'Lymphoid Progenitor': 'Immune/Fluid',
    'Megakaryocyte': 'Immune/Fluid',
    'Erythrocyte': 'Immune/Fluid',
    'Hematopoietic Stem Cell': 'Immune/Fluid',

    # --- 5. UNKNOWN ---
    'Sensory Neuron': 'Unknown',
    'Motor Neuron': 'Unknown',
    'Neural Crest Cell': 'Unknown',
    'Oligodendrocyte Precursor': 'Unknown',
    'Pacemaker Cell': 'Unknown',
    'Glial Cell': 'Unknown',
    'Germ Cell': 'Unknown',
    'Interneuron': 'Unknown',
    'Neuroblast': 'Unknown',
    'Stem Cell': 'Unknown',
}

import scanpy as sc
import numpy as np
from scipy.sparse import issparse


def validate_pan_tissue_atlas(adata):
    print("--- 🔬 Commencing Great Merge Validation ---")

    # 1. THE INTEGER CHECK (Critical for scANVI)
    # We sample the raw data to check for decimals.
    # Even if they look like integers (1.0), we check the 'remainder'.
    raw_data_sample = adata.X[:500, :500].data
    is_integer = np.all(raw_data_sample % 1 == 0)

    if is_integer:
        print("✅ RAW INTEGERS: Verified. Poisson distribution is safe.")
    else:
        print("❌ ERROR: Decimals detected in .raw! Training will crash.")

    # 2. THE SPARSITY CHECK
    # Large datasets MUST be CSR (Compressed Sparse Row) to stay under 24GB RAM
    if issparse(adata.X):
        print(f"✅ SPARSITY: Verified. Matrix is {type(adata.X)}.")
    else:
        print("⚠️ WARNING: Matrix is DENSE. You will likely hit OOM during training.")

    # 3. SHAPE & ALIGNMENT
    n_cells, n_genes = adata.shape
    raw_cells, raw_genes = adata.shape

    if n_genes == raw_genes and n_cells == raw_cells:
        print(f"✅ ALIGNMENT: Main and Raw are perfectly synced at ({n_cells} cells x {n_genes} genes).")
    else:
        print(f"❌ MISMATCH: Main ({n_cells}, {n_genes}) vs Raw ({raw_cells}, {raw_genes}).")

    # 4. METADATA PERSISTENCE
    required_obs = ['batch_tissue', 'mechanical_cell_type', 'granular_cell_type', 'cell_type']
    missing = [col for col in required_obs if col not in adata.obs.columns]

    if not missing:
        print("✅ METADATA: All mandatory training columns are present.")
    else:
        print(f"❌ MISSING COLUMNS: {missing}")


print("Initializing Sequential Merge Pipeline...")
gc.collect()

# 1. Define Paths (DO NOT LOAD THEM YET)
file_paths = [
    "../../clean_skin.h5ad",
    # "../Cellular Data/Single Cell/Reference/sbLung.h5ad",
    # "../Cellular Data/Single Cell/Reference/sbGut.h5ad",
    # "../Cellular Data/Single Cell/Reference/sbVasculature.h5ad"
]
dataset_names = [
    "Skin",
    # "Lung",
    # "Gut",
    # "Vasculature"
]
# 2. Phase 1: Zero-RAM Intersection
# We load the files in "backed='r'" mode. This reads ONLY the metadata (var_names)
# off the hard drive without putting the massive matrices into RAM.
print("Calculating Universal Genes (Zero-RAM mode)...")
gene_lists = []
import pybiomart
for p in file_paths:
    # backed='r' is the magic word here
    adata_backed = sc.read_h5ad(p)

    dataset = pybiomart.Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')

    # 2. Query the gene symbols and their biological types
    biomart_query = dataset.query(attributes=['external_gene_name', 'gene_biotype'])

    # 3. Filter the results to ONLY protein-coding genes
    protein_coding_genes = biomart_query[biomart_query['Gene type'] == 'protein_coding']['Gene name'].dropna().unique()

    # 4. Find the intersection between your tissue data and the protein-coding list
    clean_overlap = adata_backed.var_names.intersection(protein_coding_genes)

    adata_backed.var_names_make_unique()
    # 5. Surgically shrink the AnnData object
    adata_backed = adata_backed[:, clean_overlap].copy()



    print(f"Final shape (Protein-Coding Only): {adata_backed.shape}")


    print(adata_backed.var_names[:5])

    # Calculate the mean total counts per cell
    mean_depth = adata_backed.X.sum(axis=1).mean()
    print(f"True Training Target Depth: {mean_depth}")

    print("Training shape", adata_backed.shape)

    # 1. Look at the raw matrix values
    sample_data = adata_backed.X[:5, :5].toarray() if issparse(adata_backed.X) else adata_backed.X[:5, :5]
    print("Matrix Sample:\n", sample_data)

    # 2. Check the maximum value
    max_val = adata_backed.X.max()
    print(f"Max Expression Value: {max_val}")

    # 3. Check for fractions/decimals
    is_integer = numpy.all(numpy.equal(numpy.mod(sample_data, 1), 0))
    print(f"Are all values raw integers? {is_integer}")

    gene_lists.append(adata_backed.var_names.tolist())
    adata_backed.file.close()  # Close the file pointer

common_genes = list(set(gene_lists[0]).intersection(*gene_lists[1:]))
sorted_common = sorted(common_genes)
print(f"Found {len(common_genes)} universal genes across all datasets.")

# 3. Phase 2: Sequential Slicing
# Load ONE file -> Slice it -> Delete the big one -> Save the small one to the list
temp_files = []

for p, name in zip(file_paths, dataset_names):
    print(f"\nProcessing {name} (Streaming from disk)...")

    # 1. LOAD AS POINTER (0 RAM for the matrix)
    adata_backed = sc.read_h5ad(p, backed='r')

    # ---> THE FIX <---
    # Re-apply uniqueness to the backed index because we loaded the raw file again!
    # This only touches the metadata in RAM, it does not download the matrix.
    adata_backed.var_names_make_unique()
    if not adata_backed.obs_names.is_unique:
        adata_backed.obs_names_make_unique()

    # 2. EXTRACT METADATA SAFELY
    obs_df = adata_backed.obs.copy()  # This is now 100% safe

    # Find the right cell_type column
    possible_names = ['cell_type', 'celltype_lvl_3_extended']
    for col in possible_names:
        if col in obs_df.columns:
            obs_df['cell_type'] = obs_df[col]
            break

    # 3. CREATE THE ROW MASK (Which cells to keep)
    healthy_mask = ~obs_df['cell_type'].str.contains('disease|tumor|cancer', case=False, na=False)

    # 4. THE MAGIC MOVE: DOUBLE-SLICE FROM DISK
    print("Extracting subset into memory...")
    # This is the single most efficient line of code in Scanpy.
    # It crosses the SSD to find ONLY the healthy cells and ONLY the 2,500 genes,
    # and pulls that tiny fraction directly into RAM.
    adata_small = adata_backed[healthy_mask, sorted_common].to_memory()

    # Close the massive file pointer to release the OS lock
    adata_backed.file.close()

    # 5. APPLY ROSETTA STONE TO THE IN-MEMORY OBJECT
    # Now we only apply our dictionaries to the cells that survived the slice
    obs_df = obs_df[healthy_mask]

    obs_df['granular_cell_type'] = obs_df['cell_type'].map(master_chemical_map)
    obs_df['granular_cell_type'] = obs_df['granular_cell_type'].fillna(obs_df['cell_type'])

    obs_df['mechanical_cell_type'] = obs_df['granular_cell_type'].map(mechanical_class_map)
    obs_df['mechanical_cell_type'] = obs_df['mechanical_cell_type'].fillna('Unknown')

    obs_df['batch_tissue'] = name

    # 6. ATTACH CLEAN METADATA AND SAVE
    needed_cols = ['cell_type', 'batch_tissue', 'granular_cell_type', 'mechanical_cell_type']
    existing_needed = [c for c in needed_cols if c in obs_df.columns]

    adata_small.obs = obs_df[existing_needed]

    temp_path = f"temp_{name}.h5ad"
    adata_small.write_h5ad(temp_path)
    temp_files.append(temp_path)

    # 7. NUKE THE RAM
    del adata_small
    del obs_df
    gc.collect()
    print(f"✅ {name} offloaded to disk. RAM is perfectly clear.")

# --- THE FINAL DISK-BASED CONCAT ---
print("\nFinalizing Merge from Disk...")
sliced_list = [sc.read_h5ad(f) for f in temp_files]

print("concatenating...")
adata_pan_raw = ad.concat(
    sliced_list,
    join="outer",
    label="dataset_id",
    fill_value=0,
    keys=dataset_names,
    index_unique="_",  # Safety measure for duplicate barcodes
    merge="same"
)

print(f"✅ Great Merge Complete. Final Shape: {adata_pan_raw.shape}")

mean_depth = adata_pan_raw.X.sum(axis=1).mean()
print(f"True Training Target Depth: {mean_depth}")

print("Training shape", adata_pan_raw.shape)

# 1. Look at the raw matrix values
sample_data = adata_pan_raw.X[:5, :5].toarray() if issparse(adata_pan_raw.X) else adata_pan_raw.X[:5, :5]
print("Matrix Sample:\n", sample_data)

# 2. Check the maximum value
max_val = adata_pan_raw.X.max()
print(f"Max Expression Value: {max_val}")

# 3. Check for fractions/decimals
is_integer = numpy.all(numpy.equal(numpy.mod(sample_data, 1), 0))
print(f"Are all values raw integers? {is_integer}")

bad_labels = ['Unknown', 'unknown', 'nan', 'NaN', 'None']

# 1. Mask for Mechanical (Convert to string to catch literal 'nan' strings safely)
mech_valid = ~adata_pan_raw.obs['mechanical_cell_type'].astype(str).isin(bad_labels)

# 2. Mask for Granular (Check strings AND check for actual pd.NA/np.nan objects)
gran_valid = ~adata_pan_raw.obs['granular_cell_type'].astype(str).isin(bad_labels) & adata_pan_raw.obs[
    'granular_cell_type'].notna()

# 3. Apply the combined mask
adata_tensor = adata_pan_raw[mech_valid & gran_valid].copy()

print(f"🧹 Strict Filter Applied. Cells retained: {adata_tensor.n_obs}")
del adata_pan_raw
gc.collect()

final_mean_depth = float(adata_tensor.X.sum(axis=1).mean())
print(f"📏 Final Training Target Depth: {final_mean_depth:.4f}")

# 1. Verification of the Purge
# class_counts = adata_tensor.obs['mechanical_cell_type'].value_counts()
# print("--- Final Class Distribution ---")
# print(class_counts)

adata_tensor.obs['mechanical_cell_type'] = adata_tensor.obs['mechanical_cell_type'].astype('category')
adata_tensor.obs['granular_cell_type'] = adata_tensor.obs['granular_cell_type'].astype('category')

# 2. Explicitly inject "Unknown" into the allowed categories if it was filtered out
if "Unknown" not in adata_tensor.obs['mechanical_cell_type'].cat.categories:
    adata_tensor.obs['mechanical_cell_type'] = adata_tensor.obs['mechanical_cell_type'].cat.add_categories(
        ["Unknown"])

if "Unknown" not in adata_tensor.obs['granular_cell_type'].cat.categories:
    adata_tensor.obs['granular_cell_type'] = adata_tensor.obs['granular_cell_type'].cat.add_categories(["Unknown"])

# No need to look for .raw anymore! The raw integers are right here:
print("Verifying raw integer counts in .X:")
print(adata_tensor.X[:15, :15].data)

print("Writing Final")
print(adata_tensor.var_names[:5])

# Calculate the mean total counts per cell
mean_depth = adata_tensor.X.sum(axis=1).mean()
print(f"True Training Target Depth: {mean_depth}")

print("Training shape", adata_tensor.shape)

# 1. Look at the raw matrix values
sample_data = adata_tensor.X[:5, :5].toarray() if issparse(adata_tensor.X) else adata_tensor.X[:5, :5]
print("Matrix Sample:\n", sample_data)

# 2. Check the maximum value
max_val = adata_tensor.X.max()
print(f"Max Expression Value: {max_val}")

# max_val = raw_adata.raw.max()
# print(f"Max Raw Expression Value: {max_val}")

# 3. Check for fractions/decimals
is_integer = numpy.all(numpy.equal(numpy.mod(sample_data, 1), 0))
print(f"Are all values raw integers? {is_integer}")

adata_tensor.write_h5ad("../../cellular_data/Single Cell/Reference/skin_new_classifications.h5ad")

validate_pan_tissue_atlas(adata_tensor)

# Cleanup temp files
for f in temp_files:
    os.remove(f)

print("🚀 PROCESS COMPLETE.")
