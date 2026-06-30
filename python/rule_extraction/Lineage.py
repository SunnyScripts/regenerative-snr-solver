import pandas
import scanpy

adata_pan_raw = scanpy.read_h5ad("whatever I end up calling this merged file.h5ad")

# 1. Look at what detailed names you actually have
# print("Unique detailed cell types:", adata_pan_raw.obs['cell_type'].unique())

# 2. Build the mapping dictionary
# (You will need to adjust the keys to match your exact CELLxGENE labels)
lineage_map = {
    # ==========================================
    # 1. EPITHELIAL (The Barrier & Physics Target)
    # ==========================================
    # Gut & Organ Lining
    'colon epithelial cell': 'Epithelial',
    'intestine goblet cell': 'Epithelial',
    'intestinal tuft cell': 'Epithelial',
    'colonocyte': 'Epithelial',
    'type I enteroendocrine cell': 'Epithelial',
    'GIP cell': 'Epithelial',
    'type L enteroendocrine cell': 'Epithelial',
    'type N enteroendocrine cell': 'Epithelial',
    'progenitor cell of endocrine pancreas': 'Epithelial',
    'transit amplifying cell': 'Epithelial',

    # Lung / Respiratory
    'club cell': 'Epithelial',
    'serous secreting cell': 'Epithelial',
    'mucus secreting cell': 'Epithelial',
    'pulmonary alveolar epithelial cell': 'Epithelial',
    'acinar cell': 'Epithelial',
    'pulmonary alveolar type 1 cell': 'Epithelial',
    'pulmonary alveolar type 2 cell': 'Epithelial',
    'brush cell of tracheobronchial tree': 'Epithelial',
    'multiciliated columnar cell of tracheobronchial tree': 'Epithelial',
    'nasal mucosa goblet cell': 'Epithelial',
    'epithelial cell of lower respiratory tract': 'Epithelial',
    'respiratory basal cell': 'Epithelial',
    'ionocyte': 'Epithelial',
    'multiciliated epithelial cell': 'Epithelial',

    # Skin (Keratinocytes & Appendages)
    'Basal KC': 'Epithelial',
    'Cornified KC': 'Epithelial',
    'Granular KC': 'Epithelial',
    'Prolif. KC': 'Epithelial',
    'Spinous KC': 'Epithelial',
    'Merkel cell': 'Epithelial',
    'SG': 'Epithelial',  # Sebaceous Gland
    'Bulb': 'Epithelial',  # Hair follicle parts
    'Bulge': 'Epithelial',
    'Coil': 'Epithelial',
    'Duct': 'Epithelial',
    'Infundibulum': 'Epithelial',
    'Isthmus': 'Epithelial',

    # ==========================================
    # 2. ENDOTHELIAL (Blood & Lymph Vessels)
    # ==========================================
    'endothelial cell': 'Endothelial',
    'endothelial cell of lymphatic vessel': 'Endothelial',
    'capillary endothelial cell': 'Endothelial',
    'vein endothelial cell': 'Endothelial',
    'endothelial cell of artery': 'Endothelial',
    'Arterial EC': 'Endothelial',
    'Capillary EC': 'Endothelial',
    'Lymphatic EC': 'Endothelial',
    'Venous EC': 'Endothelial',

    # ==========================================
    # 3. STROMAL (Connective, Structure & Muscle)
    # ==========================================
    'fibroblast': 'Stromal',
    'smooth muscle cell': 'Stromal',
    'pericyte': 'Stromal',
    'interstitial cell of Cajal': 'Stromal',
    'enteric smooth muscle cell': 'Stromal',
    'stromal cell of lamina propria of small intestine': 'Stromal',
    'mesenchymal lymphangioblast': 'Stromal',
    'mesothelial cell': 'Stromal',
    'myofibroblast cell': 'Stromal',
    'stromal cell': 'Stromal',
    'Fibro A': 'Stromal',
    'Fibro B': 'Stromal',
    'Fibro C': 'Stromal',
    'Fibro D': 'Stromal',
    'Fibro E': 'Stromal',
    'Fibro diseased': 'Stromal',
    'SMC': 'Stromal',  # Smooth Muscle Cell
    'Skeletal muscle': 'Stromal',

    # ==========================================
    # 4. IMMUNE (White Blood Cells & Circulating)
    # ==========================================
    'T cell': 'Immune',
    'mast cell': 'Immune',
    'Mast cell': 'Immune',
    'macrophage': 'Immune',
    'B cell': 'Immune',
    'monocyte': 'Immune',
    'Monocyte': 'Immune',
    'natural killer cell': 'Immune',
    'CD4-positive, alpha-beta T cell': 'Immune',
    'CD8-positive, alpha-beta T cell': 'Immune',
    'basophil': 'Immune',
    'neutrophil': 'Immune',
    'Neutrophil': 'Immune',
    'myeloid dendritic cell': 'Immune',
    'plasma cell': 'Immune',
    'mature NK T cell': 'Immune',
    'classical monocyte': 'Immune',
    'non-classical monocyte': 'Immune',
    'intermediate monocyte': 'Immune',
    'T-helper 17 cell': 'Immune',
    'activated CD8-positive, alpha-beta T cell': 'Immune',
    'CD8-positive, alpha-beta memory T cell': 'Immune',
    'mucosal invariant T cell': 'Immune',
    'IgG plasma cell': 'Immune',
    'IgM plasma cell': 'Immune',
    'IgA plasma cell': 'Immune',
    'conventional dendritic cell': 'Immune',
    'group 2 innate lymphoid cell': 'Immune',
    'group 3 innate lymphoid cell': 'Immune',
    'CD34-positive, CD56-positive, CD117-positive common innate lymphoid precursor, human': 'Immune',
    'NKp44-positive group 3 innate lymphoid cell, human': 'Immune',
    'NKp44-negative group 3 innate lymphoid cell, human': 'Immune',
    'T follicular helper cell': 'Immune',
    'hematopoietic stem cell': 'Immune',
    'alveolar macrophage': 'Immune',
    'plasmacytoid dendritic cell': 'Immune',
    'elicited macrophage': 'Immune',
    'CD1c-positive myeloid dendritic cell': 'Immune',
    'erythrocyte': 'Immune',  # Red blood cell, grouped here for filtering
    'dendritic cell': 'Immune',
    'DC': 'Immune',
    'Mph': 'Immune',  # Macrophage
    'NK': 'Immune',

    # ==========================================
    # 5. NEURAL (Nervous System)
    # ==========================================
    'oligodendrocyte precursor cell': 'Neural',
    'migratory enteric neural crest cell': 'Neural',
    'inhibitory motor neuron': 'Neural',
    'Sensory neuron': 'Neural',

    # ==========================================
    # 6. UNKNOWN / AMBIGUOUS
    # ==========================================
    'progenitor cell': 'Unknown',
    'unknown': 'Unknown',
    'nan': 'Unknown'
}


base_column = adata_pan_raw.obs['cell_type']

# 3. Apply the map to create the new column
adata_pan_raw.obs['broad_class'] = base_column.obs['cell_type'].map(lineage_map)

# 4. The Safety Net: Catch anything you forgot to map
# If a cell wasn't in your dictionary, it gets labeled 'Unknown' instead of crashing as a NaN
adata_pan_raw.obs['broad_class'] = adata_pan_raw.obs['broad_class'].fillna('Unknown')

print("Mapping complete. Distribution:")
print(adata_pan_raw.obs['broad_class'].value_counts())