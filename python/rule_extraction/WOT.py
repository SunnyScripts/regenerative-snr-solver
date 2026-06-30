# Created by Ryan Berg March 19th, 2026
# Purpose: Determine the gene network edge decay by performing Optimal Transport on a group of Young Skin
# and a group of Old Skin to find the transition between the expression probabilities

# import wot
import ot
import scanpy

import numpy
import pandas

import scipy.sparse as sparse
from pyro.contrib.oed.glmm import epsilon
from sklearn.decomposition import PCA
import scipy.spatial.distance as dist

import json
import gc # ram management



print("Reading filtered reference file...")
fHSCA = scanpy.read_h5ad("../Cell Data/Reference/Filtered_HSCA_extended.h5ad")

import mygene

# ... [After loading fHSCA but BEFORE filtering genes] ...

mg = mygene.MyGeneInfo()
all_genes = fHSCA.var_names.tolist()

print("Querying Gene Ontology for Bioelectric Hardware...")
results = mg.querymany(all_genes, scopes='symbol', fields='go', species='human')

target_go_terms = {
    'GO:0005267': 'Potassium', 'GO:0005272': 'Sodium', 'GO:0005254': 'Chloride',
    'GO:0005324': 'Pumps', 'GO:0005921': 'Connexins', 'GO:0005198': 'Keratins',
    'GO:0007156': 'Cadherins'
}

hardware_genes = set()
for res in results:
    if 'go' in res:
        for cat in ['MF', 'BP', 'CC']:
            if cat in res['go']:
                terms = res['go'][cat] if isinstance(res['go'][cat], list) else [res['go'][cat]]
                for term in terms:
                    if term['id'] in target_go_terms:
                        hardware_genes.add(res['query'])
                        break

hardware_genes = list(hardware_genes)
# Ensure we exactly hit our 32-gene cap for the GPU struct, or pad it with zeros later
hardware_genes = hardware_genes[:32]
print(f"Isolated {len(hardware_genes)} hardware genes.")

# Now proceed to find 2000 HVGs and run your PCA and Sinkhorn math as normal...

## QC ##
# check cell viability and extracellular genes
print("-Quality Control Step-")
scanpy.pp.filter_cells(fHSCA, min_counts=3)
scanpy.pp.filter_genes(fHSCA, min_cells=1)

## Preprocess ##
print("preprocessing the data...")

#HSCA dataset is already normalized
# raw_max = adata.X.max()
#
# print(f"Matrix Minimum Value: {raw_min}")
# print(f"Matrix Maximum Value: {raw_max}")
#
# if raw_min < 0:
#     print("WARNING: Data is ALREADY SCALED (contains negative numbers).")
#     print("Do NOT run normalize_total or log1p. It will destroy the matrix.")
# elif raw_max < 30:
#     print("WARNING: Data is likely ALREADY LOG-NORMALIZED.")
#     print("Do NOT run log1p again.")
# else:
#     print("Data appears to be raw counts. Safe to normalize.")
#
# scanpy.pp.normalize_total(fHSCA, target_sum=1e4)
# scanpy.pp.log1p(fHSCA)



requiredGenes = [
    'GJA1', 'GJB2', 'GJB6',  # Gap Junctions (Connexins)
    'KRT1', 'KRT5', 'KRT10', 'KRT14', # Keratinocyte markers
    'TP63', 'CDH1', 'DSP',   # Adhesion and stemness
    'KCNQ1', 'KCNJ2', 'SCN5A' # Example Ion Channels
]

# are the genes in the dataset
requiredGenes = [gene for gene in requiredGenes if gene in fHSCA.var_names]

scanpy.pp.highly_variable_genes(fHSCA, n_top_genes=2000, subset=False)

isHighlyVariableGene = fHSCA.var["highly_variable"]
isRequired = fHSCA.var_names.isin(requiredGenes)

# remove all others
print("removing noise from dataset")
fHSCA = fHSCA[:, (isHighlyVariableGene | isRequired)].copy()

print(f"Matrix shape before math: {fHSCA.shape}")
if fHSCA.shape[0] < 10:
    raise ValueError("CRITICAL ERROR: We filtered out almost all the cells! The math is crashing because the matrix is empty.")



# 2. EXTRACT RAW DATA (Abandon Scanpy memory management)
print("Extracting raw matrix...")
if sparse.issparse(fHSCA.X):
    X_raw = fHSCA.X.toarray().astype(numpy.float64)
else:
    X_raw = fHSCA.X.astype(numpy.float64)

# Calculate standard deviation and mean manually
means = numpy.mean(X_raw, axis=0)
stds = numpy.std(X_raw, axis=0)

# The Mathematical Sledgehammer: If variance is 0, force it to 1.0 to prevent division by zero
stds[stds == 0.0] = 1.0

# Scale and clip (equivalent to max_value=10)
X_scaled = (X_raw - means) / stds
X_scaled = numpy.clip(X_scaled, -10, 10)

# Final safety check: This should mathematically be impossible to fail now
if numpy.isnan(X_scaled).any() or numpy.isinf(X_scaled).any():
    raise ValueError("CRITICAL ERROR: NaNs/Infs generated during manual scaling!")
else:
    print("Matrix is 100% mathematically clean.")



# purge the undesirables nan and inf ;)
# fHSCA.X = numpy.nan_to_num(fHSCA.X, nan=0.0, posinf=0.0, neginf=0.0)

# 4. MANUAL PCA (Bypassing scanpy.tl.pca)
print("Running raw sklearn PCA...")
pca = PCA(n_components=50, svd_solver='full')
print("xpca")
X_pca = pca.fit_transform(X_scaled)

# Inject clean PCA back into the object
print("read in xpca")
fHSCA.obsm['X_pca'] = X_pca
print("PCA complete. No warnings should exist above this line.")

#data type fix
# print("begin transport mapping")
fHSCA.obs['age_years'] = pandas.to_numeric(fHSCA.obs['age_years'], errors='coerce')
fHSCA = fHSCA[~fHSCA.obs['age_years'].isna()]

ageGroups = [0, 35, 55, 100]
otLabels = [0, 1, 2]

# day is a wot object label for time steps
print("create age groups object in the day object")
fHSCA.obs["day"] = pandas.cut(fHSCA.obs["age_years"], bins=ageGroups, labels=otLabels).astype(float)

# 1. Clean the 'day' column
fHSCA = fHSCA[~fHSCA.obs['day'].isna()].copy()


def get_clean_transport(t0, t1, adata, epsilon=0.05):
    print(f"\n--- Transporting Day {t0} -> Day {t1} ---")

    # 1. Extract PCA coords
    pca0 = adata[adata.obs['day'] == t0].obsm['X_pca'].astype(numpy.float64)
    pca1 = adata[adata.obs['day'] == t1].obsm['X_pca'].astype(numpy.float64)

    # 2. Calculate Cost Matrix (Squared Euclidean)
    # Using 'sqeuclidean' is the standard for WOT trajectories
    print("Calculating cost matrix...")
    M = dist.cdist(pca0, pca1, metric='sqeuclidean')

    # 3. Scale the Cost Matrix (Crucial for Sinkhorn stability)
    # Dividing by the median prevents the exp(-M/epsilon) from overflowing
    M /= numpy.median(M)

    # 4. Create Uniform Weights
    # We assume every cell at t0 has equal 'mass' to move to t1
    n0, n1 = len(pca0), len(pca1)
    a, b = numpy.ones(n0) / n0, numpy.ones(n1) / n1

    # 5. Run Sinkhorn (The core OT solver)
    print(f"Running Sinkhorn (epsilon={epsilon})...")
    # This returns the transport plan T
    T = ot.sinkhorn(a, b, M, reg=epsilon)

    # Cleanup cost matrix to save 4GB of RAM
    del M, pca0, pca1
    gc.collect()

    return T


# --- EXECUTION ---

# 1. Get the two steps
T_01 = get_clean_transport(0.0, 1.0, fHSCA, epsilon=0.1)
T_12 = get_clean_transport(1.0, 2.0, fHSCA, epsilon=0.1)

# 2. Compute the long-range transition (0 -> 2)
print("\nMultiplying steps for final trajectory...")
T_02 = T_01 @ T_12

# 3. Row-Normalize
# Each row in T_02 represents where the 'mass' of a Day 0 cell ends up at Day 2
print("normalize rows")
T_02 = T_02 / T_02.sum(axis=1, keepdims=True)

print(f"Locked! Final transition matrix shape: {T_02.shape}")



# Continuous Rates per Lineage


import math
import numpy as np

# ... [After T_02 is calculated] ...

print("Extracting Continuous Decay Rates per Lineage...")

# Assuming your timescale from age 0 to 100 is 100 years
# todo don't assume
TOTAL_LIFESPAN_YEARS = 100.0

# We need the indices of our specific hardware genes to pull them from the massive X matrix
hardware_indices = [fHSCA.var_names.get_loc(g) for g in hardware_genes]

# Find the indices of ATP pumps and Cadherins to calculate physical hardware decay
pump_indices = [fHSCA.var_names.get_loc(g) for g in hardware_genes if 'ATP1A' in g]
cadherin_indices = [fHSCA.var_names.get_loc(g) for g in hardware_genes if 'CDH' in g]

ot_tensors_export = []

# todo finalize granular_id
# Loop through every unique cell lineage (0 to 69)
unique_lineages = sorted(fHSCA.obs['granular_id'].unique())

for lineage_id in unique_lineages:
    # 1. Isolate the cells for this specific lineage
    lineage_mask = fHSCA.obs['granular_id'] == lineage_id

    young_mask = lineage_mask & (fHSCA.obs["day"] == 0.0)
    old_mask = lineage_mask & (fHSCA.obs["day"] == 2.0)

    # If a lineage doesn't exist in both young and old (e.g., a transient state), skip or pad
    if not young_mask.any() or not old_mask.any():
        continue

    young_X_lineage = fHSCA[young_mask].X
    old_X_lineage = fHSCA[old_mask].X

    if hasattr(young_X_lineage, "toarray"): young_X_lineage = young_X_lineage.toarray()
    if hasattr(old_X_lineage, "toarray"): old_X_lineage = old_X_lineage.toarray()

    # Apply the global transition matrix specifically to this lineage's old cells
    # Note: T_02 applies to all cells. You may need to subset T_02 if you want strictly isolated paths
    avg_young = young_X_lineage.mean(axis=0)
    avg_old = old_X_lineage.mean(axis=0)

    rna_decay_rates = []

    # 2. Calculate continuous rate (k) for the 32 hardware genes
    for idx in hardware_indices:
        young_val = avg_young[idx] + 1e-6
        old_val = avg_old[idx] + 1e-6

        # k = ln(Ratio) / dt
        rate_k = math.log(old_val / young_val) / TOTAL_LIFESPAN_YEARS
        rna_decay_rates.append(float(rate_k))

    # 3. Calculate Physical Hardware Decay
    # We average the rate of all ATP pump genes to get a master atrophy rate
    pump_rates = [math.log((avg_old[i] + 1e-6) / (avg_young[i] + 1e-6)) / TOTAL_LIFESPAN_YEARS for i in pump_indices]
    pump_atrophy = float(np.mean(pump_rates)) if pump_rates else 0.0

    adhesion_rates = [math.log((avg_old[i] + 1e-6) / (avg_young[i] + 1e-6)) / TOTAL_LIFESPAN_YEARS for i in
                      cadherin_indices]
    adhesion_decay = float(np.mean(adhesion_rates)) if adhesion_rates else 0.0

    # Assume a generic 10% volume loss over a lifespan for the area decay,
    # or tie it to structural Keratin decay
    area_decay = math.log(0.9) / TOTAL_LIFESPAN_YEARS

    # 4. Package for Rust
    ot_tensors_export.append({
        "granular_id": int(lineage_id),
        "rna_decay_rates": rna_decay_rates,  # Exactly 32 floats
        "area_decay_rate": area_decay,
        "pump_atrophy_rate": pump_atrophy,
        "adhesion_decay_rate": adhesion_decay
    })

print("Writing GPU-ready tensor file...")
with open("ot_tensors_gpu.json", "w") as file:
    json.dump(ot_tensors_export, file, indent=4)

print("Optimal Transport Genetic Edge Decay Calculated and Exported ;)")


# # 2. Verify we have cells in every timepoint
# print("--- CELL COUNTS PER DAY ---")
# print(fHSCA.obs['day'].value_counts())
#
# # 3. Check for any 0-count days
# day_counts = fHSCA.obs['day'].value_counts()
# if len(day_counts) < 2:
#     raise ValueError("CRITICAL: You need at least 2 timepoints to calculate a transport map!")
#
#
# print("normalize pca space 0 to 1")
# X_pca = fHSCA.obsm['X_pca'].copy()
# X_min = X_pca.min()
# X_max = X_pca.max()
# X_pca_stable = (X_pca - X_min) / (X_max - X_min)
#
# print("building PCA only transport object..")
# pcaAnnData = scanpy.AnnData( X=fHSCA.obsm['X_pca'].copy(), obs=fHSCA.obs.copy(), var=pandas.DataFrame(index=[f"PC{i+1}" for i in range(50)]))
#
# print("set ot model") # using kwargs attribute of OTModel
# otModel = wot.ot.OTModel(pcaAnnData, day_field='day', epsilon=0.2, lambda1=1, lambda2=50, growth_iter=3)
# print("compute transport maps for optimal transport model")
# otModel.compute_all_transport_maps(tmap_out="transportMap")

# # Sparse Map Check
# print("read transport maps")
# transportMap1 = scanpy.read_h5ad("transportMap_0.0_1.0.h5ad")
# young2mid = transportMap1.X.toarray() if sparse.issparse(transportMap1.X) else transportMap1.X
# transportMap2 = scanpy.read_h5ad("transportMap_1.0_2.0.h5ad")
# mid2old = transportMap2.X.toarray() if sparse.issparse(transportMap2.X) else transportMap2.X
#
# # Matrix multiplication for young to old transport map
# print("multiply maps and normalize")
# young2old = young2mid @ mid2old
# # normalize probability distribution
# young2old = young2old / young2old.sum(axis=1, keepdims=True)
#
# # Compute gene network edge decay
# youngCells = fHSCA.obs[fHSCA.obs["day"] == 0].index
# oldCells = fHSCA.obs[fHSCA.obs["day"] == 2].index
#
# youngExpression = fHSCA[youngCells].X
# oldExpression = fHSCA[oldCells].X
#
# predictedOldExpression = (young2old @ oldExpression).mean(axis=0)
# actualYoungExpression = youngExpression.mean(axis=0)
#
# #output decay tensor JSON for the sim
# decayTensor = {}
# genes = fHSCA.var_names.tolist()
#
# for i, gene in enumerate(genes):
#     youngGene = actualYoungExpression[i, 0] if sparse.issparse(actualYoungExpression) else actualYoungExpression[i]
#     oldGene = predictedOldExpression[i, 0] if sparse.issparse(predictedOldExpression) else predictedOldExpression[i]
#
#     # Actual Math for decay
#     decayRatio = (oldGene + 1e-6) / (youngGene + 1e-6)
#     decayTensor[gene] = float(decayRatio)
#
# optimalTransportDecay = {
#     "deltaTimeEpochs": 2,
#     "edgeDecayTensor": decayTensor
# }
#
# with open("decayTensor.json", "w") as file:
#     json.dump(optimalTransportDecay, file, indent=4)
#
# print("Optimal Transport Genetic Edge Decay Calculated and Exported ;)")














