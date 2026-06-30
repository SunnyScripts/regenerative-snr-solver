import os
import json
import struct
import gc
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import squidpy as sq
import omnipath as op
import mygene
import sdevelo


# ==========================================
# 1. THE MASTER LOCK (Data Sync)
# ==========================================
def load_official_maps():
    print("🔒 1. Locking engine to Classifier Maps...")
    with open("../../models/rna_cell_classifier/Dualv2/broad_class_map.json", "r") as f:
        broad_array = json.load(f)
    with open("../../models/rna_cell_classifier/Dualv2/granular_class_map.json", "r") as f:
        gran_array = json.load(f)

    broad_dict = {name: i for i, name in enumerate(broad_array)}
    gran_dict = {name: i for i, name in enumerate(gran_array)}

    print(f"   ✅ Locked {len(broad_dict)} Broad Types and {len(gran_dict)} Granular Types.")
    return broad_dict, gran_dict


# ==========================================
# 2. DATA LOAD & GUT CHECKS
# ==========================================
def enforce_log1p_normalization(adata, name):
    print(f"   🔍 Checking {name} for log1p normalization...")

    max_val = adata.X.max() if hasattr(adata.X, "todense") else np.max(adata.X)

    if max_val > 50:
        print(f"   ⚠️ {name} appears to be RAW COUNTS (Max: {max_val:.2f}). Applying target_sum=1e4 & log1p...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print(f"   ✅ {name} appears to be normalized (Max: {max_val:.2f}).")

    return adata


def load_and_verify_h5ad(filepath, name, gran_dict, check_types=True):
    print(f"🧬 2. Loading {name} Data from {filepath}...")
    adata = sc.read_h5ad(filepath)

    if check_types:
        if adata.obs['granular_cell_type'].isna().any():
            print("   ⚠️ WARNING: Found NaN cell types. Dropping them.")
            adata = adata[adata.obs['granular_cell_type'].notna()].copy()

        h5ad_grans = set(adata.obs['granular_cell_type'].unique())
        model_grans = set(gran_dict.keys())
        rogue_types = h5ad_grans - model_grans

        if rogue_types:
            raise ValueError(f"🚨 FATAL: {name} contains unknown cell types! {rogue_types}")

    if adata.X.min() < 0:
        raise ValueError(f"🚨 FATAL: {name} expression matrix contains negative values.")

    return enforce_log1p_normalization(adata, name)


# ==========================================
# 3. SDEVELO PARAMETER EXTRACTION
# ==========================================
def extract_sde_parameters(adata):
    print("📈 3. Running SDEvelo to extract transcriptomic kinetics...")

    sde_adata = sc.pp.subsample(adata, n_obs=10000, random_state=42, copy=True)
    sde_adata.layers['spliced'] = sde_adata.layers['mature']
    sde_adata.layers['unspliced'] = sde_adata.layers['nascent']

    sc.pp.filter_cells(sde_adata, min_genes=200)
    sc.pp.filter_genes(sde_adata, min_cells=3)
    sc.pp.highly_variable_genes(sde_adata, n_top_genes=2000)
    sde_adata = sde_adata[:, sde_adata.var.highly_variable].copy()

    sc.tl.pca(sde_adata)
    sc.pp.neighbors(sde_adata, n_neighbors=30, n_pcs=30)
    scv.pp.moments(sde_adata)
    scv.tl.velocity(sde_adata, mode='stochastic')

    args = sdevelo.Config()
    args.process = False
    args.sde_mode = "torchsde"

    print("   ⏳ Training SDENN model...")
    model = sdevelo.SDENN(args, sde_adata)
    model.train(100)

    def get_param(param):
        return param.detach().cpu().numpy().tolist()

    sigma2_array = get_param(model.sigma2)
    raw_genes = sde_adata.var_names.tolist()

    if np.isnan(sigma2_array).any():
        raise ValueError("🚨 FATAL: SDEvelo returned NaN parameters. Model failed to converge.")

    # --- THE MYGENE TRANSLATOR (Version-Stripped) ---
    print("   🔤 Translating Ensembl IDs to Gene Symbols...")

    # 1. Strip the version numbers (e.g., ENSG00000142583.18 -> ENSG00000142583)
    stripped_genes = [g.split('.')[0] for g in raw_genes]

    mg = mygene.MyGeneInfo()
    mg.set_caching(cache_db='mygene_cache')

    # Query using the clean, stripped IDs
    df_mapping = mg.querymany(stripped_genes, scopes='ensembl.gene', fields='symbol', species='human',
                              as_dataframe=True)

    gene_noise_map = {}
    translated_genes = []

    for i, raw_id in enumerate(raw_genes):
        stripped_id = stripped_genes[i]
        symbol_str = raw_id  # Fallback to the raw ID if translation fails

        # Look up the translation using the stripped ID
        if stripped_id in df_mapping.index and 'symbol' in df_mapping.columns:
            symbol = df_mapping.loc[stripped_id, 'symbol']
            if isinstance(symbol, pd.Series):
                symbol = symbol.iloc[0]  # Handle duplicate mappings
            if pd.notna(symbol):
                symbol_str = str(symbol).upper()

        translated_genes.append(symbol_str)

        # Only add to the physics noise map if it successfully became a Symbol
        if symbol_str != raw_id:
            gene_noise_map[symbol_str] = sigma2_array[i]

    physics_payload = {
        "model_type": "univariate_kinetics",
        "genes": translated_genes,
        "parameters": {
            "a": get_param(model.a),
            "b": get_param(model.b),
            "c": get_param(model.c),
            "beta": get_param(model.beta),
            "gamma": get_param(model.gamma),
            "sigma1": get_param(model.sigma1),
            "sigma2": sigma2_array
        }
    }

    print(f"   ✅ Successfully mapped {len(gene_noise_map)} highly variable genes to Symbols.")
    return gene_noise_map, physics_payload


# ==========================================
# 4. BIOLOGICAL PHYSICS (Atlas Rules)
# ==========================================
def compile_gpu_interactome_and_adhesion(adata, gene_noise_map, broad_dict, gran_dict):
    print("⚙️ 4a. Compiling Mechanical Rules (MRF & Adhesion)...")

    op_intercell = op.interactions.import_intercell_network()
    secreted_genes = set(op_intercell[op_intercell['secreted_intercell_source'] == True]['genesymbol_intercell_source'])
    adata.obs["granular_cell_type"] = adata.obs["granular_cell_type"].astype(str).astype("category")

    cache_file = "ligrec_results_cache.csv"

    # --- THE LIGREC CACHE ---
    if os.path.exists(cache_file):
        print("   🚀 Loading cached Ligrec permutations...")
        sig_rules = pd.read_csv(cache_file)
    else:
        print("   ⏳ Running Ligrec (this will take a while)...")

        # ==========================================
        # STRATIFIED SUBSAMPLING (Memory Optimized)
        # ==========================================
        ACCURACY_CAP = 50000
        print(f"   ✂️ Calculating stratified indices (Cap: {ACCURACY_CAP})...")

        cells_to_keep = []

        # 1. Loop through and collect INDICES ONLY (costs almost zero RAM)
        for cluster in adata.obs["granular_cell_type"].unique():
            # Get the integer positions of all cells in this cluster
            cluster_indices = np.where(adata.obs["granular_cell_type"] == cluster)[0]

            if len(cluster_indices) > ACCURACY_CAP:
                np.random.seed(42)
                kept = np.random.choice(cluster_indices, size=ACCURACY_CAP, replace=False)
                cells_to_keep.extend(kept)
            else:
                cells_to_keep.extend(cluster_indices)

        # 2. Perform a single slice to create the new object
        print("   🔪 Slicing AnnData object...")
        adata_sub = adata[cells_to_keep].copy()

        # 3. Clean up the massive list of indices
        del cells_to_keep

        # 4. FORCE GARBAGE COLLECTION
        print("   🧹 Emptying the garbage collector...")
        gc.collect()

        print(f"   📉 Reduced memory load: {len(adata)} cells -> {len(adata_sub)} cells")

        # 5. Run Ligrec
        sq.gr.ligrec(adata_sub, n_perms=10000, cluster_key="granular_cell_type", use_raw=False, n_jobs=11)
        res = adata_sub.uns['granular_cell_type_ligrec']

        pvals = res['pvalues'].stack(level=[0, 1]).reset_index()
        means = res['means'].stack(level=[0, 1]).reset_index()
        pvals.columns = ['ligand', 'receptor', 'source', 'target', 'pvalue']
        means.columns = ['ligand', 'receptor', 'source', 'target', 'mean_expr']
        rules_df = pd.merge(pvals, means, on=['ligand', 'receptor', 'source', 'target'])

        sig_rules = rules_df[(rules_df['pvalue'] <= 0.05) & (rules_df['mean_expr'] >= 0.5)].copy()

        # Save point!
        sig_rules.to_csv(cache_file, index=False)
        print(f"   💾 Ligrec results cached to {cache_file}!")

    aggregated_sde = {}
    num_gran = len(gran_dict)
    adhesion_matrix = np.zeros((num_gran, num_gran), dtype=np.float32)

    # Note: We continue to use the original `adata` here for mapping broad/granular IDs
    # because `adata_sub` was just a temporary object for the ligrec math.
    for _, row in sig_rules.iterrows():
        ligand, receptor = row['ligand'], row['receptor']
        source_gran, target_gran = row['source'], row['target']
        weight = row['mean_expr']

        l_noise = gene_noise_map.get(ligand, 0.1)
        r_noise = gene_noise_map.get(receptor, 0.1)
        diffusion = (l_noise ** 2 + r_noise ** 2) / 2.0

        gran_id_s = gran_dict[source_gran]
        gran_id_t = gran_dict[target_gran]

        broad_name_s = adata.obs.loc[adata.obs['granular_cell_type'] == source_gran, 'mechanical_cell_type'].iloc[0]
        broad_name_t = adata.obs.loc[adata.obs['granular_cell_type'] == target_gran, 'mechanical_cell_type'].iloc[0]
        pair_idx = (broad_dict[broad_name_s] * 4) + broad_dict[broad_name_t]

        if ligand in secreted_genes:
            if pair_idx not in aggregated_sde:
                aggregated_sde[pair_idx] = {}
            if gran_id_t not in aggregated_sde[pair_idx]:
                aggregated_sde[pair_idx][gran_id_t] = [0.0, 0.0]

            aggregated_sde[pair_idx][gran_id_t][0] += weight
            aggregated_sde[pair_idx][gran_id_t][1] += diffusion
        else:
            adhesion_matrix[gran_id_s, gran_id_t] += weight
            adhesion_matrix[gran_id_t, gran_id_s] += weight

    max_adhesion = adhesion_matrix.max()
    if max_adhesion > 0:
        adhesion_matrix = (adhesion_matrix / max_adhesion) * 20.0

    return aggregated_sde, adhesion_matrix


def extract_gap_junction_matrix(adata, gran_dict):
    print("⚡ 4b. Extracting Bioelectric Conductance (Connexins)...")
    num_gran = len(gran_dict)
    conductance_matrix = np.zeros((num_gran, num_gran), dtype=np.float32)

    all_genes = set(adata.var_names)
    cx_genes = [g for g in all_genes if g.startswith("GJA") or g.startswith("GJB") or g.startswith("GJC")]

    if not cx_genes:
        print("   ⚠️ No Gap Junction genes found!")
        return conductance_matrix

    cx_profiles = {}
    for gran_name, gran_id in gran_dict.items():
        subset = adata[adata.obs['granular_cell_type'] == gran_name]
        if len(subset) == 0:
            cx_profiles[gran_id] = np.zeros(len(cx_genes))
            continue

        means = np.array(subset[:, cx_genes].X.mean(axis=0)).flatten() if hasattr(subset.X, "toarray") else np.array(
            subset[:, cx_genes].X.mean(axis=0))
        cx_profiles[gran_id] = means

    for id_a in range(num_gran):
        for id_b in range(num_gran):
            conductance_matrix[id_a, id_b] = np.dot(cx_profiles[id_a], cx_profiles[id_b])

    max_c = conductance_matrix.max()
    if max_c > 0:
        conductance_matrix = conductance_matrix / max_c

    return conductance_matrix


# ==========================================
# 5. SPATIAL PHYSICS (Depth & Stratification)
# ==========================================
def get_stratification_params(ct_name):
    name = str(ct_name).lower()
    if "cornified kc" in name:
        return (0.95, 1.0)
    elif "granular kc" in name:
        return (0.80, 1.0)
    elif "spinous kc" in name:
        return (0.55, 1.0)
    elif "basal kc" in name or "prolif. kc" in name:
        return (0.30, 1.5)
    elif "fibro" in name:
        return (0.10, 0.6)
    elif any(x in name for x in ["ec", "venous", "arterial", "capillary", "lymphatic"]):
        return (0.05, 0.8)
    elif any(x in name for x in ["duct", "sg", "isthmus", "bulb", "bulge", "infundibulum", "coil"]):
        return (0.25, 0.8)
    elif any(x in name for x in ["t cell", "b cell", "dc", "mph", "nk", "mast", "neutro", "mono"]):
        return (0.50, 0.05)
    elif "merkel" in name:
        return (0.30, 1.0)
    elif any(x in name for x in ["neuron", "smc", "skeletal"]):
        return (0.02, 0.9)
    return (0.5, 0.1)


def extract_physics_1d_arrays(gran_dict):
    print("📏 4c. Extracting Spatial Physics 1D Arrays...")
    num_gran = len(gran_dict)
    ideal_depths = np.zeros(num_gran, dtype=np.float32)
    strat_weights = np.zeros(num_gran, dtype=np.float32)

    for name, g_id in gran_dict.items():
        depth, weight = get_stratification_params(name)
        ideal_depths[g_id] = depth
        strat_weights[g_id] = weight

    return ideal_depths, strat_weights


# ==========================================
# 6. THE BINARY COMPILER
# ==========================================
def write_gpu_binaries(aggregated_sde, adhesion_matrix, conductance_matrix, ideal_depths, strat_weights):
    print("💾 5. Packing strict C-Structs for WebGPU...")

    adhesion_matrix.tofile("adhesion_matrix.bin")
    conductance_matrix.tofile("conductance_matrix.bin")
    ideal_depths.tofile("ideal_depths.bin")
    strat_weights.tofile("strat_weights.bin")

    # --- DYNAMIC DRIFT SQUASHING ---
    max_original_drift = 0.0
    for rules in aggregated_sde.values():
        for drift, diff in rules.values():
            if drift > max_original_drift:
                max_original_drift = drift

    max_allowed_drift = 20.0
    scale_factor = max_original_drift / max_allowed_drift if max_original_drift > 0 else 1.0
    print(f"   📉 Dynamic Squash: Max Original Drift {max_original_drift:.2f} -> Scaling by {scale_factor:.2f}")

    offset_bytes = bytearray()
    rule_bytes = bytearray()
    current_rule_idx = 0

    for pair_idx in range(16):
        if pair_idx in aggregated_sde:
            rules = aggregated_sde[pair_idx]
            count = len(rules)
            offset_bytes += struct.pack('<IIII', current_rule_idx, count, 0, 0)

            for target_gran_id, (drift, diffusion) in rules.items():
                squashed_drift = drift / scale_factor
                rule_bytes += struct.pack('<IffI', target_gran_id, squashed_drift, diffusion, 0)
                current_rule_idx += 1
        else:
            offset_bytes += struct.pack('<IIII', 0, 0, 0, 0)

    with open("gpu_interactome.bin", "wb") as f:
        f.write(offset_bytes)
        f.write(rule_bytes)

    print("   ✅ Binaries written successfully.")


# ==========================================
# 7. THE FINAL VALIDATION GATE (Rust-Style)
# ==========================================
def validate_world_integrity(gene_noise_map, adhesion_mat, sde_rules):
    print("🛡️  Running Final Integrity Audit...")

    # 1. Did translation actually work?
    coverage = len([g for g in gene_noise_map if not g.startswith("ENSG")])
    if coverage < 500:  # Threshold check
        raise RuntimeError(f"🚨 PIPELINE FAILURE: Only {coverage} genes translated. SDE Noise is missing!")

    # 2. Are there NaNs in the matrices?
    if np.isnan(adhesion_mat).any():
        raise RuntimeError("🚨 PIPELINE FAILURE: Adhesion matrix contains NaNs. GPU will crash.")

    # 3. Is the Interactome empty?
    if len(sde_rules) == 0:
        raise RuntimeError("🚨 PIPELINE FAILURE: Zero interactome rules generated. Check your P-value filters.")

    print("✅ Integrity Audit Passed. Data is safe for Metal ingestion.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    PATH_KINETICS = "../../cellular_data/Single Cell/GSE130973/SRR9036396/counts_unfiltered/annSkin25.h5ad"
    PATH_ATLAS = "../../cellular_data/Single Cell/Reference/skin_new_classifications.h5ad"

    b_dict, g_dict = load_official_maps()

    kinetics_adata = load_and_verify_h5ad(PATH_KINETICS, "Kinetics", g_dict, check_types=False)
    gene_noise_map, sde_payload = extract_sde_parameters(kinetics_adata)

    with open("sde_physics_constants.json", "w") as f:
        json.dump(sde_payload, f, indent=4)

    atlas_adata = load_and_verify_h5ad(PATH_ATLAS, "Atlas", g_dict, check_types=True)

    # This will print every granular cell type and its exact cell count, sorted highest to lowest.
    print(atlas_adata.obs["granular_cell_type"].value_counts())

    sde_rules, adhesion_mat = compile_gpu_interactome_and_adhesion(atlas_adata, gene_noise_map, b_dict, g_dict)
    conductance_mat = extract_gap_junction_matrix(atlas_adata, g_dict)
    ideal_depths, strat_weights = extract_physics_1d_arrays(g_dict)

    write_gpu_binaries(sde_rules, adhesion_mat, conductance_mat, ideal_depths, strat_weights)

    master_gene_map = {gene: i for i, gene in enumerate(atlas_adata.var_names)}
    with open("master_gene_map.json", "w") as f:
        json.dump(master_gene_map, f, indent=4)

    validate_world_integrity(gene_noise_map, adhesion_mat, sde_rules)

    print("🎉 World Compiled. WebGPU Engine is ready for launch.")

