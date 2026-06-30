import omnipath as op
import scanpy as sc
# import pandas as pd
import json
import numpy as np
import itertools

def get_stratification_params(ct_name: str) -> tuple[float, float]:
    """
    Returns (ideal_depth, stratification_weight) based on skin biology.
    Depth: 0.0 = Deep Dermis, 1.0 = Skin Surface.
    """
    name = str(ct_name).lower()

    # --- 1. The Epidermal Layers (Strictly layered) ---
    if "cornified kc" in name:
        return (0.95, 1.0)
    elif "granular kc" in name:
        return (0.80, 1.0)
    elif "spinous kc" in name:
        return (0.55, 1.0)
    elif "basal kc" in name or "prolif. kc" in name:
        return (0.30, 1.5) # The basement membrane anchor

    # --- 2. The Dermis (Deeper, more flexible) ---
    elif "fibro" in name:
        return (0.10, 0.6)
    elif any(x in name for x in ["ec", "venous", "arterial", "capillary", "lymphatic"]):
        return (0.05, 0.8)

    # --- 3. Appendages (Hair follicle / Sweat gland parts) ---
    elif any(x in name for x in ["duct", "sg", "isthmus", "bulb", "bulge", "infundibulum", "coil"]):
        return (0.25, 0.8)

    # --- 4. Wanderers (Immune cells) ---
    elif any(x in name for x in ["t cell", "b cell", "dc", "mph", "nk", "mast", "neutro", "mono"]):
        return (0.50, 0.05)

        # --- 5. Others ---
    elif "merkel" in name:
        return (0.30, 1.0)
    elif any(x in name for x in ["neuron", "smc", "skeletal"]):
        return (0.02, 0.9)

    # Default fallback
    return (0.5, 0.1)


def generate_rulebook(h5ad_path: str, output_json: str):
    print("1. Loading HSCA Extended Data & OmniPath...")
    adata = sc.read_h5ad(h5ad_path)

    # Filter out 'nan' cell types and solidify categories
    obs_col = 'celltype_lvl_3_extended'
    adata = adata[adata.obs[obs_col].notna()].copy()
    categories = adata.obs[obs_col].astype('category').cat.categories

    # Gene symbols are stored in the index
    dataset_symbols = set(adata.var_names)
    print(f"Dataset contains {len(dataset_symbols)} unique gene symbols.")

    print("2. Fetching & Filtering OmniPath Intercell Network...")
    interactions = op.interactions.import_intercell_network()

    # Filter for High Confidence (Curation Effort >= 2)
    interactions = interactions[interactions['curation_effort'] >= 2].copy()

    # Intersect with our skin sample genes using the specific intercell columns
    valid_lr = interactions[
        interactions['genesymbol_intercell_source'].isin(dataset_symbols) &
        interactions['genesymbol_intercell_target'].isin(dataset_symbols)
        ].copy()

    # Normalize Curation Effort to a weight of [0.1, 1.0]
    min_eff = valid_lr['curation_effort'].min()
    max_eff = valid_lr['curation_effort'].max()
    valid_lr['norm_weight'] = 0.1 + (valid_lr['curation_effort'] - min_eff) / (max_eff - min_eff) * 0.9

    # Determine the unique list of genes required for adhesion math
    active_genes = sorted(list(set(valid_lr['genesymbol_intercell_source']) |
                               set(valid_lr['genesymbol_intercell_target'])))

    print(f"3. Building Internal Expression Profiles for {len(active_genes)} genes...")
    all_profiles = {}

    for ct in categories:
        ct_adata = adata[adata.obs[obs_col] == ct]
        if hasattr(ct_adata.X, "toarray"):
            means = np.array(ct_adata[:, active_genes].X.mean(axis=0)).flatten()
        else:
            means = np.array(ct_adata[:, active_genes].X.mean(axis=0))

        all_profiles[str(ct)] = dict(zip(active_genes, means))

    print("4. Computing Type-to-Type Adhesion Matrix (Penalties)...")
    pairwise_results = []

    for ct_a, ct_b in itertools.combinations_with_replacement(categories, 2):
        p_a = all_profiles[str(ct_a)]
        p_b = all_profiles[str(ct_b)]

        total_stickiness = 0.0

        # Calculate biological affinity based on ligand-receptor matches
        for _, row in valid_lr.iterrows():
            l = str(row['genesymbol_intercell_source'])
            r = str(row['genesymbol_intercell_target'])
            w = float(row['norm_weight'])

            # A binding to B
            total_stickiness += p_a.get(l, 0.0) * p_b.get(r, 0.0) * w

            # B binding to A (skip if same type to avoid double counting)
            if ct_a != ct_b:
                total_stickiness += p_b.get(l, 0.0) * p_a.get(r, 0.0) * w

        pairwise_results.append({
            "type_a": str(ct_a),
            "type_b": str(ct_b),
            "raw_stickiness": total_stickiness
        })

    # Invert "Stickiness" to "Energy Penalty" for the MRF physics engine
    s_vals = [res['raw_stickiness'] for res in pairwise_results]
    s_min, s_max = min(s_vals), max(s_vals)

    MIN_PENALTY = 1.0  # Max stickiness = Min penalty
    MAX_PENALTY = 20.0 # Min stickiness = Max penalty

    adhesion_rules = []
    for res in pairwise_results:
        if s_max > s_min:
            norm_s = (res['raw_stickiness'] - s_min) / (s_max - s_min)
        else:
            norm_s = 0.5

        penalty = MAX_PENALTY - (norm_s * (MAX_PENALTY - MIN_PENALTY))

        adhesion_rules.append({
            "type_a": res['type_a'],
            "type_b": res['type_b'],
            "penalty": round(penalty, 4)
        })

    print("5. Assembling Lightweight JSON for Rust...")
    gap_junction_genes = ["GJA1", "GJA3", "GJA4", "GJA5", "GJA8", "GJB1", "GJB2", "GJB3", "GJB4", "GJB5", "GJB6", "GJC1", "GJC2", "GJC3"]
    cell_types_json = {}

    for ct in categories:
        ct_adata = adata[adata.obs[obs_col] == ct]

        # Extract ONLY dynamic genes for Rust memory efficiency
        dynamic_genes = {}
        for g in gap_junction_genes:
            if g in dataset_symbols:
                dynamic_genes[g] = float(ct_adata[:, g].X.mean())
            else:
                dynamic_genes[g] = 0.0

        ideal_depth, strat_weight = get_stratification_params(ct)

        cell_types_json[str(ct)] = {
            "type_id": int(list(categories).index(ct) + 1),
            "starting_genes": dynamic_genes, # Super lightweight array!
            "ideal_depth": ideal_depth,
            "stratification_weight": strat_weight,
            "base_radius": 5.0,
            "division_rate": 0.02 if "Prolif" in str(ct) else 0.0
        }

    rulebook = {
        "cell_types": cell_types_json,
        "adhesion_rules": adhesion_rules,
        "gap_junction_genes": gap_junction_genes
    }

    with open(output_json, 'w') as f:
        json.dump(rulebook, f, indent=4)

    print(f"Success! Rulebook saved to {output_json}")

if __name__ == "__main__":
    generate_rulebook("../Cellular Data/Single Cell/Reference/HSCA_extended.h5ad", "interactome.json")