# import scanpy
# trash_labels = ['nan', 'Unknown', 'unknown', 'None', 'NA']
# print("reading raw mechanical h5ad")
# adata_final = scanpy.read_h5ad("../Cellular Data/Single Cell/Reference/raw_mechanical.h5ad")
# print("updating data")
# adata_final = adata_final[
#     (~adata_final.obs['mechanical_cell_type'].isin(trash_labels)) &
#     (adata_final.obs['mechanical_cell_type'].notna())
# ].copy()
#
# # 3. CRITICAL: Prune the unused categories
# # Even if the rows are gone, 'Unknown' stays in the category list unless you do this:
# adata_final.obs['granular_cell_type'] = adata_final.obs['granular_cell_type'].cat.remove_unused_categories()
# adata_final.obs['mechanical_cell_type'] = adata_final.obs['mechanical_cell_type'].cat.remove_unused_categories()
#
# print(f"✅ Cleaned Anndata: {len(adata_final.obs['granular_cell_type'].unique())} unique cell types remain.")
# print("saving file")
# adata_final.write_h5ad("sc_mech.h5ad")

import scanpy as sc
# from scipy.sparse import issparse
import squidpy as sq
import omnipath as op
import pandas as pd
import json
import mygene

def main():

    reference_file = "sc_mech"
    print(f"reading {reference_file} from the dard drive")
    sc_mech = sc.read_h5ad(f"../Cellular Data/Single Cell/Reference/{reference_file}.h5ad")


    mg = mygene.MyGeneInfo()
    mg.set_caching(cache_db='mygene_cache')

    # 2. Query your Entrez IDs
    # Assuming Entrez IDs are currently in adata.var_names
    entrez_list = sc_mech.var_names.tolist()

    # scopes='entrezgene' tells it the input type
    # fields='symbol' tells it what you want back
    df_mapping = mg.querymany(entrez_list,
                              scopes='entrezgene',
                              fields='symbol',
                              species='human',
                              as_dataframe=True)

    # 3. Clean up the results
    # mygene returns a dataframe; we filter for valid symbols and convert to uppercase
    df_mapping = df_mapping.dropna(subset=['symbol'])
    df_mapping['symbol'] = df_mapping['symbol'].str.upper()

    # 4. Update your AnnData object
    # It's safest to add symbols as a column first, then switch the index
    # Note: Some Entrez IDs might map to the same Symbol; make_unique handles this
    sc_mech.var['gene_symbols'] = df_mapping['symbol']
    sc_mech = sc_mech[:, ~sc_mech.var['gene_symbols'].isna()].copy()
    sc_mech.var_names = sc_mech.var['gene_symbols']
    sc_mech.var_names_make_unique()


    # ==========================================
    # 1. Stratified Sub-Sampling (To prevent memory crash)
    # ==========================================
    print("1. Subsampling 1.8M cells for Permutation Testing...")
    # Sample roughly 50k-100k cells, keeping the granular distributions intact
    sc.pp.subsample(sc_mech, n_obs=500000, random_state=42)

    # ==========================================
    # 2. Build the Granular -> Broad Mapping
    # ==========================================
    print("2. Mapping Granular classes to Mechanical classes...")
    # Create a dictionary to instantly look up the Broad class of any Granular class
    # e.g., mapping_dict['Hair Follicle Stem Cell'] returns 'Epithelial'
    mapping_df = sc_mech.obs[['granular_cell_type', 'mechanical_cell_type']].drop_duplicates()
    mapping_dict = dict(zip(mapping_df['granular_cell_type'], mapping_df['mechanical_cell_type']))

    # ==========================================
    # 3. Run the Squidpy Ligand-Receptor Analysis
    # ==========================================
    print("3. Running Squidpy Permutation Tests...")
    #todo Make sure .X is normalized/log-transformed before running this

    # This uses OmniPath under the hood
    sq.gr.ligrec(
        sc_mech,
        n_perms=10000,
        cluster_key="granular_cell_type",
        use_raw=False,
        n_jobs=8
    )

    print(sc_mech.uns.keys())

    # sc_mech.write_h5ad(f"../Cellular Data/Single Cell/Reference/{reference_file}_ligrec.h5ad")

    res = sc_mech.uns['granular_cell_type_ligrec']

    # ==========================================
    # 4. Parse the Multi-Index Output into a Flat Table
    # ==========================================
    print("4. Parsing Multi-Index DataFrames...")
    # Squidpy outputs dense Multi-Index DataFrames. We need to flatten them.
    pvals_series = res['pvalues'].stack(level=[0, 1])
    means_series = res['means'].stack(level=[0, 1])

    # 2. Reset the index to turn the Series back into a flat DataFrame
    pvals = pvals_series.reset_index()
    means = means_series.reset_index()

    # 3. Rename the 5 resulting columns
    # The resulting order is always: ligand, receptor, source, target, value
    pvals.columns = ['ligand', 'receptor', 'source', 'target', 'pvalue']
    means.columns = ['ligand', 'receptor', 'source', 'target', 'mean_expr']

    # 4. Merge them together
    rules_df = pd.merge(pvals, means, on=['ligand', 'receptor', 'source', 'target'])

    # ==========================================
    # 4.5 Run Differential Expression (The DE Filter)
    # ==========================================
    print("4.5 Running Differential Expression to find Unique Markers...")

    # Run Wilcoxon test to find genes highly expressed in one group vs the rest
    sc.tl.rank_genes_groups(
        sc_mech,
        groupby='granular_cell_type',
        method='wilcoxon',
        use_raw=False
    )

    # Build a dictionary mapping each cell type to its top 100 defining genes
    marker_genes_dict = {}
    for cell_type in sc_mech.obs['granular_cell_type'].cat.categories:
        markers = sc.get.rank_genes_groups_df(sc_mech, group=cell_type).head(100)['names'].tolist()
        marker_genes_dict[cell_type] = set(markers)

    # Helper function to apply the strict biological filter
    def passes_de_filter(row):
        # A rule is only kept if the Ligand defines the Source
        # OR the Receptor defines the Target.
        source_markers = marker_genes_dict.get(row['source'], set())
        target_markers = marker_genes_dict.get(row['target'], set())

        return (row['ligand'] in source_markers) or (row['receptor'] in target_markers)

    # ==========================================
    # 5. Filter for Biological Significance & Add Metadata
    # ==========================================
    print("5. Filtering for Significant Interactions and determining Contact Rules...")

    op_intercell = op.interactions.import_intercell_network()
    secreted_ligands = op_intercell[op_intercell['secreted_intercell_source'] == True][
        'genesymbol_intercell_source'].unique()

    # 1. Apply baseline statistical thresholds
    baseline_rules = rules_df[
        (rules_df['pvalue'] <= 0.05) &
        (rules_df['mean_expr'] >= 0.5)
        ].copy()

    # 2. Apply the strict DE Biological Filter
    # This wipes out generic housekeeping interactions
    significant_rules = baseline_rules[baseline_rules.apply(passes_de_filter, axis=1)].copy()

    print(f"   - Rules passing basic stats: {len(baseline_rules)}")
    print(f"   - Rules surviving DE Filter: {len(significant_rules)}")

    # ==========================================
    # 6. Build the Rust JSON Structure
    # ==========================================
    print("6. Formatting JSON for the Rust SDE Engine...")

    interaction_rules = []
    max_expr = significant_rules['mean_expr'].max()
    min_expr = significant_rules['mean_expr'].min()

    for _, row in significant_rules.iterrows():
        # UPDATED: Keys changed to match the stacked DataFrame
        source_gran = row['source']
        target_gran = row['target']
        ligand = row['ligand']

        norm_weight = 0.1 + ((row['mean_expr'] - min_expr) / (max_expr - min_expr)) * 0.9

        # THE CRITICAL CHECK: Is this Ligand Secreted?
        # If it is NOT secreted, it is bound to the membrane, so it requires physical contact.
        requires_contact = bool(ligand not in secreted_ligands)

        rule = {
            "source_broad": mapping_dict[source_gran],
            "target_broad": mapping_dict[target_gran],
            "ligand_gene": ligand,
            "receptor_gene": row['receptor'],
            "curation_weight": round(norm_weight, 4),
            "target_granular_vote": target_gran,
            "requires_contact": requires_contact
        }
        interaction_rules.append(rule)

    final_json = {
        "interaction_rules": interaction_rules
    }

    # Export to disk
    with open('interactome.json', 'w') as f:
        json.dump(final_json, f, indent=2)

    print(f"✅ Success! Exported {len(interaction_rules)} rules to interactome.json")

    # 1. Create the Granular Map (String -> u16)
    # Rust needs: {"Basal Keratinocyte": 0, "Fibroblast": 1, ...}
    granular_categories = sc_mech.obs['granular_cell_type'].cat.categories
    granular_map = {name: i for i, name in enumerate(granular_categories)}

    # 2. Create the Broad Map (String -> u8)
    # Rust needs: {"Epithelial": 1, "Mesenchymal": 2, ...}
    broad_categories = sc_mech.obs['mechanical_cell_type'].cat.categories
    broad_map = {name: i for i, name in enumerate(broad_categories)}

    # Export for the Rust Engine
    with open('granular_classes.json', 'w') as f:
        json.dump(granular_map, f, indent=2)

    with open('broad_classes.json', 'w') as f:
        json.dump(broad_map, f, indent=2)

    print("🚀 MapBook JSONs exported for Rust!")

if __name__ == "__main__":
    main()