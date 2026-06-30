import scanpy as sc
import mygene

import pandas as pd

def find_hardware_overlap(xenium_path, cosmx_path):
    print("vendor panel genes")
    # 1. Load the CSVs provided by the vendors
    # (Usually a single column named 'gene_symbol' or 'gene_id')
    xenium_list = set(pd.read_csv(xenium_path)['gene_name'])
    cosmx_list = set(pd.read_csv(cosmx_path)['Symbol_Name'])

    print("xenium", len(xenium_list))
    print("cosmx", len(cosmx_list))

    # merfish_list = set(pd.read_csv(merfish_path)['gene_symbol'])

    mg = mygene.MyGeneInfo()
    mg.set_caching(cache_db='mygene_cache')
    print("Querying mygene database...")

    # querymany is highly optimized for large lists.
    # Returning a dataframe makes it much easier to filter.
    result = mg.querymany(
        xenium_list,
        scopes=['symbol'],  # ensembl.gene
        fields='entrezgene',
        species='human',
        as_dataframe=True
    )

    # 3. Clean up the results
    # Drop queries that didn't find an Entrez ID
    xGenes = result.dropna(subset=['entrezgene']).copy()

    # Convert Entrez IDs to strings (they often return as floats in pandas)
    xGenes['entrez_str'] = xGenes['entrezgene'].astype(int).astype(str)


    # querymany is highly optimized for large lists.
    # Returning a dataframe makes it much easier to filter.
    result = mg.querymany(
        cosmx_list,
        scopes=['symbol'],  # ensembl.gene
        fields='entrezgene',
        species='human',
        as_dataframe=True
    )

    # 3. Clean up the results
    # Drop queries that didn't find an Entrez ID
    nGenes = result.dropna(subset=['entrezgene']).copy()

    # Convert Entrez IDs to strings (they often return as floats in pandas)
    nGenes['entrez_str'] = nGenes['entrezgene'].astype(int).astype(str)

    # 2. Find the "Triple Crown" intersection
    # Genes that are present on ALL THREE platforms
    xGenes_clean = xGenes[~xGenes.index.duplicated(keep='first')]
    nGenes_clean = nGenes[~nGenes.index.duplicated(keep='first')]

    # 2. Extract strictly the 'entrez_str' columns as sets
    xenium_entrez_set = set(xGenes_clean['entrez_str'])
    cosmx_entrez_set = set(nGenes_clean['entrez_str'])

    # 3. Find the True Biological Intersection
    universal_overlap = list(xenium_entrez_set.intersection(cosmx_entrez_set))
    # Force the overlap list to be strings
    universal_overlap = [str(g) for g in universal_overlap]

    print(f"Final overlap count:", len(universal_overlap))
    return universal_overlap

# def find_hardware_overlap(xenium_path, cosmx_path):
#     print("vendor panel genes")
#     # 1. Load the CSVs provided by the vendors
#     # (Usually a single column named 'gene_symbol' or 'gene_id')
#     xenium_list = set(pd.read_csv(xenium_path)['gene_name'])
#     cosmx_list = set(pd.read_csv(cosmx_path)['Symbol_Name'])
#
#     print("xenium", len(xenium_list))
#     print("cosmx", len(cosmx_list))
#
#     # merfish_list = set(pd.read_csv(merfish_path)['gene_symbol'])
#
#     mg = mygene.MyGeneInfo()
#     mg.set_caching(cache_db='mygene_cache')
#     print("Querying mygene database...")
#
#     # querymany is highly optimized for large lists.
#     # Returning a dataframe makes it much easier to filter.
#     result = mg.querymany(
#         xenium_list,
#         scopes=['symbol'],  # ensembl.gene
#         fields='entrezgene',
#         species='human',
#         as_dataframe=True
#     )
#
#     # 3. Clean up the results
#     # Drop queries that didn't find an Entrez ID
#     xGenes = result.dropna(subset=['entrezgene']).copy()
#
#     # Convert Entrez IDs to strings (they often return as floats in pandas)
#     xGenes['entrez_str'] = xGenes['entrezgene'].astype(int).astype(str)
#
#
#     # querymany is highly optimized for large lists.
#     # Returning a dataframe makes it much easier to filter.
#     result = mg.querymany(
#         cosmx_list,
#         scopes=['symbol'],  # ensembl.gene
#         fields='entrezgene',
#         species='human',
#         as_dataframe=True
#     )
#
#     # 3. Clean up the results
#     # Drop queries that didn't find an Entrez ID
#     nGenes = result.dropna(subset=['entrezgene']).copy()
#
#     # Convert Entrez IDs to strings (they often return as floats in pandas)
#     nGenes['entrez_str'] = nGenes['entrezgene'].astype(int).astype(str)
#
#     # 2. Find the "Triple Crown" intersection
#     # Genes that are present on ALL THREE platforms
#     xGenes_clean = xGenes[~xGenes.index.duplicated(keep='first')]
#     nGenes_clean = nGenes[~nGenes.index.duplicated(keep='first')]
#
#     # 2. Extract strictly the 'entrez_str' columns as sets
#     xenium_entrez_set = set(xGenes_clean['entrez_str'])
#     cosmx_entrez_set = set(nGenes_clean['entrez_str'])
#
#     # 3. Find the True Biological Intersection
#     universal_overlap = list(xenium_entrez_set.intersection(cosmx_entrez_set))
#     # Force the overlap list to be strings
#     universal_overlap = [str(g) for g in universal_overlap]
#
#     print(f"Xenium Genes: {len(xenium_list)}")
#     print(f"CosMx Genes: {len(cosmx_list)}")
#     # print(f"MERFISH Genes: {len(merfish_list)}")
#     print(f"---")
#     print(f"Universal Hardware Overlap: {len(universal_overlap)} genes")
#
#     adata_pan_raw = sc.read_h5ad("../Cellular Data/Single Cell/Reference/Mechanical.h5ad")
#     print(adata_pan_raw.var.head())
#
#     adata_golden = adata_pan_raw[:, adata_pan_raw.var_names.isin(universal_overlap)].copy()
#
#     print(f"Final Golden Pool Shape: {adata_golden.shape}")
#     adata_golden.write_h5ad("mechVendor.h5ad")

    # return list(universal_overlap)

# find_hardware_overlap("../Cellular Data/Single Cell/Reference/10x5kPanel.csv",
#                       "../Cellular Data/Single Cell/Reference/nano6kPanel.csv")


# 3. Prioritize by Variance (Integrated into the Bucket Filler)
def get_hardware_bucket(adata_pan, overlap_genes, quota=77):
    # Slice the Pan-Tissue atlas to only the genes that exist on all hardware
    adata_overlap = adata_pan[:, adata_pan.var_names.isin(overlap_genes)].copy()

    # Let math find the 77 most informative genes within this safe overlap
    sc.pp.highly_variable_genes(adata_overlap, n_top_genes=quota, batch_key="batch_tissue")

    return adata_overlap.var[adata_overlap.var['highly_variable']].index.tolist()



def calculate_lineage_anchors():
    adata_pan_raw = sc.read_h5ad("../Cellular Data/Single Cell/Reference/raw_mechanical.h5ad")
    # vendor_pool = find_hardware_overlap("../Cellular Data/Single Cell/Reference/10x5kPanel.csv",
    #                    "../Cellular Data/Single Cell/Reference/nano6kPanel.csv")

    TARGET_SIZE = 768
    TEN_PERCENT = int(TARGET_SIZE * 0.10)

    # 1. Save the pristine raw counts
    # This ensures scANVI can still see the original integers later
    adata_pan_raw.layers["counts"] = adata_pan_raw.X.copy()

    # 2. Normalize
    # Target_sum=1e4 is the standard (scales every cell to 10,000 counts)
    sc.pp.normalize_total(adata_pan_raw, target_sum=1e4)

    # 3. Log-transform
    # This converts the counts to log-space
    sc.pp.log1p(adata_pan_raw)

    print("Running Wilcoxon rank-sum test on mechanical classes...")
    sc.tl.rank_genes_groups(
        adata_pan_raw,
        groupby='mechanical_cell_type',
        method='wilcoxon',
        use_raw=False  # Ensures it uses your log-normalized X, not the raw counts layer
    )

    # ==========================================
    # BUCKET 1: THE 10% LINEAGE GENES (DE MATH)
    # ==========================================
    print(f"Calculating Top {TEN_PERCENT} Lineage Anchors...")

    # The 4 primary geometries your Rust engine cares about
    target_classes = ['Epithelial', 'Mesenchymal', 'Endothelial', 'Immune/Fluid']

    balanced_lineage_genes = []

    # Loop through each class and grab its top 19 hardware-compatible anchors
    for geom in target_classes:
        # 1. Get the DE dataframe for this specific geometry
        de_df = sc.get.rank_genes_groups_df(adata_pan_raw, group=geom)

        # 2. Filter for hardware compatibility
        # hardware_de_df = de_df[de_df['names'].isin(vendor_pool)].copy()

        # 3. Sort by highest significance (Wilcoxon score)
        hardware_de_df = de_df.sort_values(by='scores', ascending=False)

        # 4. Take the Top 19 for this class
        top_19 = hardware_de_df.head(19)['names'].tolist()

        print(f"Captured Top 19 anchors for {geom}.")
        balanced_lineage_genes.extend(top_19)

    # 5. Remove any accidental duplicates (in case two classes share a highly ranked gene)
    balanced_lineage_genes = list(set(balanced_lineage_genes))

    # If duplicates brought the number slightly below 76, we can just top it off
    # with the next best generic HVGs later, or leave it exactly as is.
    print(f"\n✅ Balanced Lineage Bucket complete. Total unique genes: {len(balanced_lineage_genes)}")
    print(balanced_lineage_genes)

    # ==========================================
    # BUCKET 2: THE 10% VENDOR OVERLAP GENES
    # ==========================================
    print(f"\nCalculating Top {TEN_PERCENT} Vendor Signal Genes...")

    # To get the best "generic" vendor genes for spatial alignment, we find the
    # most Highly Variable Genes (HVGs) that aren't already in the Lineage bucket.

    # 1. Calculate general variance across the whole merged atlas
    # (If you already ran this during preprocessing, you can skip this line)
    sc.pp.highly_variable_genes(adata_pan_raw, layer="counts", flavor='seurat_v3', n_top_genes=4000)

    # Extract the variance dataframe
    hvg_df = adata_pan_raw.var[adata_pan_raw.var['highly_variable'] == True].copy()
    hvg_df = hvg_df.sort_values(by='variances_norm', ascending=False)

    # 2. EXCLUDE the Lineage genes so we don't double-dip in our 768 budget
    remaining_vendor_pool = [g for g in adata_pan_raw.var_names if g not in balanced_lineage_genes]

    # 3. Filter the highly variable genes down to just the remaining hardware panel
    hardware_hvg_df = hvg_df[hvg_df.index.isin(remaining_vendor_pool)]

    # 4. Slice the top 10%
    vendor_overlap_genes = hardware_hvg_df.head(TEN_PERCENT).index.tolist()

    print(f"✅ Vendor Overlap Bucket filled: {len(vendor_overlap_genes)} genes.")

    # --- SAFETY CHECK ---
    overlap = set(balanced_lineage_genes).intersection(vendor_overlap_genes)
    print(f"\nVerification: Double-dipped genes (Should be 0): {len(overlap)}")
    print(vendor_overlap_genes)

print("building custom gene panel")
calculate_lineage_anchors()

# Usage in your Bucket Filler:
# lineage_bucket = calculate_lineage_anchors(adata_pan_raw, quotas['lineage'])
# universal_panel.update(lineage_bucket)

# import pandas as pd


def find_hardware_overlap(xenium_path, cosmx_path, merfish_path):
    # 1. Load the CSVs provided by the vendors
    # (Usually a single column named 'gene_symbol' or 'gene_id')
    xenium_list = set(pd.read_csv(xenium_path)['gene_symbol'])
    cosmx_list = set(pd.read_csv(cosmx_path)['gene_symbol'])
    merfish_list = set(pd.read_csv(merfish_path)['gene_symbol'])

    # 2. Find the "Triple Crown" intersection
    # Genes that are present on ALL THREE platforms
    universal_overlap = xenium_list.intersection(cosmx_list).intersection(merfish_list)

    print(f"Xenium Genes: {len(xenium_list)}")
    print(f"CosMx Genes: {len(cosmx_list)}")
    print(f"MERFISH Genes: {len(merfish_list)}")
    print(f"---")
    print(f"Universal Hardware Overlap: {len(universal_overlap)} genes")

    return list(universal_overlap)


# 3. Prioritize by Variance (Integrated into the Bucket Filler)
def get_hardware_bucket(adata_pan, overlap_genes, quota=77):
    # Slice the Pan-Tissue atlas to only the genes that exist on all hardware
    adata_overlap = adata_pan[:, adata_pan.var_names.isin(overlap_genes)].copy()

    # Let math find the 77 most informative genes within this safe overlap
    sc.pp.highly_variable_genes(adata_overlap, n_top_genes=quota, batch_key="batch_tissue")

    return adata_overlap.var[adata_overlap.var['highly_variable']].index.tolist()


def find_hardware_overlap(xenium_path, cosmx_path):
    print("vendor panel genes")
    # 1. Load the CSVs provided by the vendors
    # (Usually a single column named 'gene_symbol' or 'gene_id')
    xenium_list = set(pd.read_csv(xenium_path)['gene_name'])
    cosmx_list = set(pd.read_csv(cosmx_path)['Symbol_Name'])

    print("xenium", len(xenium_list))
    print("cosmx", len(cosmx_list))

    # merfish_list = set(pd.read_csv(merfish_path)['gene_symbol'])

    mg = mygene.MyGeneInfo()
    mg.set_caching(cache_db='mygene_cache')
    print("Querying mygene database...")

    # querymany is highly optimized for large lists.
    # Returning a dataframe makes it much easier to filter.
    result = mg.querymany(
        xenium_list,
        scopes=['symbol'],  # ensembl.gene
        fields='entrezgene',
        species='human',
        as_dataframe=True
    )

    # 3. Clean up the results
    # Drop queries that didn't find an Entrez ID
    xGenes = result.dropna(subset=['entrezgene']).copy()

    # Convert Entrez IDs to strings (they often return as floats in pandas)
    xGenes['entrez_str'] = xGenes['entrezgene'].astype(int).astype(str)


    # querymany is highly optimized for large lists.
    # Returning a dataframe makes it much easier to filter.
    result = mg.querymany(
        cosmx_list,
        scopes=['symbol'],  # ensembl.gene
        fields='entrezgene',
        species='human',
        as_dataframe=True
    )

    # 3. Clean up the results
    # Drop queries that didn't find an Entrez ID
    nGenes = result.dropna(subset=['entrezgene']).copy()

    # Convert Entrez IDs to strings (they often return as floats in pandas)
    nGenes['entrez_str'] = nGenes['entrezgene'].astype(int).astype(str)

    # 2. Find the "Triple Crown" intersection
    # Genes that are present on ALL THREE platforms
    xGenes_clean = xGenes[~xGenes.index.duplicated(keep='first')]
    nGenes_clean = nGenes[~nGenes.index.duplicated(keep='first')]

    # 2. Extract strictly the 'entrez_str' columns as sets
    xenium_entrez_set = set(xGenes_clean['entrez_str'])
    cosmx_entrez_set = set(nGenes_clean['entrez_str'])

    # 3. Find the True Biological Intersection
    universal_overlap = list(xenium_entrez_set.intersection(cosmx_entrez_set))
    # Force the overlap list to be strings
    universal_overlap = [str(g) for g in universal_overlap]

    print(f"Xenium Genes: {len(xenium_list)}")
    print(f"CosMx Genes: {len(cosmx_list)}")
    return universal_overlap