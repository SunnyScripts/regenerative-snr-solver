import gc
from scipy.sparse import issparse, csr_matrix
import scanpy as sc
# import numpy as np


import mygene
# import pandas as pd
# import numpy as np

def enforce_entrez_ids(adata):
    """
    Safely converts AnnData var_names from Gene Symbols to Entrez IDs.
    Drops genes that cannot be mapped.
    """
    print(f"Starting conversion. Initial genes: {adata.shape[1]}")

    # 1. Extract the current symbols
    symbols = adata.var_names.tolist()

    # 2. Query the NCBI database via mygene
    mg = mygene.MyGeneInfo()
    mg.set_caching(cache_db='mygene_cache')
    print("Querying mygene database...")

    # querymany is highly optimized for large lists.
    # Returning a dataframe makes it much easier to filter.
    results = mg.querymany(
        symbols,
        scopes=['symbol', 'alias', 'ensembl.gene', 'accessions'],# ensembl.gene
        fields='entrezgene',
        species='human',
        as_dataframe=True
    )

    # 3. Clean up the results
    # Drop queries that didn't find an Entrez ID
    valid_results = results.dropna(subset=['entrezgene']).copy()

    # Convert Entrez IDs to strings (they often return as floats in pandas)
    valid_results['entrez_str'] = valid_results['entrezgene'].astype(int).astype(str)

    # Handle duplicates (sometimes an old symbol maps to multiple IDs)
    # We keep the first valid hit to ensure strict 1-to-1 mapping
    valid_results = valid_results[~valid_results.index.duplicated(keep='first')]

    # 4. Map back to the AnnData object
    # Find which of our original symbols successfully mapped
    successful_symbols = valid_results.index.tolist()

    # Slice the anndata to ONLY include genes that successfully mapped
    adata_mapped = adata[:, successful_symbols].copy()

    # 5. Overwrite the var_names with the new Entrez strings
    new_var_names = valid_results.loc[successful_symbols, 'entrez_str'].tolist()
    adata_mapped.var_names = new_var_names

    # 6. Safety check: Ensure all new names are unique
    adata_mapped.var_names_make_unique()

    dropped_count = adata.shape[1] - adata_mapped.shape[1]
    print(
        f"✅ Conversion Complete. Final genes: {adata_mapped.shape[1]} (Dropped {dropped_count} unmapped/duplicate genes)")

    return adata_mapped

def harmonize_atlas(adata, dataset_name):
    print(f"Cleaning {dataset_name}...")

    print(adata.var_names[:20].tolist())

    # 1. Enforce Entrez IDs (Assuming your var_names are currently symbols)
    # You would use mygene here to map symbols to Entrez, drop the ones that fail
    # adata.var_names = mapped_entrez_ids

    # 2. Make names unique (drop duplicate gene columns)
    # adata.var_names_make_unique()

    # adata = enforce_entrez_ids(adata)

    print(f"Starting conversion. Initial genes: {adata.shape[1]}")

    # 1. Extract the current symbols
    symbols = adata.var_names.tolist()

    # 2. Query the NCBI database via mygene
    mg = mygene.MyGeneInfo()
    mg.set_caching(cache_db='mygene_cache')
    print("Querying mygene database...")

    # querymany is highly optimized for large lists.
    # Returning a dataframe makes it much easier to filter.
    results = mg.querymany(
        symbols,
        scopes=['symbol', 'alias', 'ensembl.gene', 'accessions'],  # ensembl.gene
        fields='entrezgene',
        species='human',
        as_dataframe=True
    )

    # 3. Clean up the results
    # Drop queries that didn't find an Entrez ID
    valid_results = results.dropna(subset=['entrezgene']).copy()

    # Convert Entrez IDs to strings (they often return as floats in pandas)
    valid_results['entrez_str'] = valid_results['entrezgene'].astype(int).astype(str)

    # Handle duplicates (sometimes an old symbol maps to multiple IDs)
    # We keep the first valid hit to ensure strict 1-to-1 mapping
    valid_results = valid_results[~valid_results.index.duplicated(keep='first')]

    # 4. Map back to the AnnData object
    # Find which of our original symbols successfully mapped
    successful_symbols = valid_results.index.tolist()

    # Slice the anndata to ONLY include genes that successfully mapped
    adata_mapped = adata[:, successful_symbols].copy()

    # 5. Overwrite the var_names with the new Entrez strings
    new_var_names = valid_results.loc[successful_symbols, 'entrez_str'].tolist()
    adata_mapped.var_names = new_var_names

    # 6. Safety check: Ensure all new names are unique
    adata_mapped.var_names_make_unique()

    dropped_count = adata.shape[1] - adata_mapped.shape[1]

    del adata
    gc.collect()

    print(
        f"✅ Conversion Complete. Final genes: {adata_mapped.shape[1]} (Dropped {dropped_count} unmapped/duplicate genes)")

    print("filter")

    # 3. Filter Empties (Crucial)
    # Drop cells with fewer than 100 total genes detected
    sc.pp.filter_cells(adata_mapped, min_genes=100)
    # Drop genes that appear in fewer than 10 cells across the dataset
    sc.pp.filter_genes(adata_mapped, min_cells=10)


    print("save raw counts")
    # 1. Ensure X is sparse BEFORE trying to move it
    if not issparse(adata_mapped.X):
        print("Matrix is dense. Converting to Sparse CSR now...")
        adata_mapped.X = csr_matrix(adata_mapped.X)

    # 2. Save to layers WITHOUT calling .toarray()
    # This uses almost ZERO additional RAM because it's just a pointer to the existing sparse data
    adata_mapped.layers["counts"] = adata_mapped.X.copy()

    # 5. Add universal metadata
    adata_mapped.obs['batch_tissue'] = dataset_name

    anchors_to_check = ['4074', '999', '3852', '1291']  # EPCAM, CDH1, KRT5, COL1A1
    exists = [a in adata_mapped.var_names for a in anchors_to_check]
    print(f"Anchors survived: {dict(zip(anchors_to_check, exists))}")

    print("write")
    return adata_mapped


import anndata as ad
import numpy as np


def the_great_merge():
    print("Initiating The Great Merge...")

    harmonized_adatas = [sc.read_h5ad("../Cellular Data/Single Cell/Reference/cSkin.h5ad"),
                         sc.read_h5ad("../Cellular Data/Single Cell/Reference/cLung.h5ad"),
                         sc.read_h5ad("../Cellular Data/Single Cell/Reference/cGut.h5ad"),
                         sc.read_h5ad("../Cellular Data/Single Cell/Reference/cVasculature.h5ad")]

    print("done reading")
    dataset_names = ["Skin", "Lung", "Gut", "Vasculature"]

    # 1. Extract the gene lists (var_names) from every dataset
    gene_lists = [adata.var_names.tolist() for adata in harmonized_adatas]

    # 2. Find the strict intersection (Genes that exist in ALL datasets)
    # This prevents NaN errors later in the neural network
    common_genes = list(set(gene_lists[0]).intersection(*gene_lists[1:]))
    print(f"Found {len(common_genes)} universal genes across all datasets.")

    # 3. Slice every dataset down to just this common pool
    sliced_adatas = []
    for adata, name in zip(harmonized_adatas, dataset_names):
        # Sort the genes so the columns align perfectly
        sliced = adata[:, sorted(common_genes)].copy()
        print("add slices")
        sliced.obs['batch_tissue'] = name  # Ensure the scVI batch key is set
        sliced_adatas.append(sliced)
        print(f"Sliced {name} to common genes.")

    del harmonized_adatas
    gc.collect()

    # 4. Concatenate into a single massive Pan-Tissue Atlas
    # We use ad.concat (the modern standard over .concatenate)
    print("Concatenating matrices. This may take a moment and spike RAM...")
    adata_pan_raw = ad.concat(
        sliced_adatas,
        join="inner",  # We already intersected, so outer is safe
        label="dataset_id",  # Creates a column tracking the source AnnData
        keys=dataset_names,  # Fills the dataset_id column
        merge="same"
    )

    print(f"✅ Great Merge Complete. Final Shape: {adata_pan_raw.shape}")
    print("writing")
    adata_pan_raw.write_h5ad("Mechanical.h5ad")
    return 1




# gut = sc.read_h5ad("../Cellular Data/Single Cell/Reference/Gut.h5ad")
# harmonize_atlas(vasculature, "Vasculature").write_h5ad("../Cellular Data/Single Cell/Reference/cVasculature.h5ad")

# Usage Example:
the_great_merge()



