import scanpy as sc
import os

def transfer_hsca_labels(xenium_folder, hsca_reference_path, output_h5ad_path):
    print("Loading Xenium Query and HSCA Reference...")

    # 1. Load the raw Xenium feature matrix from your folder
    h5_path = os.path.join(xenium_folder, "cell_feature_matrix.h5")
    adata_query = sc.read_10x_h5(h5_path)

    # 2. Load the Tolga Duz HSCA reference
    adata_ref = sc.read_h5ad(hsca_reference_path)

    # 3. Find the Intersection of Genes
    # We can only map the spatial cells using the exact genes they share with the atlas
    shared_genes = adata_query.var_names.intersection(adata_ref.var_names)
    print(f"Found {len(shared_genes)} shared genes for mapping.")

    # Subset both datasets to only the shared genes
    adata_query = adata_query[:, shared_genes].copy()
    adata_ref = adata_ref[:, shared_genes].copy()

    # 4. Process the Reference (Using ONLY the spatial panel genes)
    print("Recalculating Reference PCA on spatial panel genes...")
    sc.pp.normalize_total(adata_ref)
    sc.pp.log1p(adata_ref)
    sc.pp.pca(adata_ref)
    sc.pp.neighbors(adata_ref)
    sc.tl.umap(adata_ref)

    # 5. Process the Query identically
    sc.pp.normalize_total(adata_query)
    sc.pp.log1p(adata_query)

    # 6. THE MAGIC: Mathematical Label Transfer
    print("Ingesting Xenium query data into the HSCA reference space...")

    # NOTE: Check what the cell type column is actually named in the HSCA metadata.
    # It might be 'cell_type', 'annotation', or 'harmonized_cell_type'.
    reference_label_column = 'cell_type'

    # This projects the Xenium cells onto the HSCA PCA space and assigns labels
    sc.tl.ingest(adata_query, adata_ref, obs=reference_label_column)

    # 7. Save the mathematically annotated object
    adata_query.write_h5ad(output_h5ad_path)
    print(f"Success! Reference-mapped data saved to {output_h5ad_path}")

# Run the projection
transfer_hsca_labels(
    xenium_folder="/path/to/manchester_xenium",
    hsca_reference_path="/path/to/hsca_atlas.h5ad",
    output_h5ad_path="manchester_annotated.h5ad"
)