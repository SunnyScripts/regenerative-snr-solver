import json
import scanpy

adata = scanpy.read_h5ad("../../Cellular Data/Single Cell/Reference/sb2_mech.h5ad")

true_449_genes = adata.var_names.tolist()

# scvi-tools keeps a strict registry of the EXACT genes that survived
# filtering and made it into the neural network's input layer.
# try:
#     # Try the standard registry method first
#     true_449_genes = pan_scanvi_model.registry_["var_names"]
# except KeyError:
#     # Fallback to the AnnData object attached to the model
#     true_449_genes = pan_scanvi_model.adata.var_names.tolist()

print(f"Extracted {len(true_449_genes)} genes from the model registry.")

# Save this true list to a new JSON
with open("true_449_entrez_list.json", "w") as f:
    json.dump(true_449_genes, f)

print("✅ Saved! Use this JSON for your spatial alignment.")