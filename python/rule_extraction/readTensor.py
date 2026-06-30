import pandas as pd
import scanpy

adata_final = scanpy.read_h5ad("../Cell Classifier/adata.h5ad")
import holoviews as hv
import datashader as ds
from holoviews.operation.datashader import datashade, dynspread
from bokeh.palettes import Turbo256

# 1. Clean up the categories (Essential for 1.8M cells)
adata_final.obs['granular_cell_type'] = adata_final.obs['granular_cell_type'].cat.remove_unused_categories()

# 2. Build the high-res dataframe
df = pd.DataFrame(adata_final.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
df['granular_type'] = adata_final.obs['granular_cell_type'].values

# 3. Use a 256-color palette to cover all 60 types
# 'Turbo256' is a high-contrast rainbow that makes 60 types easy to distinguish
points = hv.Points(df, kdims=['UMAP1', 'UMAP2'], vdims=['granular_type'])
shaded = datashade(
    points,
    aggregator=ds.count_cat('granular_type'),
    cmap=Turbo256, # The "Rainbow" fix for 60+ classes
    width=3000,
    height=2000
)

# 4. Render to HTML
plot = dynspread(shaded, max_px=4).opts(
    width=1200, height=900,
    title="Granular Mechanical Atlas (60 Classes)",
    bgcolor="black"
)

hv.save(plot, 'granular_atlas_audit.html')
# 2. Run a simple Python web server in that folder
# Open a NEW terminal window, navigate to your project folder, and run:
# python -c "import http.server; http.server.test(HandlerClass=http.server.SimpleHTTPRequestHandler, port=8000)"

# 3. Open your browser to:
# http://localhost:8000


# Plot the UMAP using the pre-calculated X_umap
# scanpy.pl.umap(
#     adata_final,
#     color=['mechanical_cell_type', 'granular_cell_type'],
#     frameon=False,
#     wspace=0.4,
#     title=['Mechanical Classes', 'Interactome Classes'],
# )


# Check the exact counts of your mechanical classes
# print(adata_small.obs['mechanical_cell_type'].value_counts())
# 
# # If you want to see exactly WHICH granular cells ended up in Unknown
# unknowns = adata_small.obs[adata_small.obs['mechanical_cell_type'] == 'Unknown']
# print("\nGranular breakdown of Unknowns:")
# print(unknowns['granular_cell_type'].value_counts())


# from safetensors import safe_open
#
# file_path = "../Network/models/cell_type_classifier/mechanical.safetensors"
# with safe_open(file_path, framework="pt", device="cpu") as f:
#     for key in f.keys():
#         # Get tensor names and their metadata (shape/dtype)
#         tensor_slice = f.get_slice(key)
#         print(f"Name: {key}")
#         print(f"  Shape: {tensor_slice.get_shape()}")
#
#     # Access global file metadata (e.g., training info)
#     metadata = f.metadata()
#     print(f"Global Metadata: {metadata}")
