# import scanpy
import json
import requests

# A. Xenium Genes (541 physically visible genes)
with open("../Cellular Data/Spatial/Manchester/back/gene_panel.json", "r") as f:
    panel_data = json.load(f)
targets = panel_data.get('payload', {}).get('targets', panel_data)
xenium_symbols = [t['type']['data']['name'] for t in targets if t.get('type', {}).get('descriptor') == 'gene']

# DEBUG: Prove we actually extracted symbols
print(f"   - Extracted {len(xenium_symbols)} symbols from Xenium JSON.")

with open(f"../Cellular Data/Spatial/Manchester/back/train_station/stripped_gene_panel", "w") as f:
    json.dump(xenium_symbols, f)

# B. Map Xenium Symbols -> Entrez IDs (Bypassing hishel bug)
# print("   - Mapping Xenium Symbols to Entrez IDs via raw API call...")
# headers = {'Content-Type': 'application/x-www-form-urlencoded'}
# query_data = {
#     'q': ','.join(xenium_symbols),
#     'scopes': 'symbol',
#     'fields': 'entrezgene',
#     'species': 'human'
# }
#
# response = requests.post("https://mygene.info/v3/query", data=query_data, headers=headers)
# response.raise_for_status()
# mapping_results = response.json()
#
# # 1. Initialize collections
# symbol_to_entrez = {}
# dropped_symbols = []
#
# # 2. Iterate through the results
# for result in mapping_results:
#     symbol = result.get('query')
#
#     # Check if a mapping was actually found
#     if 'entrezgene' in result:
#         # Some symbols might return multiple hits; we'll take the first one
#         # or you can add logic here to choose the best one
#         symbol_to_entrez[symbol] = int(result['entrezgene'])
#     else:
#         dropped_symbols.append(symbol)
#
# # 3. Print the report
# print(f"📊 Mapping Summary:")
# print(f"   - Total Queried: {len(xenium_symbols)}")
# print(f"   - Successfully Mapped: {len(symbol_to_entrez)}")
# print(f"   - Dropped: {len(dropped_symbols)}")
#
# if dropped_symbols:
#     print("\n❌ Dropped Genes List:")
#     for s in dropped_symbols:
#         print(f"   - {s}")

# Containers for our two different file formats
# gene_to_entrez = {}  # For the mapping (Dictionary)
# xenium_entrez_set = set()  # For the flat list (Set ensures uniqueness)
#
# for item in mapping_results:
#     symbol = item.get('query') or item.get('symbol')
#     entrez_id = item.get('entrezgene')
#
#     if symbol and entrez_id:
#         str_id = str(entrez_id)
#
#         # 1. Populate the Mapping
#         gene_to_entrez[symbol] = str_id
#
#         # 2. Populate the Unique Set
#         xenium_entrez_set.add(str_id)
#
# # --- SAVE FILE 1: THE MAPPING (Symbol -> ID) ---
# with open("Gene Data/back_spatial_map.json", "w") as f:
#     json.dump(gene_to_entrez, f, indent=4)
#
# # --- SAVE FILE 2: THE FLAT ARRAY (Unique IDs Only) ---
# with open("Gene Data/back_entrez_list.json", "w") as f:
#     # We sort the list so the file is consistent every time you save it
#     json.dump(sorted(list(xenium_entrez_set)), f, indent=4)
#
# print(f"Done! Saved {len(gene_to_entrez)} mappings and {len(xenium_entrez_set)} unique IDs.")


# print(f"   - Successfully mapped {len(xenium_entrez_set)} Xenium genes to Entrez IDs.")
# adata = scanpy.read_h5ad("../Cellular Data/Single Cell/Reference/")
#
# model_genes = adata_subset.var_names.tolist()
#
# overlapping_entrez = set(model_genes).intersection(xenium_entrez_set)