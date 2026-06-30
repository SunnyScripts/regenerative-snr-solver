import omnipath
import json
import pandas as pd

print("Fetching OmniPath Intercell Network...")
# Pull the comprehensive intercellular communication database
db = omnipath.interactions.import_intercell_network()

# DEBUG: Uncomment this if you still get a KeyError to see all available columns
# print(db.columns.tolist())

# 1. FILTER FOR ADHESION
# We use 'category_intercell_source' which is the standard OmniPath column
adhesion_db = db[db['category_intercell_source'].str.contains('adhesion', na=False, case=False)]

# Extract the gene symbol pairs using the updated column names
adhesion_pairs = []

# Use the specific intercell gene symbol columns
src_col = 'genesymbol_intercell_source'
tgt_col = 'genesymbol_intercell_target'

for _, row in adhesion_db.iterrows():
    source_gene = row.get(src_col)
    target_gene = row.get(tgt_col)

    if pd.notna(source_gene) and pd.notna(target_gene):
        adhesion_pairs.append([str(source_gene), str(target_gene)])

# Deduplicate
adhesion_pairs = [list(x) for x in set(tuple(x) for x in adhesion_pairs)]

# 2. GAP JUNCTIONS (Conductance)
gap_junctions = ["GJA1", "GJA3", "GJA4", "GJA5", "GJA8", "GJB1", "GJB2", "GJB3", "GJB4", "GJB5", "GJB6", "GJC1", "GJC2", "GJC3"]

# 3. EXPORT FOR RUST
rulebook = {
    "adhesion_pairs": adhesion_pairs,
    "gap_junction_genes": gap_junctions
}

output_path = "universal_interactome.json"
# Explicitly setting encoding='utf-8' often clears the IDE warning
with open(output_path, "w", encoding='utf-8') as f:
    json.dump(rulebook, f, indent=2)

print(f"Success! Exported {len(adhesion_pairs)} adhesion pairs to {output_path}")