import json

# Load your newly minted universal rulebook
with open('interactome.json', 'r') as f:
    universal_rules = json.load(f)['interaction_rules']

# The 541 genes physically on your spatial slide
spatial_gene_set = set(spatial_adata.var_names)

surviving_rules = []
for rule in universal_rules:
    if rule['ligand_gene'] in spatial_gene_set and rule['receptor_gene'] in spatial_gene_set:
        surviving_rules.append(rule)

print(f"Universal Rules: {len(universal_rules)}")
print(f"Rules Surviving on this Hardware: {len(surviving_rules)}")

# Check if any cell types went "extinct"
surviving_targets = set(r['target_granular_vote'] for r in surviving_rules)
all_targets = set(r['target_granular_vote'] for r in universal_rules)

extinct_cells = all_targets - surviving_targets
if extinct_cells:
    print(f"⚠️ WARNING: These cells cannot be identified by this hardware: {extinct_cells}")