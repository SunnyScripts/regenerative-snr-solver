from scipy.stats import pearsonr
import networkx as nx

# 1. Define Core Genes
core_genes = ['TP63', 'KRT14', 'KRT10', 'LOR', 'GJA1', 'KCNJ2', 'CDKN2A']
valid_genes = [g for g in core_genes if g in adata.var_names]

# 2. Initialize a standard NetworkX Directed Graph (Cellnition's native input)
grn_graph = nx.DiGraph()
grn_graph.add_nodes_from(valid_genes)

CORR_THRESHOLD = 0.3

for source in valid_genes:
    for target in valid_genes:
        if source == target:
            continue

        source_expr = adata[:, source].layers['Ms'].flatten()
        target_velo = adata[:, target].layers['velocity'].flatten()

        correlation, p_value = pearsonr(source_expr, target_velo)

        if abs(correlation) > CORR_THRESHOLD and p_value < 0.05:
            # 3. Categorize the interaction for Cellnition's Hill/Logistic equations
            interaction_type = 'activation' if correlation > 0 else 'inhibition'

            # Add edge with required Cellnition metadata
            grn_graph.add_edge(
                source,
                target,
                weight=abs(correlation),
                type=interaction_type
            )

print(f"Extracted {grn_graph.number_of_edges()} regulatory edges.")





from cellnition.probability_net import ProbabilityNet

# 1. Initialize the Cellnition Network Machine
# You can specify the mathematical function type (e.g., 'logistic' or 'hill')
prob_net = ProbabilityNet(grn_graph, func_type='logistic')

# 2. Characterize the Graph Hierarchy
# (Finds input nodes, output nodes, and cyclic dependencies)
prob_net.characterize_graph()

# 3. Build the Analytic Model
# This translates the edges into continuous ODEs
prob_net.build_analytic_model()

# 4. Find the Attractor Basins
# This runs the continuous simulation to find the stable equilibrium states
# (Your 'Young', 'Senescent', etc. target states)
equilibrium_states = prob_net.find_attractor_sols()

print("\n--- Identified Stable Target States ---")
for i, state in enumerate(equilibrium_states):
    print(f"Attractor State {i}: {state}")

import anndata as ad
import networkx as nx
import json
import numpy as np

# Import Cellnition's continuous modeling suite
from cellnition import ProbabilityNet

print("Loading SDEvelo Data...")
adata = ad.read_h5ad("skin_sdevelo_output.h5ad")

# ==========================================
# 1. Extract the Causal Terrain (SDEvelo Jacobian)
# ==========================================
# We completely skip Pearson correlation. We pull the true drift weights.
# Assuming SDEvelo stored the Jacobian matrix in adata.uns
jacobian_matrix = adata.uns['sde_jacobian']
genes = adata.var_names.tolist()

grn_graph = nx.DiGraph()
grn_graph.add_nodes_from(genes)

# Populate Cellnition's required NetworkX format
for i, source in enumerate(genes):
    for j, target in enumerate(genes):
        weight = jacobian_matrix[i, j]

        # Filter out negligible noise to keep the analytic model fast
        if abs(weight) > 0.05:
            interaction_type = 'activation' if weight > 0 else 'inhibition'
            grn_graph.add_edge(source, target, weight=abs(weight), type=interaction_type)

print(f"Extracted {grn_graph.number_of_edges()} causal regulatory edges.")

# ==========================================
# 2. Build the Cellnition Analytical Model
# ==========================================
print("Initializing Cellnition ProbabilityNet...")
prob_net = ProbabilityNet(grn_graph)

# Characterize the topology (cycles, hierarchies) and build differential equations
prob_net.characterize_graph()
prob_net.build_analytic_model()

# ==========================================
# 3. Discover Attractors (The Wells)
# ==========================================
print("Finding steady-state equilibrium attractors...")
# Cellnition automatically scans the state space for point attractors (wells)
attractors = prob_net.find_attractor_sols()

# In a full script, you would map these raw mathematical attractors to your AnnData clusters
# For this example, let's assume Cellnition mapped out the following network transitions:

# Transition A: Differentiation (Basal -> Spinous)
# Transition B: Senescence (Spinous -> Senescent)

# ==========================================
# 4. Generate the 2D Architecture JSON for Rust
# ==========================================
# We map the NFSM transitions to your split granular_id / state_id logic

# ID Reference:
# Granular IDs: 0 = Basal, 1 = Spinous, 2 = Granular
# State IDs: 0 = Young, 1 = Old, 2 = Senescent, 99 = Dead

rust_fsm_config = {
    "fsm_rules": [
        {
            "rule_name": "Differentiation_Basal_To_Spinous",
            "trigger_type": "lineage_shift",
            "from_granular_id": 0,
            "to_granular_id": 1,
            "enforce_state_id": 0,  # Only healthy Young cells differentiate normally
            "driver_gene": "KRT10_spliced",  # A classic spinous marker
            "threshold_value": 0.650,
            "trigger_direction": "greater_than"
        },
        {
            "rule_name": "Senescence_Spinous",
            "trigger_type": "state_shift",
            "target_granular_id": 1,  # Applies specifically to Spinous cells
            "from_state_id": 0,
            "to_state_id": 2,  # Push to Senescent state
            "driver_gene": "CDKN2A_spliced",
            "threshold_value": 0.812,
            "trigger_direction": "greater_than"
        }
    ]
}

with open("fsm_thresholds.json", "w") as f:
    json.dump(rust_fsm_config, f, indent=2)

print("Exported dual-axis FSM thresholds for Rust WGSL aging_pass.")












import anndata as ad
import networkx as nx
import json
import numpy as np

# Import Cellnition's continuous modeling suite
from cellnition import ProbabilityNet

print("Loading SDEvelo Data...")
adata = ad.read_h5ad("skin_sdevelo_output.h5ad")

# ==========================================
# 1. Extract the Causal Terrain (SDEvelo Jacobian)
# ==========================================
# We completely skip Pearson correlation. We pull the true drift weights.
# Assuming SDEvelo stored the Jacobian matrix in adata.uns
jacobian_matrix = adata.uns['sde_jacobian']
genes = adata.var_names.tolist()

grn_graph = nx.DiGraph()
grn_graph.add_nodes_from(genes)

# Populate Cellnition's required NetworkX format
for i, source in enumerate(genes):
    for j, target in enumerate(genes):
        weight = jacobian_matrix[i, j]

        # Filter out negligible noise to keep the analytic model fast
        if abs(weight) > 0.05:
            interaction_type = 'activation' if weight > 0 else 'inhibition'
            grn_graph.add_edge(source, target, weight=abs(weight), type=interaction_type)

print(f"Extracted {grn_graph.number_of_edges()} causal regulatory edges.")

# ==========================================
# 2. Build the Cellnition Analytical Model
# ==========================================
print("Initializing Cellnition ProbabilityNet...")
prob_net = ProbabilityNet(grn_graph)

# Characterize the topology (cycles, hierarchies) and build differential equations
prob_net.characterize_graph()
prob_net.build_analytic_model()

# ==========================================
# 3. Discover Attractors (The Wells)
# ==========================================
print("Finding steady-state equilibrium attractors...")
# Cellnition automatically scans the state space for point attractors (wells)
attractors = prob_net.find_attractor_sols()

# In a full script, you would map these raw mathematical attractors to your AnnData clusters
# For this example, let's assume Cellnition mapped out the following network transitions:

# Transition A: Differentiation (Basal -> Spinous)
# Transition B: Senescence (Spinous -> Senescent)

# ==========================================
# 4. Generate the 2D Architecture JSON for Rust
# ==========================================
# We map the NFSM transitions to your split granular_id / state_id logic

# ID Reference:
# Granular IDs: 0 = Basal, 1 = Spinous, 2 = Granular
# State IDs: 0 = Young, 1 = Old, 2 = Senescent, 99 = Dead

rust_fsm_config = {
    "fsm_rules": [
        {
            "rule_name": "Differentiation_Basal_To_Spinous",
            "trigger_type": "lineage_shift",
            "from_granular_id": 0,
            "to_granular_id": 1,
            "enforce_state_id": 0,  # Only healthy Young cells differentiate normally
            "driver_gene": "KRT10_spliced",  # A classic spinous marker
            "threshold_value": 0.650,
            "trigger_direction": "greater_than"
        },
        {
            "rule_name": "Senescence_Spinous",
            "trigger_type": "state_shift",
            "target_granular_id": 1,  # Applies specifically to Spinous cells
            "from_state_id": 0,
            "to_state_id": 2,  # Push to Senescent state
            "driver_gene": "CDKN2A_spliced",
            "threshold_value": 0.812,
            "trigger_direction": "greater_than"
        }
    ]
}

with open("fsm_thresholds.json", "w") as f:
    json.dump(rust_fsm_config, f, indent=2)

print("Exported dual-axis FSM thresholds for Rust WGSL aging_pass.")





