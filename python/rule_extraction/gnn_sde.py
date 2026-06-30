
# Train the graph sde

import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class GraphSDE(nn.Module):
    def __init__(self, num_genes):
        super().__init__()

        # 1. The GNN Spatial Delta Coupler (The Teacher)
        # Input features: [V_mem, P_k, P_na, P_cl, Pump_Strength]
        self.gnn_conv1 = gnn.GCNConv(5, 64)
        self.gnn_conv2 = gnn.GCNConv(64, 32)

        # Output layer maps the hidden state to Channel Gating (Delta P)
        # Output: [Delta P_k, Delta P_na, Delta P_cl]
        self.gating_out = nn.Linear(32, 3)

        # 2. SDEvelo Baseline Priors (Learned parameters)
        # For a full genome, these would be vectors of size (num_genes)
        self.beta = nn.Parameter(torch.rand(num_genes))
        self.gamma = nn.Parameter(torch.rand(num_genes))

        self.dt = 0.01  # Time step for Euler-Maruyama integration

    def forward(self, x_phys, edge_index, rna_u, rna_s, env_v):
        """
        x_phys: Tensor [num_cells, 5] -> V_mem and Baseline Channel/Pump Data
        edge_index: Tensor [2, num_edges] -> The spatial gap junction graph
        rna_u, rna_s: Tensor [num_cells, num_genes] -> Current RNA state
        env_v: Tensor [num_cells, 1] -> The Eulerian fluid voltage
        """

        # ==========================================
        # STEP 1: GNN Message Passing (Spatial Intelligence)
        # ==========================================
        # The network looks at its neighbors to figure out how to alter its ion channels
        hidden = torch.relu(self.gnn_conv1(x_phys, edge_index))
        hidden = torch.relu(self.gnn_conv2(hidden, edge_index))

        # The learned spatial residual (Channel Gating)
        delta_p = torch.tanh(self.gating_out(hidden))

        # Extract the modifiers
        dp_k, dp_na, dp_cl = delta_p[:, 0], delta_p[:, 1], delta_p[:, 2]

        # ==========================================
        # STEP 2: GHK Thermodynamics + Active Pumps
        # ==========================================
        # Unpack the baseline physical features
        v_mem, base_pk, base_pna, base_pcl, pump_strength = x_phys.T

        # Apply the GNN's gating residuals to the baseline permeabilities
        p_k = torch.clamp(base_pk + dp_k, min=0.0)
        p_na = torch.clamp(base_pna + dp_na, min=0.0)
        p_cl = torch.clamp(base_pcl + dp_cl, min=0.0)

        # Simplified GHK calculation (Assuming constant extracellular ions for brevity)
        ghk_num = (p_k * 5.0) + (p_na * 145.0) + (p_cl * 10.0)
        ghk_den = (p_k * 140.0) + (p_na * 15.0) + (p_cl * 110.0)
        target_v = 26.7 * torch.log(ghk_num / ghk_den)

        # Calculate new V_mem combining GHK target and Active Pumps
        new_v_mem = v_mem + ((target_v - v_mem) - pump_strength) * self.dt

        # ==========================================
        # STEP 3: The SDE RNA Update (Euler-Maruyama)
        # ==========================================
        # Delta-coupling: Voltage alters RNA transcription rates (simplified Hill function)
        c_effective = torch.sigmoid(new_v_mem)

        # Stochastic Noise (Brownian motion)
        noise_u = torch.randn_like(rna_u) * 0.1 * torch.sqrt(torch.tensor(self.dt))
        noise_s = torch.randn_like(rna_s) * 0.1 * torch.sqrt(torch.tensor(self.dt))

        # The SDE Drift
        du = (c_effective.unsqueeze(1) - self.beta * rna_u) * self.dt + noise_u
        ds = (self.beta * rna_u - self.gamma * rna_s) * self.dt + noise_s

        new_rna_u = rna_u + du
        new_rna_s = rna_s + ds

        # Return the updated physical and RNA states
        return new_v_mem, new_rna_u, new_rna_s, delta_p


# Initialization Example
num_genes = 2000
model = GraphSDE(num_genes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# gene to ion channel, i need permeability and active pumps eg ATP1B1
# gap junctions are built into the topology shader with ligand receptor logic

import scanpy
import mygene
import numpy as np
import json

# ==========================================
# 0. Load the Data
# ==========================================
print("Reading filtered reference file...")
fHSCA = scanpy.read_h5ad("../Cell Data/Reference/Filtered_HSCA_extended.h5ad")

# ==========================================
# 1. Automate Gene Discovery via GO Terms
# ==========================================
print("Querying Gene Ontology for Passive Ion Channels...")
mg = mygene.MyGeneInfo()
all_genes = fHSCA.var_names.tolist()

results = mg.querymany(all_genes, scopes='symbol', fields='go', species='human')

# We only care about the passive leak channels for the GHK Permeability calculation
k_genes, na_genes, cl_genes = set(), set(), set()

for res in results:
    if 'go' in res:
        for cat in ['MF', 'BP', 'CC']:
            if cat in res['go']:
                terms = res['go'][cat] if isinstance(res['go'][cat], list) else [res['go'][cat]]
                for term in terms:
                    if term['id'] == 'GO:0005267': k_genes.add(res['query'])  # Potassium
                    if term['id'] == 'GO:0005272': na_genes.add(res['query'])  # Sodium
                    if term['id'] == 'GO:0005254': cl_genes.add(res['query'])  # Chloride

k_genes = list(k_genes)
na_genes = list(na_genes)
cl_genes = list(cl_genes)

print(f"Found {len(k_genes)} K+ channels, {len(na_genes)} Na+ channels, {len(cl_genes)} Cl- channels.")

# ==========================================
# 2. Calculate Transcriptomic Permeability per Cell
# ==========================================
print("Calculating raw permeabilities...")
# We use the raw counts (.X) to determine how many physical channels the cell built.
# If your .X is sparse, .sum(axis=1) returns a matrix, so we use .A1 to flatten it to a 1D array.

if hasattr(fHSCA.X, "toarray"):
    fHSCA.obs['P_k_raw'] = fHSCA[:, k_genes].X.sum(axis=1).A1
    fHSCA.obs['P_na_raw'] = fHSCA[:, na_genes].X.sum(axis=1).A1
    fHSCA.obs['P_cl_raw'] = fHSCA[:, cl_genes].X.sum(axis=1).A1
else:
    fHSCA.obs['P_k_raw'] = fHSCA[:, k_genes].X.sum(axis=1)
    fHSCA.obs['P_na_raw'] = fHSCA[:, na_genes].X.sum(axis=1)
    fHSCA.obs['P_cl_raw'] = fHSCA[:, cl_genes].X.sum(axis=1)

# ==========================================
# 3. Group by Lineage and Average
# ==========================================
print("Averaging by lineage...")
lineage_permeabilities = {}
unique_lineages = sorted(fHSCA.obs['granular_id'].unique())

for lineage_id in unique_lineages:
    # Isolate all cells belonging to this specific lineage
    lineage_mask = fHSCA.obs['granular_id'] == lineage_id
    lineage_cells = fHSCA[lineage_mask]

    # Calculate the mean raw permeability for this lineage
    mean_k = lineage_cells.obs['P_k_raw'].mean()
    mean_na = lineage_cells.obs['P_na_raw'].mean()
    mean_cl = lineage_cells.obs['P_cl_raw'].mean()

    lineage_permeabilities[lineage_id] = {
        'k': mean_k,
        'na': mean_na,
        'cl': mean_cl
    }

# ==========================================
# 4. Global Normalization (Min-Max Scaling)
# ==========================================
print("Normalizing for the GHK Physics Engine...")
# The GHK equation in WGSL will blow up if you feed it raw expression counts of 5,000.
# We scale the permeabilities relative to the max observed in the dataset (0.0 to 1.0).

max_k = max(p['k'] for p in lineage_permeabilities.values()) + 1e-6
max_na = max(p['na'] for p in lineage_permeabilities.values()) + 1e-6
max_cl = max(p['cl'] for p in lineage_permeabilities.values()) + 1e-6

for lineage_id in lineage_permeabilities:
    # We overwrite the dictionary with the normalized WGSL-ready floats
    lineage_permeabilities[lineage_id]['base_p_k'] = float(lineage_permeabilities[lineage_id]['k'] / max_k)
    lineage_permeabilities[lineage_id]['base_p_na'] = float(lineage_permeabilities[lineage_id]['na'] / max_na)
    lineage_permeabilities[lineage_id]['base_p_cl'] = float(lineage_permeabilities[lineage_id]['cl'] / max_cl)

# ==========================================
# 5. Export Readiness
# ==========================================
print("Done. Example output for Lineage 0:")
print(json.dumps(lineage_permeabilities[unique_lineages[0]], indent=4))

# From here, you can fold these values into your main ot_tensors_export JSON payload,
# or create a dedicated JSON file that Rust reads to populate the SDEParams buffer.




# Knowledge Distillation
import torch
import torch.nn as nn
import json


class DistilledStudent(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [V_mem, Env_K, Gap_Drift]
        # Output: [delta_Pk, delta_Pna, delta_Pcl]
        self.layer1 = nn.Linear(3, 3)
        self.layer2 = nn.Linear(3, 3)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        return x


# Let's assume you have 70 lineages, with 4 states each = 280 parameter sets
num_param_sets = 280
students = [DistilledStudent() for _ in range(num_param_sets)]
optimizers = [torch.optim.Adam(s.parameters(), lr=0.01) for s in students]
loss_fn = nn.MSELoss()

print("Distilling GNN knowledge into Student MLPs...")

# 1. Run the Teacher ONCE on the real graph to get the ground truth
teacher.eval()
with torch.no_grad():
    # teacher_delta_p shape: [num_cells, 3]
    _, _, _, teacher_delta_p = teacher(real_x_phys, real_edge_index, real_rna_u, real_rna_s, real_env_v)

# 2. Extract the local inputs the WGSL shader will actually have access to
# [V_mem, Env_K, Gap_Drift]
v_mem = real_x_phys[:, 0]
env_k = real_env_v[:, 0]

# You must calculate the gap_drift exactly as the WGSL shader will
# gap_drift = sum(conductance * (neighbor_v - my_v))
local_gap_drift = calculate_local_gap_drift(real_x_phys, real_edge_index)

student_inputs = torch.stack([v_mem, env_k, local_gap_drift], dim=1)  # Shape: [num_cells, 3]

# 3. Train the Students (Grouped by parameter index)
# Assuming you have a tensor `cell_param_indices` tracking which cell belongs to which param_idx
for epoch in range(1000):
    total_loss = 0.0

    for param_idx in range(num_param_sets):
        # Isolate the cells for this specific Student
        mask = (cell_param_indices == param_idx)
        if not mask.any():
            continue

        inputs = student_inputs[mask]
        targets = teacher_delta_p[mask]

        student = students[param_idx]
        optimizer = optimizers[param_idx]

        predictions = student(inputs)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

# 4. Export to WGSL Array
print("Exporting MLPLayer array for Rust...")


def export_layer(layer):
    return {
        "weights": layer.weight.detach().T.numpy().flatten().tolist(),
        "bias": layer.bias.detach().numpy().flatten().tolist()
    }


rust_mlp_array = []
for student in students:
    rust_mlp_array.append({
        "w1": export_layer(student.layer1)["weights"],
        "b1": export_layer(student.layer1)["bias"],
        "w2": export_layer(student.layer2)["weights"],
        "b2": export_layer(student.layer2)["bias"]
    })

with open("mlp_weights.json", "w") as f:
    json.dump(rust_mlp_array, f)