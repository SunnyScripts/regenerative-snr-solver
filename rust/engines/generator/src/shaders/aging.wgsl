// ==========================================
// DATA STRUCTURES (Memory Alignment)
// ==========================================

// This is the struct we generate from your Python JSON export.
// It holds the continuous decay/growth rates (k) for this specific lineage.
struct OptimalTransportTensor
{
    // The continuous rate constants for the 32 critical genes
    // e.g., if ABCA12 is index 0, rna_decay_rates[0] = 0.0038
    rna_decay_rates: array<f32, 32>,

    // Physical hardware decay rates extracted from structural/pump gene trajectories
    area_decay_rate: f32,       // How fast the cell physically shrinks
    pump_atrophy_rate: f32,     // How fast ATP1A1 capacity decays
    adhesion_decay_rate: f32,   // How fast CDH1 (Cadherin) grip weakens
    _padding: f32,              // 16-byte WGSL alignment
}

// Assuming CellNode includes:
// pub rna_spliced: array<f32, 32>
// pub area: f32
// pub pump_health_multiplier: f32 (Starts at 1.0, degrades to 0.0)
// pub adhesion_multiplier: f32    (Starts at 1.0, degrades to 0.0)

// ==========================================
// BINDINGS & CONSTANTS
// ==========================================
@group(0) @binding(0) var<storage, read_write> cells: array<CellNode>;
@group(0) @binding(1) var<storage, read> ot_tensors: array<OptimalTransportTensor>;

// The amount of macro-time passing in this single shader dispatch.
// E.g., if you are simulating 1 month per slow-clock tick, this might be 1.0.
@group(0) @binding(2) var<uniform> macro_time_step: f32;

const NUM_STATES: u32 = 4u;

@compute @workgroup_size(64)
fn aging_pass(@builtin(global_invocation_id) id: vec3<u32>) {
    let my_idx = id.x;
    if (my_idx >= arrayLength(&cells)) { return; }

    var me = cells[my_idx];

    // ==========================================
    // 1. The Void Check
    // ==========================================
    // If the cell is already dead (99), it is a permanent mathematical void.
    // It does not age, it does not shrink further. It just insulates.
    if (me.state_id == 99u) { return; }

    // Look up the specific OT decay tensor for this cell's lineage
    let ot = ot_tensors[me.granular_id];

    // ==========================================
    // 2. Transcriptomic Aging (The OT Manifold)
    // ==========================================
    // We apply your Sinkhorn-derived continuous rates (k) using Euler integration.
    // This mathematically forces the cell's RNA down the aging trajectory.
    for (var i = 0u; i < 32u; i++) {
        let decay_k = ot.rna_decay_rates[i];

        // dS = (k * S) * dt
        me.rna_spliced[i] += (decay_k * me.rna_spliced[i]) * macro_time_step;

        // Clamp to prevent negative expression due to large time steps
        me.rna_spliced[i] = max(me.rna_spliced[i], 0.0);
    }

    // ==========================================
    // 3. Hardware Degradation (Physical Reality)
    // ==========================================
    // As the tissue ages, the physical machine breaks down.
    // These multipliers directly throttle the Fast Clock's physics equations.

    // 1. Cell Shrinkage (Volume Loss)
    me.area += (ot.area_decay_rate * me.area) * macro_time_step;

    // 2. Pump Failure (ATP1A1 degradation)
    // This will cause the cell's V_mem to drift closer to 0mV in the Fast Clock
    me.pump_health_multiplier += (ot.pump_atrophy_rate * me.pump_health_multiplier) * macro_time_step;
    me.pump_health_multiplier = max(me.pump_health_multiplier, 0.0);

    // 3. Adhesion Loss (Cadherin degradation)
    // When Rust runs the topology update, this multiplier lowers the physical bond strength
    me.adhesion_multiplier += (ot.adhesion_decay_rate * me.adhesion_multiplier) * macro_time_step;
    me.adhesion_multiplier = max(me.adhesion_multiplier, 0.0);

    // ==========================================
    // 4. The Mathematical Death Boundary
    // ==========================================
    // If the OT trajectory physically shrinks the cell to nothing,
    // or if a critical driver gene (e.g., a mitochondrial marker) hits zero,
    // the cell undergoes necrosis and becomes a spatial void.

    let critical_gene_index = 0u; // Assuming index 0 is your core survival driver

    if (me.area <= 0.1 || me.rna_spliced[critical_gene_index] <= 0.01) {
        me.state_id = 99u;

        // Collapse the physical state
        me.v_mem = 0.0;
        me.area = 0.0;
        me.pump_health_multiplier = 0.0;

        // Wipe the transcriptome
        for (var i = 0u; i < 32u; i++) {
            me.rna_spliced[i] = 0.0;
            me.rna_unspliced[i] = 0.0;
        }
    }

    cells[my_idx] = me;
}