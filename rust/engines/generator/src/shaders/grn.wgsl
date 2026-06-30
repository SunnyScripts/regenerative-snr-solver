//ToDo: You will need to run a tiny compute shader (e.g., apply_turnover_deltas.wgsl) immediately after this one. That tiny shader will look at the turnover_deltas buffer, convert the i32 back to an f32 (by dividing by 10000.0), add it to me.rna_spliced[turnover_gene_idx], and zero out the delta buffer for the next frame.</GeneKinetics,>

// ==========================================
// DATA STRUCTURES
// ==========================================
struct Coord { x: f32, y: f32, z: f32 }

struct CellNode {
    pos: Coord, broad_id: u32,
    polarity: Coord, granular_id: u32,
    area: f32, ion_ca: f32, ion_cl: f32, v_mem: f32,
    ion_k: f32, ion_na: f32, exogenous_v_clamp: f32, pump_health_multiplier: f32,
    adhesion_multiplier: f32, neighbor_count: u32, edge_start: u32, _padding: u32,
    rna_unspliced: array<f32, 128>,
    rna_spliced: array<f32, 128>,
}

struct MembraneInterface {
    source_idx: u32, target_idx: u32, conductance: f32, adhesion: f32,
    distance: f32, connection_strength: f32, matrix_stiffness: f32, rl_edge_clamp: f32,
}

struct GeneKinetics {
    beta: f32, gamma: f32, c: f32, v_max: f32,
    k_half: f32, hill_n: f32, sigma1: f32, sigma2: f32,
    max_u: f32, max_s: f32, v_resting: f32,
    // New SDE Params packed into the padding
    old_target_u: f32,
    old_target_s: f32,
    wot_theta: f32,
    _pad2: f32, // Maintains 64-byte alignment per gene
}

struct SDEParams {
    base_p_k: f32, base_p_na: f32, base_p_cl: f32, pump_current: f32,
    g_leak: f32, capacitance: f32,
    idx_k: u32, idx_na: u32, idx_cl: u32, idx_pump: u32, idx_connexin: u32, _padding: u32,
    genes: array<GeneKinetics, 128>,
}

// ==========================================
// BINDINGS & CONSTANTS
// ==========================================
@group(0) @binding(0) var<storage, read_write> cells: array<CellNode>;
@group(0) @binding(1) var<storage, read> sde_params: array<SDEParams>;
@group(0) @binding(2) var<storage, read> edges: array<MembraneInterface>;
@group(0) @binding(3) var<storage, read_write> global_turnover_count: atomic<u32>;
// NEW: Float atomics aren't allowed. We use a dedicated i32 buffer for neighbor energy transfer.
@group(0) @binding(4) var<storage, read_write> turnover_deltas: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> sim_time: u32;

const GRN_DT: f32 = 0.1;
const NUM_STATES: u32 = 4u; // Assuming 4 aging states (e.g., Young, Adult, Old, Senescent)

@compute @workgroup_size(64)
fn grn_pass(@builtin(global_invocation_id) id: vec3<u32>)
{
    let my_idx = id.x;
    if (my_idx >= arrayLength(&cells)) { return; }

    var me = cells[my_idx];
    if (me.state_id == 99u) { return; }

    let param_idx = (me.granular_id * NUM_STATES) + me.state_id;
    let params = sde_params[param_idx];

    // ==========================================
    // 1. The SDE Math (128 Genes)
    // ==========================================
    for (var i = 0u; i < 128u; i++)
    {
        let gene = params.genes[i];

        // Epsilon added to prevent NaN
        let stimulus = max(1e-6, me.v_mem - gene.v_resting);
        let hill_num = pow(stimulus, gene.hill_n);
        let hill_den = pow(gene.k_half, gene.hill_n) + hill_num;
        let c_effective = gene.c + (gene.v_max * (hill_num / hill_den));

        // Temporal seed for valid random walks
        let seed_u = my_idx * 256u + i + sim_time;
        let seed_s = my_idx * 256u + i + 128u + sim_time;
        let noise_u = generate_gaussian_noise(seed_u) * gene.sigma1;
        let noise_s = generate_gaussian_noise(seed_s) * gene.sigma2;

        // The WOT Aging Pull
        let wot_pull_u = gene.wot_theta * (gene.old_target_u - me.rna_unspliced[i]);
        let wot_pull_s = gene.wot_theta * (gene.old_target_s - me.rna_spliced[i]);

        let du = (c_effective - gene.beta * me.rna_unspliced[i] + wot_pull_u) * GRN_DT + noise_u;
        let ds = (gene.beta * me.rna_unspliced[i] - gene.gamma * me.rna_spliced[i] + wot_pull_s) * GRN_DT + noise_s;

        me.rna_unspliced[i] = clamp(me.rna_unspliced[i] + du, 0.0, gene.max_u);
        me.rna_spliced[i] = clamp(me.rna_spliced[i] + ds, 0.0, gene.max_s);
    }

    // ==========================================
    // 2. Hardware Feedback
    // ==========================================
    // me.dynamic_p_k removed! The physics_pass dynamically reads me.rna_spliced[params.idx_k]
    // We only update the structural/atrophy modifiers here if needed.
    me.pump_health_multiplier = clamp(me.rna_spliced[params.idx_pump] / max(params.genes[params.idx_pump].max_s, 0.1), 0.1, 1.0);

    // ==========================================
    // 3. Eulerian State Flux (Fixed-Point Atomics)
    // ==========================================
    let turnover_gene_idx = 24u; // Make sure this index matches your 128-gene payload layout
    let overflow_threshold = params.genes[turnover_gene_idx].max_s * 0.95;

    if (me.rna_spliced[turnover_gene_idx] >= overflow_threshold) {
        var is_top_layer = true;
        var valid_apical_neighbors = 0u;
        var dot_products: array<f32, 32>; // Assumes max 32 neighbors for stack sizing
        var sum_dots: f32 = 0.0;

        for (var i = 0u; i < me.neighbor_count; i++) {
            let edge = edges[me.edge_start + i];
            let neighbor = cells[edge.target_idx];
            let dir_to_neighbor = normalize(neighbor.pos - me.pos);
            let alignment = dot(me.polarity, dir_to_neighbor);

            if (alignment > 0.2) {
                is_top_layer = false;
                dot_products[i] = alignment;
                sum_dots += alignment;
                valid_apical_neighbors++;
            } else {
                dot_products[i] = 0.0;
            }
        }

        if (is_top_layer) {
            atomicAdd(&global_turnover_count, 1u);
            me.rna_spliced[turnover_gene_idx] = 0.0;
        } else if (valid_apical_neighbors > 0u) {
            let energy_to_pass = me.rna_spliced[turnover_gene_idx];
            me.rna_spliced[turnover_gene_idx] = 0.0;

            for (var i = 0u; i < me.neighbor_count; i++) {
                if (dot_products[i] > 0.0) {
                    let fraction = dot_products[i] / sum_dots;
                    let target_idx = edges[me.edge_start + i].target_idx;

                    // The WebGPU Float-to-Int Workaround (Multiplied by 10,000 for precision)
                    let energy_int = i32(energy_to_pass * fraction * 10000.0);
                    atomicAdd(&turnover_deltas[target_idx], energy_int);
                }
            }
        }
    }

    cells[my_idx] = me;
}