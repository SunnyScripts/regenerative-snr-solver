// ==========================================
// DATA STRUCTURES
// ==========================================
struct MLPLayer {
    w1: mat3x3<f32>, b1: vec3<f32>,
    w2: mat3x3<f32>, b2: vec3<f32>,
}

struct GeneKinetics {
    beta: f32, gamma: f32, c: f32, v_max: f32,
    k_half: f32, hill_n: f32, sigma1: f32, sigma2: f32,
    max_u: f32, max_s: f32, v_resting: f32, _padding: f32,
}

struct SDEParams {
    base_p_k: f32, base_p_na: f32, base_p_cl: f32, pump_current: f32,
    g_leak: f32, capacitance: f32,
    idx_k: u32, idx_na: u32, idx_cl: u32, idx_pump: u32, idx_connexin: u32, _padding: u32,
    genes: array<GeneKinetics, 128>, // Expanded to 128
}

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

// ==========================================
// BINDINGS & CONSTANTS
// ==========================================
@group(0) @binding(0) var<storage, read_write> cells: array<CellNode>;
@group(0) @binding(1) var<storage, read> edges: array<MembraneInterface>;
@group(0) @binding(2) var<storage, read> sde_params: array<SDEParams>;
@group(0) @binding(3) var env_grid: texture_3d<f32>; // The diffusion shader grid
@group(0) @binding(4) var<storage, read_write> grid_deltas: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> convergence_counter: atomic<u32>;
@group(0) @binding(6) var<storage, read> mlp_weights: array<MLPLayer>;
@group(1) @binding(0) var<uniform> sim_time: u32; // Essential for SDE noise

const DT: f32 = 0.01;
const NUM_STATES: u32 = 4u;
const RT_F: f32 = 26.7; // 37°C
// Assumed static extracellular ocean for Na and Cl (since diffuse_pass only updates K via .r)
const NA_EXT: f32 = 145.0;
const CL_EXT: f32 = 110.0;

@compute @workgroup_size(64)
fn physics_pass(@builtin(global_invocation_id) id: vec3<u32>) {
    let my_idx = id.x;
    if (my_idx >= arrayLength(&cells)) { return; }

    var me = cells[my_idx];
    if (me.state_id == 99u) { return; } // Necrosis

    let param_idx = (me.granular_id * NUM_STATES) + me.state_id;
    let params = sde_params[param_idx];
    let grid_coord = vec3<i32>(me.pos);

    // Read the dynamic K+ from your diffuse_pass shader
    let k_out = textureLoad(env_grid, grid_coord, 0).r;

    // ==========================================
    // 1. Dynamic Topology (Gap Junctions)
    // ==========================================
    var gap_current: f32 = 0.0;

    if (me.broad_id != 5u) { // Assuming 5 is Immune/Fluid
        let my_connexin = max(0.0, me.rna_spliced[params.idx_connexin]);

        for (var i = 0u; i < me.neighbor_count; i++) {
            let edge = edges[me.edge_start + i];
            // RL override can physically sever the connection
            if (edge.rl_edge_clamp < 0.0) { continue; }

            let neighbor = cells[edge.target_idx];
            let neighbor_connexin = max(0.0, neighbor.rna_spliced[params.idx_connexin]);

            // Gap junctions require BOTH cells to express connexins
            let dynamic_conductance = edge.conductance * (my_connexin * neighbor_connexin);

            // I = g * dV
            gap_current += dynamic_conductance * (neighbor.v_mem - me.v_mem);
        }
    }

    // ==========================================
    // 2. The Neural Delta Coupling
    // ==========================================
    let mlp_input = vec3<f32>(me.v_mem, k_out, gap_current);
    let weights = mlp_weights[param_idx];

    let hidden_layer = tanh((weights.w1 * mlp_input) + weights.b1);
    let gating_residual = tanh((weights.w2 * hidden_layer) + weights.b2);

    // ==========================================
    // 3. Imputed Hardware & GHK Thermodynamics
    // ==========================================
    let rna_k = max(0.0, me.rna_spliced[params.idx_k]);
    let rna_na = max(0.0, me.rna_spliced[params.idx_na]);
    let rna_cl = max(0.0, me.rna_spliced[params.idx_cl]);
    let rna_pump = max(0.0, me.rna_spliced[params.idx_pump]);

    let p_k = max(0.0, (params.base_p_k * rna_k) + gating_residual.x);
    let p_na = max(0.0, (params.base_p_na * rna_na) + gating_residual.y);
    let p_cl = max(0.0, (params.base_p_cl * rna_cl) + gating_residual.z);

    let ghk_num = (p_k * k_out) + (p_na * NA_EXT) + (p_cl * me.ion_cl);
    let ghk_den = (p_k * me.ion_k) + (p_na * me.ion_na) + (p_cl * CL_EXT);

    let safe_ratio = max(ghk_num / max(ghk_den, 0.000001), 0.000001);
    let target_v_mem = RT_F * log(safe_ratio);
    let old_v_mem = me.v_mem;

    let i_leak = params.g_leak * (target_v_mem - me.v_mem);
    let active_pump_current = params.pump_current * rna_pump * me.pump_health_multiplier;

    let net_current = i_leak + gap_current - active_pump_current;

    // dV/dt = I / C
    me.v_mem += (net_current / max(params.capacitance, 0.1)) * DT;

    if (me.exogenous_v_clamp != 0.0) {
        me.v_mem = me.exogenous_v_clamp;
    }

    // ==========================================
    // 4. Intracellular Ion Flux
    // ==========================================
    let flux_scalar = DT / max(me.area, 0.1);
    let k_efflux = i_leak * flux_scalar * (p_k / max(p_k + p_na + p_cl, 0.001));

    let pump_k_in = active_pump_current * 2.0 * flux_scalar;
    let pump_na_out = active_pump_current * 3.0 * flux_scalar;

    me.ion_k = max(1.0, me.ion_k - k_efflux + pump_k_in);
    me.ion_na = max(1.0, me.ion_na - pump_na_out);

    // ==========================================
    // 5. 128-Gene SDEvelo Array
    // ==========================================
    var max_rna_delta: f32 = 0.0;

    for (var i = 0u; i < 128u; i++) {
        let gene = params.genes[i];

        let stimulus = max(1e-6, me.v_mem - gene.v_resting);
        let hill_num = pow(stimulus, gene.hill_n);
        let hill_den = pow(gene.k_half, gene.hill_n) + hill_num;
        let c_effective = gene.c + (gene.v_max * (hill_num / hill_den));

        // Pseudo-random Gaussian generation requires sim_time
        let seed_u = my_idx * 256u + i + sim_time;
        let seed_s = my_idx * 256u + i + 128u + sim_time;

        let noise_u = generate_gaussian_noise(seed_u) * gene.sigma1;
        let noise_s = generate_gaussian_noise(seed_s) * gene.sigma2;

        let du = (c_effective - gene.beta * me.rna_unspliced[i]) * DT + noise_u;
        let ds = (gene.beta * me.rna_unspliced[i] - gene.gamma * me.rna_spliced[i]) * DT + noise_s;

        let old_s = me.rna_spliced[i];

        me.rna_unspliced[i] = clamp(me.rna_unspliced[i] + du, 0.0, gene.max_u);
        me.rna_spliced[i] = clamp(me.rna_spliced[i] + ds, 0.0, gene.max_s);

        let delta_s = abs(me.rna_spliced[i] - old_s);
        if (delta_s > max_rna_delta) {
            max_rna_delta = delta_s;
        }
    }

    // ==========================================
    // 6. Grid Handoff & Convergence
    // ==========================================
    // Send the extracellular K+ changes back to your diffusion shader
    let pumped_out = i32((me.v_mem - old_v_mem) * 1000.0);
    let hash_idx = get_grid_hash(vec3<u32>(grid_coord));
    atomicAdd(&grid_deltas[hash_idx], pumped_out);

    if (abs(me.v_mem - old_v_mem) < 0.0001 && max_rna_delta < 0.0001) {
        atomicAdd(&convergence_counter, 1u);
    }

    cells[my_idx] = me;
}

// Pseudo-random helper (Assuming you have this defined)
fn generate_gaussian_noise(seed: u32) -> f32 {
    // Standard PCG hash to Box-Muller transform
    var state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    let u1 = f32((word >> 22u) ^ word) / 4294967295.0;

    state = (seed + 1u) * 747796405u + 2891336453u;
    let word2 = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    let u2 = f32((word2 >> 22u) ^ word2) / 4294967295.0;

    return sqrt(-2.0 * log(max(u1, 1e-7))) * cos(6.2831853 * u2);
}