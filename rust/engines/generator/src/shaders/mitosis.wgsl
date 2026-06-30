// ==========================================
// 1. STRUCTS & MEMORY ALIGNMENT
// ==========================================
const NUM_TYPES: u32 = 24u;

struct CellNode {
    // Chunk 1
    pos: vec3<f32>,
    broad_id: u32,

    // Chunk 2
    polarity: vec3<f32>,
    granular_id: u32,

    // Chunk 3
    area: f32,
    phase_buffer_1: f32,
    phase_buffer_2: f32,
    phase_buffer_3: f32,

    // Chunk 4
    v_mem: f32,
    ion_k: f32,
    ion_na: f32,
    ligand_pool: f32,

    // Chunk 5
    receptor_pool: f32,
    sde_hidden_1: f32,
    sde_hidden_2: f32,
    neighbor_count: u32,

    // Chunk 6-9
    connected_neighbors: array<u32, 16>,
}

struct MembraneInterface {
    source_idx: u32,
    target_idx: u32,
    conductance: f32,
    adhesion: f32,
    distance: f32,
    connection_strength: f32,
    edge_state_1: f32,
    edge_state_2: f32,
}

// 80-byte aligned Emitter
struct MorphogenEmitter {
    // Chunk 1 (16 bytes)
    emitter_type: u32,
    pos: vec3<f32>,

    // Chunk 2 (16 bytes)
    type_ratios: vec4<f32>,

    // Chunk 3 (16 bytes)
    dir: vec3<f32>,
    strength: f32,

    // Chunk 4 (16 bytes)
    params: vec4<f32>,

    // Chunk 5 (16 bytes)
    decay_rate: f32,
    _pad: vec3<f32>, // Required padding to hit 80 bytes perfectly
}

// Interactome Binary Buffers
struct RuleOffset {
    start_idx: u32,
    count: u32,
    _pad1: u32, // Required for 16-byte uniform alignment
    _pad2: u32,
}

struct GpuInteractomeRule {
    target_granular_id: u32,
    total_drift: f32,
    total_diffusion: f32,
    _pad: u32,
}

struct SimParams
{
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
    tile_size: f32,
    simulation_step: u32,
    current_cell_count: u32,
    _pad: vec2<u32>,
}

// ==========================================
// 2. BINDINGS
// ==========================================
@group(0) @binding(0) var<storage, read> active_emitters: array<MorphogenEmitter>;
@group(0) @binding(1) var<storage, read> emitter_count: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> cells: array<CellNode>;
@group(0) @binding(3) var<storage, read_write> global_cell_count: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> edges: array<MembraneInterface>;
@group(0) @binding(5) var<storage, read_write> global_edge_count: atomic<u32>;

// NEW: The Pre-Compiled Biology Lookups
@group(0) @binding(6) var<uniform> interaction_offsets: array<RuleOffset, 16>;
@group(0) @binding(7) var<storage, read> aggregated_rules: array<GpuInteractomeRule>;
@group(0) @binding(8) var<uniform> params: SimParams;


// ==========================================
// 3. THE COMPUTE SHADER
// ==========================================
@compute @workgroup_size(64)
fn mitosis_and_identity_pass(@builtin(global_invocation_id) id: vec3<u32>) {
    let mother_idx = id.x;
    // Threads ONLY execute if they existed at the start of this frame!
    if (mother_idx >= params.current_cell_count) { return; }

    var me = cells[mother_idx];
    var local_growth_factor = 0.0;

    var closest_emitter: MorphogenEmitter;
    var min_dist: f32 = 999999.0;
    var field_found: bool = false;

    let num_emitters = atomicLoad(&emitter_count);

    // ---------------------------------------------------------
    // PART A: SMELL THE ECM (Morphogens)
    // ---------------------------------------------------------
    for (var i = 0u; i < num_emitters; i++) {
        let em = active_emitters[i];

        if (em.emitter_type == 0u || em.emitter_type == 3u) {
            let pa = me.pos - em.pos;
            let ba = em.dir * em.params.x;
            let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
            let dist = length(pa - ba * h);

            if (dist < em.params.y) {
                let intensity = 1.0 - (dist / em.params.y);
                local_growth_factor += (em.strength * intensity);

                if (dist < min_dist) {
                    min_dist = dist;
                    closest_emitter = em;
                    field_found = true;
                }
            }
        }
    }

    // ---------------------------------------------------------
    // PART B: FRENCH FLAG MODEL (Broad Identity)
    // ---------------------------------------------------------
    var my_broad = me.broad_id;

    if (field_found) {
        let normalized_dist = min_dist / closest_emitter.params.y;
        let t0 = closest_emitter.type_ratios.x;
        let t1 = t0 + closest_emitter.type_ratios.y;
        let t2 = t1 + closest_emitter.type_ratios.z;

        if (normalized_dist <= t0)      { my_broad = 0u; }
        else if (normalized_dist <= t1) { my_broad = 1u; }
        else if (normalized_dist <= t2) { my_broad = 2u; }
        else                            { my_broad = 3u; }
    }

    // ---------------------------------------------------------
    // PART C: SDEvelo MRF (Granular Identity)
    // ---------------------------------------------------------
    var granular_energies = array<f32, NUM_TYPES>();
    var granular_noise = array<f32, NUM_TYPES>();

    // 1. Tally the Gene Expression Math using existing neighbors
    for (var i = 0u; i < me.neighbor_count; i++) {
        let neighbor_idx = me.connected_neighbors[i];
        let neighbor = cells[neighbor_idx];
        let neighbor_broad = neighbor.broad_id;

        // Lookup the Broad-to-Broad pair offset (0 to 15)
        let offset_idx = (my_broad * 4u) + neighbor_broad;
        let rules_meta = interaction_offsets[offset_idx];

        let dist = length(me.pos - neighbor.pos);
        let signal = clamp(1.0 - (dist / 10.0), 0.0, 1.0); // Assume max signal rad is 10.0

        for (var r = 0u; r < rules_meta.count; r++) {
            let rule = aggregated_rules[rules_meta.start_idx + r];
            granular_energies[rule.target_granular_id] += (rule.total_drift * signal);
            granular_noise[rule.target_granular_id] += (rule.total_diffusion * signal);
        }
    }

    // 2. The Gumbel-Softmax Flip
    var best_score = -99999.0;
    var winning_granular = (me.granular_id); // Default to current ID if isolated
    let base_seed = mother_idx ^ bitcast<u32>(me.pos.x) ^ params.simulation_step;

    for (var i = 0u; i < NUM_TYPES; i++) {
        let E = granular_energies[i];
        if (E <= 0.0) { continue; } // Skip types not supported by local Ligands

        let T = max(granular_noise[i] / 2.0, 0.001); // Prevent div-by-zero

        let U = pcg_hash_f32(base_seed ^ i);
        let gumbel = -log(-log(clamp(U, 0.0001, 0.9999)));

        let logit = (E + gumbel) / T;

        if (logit > best_score) {
            best_score = logit;
            winning_granular = i;
        }
    }

    // 3. Save the Identity back to the mother cell
    // 3. Update the Identity (but DON'T write to VRAM just yet!)
        me.broad_id = my_broad;
        me.granular_id = winning_granular;

        // BOOTSTRAP: If this is an uninitialized zygote, give it a default Xenium area
        if (me.area < 1.0) {
            me.area = 314.159; // Roughly a 10-micron radius starting size
        }

        // ---------------------------------------------------------
        // PART D: POLARIZED MITOSIS & GROWTH
        // ---------------------------------------------------------
        let random_jitter = pcg_hash_f32(base_seed ^ 999u);
        let division_threshold = 0.5;

        // BIOLOGY: Cells grow in volume while soaking in the morphogen field
        if (field_found) {
            me.area += local_growth_factor * 5.0;
        }

        if (local_growth_factor > division_threshold && random_jitter > 0.2 && field_found)
        {

            let daughter_idx = atomicAdd(&global_cell_count, 1u);
            if (daughter_idx >= arrayLength(&cells)) { return; }

            // 1. CONSERVATION OF MASS: Split the area exactly in half!
            me.area = me.area * 0.5;

            // 2. Daughter inherits the fully updated Identity and halved Area
            var daughter = me;

            // 3. Polarized Geometry
            var division_vector: vec3<f32>;
            if (closest_emitter.emitter_type == 3u) {
                division_vector = me.polarity;
            } else {
                let tangent = vec3<f32>(cos(random_jitter * 6.28), sin(random_jitter * 6.28), 0.0);
                division_vector = normalize(cross(me.polarity, tangent));
            }

            // Push both the daughter AND the mother slightly apart to aid physics
            daughter.pos += division_vector * 0.5;
            me.pos -= division_vector * 0.5;

            cells[daughter_idx] = daughter;

            // 4. Mother-Daughter Tight Junction
            let edge_idx = atomicAdd(&global_edge_count, 1u);
            if (edge_idx >= arrayLength(&edges)) { return; }

            var new_edge: MembraneInterface;
            new_edge.source_idx = mother_idx;
            new_edge.target_idx = daughter_idx;
            new_edge.conductance = 1.0;
            new_edge.adhesion = 1.0;
            new_edge.distance = 1.0;
            new_edge.connection_strength = 1.0;

            edges[edge_idx] = new_edge;
        }

        // FINAL WRITE-BACK: Save the mother cell to VRAM (including her new size/position!)
        cells[mother_idx] = me;
}

fn pcg_hash_f32(seed: u32) -> f32 {
    var state = seed * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    let result = (word >> 22u) ^ word;
    return f32(result) / 4294967295.0;
}