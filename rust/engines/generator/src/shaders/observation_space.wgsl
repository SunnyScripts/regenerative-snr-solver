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

struct SimParams {
    grid_width: u32,
    grid_height: u32,
    tile_size: f32,
    epoch: u32,
}

@group(0) @binding(0) var<storage, read> cells: array<CellNode>;
@group(0) @binding(1) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> tile_counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> obs_buffer: array<f32>;
@group(0) @binding(4) var<uniform> params: SimParams;

// UPDATED: Now expects 185 floats to accommodate the final cell count
const OBS_DIM: u32 = 185u;
const PI: f32 = 3.14159265359;

fn get_tile_index(tx: i32, ty: i32) -> u32 {
    if (tx < 0 || ty < 0 || u32(tx) >= params.grid_width || u32(ty) >= params.grid_height) {
        return 4294967295u;
    }
    return u32(tx) + (u32(ty) * params.grid_width);
}

@compute @workgroup_size(64)
fn generate_micro_tensor(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let center_tile_idx = global_id.x;
    if (center_tile_idx >= (params.grid_width * params.grid_height)) { return; }

    let ctx = i32(center_tile_idx % params.grid_width);
    let cty = i32(center_tile_idx / params.grid_width);

    let tile_min_x = f32(ctx) * params.tile_size;
    let tile_min_y = f32(cty) * params.tile_size;
    let tile_max_x = tile_min_x + params.tile_size;
    let tile_max_y = tile_min_y + params.tile_size;

    var ripley = array<f32, 20>();
    var cumulative_counts = array<f32, 20>();
    var enrichment = array<f32, 36>();
    var halo = array<f32, 128>();

    // --- PASS 1: THE HALO HEIGHT MAP ---
    for (var y: i32 = -1; y <= 1; y++) {
        for (var x: i32 = -1; x <= 1; x++) {
            if (x == 0 && y == 0) { continue; }

            let neighbor_idx = get_tile_index(ctx + x, cty + y);
            if (neighbor_idx == 4294967295u) { continue; }

            let start = tile_offsets[neighbor_idx];
            let count = tile_counts[neighbor_idx];

            for (var i: u32 = 0u; i < count; i++) {
                let cell = cells[start + i];
                // UPDATED: Direct struct access, no bitwise unpacking
                let mech_id = cell.broad_id;
//                if (mech_id == 0u) { continue; }

                var edge_offset = 0u;
                var slot_ratio = 0.0;

                // UPDATED: Using cell.pos.x and cell.pos.y
                if (cell.pos.y < tile_min_y) {
                    edge_offset = 0u;
                    slot_ratio = clamp((cell.pos.x - tile_min_x) / params.tile_size, 0.0, 0.999);
                } else if (cell.pos.x > tile_max_x) {
                    edge_offset = 32u;
                    slot_ratio = clamp((cell.pos.y - tile_min_y) / params.tile_size, 0.0, 0.999);
                } else if (cell.pos.y > tile_max_y) {
                    edge_offset = 64u;
                    slot_ratio = clamp((cell.pos.x - tile_min_x) / params.tile_size, 0.0, 0.999);
                } else if (cell.pos.x < tile_min_x) {
                    edge_offset = 96u;
                    slot_ratio = clamp((cell.pos.y - tile_min_y) / params.tile_size, 0.0, 0.999);
                }

                let local_slot = u32(slot_ratio * 32.0);
                halo[edge_offset + local_slot] += 1.0;
            }
        }
    }

    // --- PASS 2: RIPLEY & ENRICHMENT (O(N^2) Pairwise) ---
    let center_start = tile_offsets[center_tile_idx];
    let center_count = tile_counts[center_tile_idx];
    var valid_cells: u32 = center_count; // All cells in the count are valid

    for (var i: u32 = 0u; i < center_count; i++)
    {
        let cell_A = cells[center_start + i];
        let type_A = min(cell_A.broad_id, 5u);

        for (var j: u32 = 0u; j < center_count; j++)
        {
            if (i == j) { continue; }

            let cell_B = cells[center_start + j];
            let type_B = min(cell_B.broad_id, 5u);

            let dx = cell_A.pos.x - cell_B.pos.x;
            let dy = cell_A.pos.y - cell_B.pos.y;
            let dist = sqrt((dx * dx) + (dy * dy));

            // Ripley's K (Cumulative up to 30um)
            if (dist <= 30.0) {
                let bin_idx = min(u32(dist / 1.5), 19u);
                for (var b: u32 = bin_idx; b < 20u; b++) {
                    cumulative_counts[b] += 1.0;
                }
            }

            // Dynamic Interaction Threshold based on physical area
            let r_A = sqrt(max(cell_A.area, 1.0) / PI);
            let r_B = sqrt(max(cell_B.area, 1.0) / PI);
            let interaction_threshold = r_A + r_B + 2.0;

            // Dynamic Enrichment Matrix check
            if (i < j && dist < interaction_threshold)
            {
                let matrix_idx_AB = (type_A * 6u) + type_B;
                let matrix_idx_BA = (type_B * 6u) + type_A;

                enrichment[matrix_idx_AB] += 1.0;
                enrichment[matrix_idx_BA] += 1.0;
            }
        }
    }

    // --- PASS 3: BESAG'S L-VARIANT TRANSFORMATION ---
    let n = f32(valid_cells);
    var density_factor = 0.0;

    if (n > 1.0) {
        density_factor = (params.tile_size * params.tile_size) / (n * (n - 1.0));
    }

    for (var b: u32 = 0u; b < 20u; b++) {
        let r = f32(b + 1u) * 1.5;
        let k_r = cumulative_counts[b] * density_factor;
        ripley[b] = sqrt(max(k_r, 0.0) / PI) - r;
    }

    // --- WRITE OUT THE 185-FLOAT TENSOR ---
    let buffer_start = center_tile_idx * OBS_DIM;

    for (var i: u32 = 0u; i < 20u; i++) { obs_buffer[buffer_start + i] = ripley[i]; }
    for (var i: u32 = 0u; i < 36u; i++) { obs_buffer[buffer_start + 20u + i] = enrichment[i]; }
    for (var i: u32 = 0u; i < 128u; i++) { obs_buffer[buffer_start + 56u + i] = halo[i]; }

    // NEW: Write the 185th float!
    obs_buffer[buffer_start + 184u] = f32(valid_cells);
}