struct CellNode {
    // Chunk 1 (16 Bytes)
    pos: vec3<f32>,
    broad_id: u32,

    // Chunk 2 (16 Bytes)
    polarity: vec3<f32>,
    granular_id: u32,

    // Chunk 3 (16 Bytes)
    state_id: u32,
    area: f32,
    ion_ca: f32,
    ion_cl: f32,

    // Chunk 4 (16 Bytes)
    v_mem: f32,
    ion_k: f32,
    ion_na: f32,
    exogenous_v_clamp: f32,

    // Chunk 5 (16 Bytes)
    ligand_pool: f32,
    receptor_pool: f32,
    pump_health_multiplier: f32,
    adhesion_multiplier: f32,

    // Chunk 6 (16 Bytes) - CSR Pointers
    neighbor_count: u32,
    edge_start: u32,
    _padding1: u32,
    _padding2: u32,

    // Chunks 7-14 (128 Bytes)
    rna_unspliced: array<f32, 32>,

    // Chunks 15-22 (128 Bytes)
    rna_spliced: array<f32, 32>,
}

struct SimParams {
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
    tile_size: f32,
    simulation_step: u32,
    current_cell_count: u32,
    _pad: vec2<u32>,
}

//struct GridParams {
//    grid_dim: vec3<u32>, // e.g., 64x64x64 buckets
//    bucket_size: f32,    // e.g., 5.0 microns per bucket
//}

@group(0) @binding(0) var<storage, read> cells: array<CellNode>;
@group(0) @binding(1) var<storage, read> global_cell_count: atomic<u32>;
@group(0) @binding(2) var<uniform> params: SimParams; // In Hash shader (Binding 3)
@group(0) @binding(3) var<storage, read_write> grid_offsets: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> sorted_cells: array<u32>;

// Helper: Convert a 3D position into a 1D Bucket Index
fn get_bucket_idx(pos: vec3<f32>) -> u32 {
    let grid_pos = vec3<u32>(
        u32(max(pos.x / params.tile_size, 0.0)),
        u32(max(pos.y / params.tile_size, 0.0)),
        u32(max(pos.z / params.tile_size, 0.0))
    );
    // Call the grid hash!
    return get_grid_hash(grid_pos);
}

fn get_grid_hash(coord: vec3<u32>) -> u32 {
    let x = clamp(coord.x, 0u, params.grid_width - 1u);
    let y = clamp(coord.y, 0u, params.grid_height - 1u);
    let z = clamp(coord.z, 0u, params.grid_depth - 1u);

    return x + (y * params.grid_width) + (z * params.grid_width * params.grid_height);
}

// ==========================================
// PASS 1: CLEAR THE GRID
// ==========================================
@compute @workgroup_size(64)
fn clear_pass(@builtin(global_invocation_id) id: vec3<u32>) {
    // REPLACED grid_dim accessors with SimParams fields
    let total_buckets = params.grid_width * params.grid_height * params.grid_depth;
    if (id.x < total_buckets) {
        atomicStore(&grid_offsets[id.x], 0u);
    }
}

// ==========================================
// PASS 2: COUNT CELLS PER BUCKET
// ==========================================
@compute @workgroup_size(64)
fn count_pass(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell_count = atomicLoad(&global_cell_count);
    if (id.x >= cell_count) { return; }

    // ---> CALL IT HERE <---
    let bucket_idx = get_bucket_idx(cells[id.x].pos);
    atomicAdd(&grid_offsets[bucket_idx], 1u);
}

// ==========================================
// PASS 3: PREFIX SUM (Exclusive Scan)
// Note: In production with millions of buckets, this is usually
// done via a parallel block-scan. For simplicity, if your grid is
// small enough, a single workgroup can scan it, or you can do it on the CPU.
// ==========================================

//ToDo Since you are already pulling data back to Rust for the CSR edge prefix sum during the Slow Clock, you might eventually find it drastically faster to also pull the grid_offsets buffer to Rust, run iter().scan() on the CPU, and push it back, replacing Pass 3 entirely. But for now, your WGSL logic is sound and will execute as written.
@compute @workgroup_size(1)
fn scan_pass()
{
    // REPLACED grid_dim accessors with SimParams fields
    let total_buckets = params.grid_width * params.grid_height * params.grid_depth;
    var sum = 0u;
    for (var i = 0u; i < total_buckets; i++) {
        let count = atomicLoad(&grid_offsets[i]);
        atomicStore(&grid_offsets[i], sum);
        sum += count;
    }
}

// ==========================================
// PASS 4: INSERT CELLS INTO SORTED ARRAY
// ==========================================
@compute @workgroup_size(64)
fn insert_pass(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell_count = atomicLoad(&global_cell_count);
    if (id.x >= cell_count) { return; }

    // ---> CALL IT HERE <---
    let bucket_idx = get_bucket_idx(cells[id.x].pos);

    let sorted_idx = atomicAdd(&grid_offsets[bucket_idx], 1u);
    sorted_cells[sorted_idx] = id.x;
}