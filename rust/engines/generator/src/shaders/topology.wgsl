const NUM_TYPES: u32 = 24u;

// 0=Epi, 1=Endo, 2=Mes, 3=Immune, 4=Other
const BROAD_RADII: array<f32, 5> = array<f32, 5>(1.3, 1.1, 0.8, 0.4, 1.0);

//struct GridParams {
//    grid_dim: vec3<u32>,
//    cell_size: f32,
//}

struct SimParams {
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
    tile_size: f32,
    simulation_step: u32,
    current_cell_count: u32,
    _pad: vec2<u32>,
}

struct CellNode {
    // Chunk 1
    pos: vec3<f32>,
    broad_id: u32,

    // Chunk 2
    polarity: vec3<f32>,
    granular_id: u32,

    // Chunk 3
    area: f32, // Maintained for struct alignment padding
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
    matrix_stiffness: f32,
    rl_edge_clamp: f32,
}

@group(0) @binding(0) var<storage, read_write> cells: array<CellNode>;
@group(0) @binding(1) var<storage, read_write> edges: array<MembraneInterface>;
@group(0) @binding(2) var<storage, read_write> global_edge_count: atomic<u32>;

// Spatial Hash
@group(0) @binding(3) var<uniform> params: SimParams;
@group(0) @binding(4) var<storage, read> grid_offsets: array<u32>;
@group(0) @binding(5) var<storage, read> sorted_cell_indices: array<u32>;

// RULEBOOK: The compiled matrices
@group(0) @binding(6) var<storage, read> adhesion_matrix: array<f32>;
@group(0) @binding(7) var<storage, read> conductance_matrix: array<f32>;
@group(0) @binding(8) var<storage, read> global_cell_count: atomic<u32>;

fn get_grid_hash(coord: vec3<u32>) -> u32
{
    let x = clamp(coord.x, 0u, params.grid_width - 1u);
    let y = clamp(coord.y, 0u, params.grid_height - 1u);
    let z = clamp(coord.z, 0u, params.grid_depth - 1u);

    return x + (y * params.grid_width) + (z * params.grid_width * params.grid_height);
}

//fn get_bucket_idx(pos: vec3<f32>) -> u32 {
//    let grid_pos = vec2<u32>(
//        u32(max(pos.x / params.tile_size, 0.0)),
//        u32(max(pos.y / params.tile_size, 0.0))
//    );
//
//    // Clamp to boundaries
//    let clamped_x = clamp(grid_pos.x, 0u, params.grid_width - 1u);
//    let clamped_y = clamp(grid_pos.y, 0u, params.grid_height - 1u);
//
//    return clamped_x + (clamped_y * params.grid_width);
//}

fn edge_exists(me: CellNode, neighbor_id: u32) -> bool {
    for (var i = 0u; i < me.neighbor_count; i++) {
        let edge_idx = me.connected_neighbors[i];
        let e = edges[edge_idx];

        if (e.source_idx == neighbor_id || e.target_idx == neighbor_id) {
            if (e.connection_strength > 0.1) {
                return true;
            }
        }
    }
    return false;
}

@compute @workgroup_size(64)
fn topology_pass(@builtin(global_invocation_id) id: vec3<u32>) {
    let my_idx = id.x;

    // Instantly kill dead threads!
    let active_cells = atomicLoad(&global_cell_count);
    if (my_idx >= active_cells) { return; }

    var me = cells[my_idx];

    // Dynamic morphology sizing
    let my_radius = BROAD_RADII[me.broad_id];

    // ==========================================
    // RULE A: FORGE NEW EDGES (Cadherin Bind)
    // ==========================================
//    let grid_coord = vec3<u32>(me.pos / params.tile_size);
    let ix = i32(floor(me.pos.x / params.tile_size));
    let iy = i32(floor(me.pos.y / params.tile_size));
    let iz = i32(floor(me.pos.z / params.tile_size));

    for (var x = -1i; x <= 1i; x++)
    {
        let nx = ix + x;
        if (nx < 0 || nx >= i32(params.grid_width)) { continue; }

        for (var y = -1i; y <= 1i; y++)
        {
            let ny = iy + y;
            if (ny < 0 || ny >= i32(params.grid_height)) { continue; }

            for (var z = -1i; z <= 1i; z++)
            {
                let nz = iz + z;
                if (nz < 0 || nz >= i32(params.grid_depth)) { continue; }

                let neighbor_coord = vec3<u32>(u32(nx), u32(ny), u32(nz));
                let hash_idx = get_grid_hash(neighbor_coord);

                // THE FIXED SPATIAL LOOKUP
                let end_idx = grid_offsets[hash_idx];
                var start_idx = 0u;
                if (hash_idx > 0u) {
                    start_idx = grid_offsets[hash_idx - 1u];
                }

                for (var i = start_idx; i < end_idx; i++)
                {
                    let neighbor_idx = sorted_cell_indices[i];

                    if (my_idx >= neighbor_idx) { continue; }

                    let neighbor = cells[neighbor_idx];
                    let dist = distance(me.pos, neighbor.pos);

                    let neighbor_radius = BROAD_RADII[neighbor.broad_id];
                    let search_radius = (my_radius + neighbor_radius) * 1.5;

                    if (dist < search_radius) {
                        if (!edge_exists(me, neighbor_idx)) {

                            let insert_idx = atomicAdd(&global_edge_count, 1u);
                            if (insert_idx >= arrayLength(&edges)) { return; }

                            let matrix_idx = (me.granular_id * NUM_TYPES) + neighbor.granular_id;

                            var new_edge: MembraneInterface;
                            new_edge.source_idx = my_idx;
                            new_edge.target_idx = neighbor_idx;
                            new_edge.adhesion = adhesion_matrix[matrix_idx];
                            new_edge.conductance = conductance_matrix[matrix_idx];
                            new_edge.distance = dist;
                            new_edge.connection_strength = 1.0;
                            new_edge.matrix_stiffness = 0.0;
                            new_edge.rl_edge_clamp = 0.0;

                            edges[insert_idx] = new_edge;

                            if (me.neighbor_count < 16u) {
                                me.connected_neighbors[me.neighbor_count] = insert_idx;
                                me.neighbor_count++;
                            }
                        }
                    }
                }
            }
        }
    }

    // ==========================================
    // RULE B: TEAR OVERSTRETCHED EDGES
    // ==========================================
    for (var e = 0u; e < me.neighbor_count; e++) {
        let edge_idx = me.connected_neighbors[e];
        var edge = edges[edge_idx];

        if (edge.connection_strength < 0.1) { continue; }

        var target_idx = edge.target_idx;
        if (target_idx == my_idx) { target_idx = edge.source_idx; }

        let target_cell = cells[target_idx];
        let current_dist = distance(me.pos, target_cell.pos);

        let target_radius = BROAD_RADII[target_cell.broad_id];
        let yield_point = (my_radius + target_radius) * 2.5;

        if (current_dist > yield_point) {
            edge.connection_strength = 0.0;
            edge.adhesion = 0.0;
            edges[edge_idx] = edge;
        } else {
            edge.distance = current_dist;
            edges[edge_idx] = edge;
        }
    }

    cells[my_idx] = me;
}