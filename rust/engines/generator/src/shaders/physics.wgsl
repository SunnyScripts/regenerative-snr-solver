const BROAD_RADII: array<f32, 5> = array<f32, 5>(1.3, 1.1, 0.8, 0.4, 1.0);

//struct GridParams { grid_dim: vec3<u32>, cell_size: f32 }

struct CellNode {
    pos: vec3<f32>, broad_id: u32,
    polarity: vec3<f32>, granular_id: u32,
    area: f32, phase_buffer_1: f32, phase_buffer_2: f32, phase_buffer_3: f32,
    v_mem: f32, ion_k: f32, ion_na: f32, ligand_pool: f32,
    receptor_pool: f32, sde_hidden_1: f32, sde_hidden_2: f32, neighbor_count: u32,
    connected_neighbors: array<u32, 16>,
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

struct SimParams {
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
    tile_size: f32,
    simulation_step: u32,
    current_cell_count: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> cells: array<CellNode>;
@group(0) @binding(1) var<storage, read> edges: array<MembraneInterface>;
@group(0) @binding(2) var<storage, read> active_emitters: array<MorphogenEmitter>;
@group(0) @binding(3) var<storage, read> emitter_count: atomic<u32>;

// Rulebook
@group(0) @binding(4) var<storage, read> ideal_depths: array<f32>;
@group(0) @binding(5) var<storage, read> strat_weights: array<f32>;

// Spatial Hash & Params
@group(0) @binding(6) var<uniform> params: SimParams;
@group(0) @binding(7) var<storage, read> grid_offsets: array<u32>;
@group(0) @binding(8) var<storage, read> sorted_cell_indices: array<u32>; // <--- NOW THIS MATCHES RUST!
@group(0) @binding(9) var<storage, read> global_cell_count: atomic<u32>;  // <--- THREAD BOUNDARY

fn get_grid_hash(coord: vec3<u32>) -> u32 {
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

@compute @workgroup_size(64)
fn physics_pass(@builtin(global_invocation_id) id: vec3<u32>) {
    let my_idx = id.x;

    let active_cells = atomicLoad(&global_cell_count);
    if (my_idx >= active_cells) { return; }

    var me = cells[my_idx];
    var net_displacement = vec3<f32>(0.0, 0.0, 0.0);
    // We use max(..., 1.0) as a safety clamp so uninitialized cells with 0 area don't cause NaN explosions
    let my_radius = max(sqrt(me.area / 3.14159), 1.0);

    // ==========================================
    // 1. ADHESION MECHANICS (Friends / Hooke's Law)
    // ==========================================
    for (var i = 0u; i < me.neighbor_count; i++) {
        let edge_idx = me.connected_neighbors[i];
        let edge = edges[edge_idx];
        if (edge.connection_strength < 0.1) { continue; }

        var neighbor_idx = edge.target_idx;
        if (neighbor_idx == my_idx) { neighbor_idx = edge.source_idx; }

        let neighbor = cells[neighbor_idx];
        let delta = neighbor.pos - me.pos;
        let current_dist = length(delta);

        if (current_dist < 0.0001) { continue; }

        // Use deterministic biology radii
        let neighbor_radius = BROAD_RADII[neighbor.broad_id];
        let resting_dist = my_radius + neighbor_radius;
        let deformation = current_dist - resting_dist;

        let spring_force = edge.adhesion * deformation;
        net_displacement += (normalize(delta) * spring_force);
    }

    // ==========================================
    // 2. COLLISION MECHANICS (Strangers / Hash Search)
    // ==========================================
//    let grid_coord = vec3<u32>(me.pos / params.tile_size);
    let ix = i32(floor(me.pos.x / params.tile_size));
    let iy = i32(floor(me.pos.y / params.tile_size));
    let iz = i32(floor(me.pos.z / params.tile_size));

    for (var x = -1i; x <= 1i; x++) {
        let nx = ix + x; if (nx < 0 || nx >= i32(params.grid_width)) { continue; }
        for (var y = -1i; y <= 1i; y++) {
            let ny = iy + y; if (ny < 0 || ny >= i32(params.grid_height)) { continue; }
            for (var z = -1i; z <= 1i; z++) {
                let nz = iz + z; if (nz < 0 || nz >= i32(params.grid_depth)) { continue; }

                // ---> CALL IT RIGHT HERE <---
                let neighbor_coord = vec3<u32>(u32(nx), u32(ny), u32(nz));
                let hash_idx = get_grid_hash(neighbor_coord);

                let end_idx = grid_offsets[hash_idx];
                var start_idx = 0u;
                if (hash_idx > 0u) { start_idx = grid_offsets[hash_idx - 1u]; }

                for (var i = start_idx; i < end_idx; i++)
                {
                    let neighbor_idx = sorted_cell_indices[i];
                    if (my_idx == neighbor_idx) { continue; }

                    let neighbor = cells[neighbor_idx];
                    let delta = me.pos - neighbor.pos;
                    let dist = length(delta);

                    // Dynamically calculate the neighbor's true biological radius
                    let neighbor_radius = max(sqrt(neighbor.area / 3.14159), 1.0);
                    let min_dist = my_radius + neighbor_radius;

                    // Aggressive Collision + Singularity Kick
                    if (dist > 0.0001 && dist < min_dist) {
                        // The Inverse Explosion Factor
                        // Example: If min_dist is 2.0 and dist is 0.1, the push multiplier is 19.0!
                        // As dist approaches 2.0, the multiplier drops to 0.0.
                        let explosion_factor = (min_dist / dist) - 1.0;
                        let push_force = explosion_factor * 15.0;
                        net_displacement += (normalize(delta) * push_force);

                    } else if (dist <= 0.0001) {
                        // The Golden Angle Scatter (Slightly boosted)
                        let golden_angle = f32(my_idx) * 2.39996;
                        let kick = vec3<f32>(cos(golden_angle), sin(golden_angle), 0.1);
                        net_displacement += kick * 20.0;
                    }
                }
            }
        }
    }

    // ==========================================
    // 3. BIOLOGICAL STRATIFICATION (Fixed Z-Attractor)
    // ==========================================
    let my_ideal_depth = ideal_depths[me.granular_id];
    let my_strat_force = strat_weights[me.granular_id];

    // Cells push towards their ideal Z-layer.
    // Positive difference pulls them up, negative pulls them down.
    let z_error = my_ideal_depth - me.pos.z;
    let stratification_velocity = vec3<f32>(0.0, 0.0, z_error * my_strat_force);
    net_displacement += stratification_velocity;


    // ==========================================
    // 4. ENVIRONMENTAL MECHANICS (ECM Repulsion)
    // ==========================================
    let num_emitters = atomicLoad(&emitter_count);
    for (var e = 0u; e < num_emitters; e++) {
        let em = active_emitters[e];

        if (em.emitter_type == 1u) { // Repulsion Field
            let delta = me.pos - em.pos;
            let dist = length(delta);
            let radius = em.params.x;

            if (dist < radius) {
                let push_dir = normalize(delta);
                let intensity = 1.0 - (dist / radius);
                net_displacement += (push_dir * (em.strength * intensity));
            }
        }
    }

    // ==========================================
    // 5. INTEGRATION (Explosive Relaxation)
    // ==========================================
    let dt = 0.1; // Doubled the time step for the RL resolution!
    let fluid_viscosity = 0.8;

    // STRESS JITTER: If the cell is trapped in a canceled-out core,
    // the massive overlapping forces will trigger violent vibration to break symmetry.
    let stress = length(net_displacement);
    if (stress > 10.0) {
        let seed = f32(my_idx + (params.simulation_step * 100u));
        let jitter_x = fract(sin(seed * 1.23) * 43758.54) * 2.0 - 1.0;
        let jitter_y = fract(sin(seed * 4.56) * 43758.54) * 2.0 - 1.0;
        let jitter_z = fract(sin(seed * 7.89) * 43758.54) * 2.0 - 1.0;
        net_displacement += vec3<f32>(jitter_x, jitter_y, jitter_z) * (stress * 0.1);
    }

    // THE SPEED LIMIT (Scaled dynamically to the cell's true size)
    // A cell is allowed to move a distance equal to half its own radius in a single frame
    let max_move = my_radius * 0.5;

    var final_move = net_displacement * dt * fluid_viscosity;
    if (length(final_move) > max_move) {
        final_move = normalize(final_move) * max_move;
    }

    me.pos += final_move;
    cells[my_idx] = me;
}