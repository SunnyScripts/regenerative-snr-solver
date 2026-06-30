struct MicrotomeConfig {
    blade_center: vec4<f32>,
    normal_thick: vec4<f32>,
}

struct CellNode {
    pos: vec3<f32>, broad_id: u32,
    polarity: vec3<f32>, granular_id: u32,
    area: f32, phase_buffer_1: f32, phase_buffer_2: f32, phase_buffer_3: f32,
    v_mem: f32, ion_k: f32, ion_na: f32, ligand_pool: f32,
    receptor_pool: f32, sde_hidden_1: f32, sde_hidden_2: f32, neighbor_count: u32,
    connected_neighbors: array<u32, 16>,
}

@group(0) @binding(0) var<storage, read> cells_3d: array<CellNode>;
@group(0) @binding(1) var<storage, read> global_cell_count: array<u32>; // Fixed type!

@group(0) @binding(2) var<storage, read_write> cells_2d_slice: array<CellNode>;
@group(0) @binding(3) var<storage, read_write> slice_cell_count: atomic<u32>;

@group(0) @binding(4) var<uniform> config: MicrotomeConfig;

@compute @workgroup_size(64)
fn microtome_pass(@builtin(global_invocation_id) id: vec3<u32>) {
    let my_idx = id.x;

    // FAST EXIT: Check against the actual live cell count, not the buffer capacity
    let live_cells = global_cell_count[0];
    if (my_idx >= live_cells) { return; }

    let cell = cells_3d[my_idx];

    let normal = config.normal_thick.xyz;
    let thickness = config.normal_thick.w;

    // 1. Vector from the blade's anchor point to the cell
    let point_vector = cell.pos - config.blade_center.xyz;

    // 2. Project the cell onto the normal axis
    let distance_to_plane = dot(point_vector, normal);

    // 3. The Blade Check
    let half_thick = thickness * 0.5;

    if (abs(distance_to_plane) <= half_thick) {

        // I SURVIVED! Claim a spot in the output buffer.
        let out_idx = atomicAdd(&slice_cell_count, 1u);
        if (out_idx >= arrayLength(&cells_2d_slice)) { return; }

        var out_cell = cell;

        // 4. Flatten the 3D coordinates into a 2D plane for the Observation Shader
        let projected_3d = cell.pos - (normal * distance_to_plane);

        var flat_x = 0.0;
        var flat_y = 0.0;

        if (abs(normal.z) > 0.5) {
            // Horizontal cut (En Face)
            flat_x = projected_3d.x; flat_y = projected_3d.y;
        } else if (abs(normal.x) > 0.5) {
            // Sagittal cut
            flat_x = projected_3d.y; flat_y = projected_3d.z;
        } else {
            // Vertical cut (H&E Standard)
            flat_x = projected_3d.x; flat_y = projected_3d.z;
        }

        out_cell.pos = vec3<f32>(flat_x, flat_y, 0.0);
        cells_2d_slice[out_idx] = out_cell;
    }
}