@group(0) @binding(0) var<storage, read> cells: array<CellNode>;
@group(0) @binding(1) var<storage, read> grid_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> sorted_cells: array<u32>;
// UPGRADE 1: Rename to env_mask and change format to rgba16float
@group(0) @binding(3) var env_mask: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: SimParams;

@compute @workgroup_size(8, 8, 8)
fn bake_environment_pass(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.grid_width || id.y >= params.grid_height || id.z >= params.grid_depth) { return; }

    let hash_idx = get_grid_hash(id);
    let end_idx = grid_offsets[hash_idx];

    var start_idx = 0u;
    if (hash_idx > 0u) {
        start_idx = grid_offsets[hash_idx - 1u];
    }

    // UPGRADE 2: Setup the three color channels for our fluid behaviors
    var is_clamp: f32 = 0.0;  // Red Channel (Blood Vessels)
    var is_drain: f32 = 0.0;  // Green Channel (Lymphatics)
    var is_source: f32 = 0.0; // Blue Channel (Glands)

    for (var i = start_idx; i < end_idx; i++) {
        let cell = cells[sorted_cells[i]];

        // ----------------------------------------------------
        // BEHAVIOR R (Clamp): Arterial, Venous, and Capillary Endothelial
        // Replace 22u, 23u, 24u with your actual integer mapping
        // ----------------------------------------------------
        if (cell.granular_id == 22u || cell.granular_id == 23u || cell.granular_id == 24u) {
            is_clamp = 1.0;
        }
        // ----------------------------------------------------
        // BEHAVIOR G (Drain): Lymphatic Endothelial Cell
        // Replace 25u with your actual integer mapping
        // ----------------------------------------------------
        else if (cell.granular_id == 25u) {
            is_drain = 1.0;
        }
        // ----------------------------------------------------
        // BEHAVIOR B (Source): Sweat Gland & Sebaceous Gland
        // Replace 6u, 8u with your actual integer mapping
        // ----------------------------------------------------
        else if (cell.granular_id == 6u || cell.granular_id == 8u) {
            is_source = 1.0;
        }

        // Optimization: If a bucket happens to have all three tissue types crammed
        // inside it, we can stop checking the rest of the cells in this bucket.
        if (is_clamp > 0.5 && is_drain > 0.5 && is_source > 0.5) {
            break;
        }
    }

    // UPGRADE 3: Write the 4-channel vector to the 3D texture
    textureStore(env_mask, vec3<i32>(id), vec4<f32>(is_clamp, is_drain, is_source, 0.0));
}