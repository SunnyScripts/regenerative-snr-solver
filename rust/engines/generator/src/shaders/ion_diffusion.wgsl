@group(0) @binding(0) var grid_in: texture_3d<f32>;
@group(0) @binding(1) var grid_out: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var vessel_mask: texture_3d<rgba16float>; // Baked at Frame 0

const DIFFUSION_RATE: f32 = 0.15;
const BASELINE_K: f32 = 4.0;
const LYMPH_DRAIN_RATE: f32 = 0.05; // Drains 5% of excess per tick
const SWEAT_SOURCE_RATE: f32 = 0.1; // Secretes fixed amount per tick

@compute @workgroup_size(8, 8, 8)
fn diffuse_pass(@builtin(global_invocation_id) id: vec3<u32>) {
    let grid_dim = textureDimensions(grid_in);
    if (id.x >= grid_dim.x || id.y >= grid_dim.y || id.z >= grid_dim.z) { return; }

    let coord = vec3<i32>(id);
    let mask = textureLoad(vessel_mask, coord, 0);

    // ==========================================
    // 1. BEHAVIOR R: Venous/Arterial Clamp
    // ==========================================
    if (mask.r > 0.5) {
        // Blood flow dictates the environment. Absolute clamp.
        textureStore(grid_out, coord, vec4<f32>(BASELINE_K, 0.0, 0.0, 0.0));
        return;
    }

    var current_val = textureLoad(grid_in, coord, 0).r;

    // ==========================================
    // 2. BEHAVIOR G: Lymphatic One-Way Drain
    // ==========================================
    if (mask.g > 0.5) {
        if (current_val > BASELINE_K) {
            let excess = current_val - BASELINE_K;
            current_val -= excess * LYMPH_DRAIN_RATE;
        }
    }

    // ==========================================
    // 3. BEHAVIOR B: Sweat Gland Active Source
    // ==========================================
    if (mask.b > 0.5) {
        // Actively pumping ions into the interstitial fluid
        current_val += SWEAT_SOURCE_RATE;
    }

    // ==========================================
    // 4. Eulerian Von Neumann Diffusion
    // ==========================================
    var sum_neighbors: f32 = 0.0;
    var valid_neighbors: f32 = 0.0;

    let offsets = array<vec3<i32>, 6>(
        vec3<i32>( 1,  0,  0), vec3<i32>(-1,  0,  0),
        vec3<i32>( 0,  1,  0), vec3<i32>( 0, -1,  0),
        vec3<i32>( 0,  0,  1), vec3<i32>( 0,  0, -1)
    );

    for (var i = 0; i < 6; i++) {
        let n_coord = coord + offsets[i];
        if (n_coord.x >= 0 && n_coord.x < i32(grid_dim.x) &&
            n_coord.y >= 0 && n_coord.y < i32(grid_dim.y) &&
            n_coord.z >= 0 && n_coord.z < i32(grid_dim.z)) {

            sum_neighbors += textureLoad(grid_in, n_coord, 0).r;
            valid_neighbors += 1.0;
        }
    }

    let laplacian = sum_neighbors - (valid_neighbors * current_val);
    let new_val = current_val + (DIFFUSION_RATE * laplacian);

    textureStore(grid_out, coord, vec4<f32>(max(new_val, 0.0), 0.0, 0.0, 0.0));
}