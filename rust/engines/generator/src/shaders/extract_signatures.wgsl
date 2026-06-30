const PI: f32 = 3.14159265359;

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
    pos: vec3<f32>, broad_id: u32,
    polarity: vec3<f32>, granular_id: u32,
    area: f32,
    eccentricity: f32,     // Hijacked phase_buffer_1
    cell_angle: f32,       // Hijacked phase_buffer_2
    sdf_to_void: f32,      // Hijacked phase_buffer_3
    v_mem: f32, ion_k: f32, ion_na: f32, ligand_pool: f32,
    receptor_pool: f32, sde_hidden_1: f32, sde_hidden_2: f32, neighbor_count: u32,
    connected_neighbors: array<u32, 16>,
}

// Exactly matches the 136-byte Rust struct
struct GeometricSignature {
    esr: f32,
    eccentricity: f32,
    polarity_alignment: f32,
    sdf_to_void: f32,
    ripleys_l: array<f32, 20>,
    radial_shells: array<f32, 10>,
}

@group(0) @binding(0) var<storage, read> cells: array<CellNode>;
@group(0) @binding(1) var<storage, read_write> signatures: array<GeometricSignature>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<storage, read> grid_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> sorted_cell_indices: array<u32>;
@group(0) @binding(5) var<storage, read> global_cell_count: atomic<u32>;

fn get_grid_hash(coord: vec3<u32>) -> u32 {
    let x = clamp(coord.x, 0u, params.grid_width - 1u);
    let y = clamp(coord.y, 0u, params.grid_height - 1u);
    let z = clamp(coord.z, 0u, params.grid_depth - 1u);
    return x + (y * params.grid_width) + (z * params.grid_width * params.grid_height);
}

@compute @workgroup_size(64)
fn extract_signatures_pass(@builtin(global_invocation_id) id: vec3<u32>) {
    let my_idx = id.x;
    let active_cells = atomicLoad(&global_cell_count);
    if (my_idx >= active_cells) { return; }

    let me = cells[my_idx];
    var sig: GeometricSignature;

    // ==========================================
    // 1. INTRINSIC PROPERTIES
    // ==========================================
    // Safely calculate ESR (Equivalent Spherical Radius)
    sig.esr = sqrt(max(me.area, 1.0) / PI);
    sig.eccentricity = me.eccentricity;
    sig.sdf_to_void = me.sdf_to_void;

    // ==========================================
    // 2. SPATIAL ACCUMULATION LOOP
    // ==========================================
    var cumulative_ripley = array<f32, 20>();
    var shell_counts = array<f32, 10>();

    // Nematic Polarity Tracking
    var sum_cos_2theta: f32 = cos(2.0 * me.cell_angle);
    var sum_sin_2theta: f32 = sin(2.0 * me.cell_angle);
    var neighbor_count: f32 = 1.0; // Includes self

    let ix = i32(floor(me.pos.x / params.tile_size));
    let iy = i32(floor(me.pos.y / params.tile_size));
    let iz = i32(floor(me.pos.z / params.tile_size));

    for (var x = -1i; x <= 1i; x++) {
        let nx = ix + x; if (nx < 0 || nx >= i32(params.grid_width)) { continue; }
        for (var y = -1i; y <= 1i; y++) {
            let ny = iy + y; if (ny < 0 || ny >= i32(params.grid_height)) { continue; }
            for (var z = -1i; z <= 1i; z++) {
                let nz = iz + z; if (nz < 0 || nz >= i32(params.grid_depth)) { continue; }

                let hash_idx = get_grid_hash(vec3<u32>(u32(nx), u32(ny), u32(nz)));
                let end_idx = grid_offsets[hash_idx];
                var start_idx = 0u;
                if (hash_idx > 0u) { start_idx = grid_offsets[hash_idx - 1u]; }

                for (var i = start_idx; i < end_idx; i++) {
                    let neighbor_idx = sorted_cell_indices[i];
                    if (my_idx == neighbor_idx) { continue; }

                    let neighbor = cells[neighbor_idx];
                    let dist = distance(me.pos, neighbor.pos);

                    // A. RADIAL SHELLS (0 to 100um, 10um steps)
                    if (dist < 100.0) {
                        let shell_bin = min(u32(dist / 10.0), 9u);
                        shell_counts[shell_bin] += 1.0;

                        // B. NEMATIC POLARITY (Only evaluate closest neighbors, e.g., within 30um)
                        if (dist < 30.0) {
                            sum_cos_2theta += cos(2.0 * neighbor.cell_angle);
                            sum_sin_2theta += sin(2.0 * neighbor.cell_angle);
                            neighbor_count += 1.0;
                        }
                    }

                    // C. RIPLEY'S K (0 to 30um, 1.5um steps)
                    if (dist <= 30.0) {
                        let rip_bin = min(u32(dist / 1.5), 19u);
                        for (var b = rip_bin; b < 20u; b++) {
                            cumulative_ripley[b] += 1.0;
                        }
                    }
                }
            }
        }
    }

    // ==========================================
    // 3. NORMALIZATION MATH
    // ==========================================

    // Polarity Alignment (Magnitude of the averaged doubled-angle vectors)
    let avg_cos = sum_cos_2theta / neighbor_count;
    let avg_sin = sum_sin_2theta / neighbor_count;
    sig.polarity_alignment = sqrt((avg_cos * avg_cos) + (avg_sin * avg_sin));

    // Radial Shells (Divide count by the geometric area of the ring)
    for (var i = 0u; i < 10u; i++) {
        let w = 10.0; // 10um step width
        let shell_area = PI * (w * w) * f32((2u * i) + 1u);
        sig.radial_shells[i] = shell_counts[i] / shell_area;
    }

    // Ripley's L (Besag's transformation)
    // We estimate local density using the 100um search radius area
    let local_area = PI * 100.0 * 100.0;
    var density_factor = 0.0;
    if (neighbor_count > 1.0) {
        density_factor = local_area / (neighbor_count * (neighbor_count - 1.0));
    }

    for (var b = 0u; b < 20u; b++) {
        let r = f32(b + 1u) * 1.5;
        let k_r = cumulative_ripley[b] * density_factor;
        sig.ripleys_l[b] = sqrt(max(k_r, 0.0) / PI) - r;
    }

    // Write the finalized signature to VRAM
    signatures[my_idx] = sig;
}