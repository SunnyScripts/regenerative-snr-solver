struct Camera {
    view_proj: mat4x4<f32>,
    cell_scale: f32,
    mouse_x: f32,
    mouse_y: f32,
    _pad: f32,
    tile_bounds: vec4<f32>, // [min_x, min_y, max_x, max_y]
}

struct CellNode {
    // Chunk 1
    pos: vec3<f32>,
    broad_id: u32,

    // Chunk 2
    polarity: vec3<f32>,
    granular_id: u32,

    // Chunk 3
    area: f32,
    ion_ca: f32,
    ion_cl: f32,
    v_mem: f32,

    // Chunk 4
    ion_k: f32,
    ion_na: f32,
    exogenous_v_clamp: f32,
    pump_health_multiplier: f32,

    // Chunk 5
    adhesion_multiplier: f32,
    neighbor_count: u32,
    edge_start: u32,
    _padding: u32,

    // Chunks 6-21
    rna_unspliced: array<f32, 32>,
    rna_spliced: array<f32, 32>,
}


@group(0) @binding(0) var<storage, read> cells: array<CellNode>;
@group(1) @binding(0) var<uniform> camera: Camera;


// 4 corners of a flat square
var<private> pos: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>(1.0,  1.0)
);

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) seed: vec2<f32>, // NEW: Used to generate organic randomness
};

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32, @builtin(instance_index) i_idx: u32) -> VertexOutput {
    let cell = cells[i_idx];

    // ==========================================
    // 1. TRUE BIOLOGICAL RADIUS
    // ==========================================
    // Calculate exact radius from Xenium area (with a 1.0 minimum safety clamp)
    var radius = max(sqrt(cell.area / 3.14159), 1.0);

    // Apply the global Slint UI scale slider
    radius = radius * camera.cell_scale;

    // ==========================================
    // 2. THE MAGNIFICATION LENS
    // ==========================================
    let mouse_pos = vec2<f32>(camera.mouse_x, camera.mouse_y);
    let dist = distance(cell.pos.xy, mouse_pos);
    let lens_radius = 80.0;

//    if (dist < lens_radius) {
//        let magnification_curve = 1.0 - smoothstep(0.0, lens_radius, dist);
//        let max_zoom_multiplier = 1.5;
//        radius = radius + (radius * magnification_curve * max_zoom_multiplier);
//    }

    // ==========================================
    // 3. GEOMETRY & PROJECTION
    // ==========================================
    let vertex_pos = vec3<f32>(pos[v_idx] * radius, 0.0);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(cell.pos + vertex_pos, 1.0);
    out.uv = pos[v_idx] * 0.5 + 0.5;
    out.seed = cell.pos.xy;

    // ==========================================
    // 4. APPLE COLOR MAPPING
    // ==========================================
    if (cell.broad_id == 0u) { out.color = vec3<f32>(0.0, 0.48, 1.0); }
    else if (cell.broad_id == 1u) { out.color = vec3<f32>(1.0, 0.23, 0.19); }
    else if (cell.broad_id == 2u) { out.color = vec3<f32>(1.0, 0.95, 0.96); }
    else if (cell.broad_id == 3u) { out.color = vec3<f32>(1.0, 0.95, 0.19); }
    else { out.color = vec3<f32>(0.55, 0.55, 0.57); }

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32>
{
    // Shift UV to be center-origin (-0.5 to 0.5) for easier math
    var centered_uv = in.uv - vec2<f32>(0.5, 0.5);

    // ==========================================
    // 1. ORGANIC SQUISH (Perturb the shape)
    // ==========================================
    // Calculate the angle of the pixel to apply a sine wave to the radius
    let angle = atan2(centered_uv.y, centered_uv.x);

    // Mix the cell's position (seed) into the sine wave so every cell is uniquely deformed
    let squish = sin(angle * 4.0 + in.seed.x * 0.1) * 0.03;
    let squish2 = cos(angle * 7.0 + in.seed.y * 0.1) * 0.02;

    // The base radius is normally 0.5 (the edge of the UV box)
    let membrane_radius = 0.45 + squish + squish2;
    let dist = length(centered_uv);

    // Discard pixels outside the wobbly cell boundary
    if (dist > membrane_radius) { discard; }

    // ==========================================
    // 2. THE CYTOPLASM (Volume)
    // ==========================================
    var final_color = in.color;

    // Make the center brighter and the edges darker (Ambient Occlusion)
    let volume_glow = smoothstep(membrane_radius, 0.0, dist);
    final_color = final_color * (0.5 + 0.5 * volume_glow);

    // ==========================================
    // 3. THE NUCLEUS (Dense Core)
    // ==========================================
    // Offset the nucleus slightly from the center based on the seed
    let nuc_offset = vec2<f32>(sin(in.seed.x), cos(in.seed.y)) * 0.08;
    let nuc_dist = length(centered_uv - nuc_offset);

    // Soft-edge nucleus boundary
    let nuc_radius = 0.12;
    let in_nucleus = smoothstep(nuc_radius + 0.03, nuc_radius, nuc_dist);

    // Darken and desaturate the color for the nucleus
    let nuc_color = in.color * 0.2;
    final_color = mix(final_color, nuc_color, in_nucleus);

    // ==========================================
    // 4. THE MEMBRANE (Fresnel Rim Light)
    // ==========================================
    // Creates a glowing "lipid bilayer" edge
    let rim = smoothstep(membrane_radius - 0.06, membrane_radius, dist);

    // Add bright, saturated color to the very edge of the cell
    final_color = final_color + (in.color * rim * 1.8);

//    if (in.seed.x >= camera.tile_bounds.x && in.seed.x <= camera.tile_bounds.z &&
//        in.seed.y >= camera.tile_bounds.y && in.seed.y <= camera.tile_bounds.w)
//        {
//            // Add a glowing green rim-light to all cells in the selected tile!
//            final_color = final_color + vec3<f32>(0.0, 0.8, 0.2) * rim * 5.0;
//        }

    return vec4<f32>(final_color, 1.0); // If you enable WGPU alpha blending, change 1.0 to 0.8
}