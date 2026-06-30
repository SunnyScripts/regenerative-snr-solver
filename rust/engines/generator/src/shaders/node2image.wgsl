// ---------------------------------------------------------
// VERTEX SHADER: Maps the 2D cell coordinate to the screen
// ---------------------------------------------------------
struct CellNode {
    pos: vec3<f32>,
    area: f32,
    polarity: vec3<f32>,
    packed_ids: u32,
    // ... padding
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) cell_type: u32,
}

@group(0) @binding(0) var<storage, read> slice_cells: array<CellNode>;

// We draw a simple quad (4 vertices) for every cell
@vertex
fn vs_main(
    @builtin(vertex_index) v_idx: u32,
    @builtin(instance_index) inst_idx: u32
) -> VertexOutput {
    var out: VertexOutput;
    let cell = slice_cells[inst_idx];

    // Quad coordinates (-1 to 1) mapped to the cell's radius
    var quad_pos = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0,  1.0)
    );

    let local_pos = quad_pos[v_idx];
    out.uv = local_pos; // Pass UV to fragment shader for circle drawing
    out.cell_type = cell.packed_ids;

    // Transform biological 2D coordinates to WGPU Screen Space (-1.0 to 1.0)
    // (Assuming a 200x200um viewing window)
    let screen_x = (cell.pos.x / 100.0) - 1.0 + (local_pos.x * (cell.area / 100.0));
    let screen_y = (cell.pos.y / 100.0) - 1.0 + (local_pos.y * (cell.area / 100.0));

    out.clip_position = vec4<f32>(screen_x, screen_y, 0.0, 1.0);
    return out;
}

// ---------------------------------------------------------
// FRAGMENT SHADER: The Biological Paint (H&E)
// ---------------------------------------------------------
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Calculate distance from center of the cell
    let dist = length(in.uv);

    // If outside the circle, discard the pixel (transparency)
    if (dist > 1.0) { discard; }

    // Standard H&E Base Colors
    let nucleus_color = vec3<f32>(0.2, 0.0, 0.4); // Dark Purple (Hematoxylin)
    var cyto_color = vec3<f32>(0.9, 0.4, 0.6);    // Pink (Eosin)

    // Adjust morphology based on cell type!
    var nucleus_size = 0.3; // Default 30% of radius

    if (in.cell_type == 1u) { // Epithelial (Dense, blocky)
        nucleus_size = 0.5;
        cyto_color = vec3<f32>(0.8, 0.3, 0.5); // Darker pink
    } else if (in.cell_type == 3u) { // Immune (Small, all nucleus)
        nucleus_size = 0.8;
        cyto_color = vec3<f32>(0.95, 0.8, 0.8); // Barely any cytoplasm
    }

    // Paint the pixel
    if (dist < nucleus_size) {
        return vec4<f32>(nucleus_color, 1.0);
    } else {
        return vec4<f32>(cyto_color, 0.7); // 0.7 Alpha allows overlapping transparency!
    }
}
//todo update formula
struct GeneKinetics {
    // ... [existing beta, gamma, hill variables] ...
    sigma1: f32,
    sigma2: f32,

    // NEW: The Empirical Aging Variables
    old_target_u: f32, // The empirical mean of the Old dataset (unspliced)
    old_target_s: f32, // The empirical mean of the Old dataset (spliced)
    wot_theta: f32,    // The strength of the aging "rubber band"
}

// Inside grn_pass.wgsl SDE loop:
let wot_pull_u = gene.wot_theta * (gene.old_target_u - me.rna_unspliced[i]);
let wot_pull_s = gene.wot_theta * (gene.old_target_s - me.rna_spliced[i]);

let du = (c_effective - gene.beta * me.rna_unspliced[i]) * GRN_DT + wot_pull_u + noise_u;
let ds = (gene.beta * me.rna_unspliced[i] - gene.gamma * me.rna_spliced[i]) * GRN_DT + wot_pull_s + noise_s;