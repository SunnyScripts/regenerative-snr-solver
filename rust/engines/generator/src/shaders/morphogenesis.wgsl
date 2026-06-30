struct BrainOpcode {
     cmd_id: u32,
     p1: f32,
     p2: f32,
     p3: f32,
     p4: f32,
     type_ratios: vec4<f32>, // NEW: The French Flag [Epi, Endo, Mes, Imm]
     _pad: vec3<f32>,        // Padding to hit exactly 48 bytes
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

struct SimParams {
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
    tile_size: f32,
    simulation_step: u32,
    current_cell_count: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> raw_opcodes: array<BrainOpcode>;
@group(0) @binding(1) var<storage, read_write> active_emitters: array<MorphogenEmitter>;
@group(0) @binding(2) var<storage, read_write> emitter_count: atomic<u32>;
@group(0) @binding(3) var<uniform> params: SimParams;

fn get_tile_center(idx: u32) -> vec3<f32> {
    // 1. Convert 1D index back to 2D grid coordinates
    let tx = f32(idx % params.grid_width);
    let ty = f32(idx / params.grid_width);

    // 2. Calculate world space (X, Y)
    // We add 0.5 to center the position within the tile
    let x = (tx + 0.5) * params.tile_size;
    let y = (ty + 0.5) * params.tile_size;

    // We keep Z at 0.0 for the baseline,
    // though your specific opcodes may override this later.
    return vec3<f32>(x, y, 0.0);
}

// Thread Count = Number of Tiles
@compute @workgroup_size(64)
fn paint_ecm(@builtin(global_invocation_id) id: vec3<u32>) {
    let tile_idx = id.x;

    // ---------------------------------------------------------
    // 1. THE DECAY STEP (Only Thread 0 does the cleanup)
    // ---------------------------------------------------------
    if (tile_idx == 0u) {
        let current_count = atomicLoad(&emitter_count);
        var alive = 0u;

        for (var i = 0u; i < current_count; i++) {
            var em = active_emitters[i];
            em.strength -= em.decay_rate; // The signal fades over time

            // If it's still biologically active, pack it down
            if (em.strength > 0.01) {
                active_emitters[alive] = em;
                alive++;
            }
        }
        atomicStore(&emitter_count, alive);
    }

    // Ensure cleanup is done before injecting new ones
    workgroupBarrier();

    // ---------------------------------------------------------
    // 2. THE INJECTION STEP
    // ---------------------------------------------------------
    let op = raw_opcodes[tile_idx];
    if (op.cmd_id == 7u) { return; } // Halt command = Do nothing

    var new_em: MorphogenEmitter;
    new_em.strength = 1.0;
    new_em.decay_rate = 0.05; // Fades out completely in 20 steps

    // Using your exact 4 parameters to shape the SDF Math!
    switch op.cmd_id {
        // Morph::Invaginate { depth, width, target_x, target_y }
        case 1u: {
            new_em.emitter_type = 1u; // Repulsion Field
            new_em.pos = vec3<f32>(op.p3, op.p4, 50.0); // Z is derived from tissue depth
            new_em.params = vec4<f32>(op.p1, op.p2, 0.0, 0.0); // Radius and falloff
        }
        // Morph::Elongate { length, radius, target_x, target_y }
        case 2u: {
            new_em.emitter_type = 0u; // Growth Factor
            // We set the start position at the center of this Tile
            new_em.pos = get_tile_center(tile_idx);
            // The vector points toward the target
            new_em.dir = normalize(vec3<f32>(op.p3 - new_em.pos.x, op.p4 - new_em.pos.y, 0.0));
            new_em.params = vec4<f32>(op.p1, op.p2, 0.0, 0.0); // Length, Radius
        }
        // Morph::Condense { radius, density, z_stretch, hollow_core }
        case 4u: {
            new_em.emitter_type = 2u; // Adhesion Modifier
            new_em.pos = get_tile_center(tile_idx);
            new_em.params = vec4<f32>(op.p1, op.p2, op.p3, op.p4);
        }
        // Morph::Intercalate { amplitude, frequency, phase_x, phase_y }
        case 0u: {
            new_em.emitter_type = 4u; // Frequency Field (Sine Wave Overlay)
            new_em.pos = get_tile_center(tile_idx);
            // Stores the wave parameters. The Mitosis pass will run sin() math on these
            // to create alternating bands of high/low cell affinity.
            new_em.params = vec4<f32>(op.p1, op.p2, op.p3, op.p4);
        }

        // Morph::Bifurcate { angle, radius_decay, yaw, asymmetry }
        case 3u: {
            new_em.emitter_type = 0u; // Growth Factor
            new_em.pos = get_tile_center(tile_idx);
            // We store the primary branch direction in `dir`
            new_em.dir = normalize(vec3<f32>(cos(op.p3), sin(op.p3), 0.0));
            // p1 (angle) and p4 (asymmetry) will be used by the cells to calculate
            // distance to TWO intersecting capsules (a 'Y' shape SDF).
            new_em.params = vec4<f32>(op.p1, op.p2, 0.0, op.p4);
        }

        // Morph::Disperse { scatter_radius, density, z_stretch, hollow_core }
        case 5u: {
            new_em.emitter_type = 2u; // Adhesion Modifier
            new_em.pos = get_tile_center(tile_idx);
            new_em.params = vec4<f32>(op.p1, op.p2, op.p3, op.p4);
            // The key difference from Condense: Negative strength!
            // This tells the spring physics to push apart instead of pull together.
            new_em.strength = -1.0;
        }

        // Morph::Stratify { layers, z_compression, noise, gradient }
        case 6u: {
            new_em.emitter_type = 3u; // Planar Field
            new_em.pos = get_tile_center(tile_idx);
            // Forces the gradient strictly along the Z-axis to squash the tissue flat
            new_em.dir = vec3<f32>(0.0, 0.0, 1.0);
            new_em.params = vec4<f32>(op.p1, op.p2, op.p3, op.p4);
        }
        default: {}
    }

    // Atomically claim a slot in the buffer and save the math formula
    let insert_idx = atomicAdd(&emitter_count, 1u);
    active_emitters[insert_idx] = new_em;
}