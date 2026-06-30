use bytemuck::{Pod, Zeroable};

pub const GPU_OBS_DIM: usize = 185; // Ripley(20) + Enrich(36) + Halo(128) + Count(1)
pub const BRAIN_INPUT_DIM: usize = 189; // GPU_OBS_DIM + Age(1) + Normal(3)

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BrainOpcode {
    pub cmd_id: u32,
    pub p1: f32,
    pub p2: f32,
    pub p3: f32,
    pub p4: f32,
    pub type_ratios: [f32; 4],
    pub _pad: [f32; 3], // Pads from 36 bytes to 48 bytes
}

impl Default for BrainOpcode {
    fn default() -> Self {
        Self {
            cmd_id: 7, // Default to Halt
            p1: 0.0, p2: 0.0, p3: 0.0, p4: 0.0,
            type_ratios: [0.0; 4],
            _pad: [0.0; 3],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SimParams {
    pub grid_width: u32,
    pub grid_height: u32,
    pub grid_depth: u32,  // RESTORED: Your 3D depth!
    pub tile_size: f32,

    pub epoch: u32,
    pub current_cell_count: u32,
    pub _pad: [u32; 6],   // REDUCED: Padding is now 2 to hit exactly 32 bytes
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MembraneInterface {
    // vec4 1: Topology & Bioelectrics (16 bytes)
    pub source_idx: u32,
    pub target_idx: u32,
    pub conductance: f32,          // Gap junction strength (from Connexin expression)
    pub adhesion: f32,             // Physical bond strength (from Cadherin expression)

    // vec4 2: Spatial Physics & Signaling (16 bytes)
    pub distance: f32,             // Raw distance for Paracrine diffusion math
    pub connection_strength: f32,  // 0.0 to 1.0 for Juxtacrine/Contact math
    pub matrix_stiffness: f32,     // (Was edge_state_1) - Extracellular matrix rigidity / fibrosis
    pub rl_edge_clamp: f32,        // (Was edge_state_2) - For the RL Agent to artificially sever/boost this specific junction
}

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, Pod, Zeroable)]
pub struct Coord {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, Pod, Zeroable)]
pub struct CellNode
{
    // Chunk 1 (16 Bytes)
    pub pos: Coord,                // 12 bytes
    pub broad_id: u32,             // 4 bytes (0=Epi, 1=Endo, etc.)

    // Chunk 2 (16 Bytes)
    pub polarity: Coord,           // 12 bytes
    pub granular_id: u32,          // 4 bytes (Lineage ID for OT tensors)

    // Chunk 3 (16 Bytes)
    pub area: f32,                 // 4 bytes (Degrades via OT area_decay)
    pub ion_ca: f32,               // 4 bytes
    pub ion_cl: f32,               // 4 bytes (For GHK)
    pub v_mem: f32,                // 4 bytes (Current Voltage)

    // Chunk 4 (16 Bytes)
    pub ion_k: f32,                // 4 bytes (For GHK)
    pub ion_na: f32,               // 4 bytes (For GHK)
    pub exogenous_v_clamp: f32,    // 4 bytes (RL Intervention Float)
    pub pump_health_multiplier: f32, // 4 bytes (Degrades via OT pump_atrophy)

    // Chunk 5 (16 Bytes)
    pub adhesion_multiplier: f32,  // 4 bytes (Degrades via OT adhesion_decay)
    pub neighbor_count: u32,       // 4 bytes (CSR active bonds)
    pub edge_start: u32,           // 4 bytes (CSR Arena index)
    pub _padding: u32,             // 4 bytes (REQUIRED to finish the 16-byte block)

    // Chunk 6-13 (128 Bytes)
    pub rna_unspliced: [f32; 32],  // The SDE Fast Clock Variables

    // Chunk 14-21 (128 Bytes)
    pub rna_spliced: [f32; 32],    // The SDE Slow Clock Variables
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MLPLayer {
    pub w1: [f32; 9], // mat3x3
    pub padding1: [f32; 3], // WebGPU requires 16-byte alignment for matrices
    pub b1: [f32; 3], // vec3
    pub padding2: f32,

    pub w2: [f32; 9],
    pub padding3: [f32; 3],
    pub b2: [f32; 3],
    pub padding4: f32,
}