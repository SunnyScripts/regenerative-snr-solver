use super::types::*;
use wgpu::util::DeviceExt;
// use std::fs;
use shared_biology::GPU_OBS_DIM;
use glam::{Mat4, Vec4};

pub struct NativeWgpuEngine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub sim_params: SimParams,

    // --- OBSERVATION BUFFERS ---
    pub observation_buffer: wgpu::Buffer,
    pub staging_buffer: wgpu::Buffer,
    pub param_buffer: wgpu::Buffer,

    // Dynamic Buffers (Built frame-to-frame in io.rs)
    pub cell_buffer: Option<wgpu::Buffer>,
    pub offset_buffer: Option<wgpu::Buffer>,
    pub count_buffer: Option<wgpu::Buffer>,
    pub obs_bind_group: Option<wgpu::BindGroup>,

    // --- 3D GENERATION BUFFERS ---
    pub volume_cells: wgpu::Buffer,
    pub opcodes: wgpu::Buffer,
    pub global_cell_count: wgpu::Buffer,
    pub cpu_cell_count: u32,

    pub active_emitters: wgpu::Buffer,
    pub emitter_count: wgpu::Buffer,
    pub edges: wgpu::Buffer,
    pub global_edge_count: wgpu::Buffer,

    // --- SPATIAL HASH BUFFERS ---
    pub grid_params_buffer: wgpu::Buffer,
    pub grid_offsets_buffer: wgpu::Buffer,
    pub sorted_cells_buffer: wgpu::Buffer,

    // --- MICROTOME BUFFERS ---
    pub microtome_config_buffer: wgpu::Buffer,
    pub slice_cells: wgpu::Buffer,
    pub slice_count: wgpu::Buffer,
    pub slice_staging_buffer: wgpu::Buffer,

    // --- PIPELINES ---
    pub morphogenesis_pipeline: wgpu::ComputePipeline,
    pub mitosis_pipeline: wgpu::ComputePipeline,
    pub topology_pipeline: wgpu::ComputePipeline,
    pub physics_pipeline: wgpu::ComputePipeline,
    pub microtome_pipeline: wgpu::ComputePipeline,
    pub observation_pipeline: wgpu::ComputePipeline,

    // --- STATIC BIND GROUPS ---
    // (Notice these are no longer Options! They are guaranteed to exist)
    pub morphogenesis_bind_group: wgpu::BindGroup,
    pub mitosis_bind_group: wgpu::BindGroup,
    pub topology_bind_group: wgpu::BindGroup,
    pub physics_bind_group: wgpu::BindGroup,
    pub microtome_bind_group: wgpu::BindGroup,

    // --- Spatial Hash ---- \\
    pub hash_bind_group: wgpu::BindGroup,
    pub hash_clear_pipeline: wgpu::ComputePipeline,
    pub hash_count_pipeline: wgpu::ComputePipeline,
    pub hash_scan_pipeline: wgpu::ComputePipeline,
    pub hash_insert_pipeline: wgpu::ComputePipeline,

    // --- Renderer --- \\
    pub render_pipeline: wgpu::RenderPipeline,
    pub cell_bind_group: wgpu::BindGroup,
    pub camera: Camera,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub camera_needs_update: bool,
    pub selected_tile_bounds: Vec4,

    // UI Data Transport
    pub class_counts: [usize; 4],
    pub total_cells: usize,

    // Mouse Tracking State
    pub mouse_world_x: f32,
    pub mouse_world_y: f32,

    pub num_tiles: usize,
    pub output_size_bytes: u64,
    pub count_staging_buffer: wgpu::Buffer,
}

impl NativeWgpuEngine {
    pub async fn new(params: SimParams, resources: EngineResources) -> Self
    {
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default())
            .await.expect("Failed to find Apple Silicon Metal adapter");

        let (device, queue) = match (resources.device, resources.queue) {
            (Some(d), Some(q)) => (d, q),
            _ => adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Simulation Device"),
                    required_features: wgpu::Features::empty(), // Add features here if needed
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),

                    // THE FIX: Stop using Default::default().
                    // Tell wgpu to use the maximum limits the M4 Pro supports!
                    required_limits: adapter.limits(),
                    trace: wgpu::Trace::Off,
                    memory_hints: Default::default(),
                }
            ).await.expect("Failed to create device"),
        };

        // ==========================================
        // 1. LOAD THE STATIC GPU BINARIES
        // ==========================================
        println!("Loading pre-compiled physics binaries...");

        let adhesion_bytes = include_bytes!("physics_rules/adhesion_matrix.bin");
        let conductance_bytes = include_bytes!("physics_rules/conductance_matrix.bin");
        let depths_bytes = include_bytes!("physics_rules/ideal_depths.bin");
        let strat_bytes = include_bytes!("physics_rules/strat_weights.bin");

        let interactome_bytes = include_bytes!("physics_rules/gpu_interactome.bin");
        let (offset_bytes, rule_bytes) = interactome_bytes.split_at(256);
        assert!(rule_bytes.len() % 16 == 0, "FATAL: Interactome rules are not 16-byte aligned!");

        // ==========================================
        // 2. COMPILE PIPELINES FIRST
        // (Must be done before building Bind Groups!)
        // ==========================================
        let obs_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Observation Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/observation_space.wgsl").into()),
        });
        let observation_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Observation Pipeline"), layout: None, module: &obs_shader,
            entry_point: Some("generate_micro_tensor"), compilation_options: Default::default(), cache: None,
        });

        let morphogenesis_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Morphogenesis Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/morphogenesis.wgsl").into()),
        });
        let morphogenesis_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Morphogenesis Pipeline"), layout: None, module: &morphogenesis_shader,
            entry_point: Some("paint_ecm"), compilation_options: Default::default(), cache: None,
        });

        let mitosis_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mitosis Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mitosis.wgsl").into()),
        });
        let mitosis_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mitosis Pipeline"), layout: None, module: &mitosis_shader,
            entry_point: Some("mitosis_and_identity_pass"), compilation_options: Default::default(), cache: None,
        });

        let topology_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Topology Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/topology.wgsl").into()),
        });
        let topology_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Topology Pipeline"), layout: None, module: &topology_shader,
            entry_point: Some("topology_pass"), compilation_options: Default::default(), cache: None,
        });

        let physics_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Physics Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/physics.wgsl").into()),
        });
        let physics_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Physics Pipeline"), layout: None, module: &physics_shader,
            entry_point: Some("physics_pass"), compilation_options: Default::default(), cache: None,
        });

        let microtome_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Microtome Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/microtome.wgsl").into()),
        });
        let microtome_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Microtome Pipeline"), layout: None, module: &microtome_shader,
            entry_point: Some("microtome_pass"), compilation_options: Default::default(), cache: None,
        });

        // ==========================================
        // 3. CREATE ALL BUFFERS
        // ==========================================
        let count_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Count Staging Buffer"),
            size: 4, // Exactly 4 bytes for a single u32
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // let ideal_areas_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //     label: Some("Ideal Areas Buffer"),
        //     contents: &ideal_areas_bytes,
        //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        // });

        let adhesion_matrix_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Adhesion"), contents: adhesion_bytes, usage: wgpu::BufferUsages::STORAGE });
        let conductance_matrix_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Conductance"), contents: conductance_bytes, usage: wgpu::BufferUsages::STORAGE });
        let ideal_depths_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Depths"), contents: depths_bytes, usage: wgpu::BufferUsages::STORAGE });
        let strat_weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Strat Weights"), contents: strat_bytes, usage: wgpu::BufferUsages::STORAGE });
        let interactome_offset_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Offsets"), contents: offset_bytes, usage: wgpu::BufferUsages::UNIFORM });
        let interactome_rules_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Rules"), contents: rule_bytes, usage: wgpu::BufferUsages::STORAGE });

        let grid_params_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Grid Params"), size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let grid_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Grid Offsets"), size: ((MAX_VOLUME_CELLS + 1) * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let sorted_cells_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Sorted Cells"), size: (MAX_VOLUME_CELLS * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let max_emitters = 10_000;
        let max_edges = MAX_VOLUME_CELLS * 8;
        let active_emitters = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Active Emitters"), size: (max_emitters * std::mem::size_of::<MorphogenEmitter>()) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let emitter_count = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Emitter Count"), size: 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let edges = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Membrane Interfaces"), size: (max_edges * std::mem::size_of::<MembraneInterface>()) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let global_edge_count = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Edge Count"), size: 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let microtome_config_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Microtome Config"), size: std::mem::size_of::<MicrotomeConfig>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let slice_count = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Slice Count"), size: 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let slice_cells = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Slice Cells"), size: (MAX_VOLUME_CELLS * std::mem::size_of::<CellNode>()) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let slice_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Slice Staging"), size: (MAX_VOLUME_CELLS * std::mem::size_of::<CellNode>()) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let num_tiles = (params.grid_width * params.grid_height) as usize;
        let output_size_bytes = (num_tiles * GPU_OBS_DIM * 4) as u64; // Now 740 bytes per tile

        let observation_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Observation Storage"), size: output_size_bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Observation Staging"), size: output_size_bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Params Uniform"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE });

        let volume_cells = device.create_buffer(&wgpu::BufferDescriptor { label: Some("3D Volume Cells"), size: (MAX_VOLUME_CELLS * std::mem::size_of::<CellNode>()) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let opcodes = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Candle Opcodes"), size: (num_tiles * 20 * std::mem::size_of::<GpuOpcode>()) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let global_cell_count = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Atomic Global Count"), size: 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });

        // ==========================================
        // 4. BUILD STATIC BIND GROUPS
        // ==========================================
        let morphogenesis_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Morphogenesis Bind Group"),
            layout: &morphogenesis_pipeline.get_bind_group_layout(0),
            entries: &[
                // Binding 0: The instructions from the AI (Candle/ONNX)
                wgpu::BindGroupEntry { binding: 0, resource: opcodes.as_entire_binding() },

                // Binding 1: The ECM array where the GPU writes the output
                wgpu::BindGroupEntry { binding: 1, resource: active_emitters.as_entire_binding() },

                // Binding 2: The atomic counter for how many emitters currently exist
                wgpu::BindGroupEntry { binding: 2, resource: emitter_count.as_entire_binding() },

                // Binding 3: Global simulation parameters (Optional, but usually needed)
                wgpu::BindGroupEntry { binding: 3, resource: param_buffer.as_entire_binding() },
            ],
        });

        let mitosis_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mitosis Bind Group"), layout: &mitosis_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: active_emitters.as_entire_binding() }, // Check these match your mitosis.wgsl @binding tags perfectly!
                wgpu::BindGroupEntry { binding: 1, resource: emitter_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: volume_cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: global_cell_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: edges.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: global_edge_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: interactome_offset_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: interactome_rules_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: param_buffer.as_entire_binding() }
            ],
        });

        let topology_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Topology Bind Group"),
            layout: &topology_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: volume_cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: edges.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: global_edge_count.as_entire_binding() },

                // USE THE UNIFIED SIM PARAMS BUFFER!
                wgpu::BindGroupEntry { binding: 3, resource: param_buffer.as_entire_binding() },

                wgpu::BindGroupEntry { binding: 4, resource: grid_offsets_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: sorted_cells_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: adhesion_matrix_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: conductance_matrix_buffer.as_entire_binding() },

                // FIX THE ATOMIC COUNTER!
                wgpu::BindGroupEntry { binding: 8, resource: global_cell_count.as_entire_binding() }
            ],
        });

        let physics_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Physics Bind Group"),
            layout: &physics_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: volume_cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: edges.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: active_emitters.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: emitter_count.as_entire_binding() },

                // --- BIOLOGY RULEBOOK ---
                wgpu::BindGroupEntry { binding: 4, resource: ideal_depths_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: strat_weights_buffer.as_entire_binding() },

                // --- SPATIAL HASH ---
                wgpu::BindGroupEntry { binding: 6, resource: param_buffer.as_entire_binding() }, // SimParams
                wgpu::BindGroupEntry { binding: 7, resource: grid_offsets_buffer.as_entire_binding() },

                // THE MISSING LINK:
                wgpu::BindGroupEntry { binding: 8, resource: sorted_cells_buffer.as_entire_binding() },

                // THE THREAD KILLER:
                wgpu::BindGroupEntry { binding: 9, resource: global_cell_count.as_entire_binding() },
            ],
        });

        let microtome_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Microtome Bind Group"),
            layout: &microtome_pipeline.get_bind_group_layout(0),
            entries: &[
                // 0 & 1: The Input
                wgpu::BindGroupEntry { binding: 0, resource: volume_cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: global_cell_count.as_entire_binding() },

                // 2 & 3: The Output (Observation/Slice)
                wgpu::BindGroupEntry { binding: 2, resource: observation_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: slice_count.as_entire_binding() },

                // 4: The Config
                wgpu::BindGroupEntry { binding: 4, resource: microtome_config_buffer.as_entire_binding() },
            ],
        });
        // 1. Compile the Shader
        let hash_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spatial Hash Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/spatial_hash.wgsl").into()),
        });

        // 1. EXPLICITLY DEFINE THE LAYOUT
        let hash_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Spatial Hash Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { // 0: cells
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { // 1: global_cell_count
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { // 2: grid_params
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { // 3: grid_offsets (Atomics require read_write storage)
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { // 4: sorted_cells
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        // 1. Update the layout descriptor
        let hash_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Spatial Hash Pipeline Layout"),

            // NEW: bind_group_layouts now takes &[Option<&BindGroupLayout>]
            bind_group_layouts: &[&hash_bind_group_layout],

            // NEW: Replaces push_constant_ranges. Since we aren't using
            // small-constant-injection here, we set this to 0.
            immediate_size: 0,
        });

        // 2. APPLY THE EXPLICIT LAYOUT TO ALL 4 PIPELINES
        let hash_clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Hash Clear"), layout: Some(&hash_pipeline_layout), module: &hash_shader, entry_point: Some("clear_pass"), compilation_options: Default::default(), cache: None,
        });
        let hash_count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Hash Count"), layout: Some(&hash_pipeline_layout), module: &hash_shader, entry_point: Some("count_pass"), compilation_options: Default::default(), cache: None,
        });
        let hash_scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Hash Scan"), layout: Some(&hash_pipeline_layout), module: &hash_shader, entry_point: Some("scan_pass"), compilation_options: Default::default(), cache: None,
        });
        let hash_insert_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Hash Insert"), layout: Some(&hash_pipeline_layout), module: &hash_shader, entry_point: Some("insert_pass"), compilation_options: Default::default(), cache: None,
        });

        // 3. CREATE THE BIND GROUP USING THE EXPLICIT LAYOUT
        let hash_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Spatial Hash Bind Group"),
            layout: &hash_bind_group_layout, // Use the layout directly!
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: volume_cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: global_cell_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: param_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: grid_offsets_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: sorted_cells_buffer.as_entire_binding() },
            ],
        });

        // 1. Create the Buffer
        let camera_uniform = CameraUniform { view_proj: Mat4::IDENTITY, cell_scale: 1.0, mouse_x: 0.0, mouse_y: 0.0, _pad: 0.0, tile_bounds: Vec4::ZERO };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 2. Define the Layout
        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_bind_group_layout"),
        });

        // 3. Create the Bind Group
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let cell_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Renderer Cell Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // Visibilty must include VERTEX so the vs_main shader can see it
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let cell_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Renderer Cell Bind Group"),
            layout: &cell_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: volume_cells.as_entire_binding(), // Your VRAM buffer
                },
            ],
        });

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Instanced Cell Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/renderer.wgsl").into()),
        });

        // 2. Create the Pipeline Layout (telling WGPU about your bind groups)
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &cell_bind_group_layout,   // group(0)
                &camera_bind_group_layout, // group(1)
            ],
            immediate_size: 0,
        });

        // 3. Build the actual Pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor
        {
            label: Some("Cell Render Pipeline"),
            layout: Some(&render_pipeline_layout),

            // Wire up the Vertex Shader
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[], // EMPTY! Because we use Storage Buffers, not Vertex Buffers.
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },

            // Wire up the Fragment Shader
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    // CRITICAL: This MUST match the format of the Slint texture we created
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default()
            }),

            // We render triangles (our squares)
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                ..Default::default()
            },

            // (Optional) Add depth buffer state here later
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None
            // multiview: None,
        });

        Self {
            device,
            queue,
            sim_params: params,

            observation_buffer,
            staging_buffer,
            param_buffer,
            cell_buffer: None,
            offset_buffer: None,
            count_buffer: None,
            obs_bind_group: None,

            volume_cells,
            opcodes,
            global_cell_count,
            cpu_cell_count: 0,
            active_emitters,
            emitter_count,
            edges,
            global_edge_count,

            grid_params_buffer,
            grid_offsets_buffer,
            sorted_cells_buffer,

            microtome_config_buffer,
            slice_cells,
            slice_count,
            slice_staging_buffer,

            morphogenesis_pipeline,
            mitosis_pipeline,
            topology_pipeline,
            physics_pipeline,
            microtome_pipeline,
            observation_pipeline,

            morphogenesis_bind_group,
            mitosis_bind_group,
            topology_bind_group,
            physics_bind_group,
            microtome_bind_group,

            hash_bind_group,
            hash_clear_pipeline,
            hash_count_pipeline,
            hash_scan_pipeline,
            hash_insert_pipeline,

            render_pipeline,
            cell_bind_group,
            camera_buffer,
            camera: Camera{..Default::default()},
            camera_bind_group,

            class_counts: [0; 4],
            total_cells: 0,
            mouse_world_x: 0.0,
            mouse_world_y: 0.0,
            camera_needs_update: false,
            num_tiles,
            output_size_bytes,
            selected_tile_bounds: Vec4::ZERO,
            count_staging_buffer,
        }
    }
}