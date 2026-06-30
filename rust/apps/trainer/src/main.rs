
use std::time::Instant;
//
// mod data_loader;
// // mod cell_classifier;
// mod graph;
// mod wire_modes;
// mod rna_broad_cell_type_classifier_onnx;
// mod gpu_setup;
// mod graphics_engine;
// mod generator;
// mod split_brain;
// mod tissue_training;
// mod interactome;
//
// use data_loader::{SpatialMetadata, FastInteractome, XeniumData};
// use rna_broad_cell_type_classifier_onnx::CellClassifier;
// use graph::GroundTruthEnvironment;
// use crate::data_loader::save_tensor_to_bin;
// use crate::graph::{Coord, CellNode, SimParams};
// use crate::graphics_engine::NativeWgpuEngine;


use shared_biology::{CellNode, Coord, SimParams};
use generator::NativeWgpuEngine;

use data_compiler::gt_environment::GroundTruthEnvironment;
use data_compiler::rna_dual_head_cell_type_classifier::CellClassifier;
use data_compiler::data_io::save_tensor_to_bin;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>>
{
    // run_validation_test().await.expect("Validation tests failed");


    let granular_class: [&str; 24] = ["Arterial Endothelial Cell", "B Cell", "Basal Keratinocyte", "Capillary Endothelial Cell", "Cornified Keratinocyte", "Dendritic Cell", "Fibroblast", "Granular Keratinocyte", "Hair Follicle Cell", "Lymphatic Endothelial Cell", "Macrophage", "Mast Cell", "Merkel Cell", "Monocyte", "NK Cell", "Neutrophil", "Proliferating Keratinocyte", "Sebaceous Gland Cell", "Skeletal Muscle Cell", "Smooth Muscle Cell", "Spinous Keratinocyte", "Sweat Gland Cell", "T Cell", "Venous Endothelial Cell"];

    println!("🚀 Booting Bioelectric SDE Engine...");
    let start_pipeline = Instant::now();

    // 1. Instantiate the Environment Object (Loads all files internally)
    let mut environment = GroundTruthEnvironment::new(
        "../../../cellular_data/Spatial/Manchester/back/cell_feature_matrix.h5",
        "../../../cellular_data/Spatial/Manchester/back/cells.parquet",
        // "morphology_fov_locations.json",
        // "../H&E Computer Vision/RNA_Cell_Classification/mechanical_classes.json",
        // "../H&E Computer Vision/RNA_Cell_Classification/granular_classes.json",
        // "model_means.json",
        // "interactome.json",
        150.0, // Interaction Radius
    )?;

    let mut classifier = CellClassifier::load("../../../models/rna_cell_classifier/Dualv2/dual_head_classifier.onnx")?;

    environment.identify_broad_cell_types(&mut classifier, "../../../python/cell_type_model_builder/symbol2entrez.json",
                         "../../../python/cell_type_model_builder/true_entrez_list.json")?;

    // 1. Resolve the Future into actual data (Vec<Vec<f32>>)
    // Note: You must be inside an async function or block to use .await
    let ground_truth_tensor = environment.extract_observation_tensor().await?;


    println!("✅ Ground Truth setup in {:.2?}", start_pipeline.elapsed());

    // ==========================================
    // 📊 BROAD CLASS DISTRIBUTION
    // ==========================================
    let mut class_counts = [0usize; 4];
    let mut granular_counts = [0usize; 25];
    let mut unknown_count = 0usize;
    let mut lr_voting_queue = 0usize; // Let's track how many cells need LR physics!
    let total_cells = environment.gpu_nodes.len();

    for node in &environment.gpu_nodes {
        // --- BITWISE UNPACKING ---
        // Lower 16 bits
        let broad_id = node.broad_id;
        // Shift down 16 bits to get the Granular ID
        let granular_id = node.granular_id;

        // Track the Broad Structure
        match broad_id {
            0 => class_counts[0] += 1, // Endothelial
            1 => class_counts[1] += 1, // Epithelial
            2 => class_counts[2] += 1, // Immune/Fluid
            3 => class_counts[3] += 1, // Mesenchymal
            4 => unknown_count += 1,   // Low Confidence Broad
            _ => unknown_count += 1,   // Catch-all
        }

        if granular_id < 25 {
            granular_counts[granular_id as usize] += 1;
        }
        // Track how many structurally valid cells need LR Voting
        if broad_id != 4 && granular_id == 25 {
            lr_voting_queue += 1;
        }
    }

    println!("\n✅ Classification Complete! Broad Identity Distribution:");
    println!("------------------------------------------------------");

    let class_names = ["Endothelial", "Epithelial", "Immune/Fluid", "Mesenchymal"];

    for i in 0..4 {
        let count = class_counts[i];
        let pct = (count as f64 / total_cells as f64) * 100.0;
        println!("  - {:<15} : {:>8} cells ({:>5.2}%)", class_names[i], count, pct);
    }

    let unknown_pct = (unknown_count as f64 / total_cells as f64) * 100.0;
    println!("  - {:<15} : {:>8} cells ({:>5.2}%)", "Unknown (4)", unknown_count, unknown_pct);
    println!("------------------------------------------------------");
    println!("  - {:<15} : {:>8} total", "Total Processed", total_cells);

    // Put this right below your Broad distribution printout
    println!("\n✅ Granular Identity Distribution (High Confidence):");
    println!("------------------------------------------------------");

    // Since you already loaded map_book.granular_map, you can iterate through it!
    // Note: Depending on how your map is structured, you might need to invert it
    // (from String->ID to ID->String) or just iterate up to 25.

    for i in 0..25 {
        let count = granular_counts[i];
        if count > 0 { // Only print classes that actually exist in this slice
            // If you have an array of granular names, use it here instead of format!
            println!("  - Class {:<13} : {:>8} cells", granular_class[i], count);
        }
    }
    println!("------------------------------------------------------");

    // Print out the physics queue!
    println!("\n⚛️ Physics Engine Triggers:");
    println!("  - LR Voting Queue : {:>8} cells require neighbor resolution", lr_voting_queue);

    println!("Saving test.bin of tensors boi ;)");
    // 2. Pass the resolved data to your batched saver
    // ground_truth_tensor is Vec<Vec<f32>>, so &ground_truth_tensor is &[Vec<f32>]
    save_tensor_to_bin("test.bin", &ground_truth_tensor)?;

    /*
    // ========================================================
    //  3. GPU! Score ground truth to generate observation space
    // ========================================================




    // 4. Feed the tensor directly into your SDE Generator model
    // let synthetic_tissue = generator_model.forward(obs_tensor);

    // ========================================================
    // 4. Train
    // ========================================================
    // 💪 training regimen for geometry model
     */

    Ok(())
}

pub async fn run_validation_test() -> anyhow::Result<()> {
    println!("🧪 Booting 4-Cell GPU Unit Test...");

    // 1. Manually construct our custom cells
    // Remember: packed_ids holds the mechanical_id (Broad ID) in the lower 16 bits.
    let custom_cells = vec![
        CellNode { pos: Coord{ x: 10.0, y: 10.0, z: 0.0 }, broad_id: 1, ..Default::default() }, // Type 1
        CellNode { pos: Coord{ x: 11.0, y: 11.0, z: 0.0 }, broad_id: 1, ..Default::default() }, // Type 1 (Touching Cell 0)
        CellNode { pos: Coord{ x: 10.0, y: 20.0, z: 0.0 }, broad_id: 2, ..Default::default() }, // Type 2 (10um away)
        CellNode { pos: Coord{ x: 140.0, y: 140.0, z: 0.0 }, broad_id: 2, ..Default::default() },// Type 2 (Far corner)
    ];

    // 2. Define a single 150um tile
    let params = SimParams {
        grid_width: 1,
        grid_height: 1,
        tile_size: 150.0,
        epoch: 0,
    };

    // // 3. Run the GPU Engine directly
    let mut wgpu_engine = NativeWgpuEngine::new(params).await;



    let (gpu_cells, offsets, counts) = NativeWgpuEngine::prepare_compute_buffers(&custom_cells, &params);
    //
    wgpu_engine.write_buffers(&gpu_cells, &offsets, &counts);
    wgpu_engine.dispatch_observation_shader();
    //
    // // 4. Read the Tensor Output
    let tensor = wgpu_engine.read_observation_buffer().await;
    //
    // // 5. Parse and Inspect the Data
    inspect_tile_tensor(&tensor[0], 0);

    Ok(())
}


fn inspect_tile_tensor(tensor: &[f32], tile_idx: usize) {
    let start = tile_idx * 184;
    let end = start + 184;

    if tensor.len() < end {
        println!("❌ Tensor is too small! Length: {}", tensor.len());
        return;
    }

    let tile_data = &tensor[start..end];

    let ripley = &tile_data[0..20];
    let enrichment = &tile_data[20..56];
    let halo = &tile_data[56..184];

    println!("\n========================================");
    println!("🔍 INSPECTING TILE {}", tile_idx);
    println!("========================================\n");

    // 1. Check for NaNs (The Silent Killer)
    for (i, &val) in tile_data.iter().enumerate() {
        if val.is_nan() || val.is_infinite() {
            println!("🔥 FATAL MATH ERROR: Found {} at index {}", val, i);
            return;
        }
    }

    // 2. Print Ripley's L-Curve (20 bins)
    println!("📈 Ripley's L-Curve (0 to 30um):");
    println!("{:?}", ripley);

    // 3. Print Enrichment Matrix (6x6)
    println!("\n🧬 Enrichment Adjacency Matrix:");
    for row in 0..6 {
        let row_start = row * 6;
        let row_end = row_start + 6;
        println!("{:?}", &enrichment[row_start..row_end]);
    }

    // 4. Print Halo Map Summaries
    println!("\n🛡️ Halo Boundary Pressure:");
    let top = halo[0..32].iter().sum::<f32>();
    let right = halo[32..64].iter().sum::<f32>();
    let bottom = halo[64..96].iter().sum::<f32>();
    let left = halo[96..128].iter().sum::<f32>();

    println!("Top: {}, Right: {}, Bottom: {}, Left: {}", top, right, bottom, left);
    println!("========================================\n");
}


/* use candle_core::{Tensor, Device};
// use candle_nn::{AdamW, Optimizer};
//
// #[tokio::main]
// async fn train_synthetic_tissue_generator() -> anyhow::Result<()>
// {
//     // ==========================================
//     // 1. LOAD BRAIN & STATE
//     // ==========================================
//     let mut varmap = VarMap::new();
//     let candle_device = Device::new_metal(0).unwrap_or(Device::Cpu);
//
//     let weights_path = "generator_brain.safetensors";
//     let state_path = "training_state.json";
//
//     let mut state = TrainingState::default();
//
//     if std::path::Path::new(weights_path).exists() {
//         println!("🧠 Loading existing brain from disk...");
//         varmap.load(weights_path)?;
//
//         // Load the sidecar JSON
//         if let Ok(json_data) = fs::read_to_string(state_path) {
//             state = serde_json::from_str(&json_data)?;
//             println!("📅 Resuming at Epoch {} (Last Target: {})", state.epoch, state.last_target_file);
//         }
//     } else {
//         println!("🌱 Initializing fresh brain...");
//     }
//
//     let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &candle_device);
//     let mut policy_network = MorphologyBrain::new(vb);
//     let mut optimizer = AdamW::new(policy_network.parameters(), 1e-4)?;
//
//     // ==========================================
//     // 2. THE CONTINUOUS LEARNING LOOP
//     // ==========================================
//     let target_files = get_all_target_files("./target_bank/");
//
//     // Notice we start the loop at `state.epoch` instead of 0
//     for epoch in state.epoch..MAX_EPOCHS {
//
//         // --- A. TARGET SELECTION & CONFIDENCE ---
//         let active_target_file = target_files.choose(&mut rand::thread_rng()).unwrap();
//         let is_he_slide = active_target_file.contains("he_");
//
//         // Here is the 30% confidence shield for H&E
//         let confidence_scalar = if is_he_slide { 0.3_f32 } else { 1.0_f32 };
//
//         // Update state tracking
//         state.epoch = epoch;
//         state.last_target_file = active_target_file.clone();
//
//         // Load the bytes and instantly swap the WGPU VRAM Buffer
//         let target_data = load_target_from_disk(&active_target_file);
//         wgpu_env.queue.write_buffer(&wgpu_env.ground_truth_buffer, 0, bytemuck::cast_slice(&target_data));
//
//         // --- B. FORWARD PASS (Candle) ---
//         // ... (#[derive(Default)]
//
//  Get states, run policy network, get flat actions) ...
//
//         // --- C. PHYSICS ENGINE (WGPU) ---
//         // ... (Upload actions, run generation pass, run #[derive(Default)]
// observation pass) ...
//
//         // --- D. REWARD & LOSS ---
//         // ... (Read raw errors, apply curriculum weights to get standard reward) ...
//
//         let normalized_rewards = normalize_batch(&rewards);
//         let advantages = calculate_advantages(&normalized_rewards);
//         let base_loss = calculate_ppo_loss(log_probabilities, advantages, 0.20);
//
//         // --- E. THE CONFIDENCE MODIFIER ---
//         // Multiply the entire loss tensor by our confidence scalar.
//         // If H&E, the gradients are crushed to 30% strength!
//         let final_scaled_loss = (base_loss * confidence_scalar as f64)?;
//
//         optimizer.backward_step(&final_scaled_loss)?;
//
//         // --- F. SAVING CHECKPOINTS ---
//         if epoch % 50 == 0 {
//             println!("💾 Saving Checkpoint (Epoch {})...", epoch);
//             varmap.save(weights_path)?;
//
//             // Save the sidecar state
//             let state_json = serde_json::to_string_pretty(&state)?;
//             fs::write(state_path, state_json)?;
//         }
//     }
//
//     Ok(())
// }


// #[repr(C)]
// #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
// pub struct SimParams {
//     pub grid_width: u32,
//     pub grid_height: u32,
//     pub tile_size: f32,
//     pub _pad: u32,
// }
// pub async fn extract_ground_truth(xenium_path: &str, tile_size: f32) -> SimParams {
//     let real_tissue = load_xenium_data(xenium_path);
//
//     // 1. Calculate dynamic bounding box
//     let min_x = real_tissue.iter().map(|c| c.x).reduce(f32::min).unwrap();
//     let max_x = real_tissue.iter().map(|c| c.x).reduce(f32::max).unwrap();
//     let min_y = real_tissue.iter().map(|c| c.y).reduce(f32::min).unwrap();
//     let max_y = real_tissue.iter().map(|c| c.y).reduce(f32::max).unwrap();
//
//     let grid_width = ((max_x - min_x) / tile_size).ceil() as u32;
//     let grid_height = ((max_y - min_y) / tile_size).ceil() as u32;
//     let num_tiles = (grid_width * grid_height) as usize;
//
//     let params = SimParams { grid_width, grid_height, tile_size, _pad: 0 };
//
//     // 2. Dispatch native WGPU observation shader
//     let wgpu_engine = NativeWgpuEngine::new(&params).await;
//     wgpu_engine.write_cells(real_tissue);
//     wgpu_engine.dispatch_observation_shader();
//
//     // 3. Save to disk
//     let ground_truth_tensor: Vec<f32> = wgpu_engine.read_observation_buffer().await;
//     save_to_disk("ground_truth_target.bin", ground_truth_tensor);
//
//     println!("Extracted Ground Truth: {}x{} grid ({} tiles)", grid_width, grid_height, num_tiles);
//     params // Return params so the training loop knows the exact shape
// }
 */

// impl TissueSimulator {
/*     pub async fn run_step(&mut self, opcodes: Vec<f32>, epoch: u32) -> Vec<TileErrors> {
//         // 1. Update SimParams with the current epoch for any GPU-side curriculum/logging
//         self.params.epoch = epoch;
//         self.queue.write_buffer(&self.param_buffer, 0, bytemuck::bytes_of(&self.params));
//
//         // 2. Upload the new Opcodes
//         self.queue.write_buffer(&self.opcode_buffer, 0, bytemuck::cast_slice(&opcodes));
//
//         let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
//
//         // PASS 1: GENERATION (The Grammar)
//         // One thread per tile: reads 20 opcodes and updates N cells
//         {
//             let mut gpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
//             gpass.set_pipeline(&self.generation_pipeline);
//             gpass.set_bind_group(0, &self.generation_bind_group, &[]);
//             gpass.dispatch_workgroups(self.num_tiles_x, self.num_tiles_y, 1);
//         }
//
//         // PASS 2: OBSERVATION & ERROR CALCULATION
//         // Reads the new geometry and writes [CountErr, RipleyErr, EnrichErr, HaloErr]
//         {
//             let mut opass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
//             opass.set_pipeline(&self.observation_pipeline);
//             opass.set_bind_group(0, &self.observation_bind_group, &[]);
//             opass.dispatch_workgroups(self.num_tiles_x, self.num_tiles_y, 1);
//         }
//
//         // 3. Resolve the Copy for Double Buffering
//         encoder.copy_buffer_to_buffer(&self.error_storage_buffer, 0, &self.staging_buffer, 0, self.error_size);
//         self.queue.submit(Some(encoder.finish()));
//
//         // 4. Return the errors from the PREVIOUS pass (MapAsync)
//         self.read_errors_from_staging().await
//     }
// }
 */