use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

// --- 1. The Deserialization Structs ---
#[derive(Deserialize)]
struct SdePhysics {
    genes: Vec<String>,
    parameters: SdeParams,
}

#[derive(Deserialize)]
struct SdeParams {
    sigma2: Vec<f32>, // Spliced noise (the primary driver of phenotype variance)
}

#[derive(Deserialize)]
struct Interactome {
    interaction_rules: Vec<RuleInput>,
}

#[derive(Deserialize)]
struct RuleInput {
    source_broad: String,
    target_broad: String,
    ligand_gene: String,
    receptor_gene: String,
    curation_weight: f32,
    target_granular_vote: String,
}

// --- 2. The GPU Memory Structs (Must match WGSL exactly) ---
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RuleOffset {
    pub start_idx: u32,
    pub count: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuInteractomeRule {
    pub target_granular_id: u32,
    pub total_drift: f32,
    pub total_diffusion: f32,
    pub _pad: u32, // 16-byte alignment
}

// --- 3. The Compiler Function ---
pub fn compile_interactome_to_disk(maps: &MasterMaps) {
    // 1. Load JSONs
    let sde_data: SdePhysics = serde_json::from_reader(File::open("sde_physics_constants.json").unwrap()).unwrap();
    let int_data: Interactome = serde_json::from_reader(File::open("interactome.json").unwrap()).unwrap();

    // Create a fast lookup for gene noise: gene_name -> sigma2
    let mut gene_noise: HashMap<String, f32> = HashMap::new();
    for (i, gene) in sde_data.genes.iter().enumerate() {
        gene_noise.insert(gene.clone(), sde_data.parameters.sigma2[i]);
    }

    // Maps to convert strings to your hardcoded u32 IDs
    // let broad_map = HashMap::from([("Epithelial", 0), ("Endothelial", 1), ("Mesenchymal", 2), ("Immune", 3)]);
    // Assume granular_map contains "Basal Keratinocyte" -> 12, etc.
    // let granular_map: HashMap<String, u32> = load_granular_map();

    // Temporary nested map: [Broad_Pair_Index] -> { Target_Granular_ID -> (Sum_Drift, Sum_Diffusion) }
    let mut aggregated_rules: HashMap<u32, HashMap<u32, (f32, f32)>> = HashMap::new();

    // 2. Process and Aggregate
    for rule in int_data.interaction_rules {
        let src_id = *maps.broad_type_map.get(&rule.source_broad).expect("Unknown Broad Source!");
        let tgt_id = *maps.broad_type_map.get(&rule.target_broad).expect("Unknown Broad Target!");
        let pair_idx = (src_id * 4) + tgt_id; // 0 to 15 index

        let gran_id = *maps.granular_type_map.get(&rule.target_granular_vote).expect("Unknown Granular Target!");

        // Lookup the SDE noise. Default to 0.1 if gene is missing.
        let lig_sigma = gene_noise.get(&rule.ligand_gene).unwrap_or(&0.1);
        let rec_sigma = gene_noise.get(&rule.receptor_gene).unwrap_or(&0.1);

        // Variance is standard deviation squared. We average the variance of the ligand and receptor.
        let rule_diffusion = (lig_sigma.powi(2) + rec_sigma.powi(2)) / 2.0;

        // Accumulate the math!
        let entry = aggregated_rules.entry(pair_idx).or_default().entry(gran_id).or_insert((0.0, 0.0));
        entry.0 += rule.curation_weight; // Add to total drift
        entry.1 += rule_diffusion;       // Add to total diffusion
    }

    // 3. Flatten for the GPU
    let mut final_offsets = vec![RuleOffset { start_idx: 0, count: 0 }; 16];
    let mut final_rules = Vec::new();
    let mut current_idx = 0;

    for pair_idx in 0..16u32 {
        if let Some(gran_map) = aggregated_rules.get(&pair_idx) {
            final_offsets[pair_idx as usize] = RuleOffset { start_idx: current_idx, count: gran_map.len() as u32 };

            for (&gran_id, &(drift, diffusion)) in gran_map.iter() {
                final_rules.push(GpuInteractomeRule {
                    target_granular_id: gran_id,
                    total_drift: drift,
                    total_diffusion: diffusion,
                    _pad: 0,
                });
                current_idx += 1;
            }
        }
    }

    // 4. Write pure bytes to disk
    let mut file = File::create("gpu_interactome.bin").unwrap();
    file.write_all(bytemuck::cast_slice(&final_offsets)).unwrap();
    file.write_all(bytemuck::cast_slice(&final_rules)).unwrap();
}


// use crate::trainer::TargetOrientation;
//
// #[tokio::main]
// async fn main() -> anyhow::Result<()> {
//     // ... [Setup Candle Device, Agent, etc.] ...
//
//     // 1. You explicitly annotate your target files based on lab knowledge!
//     let training_targets = vec![
//         // The H&E or cross-section slides are Vertical
//         ("data/targets/skin_cross_section_1.bin".to_string(), TargetOrientation::Vertical),
//         ("data/targets/skin_cross_section_2.bin".to_string(), TargetOrientation::Vertical),
//
//         // The spatial transcriptomic (Xenium) slides cut from the top down are En Face
//         ("data/targets/xenium_basal_layer.bin".to_string(), TargetOrientation::EnFace),
//         ("data/targets/xenium_spinous_layer.bin".to_string(), TargetOrientation::EnFace),
//     ];
//
//     // 2. Pass the annotated vector into your training loop
//     let fitness = train_agent_for_generation(
//         &mut agent,
//         1000, // epochs
//         &training_targets, // <-- This is where the match statement gets it!
//         &candle_device
//     ).await?;
//
//     println!("Training Complete. Agent Fitness: {}", fitness);
//     Ok(())
// }