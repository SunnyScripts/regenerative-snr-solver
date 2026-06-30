use bytemuck::Zeroable;
use std::collections::HashMap;
use crate::data_io::{XeniumData, SpatialMetadata, self, load_json};
use crate::rna_dual_head_cell_type_classifier::CellClassifier;
use serde::Deserialize;

use shared_biology::{CellNode, SimParams, Coord};
use generator::NativeWgpuEngine;
use generator::types::{BoundingBox, EngineResources};
use glam::Vec3;
use std::collections::VecDeque;
// use ndarray::array;


// use crate::wire_modes::{wire_graph, WiringMode, WiringResult};
const CHUNK_SIZE: usize = 4096; //or 8192

#[derive(Deserialize, Debug)]
pub struct FovMetadata {
    pub height: f32,
    pub width: f32,
    pub x: f32,
    pub y: f32,
}

#[derive(Deserialize, Debug)]
pub struct FovLocationsFile {
    pub fov_locations: HashMap<String, FovMetadata>,
    pub units: String,
}



pub struct GroundTruthEnvironment {
    // ==========================================
    // PRIVATE STATE: Only accessible inside this struct
    // ==========================================
    xenium_data: XeniumData,
    spatial_metadata: Vec<SpatialMetadata>,
    // model_means: Vec<f32>,
    // interactome: FastInteractome,
    // interaction_radius: f32,
    //
    // // ==========================================
    // // PUBLIC OUTPUTS: Accessible after calling `identify()`
    // // ==========================================
    pub gpu_nodes: Vec<CellNode>,
    pub sim_params: SimParams,
    pub bounding_box: BoundingBox,

    pub class_counts: [usize; 4],
    pub granular_counts: [usize; 25],
    pub unknown_count: usize,
    pub lr_voting_queue: usize,

    pub spatial_grid: HashMap<u32, Vec<usize>>,

    // pub mrf_priors: Vec<HashMap<u32, f32>>,
}

impl GroundTruthEnvironment
{
    /// The "Constructor". Takes the paths, loads everything, and saves it privately.
    pub fn new(
        h5_path: &str,
        metadata_path: &str,
        // morphology_fov_locations_path: &str,
        // broad_map_path: &str,
        // granular_map_path: &str,
        // model_means_path: &str,
        // interactome_path: &str,
        interaction_radius: f32,
    ) -> anyhow::Result<Self> {
        println!("📂 Initializing GroundTruthEnvironment...");
        // 1. Load the core spatial and expression data
        let xenium_data = XeniumData::load_from_h5(h5_path)?;
        let (spatial_metadata, bounding_box)  = data_io::extract_spatial_data(metadata_path)?;

        let grid_width = ( (bounding_box.max_x - bounding_box.min_x).abs() / interaction_radius).ceil() as u32;
        let grid_height = ((bounding_box.max_y - bounding_box.min_y).abs() / interaction_radius).ceil() as u32;

        /* if spatial_metadata.len() != xenium_data.dense_matrix.nrows() {
        //     return Err(anyhow::anyhow!("❌ Mismatch! Metadata and Count Matrix have different cell counts."));
        // }
        // 2. Load the maps and means
        // let map_book = data_loader::build_maps(broad_map_path, granular_map_path)?;
        // let model_means: Vec<f32> = load_json(model_means_path)?;
        //
        // let interactome = FastInteractome::load_from_json(
        //     interactome_path,
        //     &xenium_data.gene_to_idx,
        //     &map_book.broad_map,
        //     &map_book.granular_map,
        // )?;
         */

        println!("✅ Environment successfully loaded into memory.");
        Ok(Self {
            xenium_data,
            spatial_metadata,
            // model_means,
            // interactome,
            // interaction_radius,
            //
            // // Initialize outputs as empty vectors until `identify` is called
            gpu_nodes: Vec::new(),
            sim_params: SimParams {
                grid_width,
                grid_height,
                grid_depth: 1,
                tile_size: interaction_radius,
                epoch: 0,
                current_cell_count: 0,
                _pad: [0; 6],
            },
            bounding_box,
            // mrf_priors: Vec::new(),
            class_counts: [0; 4],
            granular_counts: [0; 25],
            unknown_count: 0,
            lr_voting_queue: 0,
            spatial_grid: HashMap::new(),
        })
    }

    /// Private internal utility for the alignment map
    fn build_alignment_map(
        xenium_symbols: &[String],
        symbol_map_path: &str,
        expected_entrez_path: &str,
    ) -> anyhow::Result<Vec<Option<usize>>> {
        println!("⚙️ Compiling Feature Alignment Map...");

        let symbol_to_entrez: HashMap<String, u32> = load_json::<HashMap<String, u32>>(symbol_map_path)?
            .into_iter()
            .map(|(k, v)| (k.trim().to_uppercase(), v)) // Trim and uppercase
            .collect();
        let model_expected_entrez_ids: Vec<u32> = load_json(expected_entrez_path)?;

        let model_lookup: HashMap<u32, usize> = model_expected_entrez_ids
            .iter()
            .enumerate()
            .map(|(idx, &entrez_id)| (entrez_id, idx))
            .collect();

        let mut mapped_xenium_genes = 0;

        let alignment_map: Vec<_> = xenium_symbols.iter().map(|symbol| {
            // Quick trick: if case sensitivity is the issue, you could try
            // matching on symbol.to_uppercase() here and in the JSON map.
            if let Some(entrez_id) = symbol_to_entrez.get(symbol) {
                if let Some(&model_idx) = model_lookup.get(entrez_id) {
                    mapped_xenium_genes += 1;
                    return Some(model_idx);
                }
            }
            None
        }).collect();

        let mapped_indices: std::collections::HashSet<usize> = alignment_map.iter().filter_map(|&x| x).collect();

        println!("🔍 Missing Model Genes:");
        for (entrez_id, &model_idx) in &model_lookup {
            if !mapped_indices.contains(&model_idx) {
                // Find what symbol the JSON *wanted* this Entrez ID to be
                let expected_symbol = symbol_to_entrez.iter()
                    .find(|&(_, &e)| e == *entrez_id)
                    .map(|(s, _)| s.as_str())
                    .unwrap_or("Unknown Symbol");

                println!("   - Missing Entrez ID: {} (Expected Symbol: {})", entrez_id, expected_symbol);
            }
        }

        println!("📊 Alignment Report:");
        println!("   - Xenium Panel Size: {}", xenium_symbols.len());
        println!("   - Model Expected:    {}", model_expected_entrez_ids.len());
        println!("   - Successfully Mapped: {} / {}", mapped_xenium_genes, model_expected_entrez_ids.len());

        Ok(alignment_map)
    }

    // Executes the Deep Learning inference and Topography wiring.
    // Mutates `self` to populate the public `gpu_nodes` and `mrf_priors`.
    pub fn identify_broad_cell_types(
        &mut self,
        classifier: &mut CellClassifier,
        symbol_map_path: &str,
        expected_entrez_path: &str
    ) -> anyhow::Result<()>
    {
        println!("🧠 Running Deep Learning Classifier (Broad Identity)...");

        let alignment_map = Self::build_alignment_map(
            &self.xenium_data.gene_names,
            symbol_map_path,
            expected_entrez_path
        )?;

        let total_cells = self.spatial_metadata.len();

        // Allocate the nodes array on `self`
        self.gpu_nodes = vec![CellNode::zeroed(); total_cells];

        // ========================================================
        // 0. INITIALIZE STAT TRACKERS
        // ========================================================
        let mut class_counts = [0usize; 4];
        let mut granular_counts = [0usize; 25];
        let mut unknown_count = 0usize;
        let mut lr_voting_queue = 0usize;

        // ========================================================
        // 1. CHUNKING & INFERENCE
        // ========================================================
        for chunk_start in (0..total_cells).step_by(CHUNK_SIZE)
        {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(total_cells);
            let current_chunk_size = chunk_end - chunk_start;

            let mut aligned_chunk = ndarray::Array2::<f32>::zeros((current_chunk_size, 473));

            for local_idx in 0..current_chunk_size
            {
                let global_cell_idx = chunk_start + local_idx;

                let smd = &self.spatial_metadata[global_cell_idx];
                self.gpu_nodes[global_cell_idx].pos = Coord{
                    x: smd.x, y: smd.y, z: 0.0
                };
                self.gpu_nodes[global_cell_idx].area = smd.area;

                let ptr_start = self.xenium_data.indptr[global_cell_idx];
                let ptr_end = self.xenium_data.indptr[global_cell_idx + 1];

                let mut cell_total_counts = 0.0;
                for ptr in ptr_start..ptr_end {
                    cell_total_counts += self.xenium_data.sparse_counts[ptr] as f32;
                }

                let target_depth = 369.7979f32;
                let scale_factor = if cell_total_counts > 0.0 { target_depth / cell_total_counts } else { 0.0 };

                for ptr in ptr_start..ptr_end
                {
                    let xenium_gene_idx = self.xenium_data.indices[ptr];
                    if let Some(model_idx) = alignment_map[xenium_gene_idx]
                    {
                        let raw_count = self.xenium_data.sparse_counts[ptr] as f32;
                        aligned_chunk[[local_idx, model_idx]] = raw_count * scale_factor;
                    }
                }
            }

            let chunk_predictions = classifier.predict_batch(aligned_chunk)?;

            for local_idx in 0..current_chunk_size {
                let global_cell_idx = chunk_start + local_idx;
                let pred = &chunk_predictions[local_idx];

                let mut final_broad = pred.broad_class;
                let mut final_granular = pred.granular_class;

                // --- THE DUAL-GATE FILTER ---
                if pred.broad_confidence < 0.85 {
                    final_broad = 4;
                    final_granular = 24;
                } else if pred.granular_confidence < 0.70 {
                    final_granular = 24;
                }

                // --- DIRECT ASSIGNMENT ---
                self.gpu_nodes[global_cell_idx].broad_id = final_broad;
                self.gpu_nodes[global_cell_idx].granular_id = final_granular;

                // --- TRACK STATS ON THE FLY ---
                match final_broad {
                    0..=3 => class_counts[final_broad as usize] += 1,
                    _ => unknown_count += 1, // Catches 4 and any anomalous bounds
                }

                // ACCUMULATE AREA (Only for confidently classified cells)
                if final_granular < 24 {
                    granular_counts[final_granular as usize] += 1;

                    // Add the actual Xenium area to the global sum
                    // self.global_granular_area_sums[final_granular as usize] += smd.area as f64;
                    // self.global_granular_counts[final_granular as usize] += 1;
                }

                if final_granular < 25 {
                    granular_counts[final_granular as usize] += 1;
                }

                if final_broad != 4 && final_granular == 24 {
                    lr_voting_queue += 1;
                }
            }
        }

        // ========================================================
        // 2. SAVE STATS TO SELF
        // ========================================================
        self.class_counts = class_counts;
        self.granular_counts = granular_counts;
        self.unknown_count = unknown_count;
        self.lr_voting_queue = lr_voting_queue;

        // ========================================================
        // 3. CONSOLE REPORTING
        // ========================================================
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

        println!("\n✅ Granular Identity Distribution (High Confidence):");
        println!("------------------------------------------------------");

        // NOTE: Replace `granular_class` array with however you are mapping IDs back to Strings
        for i in 0..25 {
            let count = granular_counts[i];
            if count > 0 {
                // Using placeholder `Class X` here since the granular name array wasn't in scope
                println!("  - Class {:<13} : {:>8} cells", i, count);
            }
        }
        println!("------------------------------------------------------");

        println!("\n⚛️ Physics Engine Triggers:");
        println!("  - LR Voting Queue : {:>8} cells require neighbor resolution", lr_voting_queue);

        Ok(())
    }

    // Calculates Geometry Signatures and stores them directly into the VRAM-bound `gpu_nodes`
    // pub fn compute_geometric_signatures(&mut self) -> anyhow::Result<()>
    // {
    //     println!("📐 Computing Geometric Signatures (Eccentricity, Angle, SDF)...");
    //
    //     let total_cells = self.spatial_metadata.len();
    //     if self.gpu_nodes.len() != total_cells {
    //         anyhow::bail!("gpu_nodes must be allocated before computing geometry!");
    //     }
    //
    //     // ========================================================
    //     // PART 1: ECCENTRICITY & POLARITY ANGLE (Covariance Math)
    //     // ========================================================
    //     for i in 0..total_cells {
    //         let smd = &self.spatial_metadata[i];
    //
    //         // Assume smd.polygon is a Vec<[f32; 2]> of the boundary vertices
    //         let pts = &smd.polygon;
    //         let n = pts.len() as f32;
    //
    //         if n < 3.0 {
    //             // Fallback for malformed cells
    //             self.gpu_nodes[i].phase_buffer_1 = 1.0; // Perfect circle
    //             self.gpu_nodes[i].phase_buffer_2 = 0.0; // 0 radians
    //             continue;
    //         }
    //
    //         // 1. Find the Centroid
    //         let mut cx = 0.0;
    //         let mut cy = 0.0;
    //         for p in pts {
    //             cx += p[0];
    //             cy += p[1];
    //         }
    //         cx /= n;
    //         cy /= n;
    //
    //         // 2. Calculate the Covariance Matrix (Mxx, Myy, Mxy)
    //         let mut mxx = 0.0;
    //         let mut myy = 0.0;
    //         let mut mxy = 0.0;
    //
    //         for p in pts {
    //             let dx = p[0] - cx;
    //             let dy = p[1] - cy;
    //             mxx += dx * dx;
    //             myy += dy * dy;
    //             mxy += dx * dy;
    //         }
    //         mxx /= n;
    //         myy /= n;
    //         mxy /= n;
    //
    //         // 3. Calculate Eigenvalues
    //         let trace = mxx + myy;
    //         let det = (mxx * myy) - (mxy * mxy);
    //
    //         // Quadratic formula for eigenvalues
    //         let root = ((trace * trace) - 4.0 * det).max(0.0).sqrt();
    //         let lambda1 = (trace + root) / 2.0; // Major axis variance
    //         let lambda2 = (trace - root) / 2.0; // Minor axis variance
    //
    //         // 4. Calculate Eccentricity (1.0 = Circle, higher = more oval)
    //         // Using Aspect Ratio formulation for simplicity in WGSL later
    //         let eccentricity = if lambda2 > 0.0001 {
    //             (lambda1 / lambda2).sqrt()
    //         } else {
    //             1.0
    //         };
    //
    //         // 5. Calculate Major Axis Angle (in Radians)
    //         let angle = 0.5 * (2.0 * mxy).atan2(mxx - myy);
    //
    //         // 6. Save to the GPU buffers!
    //         self.gpu_nodes[i].phase_buffer_1 = eccentricity;
    //         self.gpu_nodes[i].phase_buffer_2 = angle;
    //     }
    //
    //     // ========================================================
    //     // PART 2: SDF TO VOID (Rasterized Multi-Source BFS)
    //     // ========================================================
    //     let grid_res = 2.0; // 2 microns per pixel (Balance of speed and accuracy)
    //
    //     // Calculate Grid Dimensions
    //     let width = ((self.bounding_box.max_x - self.bounding_box.min_x) / grid_res).ceil() as usize + 10;
    //     let height = ((self.bounding_box.max_y - self.bounding_box.min_y) / grid_res).ceil() as usize + 10;
    //
    //     let mut grid = vec![false; width * height]; // false = VOID, true = TISSUE
    //
    //     // 1. Burn the cells into the grid (Rasterization)
    //     for i in 0..total_cells {
    //         let smd = &self.spatial_metadata[i];
    //
    //         // Convert physical coordinates to grid indices
    //         let gx = ((smd.x - self.bounding_box.min_x) / grid_res) as isize;
    //         let gy = ((smd.y - self.bounding_box.min_y) / grid_res) as isize;
    //
    //         // Calculate a grid radius based on cell area.
    //         // We pad it slightly (+1) so neighboring cells merge into a solid tissue block.
    //         let radius = (smd.area / 3.14159).sqrt() / grid_res;
    //         let r_int = radius.ceil() as isize + 1;
    //
    //         // Draw a crude filled circle into the grid
    //         for dy in -r_int..=r_int {
    //             for dx in -r_int..=r_int {
    //                 if dx * dx + dy * dy <= r_int * r_int {
    //                     let nx = gx + dx;
    //                     let ny = gy + dy;
    //                     if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
    //                         grid[(ny as usize) * width + (nx as usize)] = true;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //
    //     // 2. Initialize the Distance Field
    //     // Distance is initialized to Infinity. Voids (false) start at 0.0.
    //     let mut sdf = vec![f32::MAX; width * height];
    //     let mut queue = VecDeque::new();
    //
    //     for y in 0..height {
    //         for x in 0..width {
    //             let idx = y * width + x;
    //             if !grid[idx] { // If this is a Void...
    //                 sdf[idx] = 0.0;
    //                 queue.push_back((x, y));
    //             }
    //         }
    //     }
    //
    //     // 3. Run the Multi-Source BFS (Manhattan expansion)
    //     let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
    //
    //     while let Some((x, y)) = queue.pop_front() {
    //         let current_dist = sdf[y * width + x];
    //
    //         for (dx, dy) in directions.iter() {
    //             let nx = x as isize + dx;
    //             let ny = y as isize + dy;
    //
    //             if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
    //                 let n_idx = (ny as usize) * width + (nx as usize);
    //
    //                 // If we find a shorter path, update it and queue it
    //                 if sdf[n_idx] > current_dist + grid_res {
    //                     sdf[n_idx] = current_dist + grid_res;
    //                     queue.push_back((nx as usize, ny as usize));
    //                 }
    //             }
    //         }
    //     }
    //
    //     // 4. Sample the SDF back to the Cells
    //     for i in 0..total_cells {
    //         let smd = &self.spatial_metadata[i];
    //
    //         let gx = ((smd.x - self.bounding_box.min_x) / grid_res).clamp(0.0, (width - 1) as f32) as usize;
    //         let gy = ((smd.y - self.bounding_box.min_y) / grid_res).clamp(0.0, (height - 1) as f32) as usize;
    //
    //         let distance_to_void = sdf[gy * width + gx];
    //
    //         self.gpu_nodes[i].phase_buffer_3 = distance_to_void;
    //     }
    //
    //     println!("✅ Geometric Signatures successfully written to VRAM buffers.");
    //     Ok(())
    // }

    pub async fn extract_observation_tensor(&self) -> anyhow::Result<Vec<Vec<f32>>>
    {
        // 1. Initialize a fresh GPU engine for this extraction
        // (Note: Adding '?' here in case WGPU fails to find a device)
        let mut wgpu_engine = NativeWgpuEngine::new(self.sim_params, EngineResources{..Default::default()}).await;

        // 2. Prepare the spatial hashing buffers
        let (gpu_cells, offsets, counts) = NativeWgpuEngine::prepare_compute_buffers(&self.gpu_nodes, &self.sim_params);

        // 3. Upload data to the M4 Pro GPU
        wgpu_engine.write_buffers(&gpu_cells, &offsets, &counts);

        // 4. Run the observation shader we just fixed
        wgpu_engine.dispatch_observation_shader();

        // 5. Read back the results (This is where the magic/waiting happens)
        let observations = wgpu_engine.read_observation_buffer().await;

        Ok(observations)
    }
    pub fn build_spatial_hash(&mut self) {
        println!("🗺️ Building CPU Spatial Hash for UI...");

        let mut grid: HashMap<u32, Vec<usize>> = HashMap::new();
        let bucket_size = self.sim_params.tile_size;
        let grid_width = self.sim_params.grid_width;

        for (idx, node) in self.gpu_nodes.iter().enumerate() {
            let bx = (node.pos.x / bucket_size).max(0.0) as u32;
            let by = (node.pos.y / bucket_size).max(0.0) as u32;
            let bucket_idx = bx + (by * grid_width);

            grid.entry(bucket_idx).or_default().push(idx);
        }

        self.spatial_grid = grid;
    }
    // pub fn export_ideal_areas(&self, output_path: &str) -> anyhow::Result<()> {
    //     println!("📊 Compiling Ground-Truth Granular Areas...");
    //
    //     // We only need 24 floats for the GPU (0 to 23)
    //     let mut ideal_areas = vec![0.0f32; 24];
    //
    //     for i in 0..24 {
    //         // let count = self.global_granular_counts[i];
    //         // let sum = self.global_granular_area_sums[i];
    //
    //         if count > 0 {
    //             ideal_areas[i] = (sum / count as f64) as f32;
    //             println!("  - Class {:<2}: Average Area = {:.2} µm²", i, ideal_areas[i]);
    //         } else {
    //             // Safety fallback if a class was completely missing from all 7 datasets
    //             ideal_areas[i] = 200.0;
    //             println!("  - Class {:<2}: Missing! Defaulting to 200.0 µm²", i);
    //         }
    //     }
    //
    //     // Your function expects a slice of Vecs: &[Vec<f32>]
    //     let batched_tensor = vec![ideal_areas];
    //
    //     // save_tensor_to_bin(output_path, &batched_tensor)?;
    //
    //     Ok(())
    // }
}

/* impl CellNode
// {
//     // ==========================================
//     // PHASE 1: TISSUE GENERATION
//     // ==========================================
//     #[inline] pub fn dir_x(&self) -> f32 { self.phase_buffer_1 }
//     #[inline] pub fn set_dir_x(&mut self, val: f32) { self.phase_buffer_1 = val; }
//
//     #[inline] pub fn dir_y(&self) -> f32 { self.phase_buffer_2 }
//     #[inline] pub fn set_dir_y(&mut self, val: f32) { self.phase_buffer_2 = val; }
//
//     #[inline] pub fn dir_z(&self) -> f32 { self.phase_buffer_3 }
//     #[inline] pub fn set_dir_z(&mut self, val: f32) { self.phase_buffer_3 = val; }
//
//     // ==========================================
//     // PHASE 2: BIOELECTRIC PHYSICS
//     // ==========================================
//     #[inline] pub fn ion_ca(&self) -> f32 { self.phase_buffer_1 }
//     #[inline] pub fn set_ion_ca(&mut self, val: f32) { self.phase_buffer_1 = val; }
//
//     #[inline] pub fn ion_cl(&self) -> f32 { self.phase_buffer_2 }
//     #[inline] pub fn set_ion_cl(&mut self, val: f32) { self.phase_buffer_2 = val; }
//
//     #[inline] pub fn gap_junction_state(&self) -> f32 { self.phase_buffer_3 }
//     #[inline] pub fn set_gap_junction_state(&mut self, val: f32) { self.phase_buffer_3 = val; }
// }
 */



// use std::collections::HashMap;
// use serde::Deserialize;
// use crate::graphics_engine::NativeWgpuEngine;

// A simple structure to hold our spatial bins
