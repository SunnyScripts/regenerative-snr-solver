use ndarray::Array2;
use std::collections::HashMap;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use std::fs;
use std::fs::File;
use std::io::Write;
use polars::prelude::*;
use anyhow::Result;

pub struct XeniumData {
    pub gene_names: Vec<String>,
    pub gene_to_idx: HashMap<String, usize>, // For O(1) lookups of "GJA1"
    pub indptr: Vec<usize>,                  // Cast to usize for safe indexing
    pub indices: Vec<usize>,                 // Cast to usize for safe indexing
    pub sparse_counts: Vec<f32>,
    pub dense_matrix: Array2<f32>,           // For the edge-wiring lookups
}

impl XeniumData
{
    pub fn load_from_h5(filepath: &str) -> Result<Self>
    {
        let h5_file = hdf5::File::open(filepath)?;
        let matrix_group = h5_file.group("matrix")?;

        // 1. DYNAMICALLY DETECT SHAPE
        // indptr length is always (N_cells + 1)
        let indptr_ds = matrix_group.dataset("indptr")?;
        let num_cells = indptr_ds.shape()[0] - 1;

        // features/name length is N_genes
        let features_group = matrix_group.group("features")?;
        let names_ds = features_group.dataset("name")?;
        let num_genes = names_ds.shape()[0];

        println!("📊 Detected Dataset Shape: {} cells x {} genes", num_cells, num_genes);

        // 2. Load Gene Names (Dynamic Allocation)
        let raw_records = names_ds.read_raw::<hdf5::types::FixedAscii<{ 23 }>>()?;
        let mut gene_names = Vec::with_capacity(num_genes);
        let mut gene_to_idx = HashMap::with_capacity(num_genes);

        for (i, fixed_str) in raw_records.into_iter().enumerate() {
            let clean_name = fixed_str.as_str().trim_matches(char::from(0)).trim().to_uppercase();
            gene_names.push(clean_name.clone());
            gene_to_idx.insert(clean_name, i);
        }

        // 3. Load Sparse Data
        let sparse_counts: Vec<f32> = matrix_group.dataset("data")?.read_raw()?;
        let indices: Vec<usize> = matrix_group.dataset("indices")?.read_raw::<i32>()?.into_iter().map(|x| x as usize).collect();
        let indptr: Vec<usize> = indptr_ds.read_raw::<i32>()?.into_iter().map(|x| x as usize).collect();

        // 4. DYNAMIC DENSE INFLATION
        // Now using the detected num_cells instead of a hardcoded 1.8M
        let mut dense_matrix = ndarray::Array2::<f32>::zeros((num_cells, num_genes));

        for cell_idx in 0..num_cells {
            let start = indptr[cell_idx];
            let end = indptr[cell_idx + 1];
            for ptr in start..end {
                dense_matrix[[cell_idx, indices[ptr]]] = sparse_counts[ptr];
            }
        }

        Ok(Self {
            gene_names,
            gene_to_idx,
            indptr,
            indices,
            sparse_counts,
            dense_matrix,
        })
    }
}

// We use the MapBook from your earlier data_loader.rs
pub struct MasterMaps {
    pub gene_map: HashMap<String, usize>,
    pub broad_type_map: HashMap<String, u32>,
    pub granular_type_map: HashMap<String, u32>,
}

impl MasterMaps {
    pub fn load() -> Self {
        // These JSONs are generated once by your Python preprocessing scripts
        let gene_map = load_json("master_gene_map.json").unwrap();
        let broad_type_map = load_json("master_broad_types.json").unwrap();
        let granular_type_map = load_json("master_granular_types.json").unwrap();

        Self { gene_map, broad_type_map, granular_type_map }
    }
}

// ==========================================
// 1. THE ENGINE STRUCT (Fast, Uses Integers)
// ==========================================
pub struct InteractionRule {
    pub ligand_idx: usize,
    pub receptor_idx: usize,
    pub curation_weight: f32,
    pub target_granular_id: u16,
    pub requires_contact: bool,
}

pub struct FastInteractome {
    pub grouped_rules: HashMap<(u8, u8), Vec<InteractionRule>>,
}

// ==========================================
// 2. THE PARSER STRUCT (Temporary, Uses Strings)
// ==========================================
#[derive(Deserialize)]
struct InteractomeFile {
    pub interaction_rules: Vec<RawJsonRule>,
}

#[derive(Deserialize)]
struct RawJsonRule {
    pub source_broad: String,
    pub target_broad: String,
    pub ligand_gene: String,
    pub receptor_gene: String,
    pub curation_weight: f32,
    pub target_granular_vote: String,
    pub requires_contact: bool,
}

// ==========================================
// 3. THE LOADER LOGIC
// ==========================================
impl FastInteractome {
    pub fn load_from_json(
        filepath: &str,
        gene_map: &HashMap<String, usize>,
        broad_map: &HashMap<String, u8>,
        granular_map: &HashMap<String, u16>
    ) -> Result<Self, Box<dyn std::error::Error>> {

        let data = std::fs::read_to_string(filepath)?;

        // Step 1: Catch the strings using the temporary Raw struct
        let parsed_file: InteractomeFile = serde_json::from_str(&data)?;

        let mut grouped_rules: HashMap<(u8, u8), Vec<InteractionRule>> = HashMap::new();

        // Step 2: Translate the strings into your engine's integers
        for raw_rule in parsed_file.interaction_rules {
            let ligand_idx = gene_map.get(&raw_rule.ligand_gene);
            let receptor_idx = gene_map.get(&raw_rule.receptor_gene);

            if let (Some(&l_idx), Some(&r_idx)) = (ligand_idx, receptor_idx) {

                let src_broad_id = broad_map.get(&raw_rule.source_broad).expect("Unknown Broad Class");
                let tgt_broad_id = broad_map.get(&raw_rule.target_broad).expect("Unknown Broad Class");
                let tgt_granular_id = granular_map.get(&raw_rule.target_granular_vote).expect("Unknown Granular Class");

                // Populate the actual engine struct
                let engine_rule = InteractionRule {
                    ligand_idx: l_idx,
                    receptor_idx: r_idx,
                    curation_weight: raw_rule.curation_weight,
                    target_granular_id: *tgt_granular_id,
                    requires_contact: raw_rule.requires_contact,
                };

                grouped_rules
                    .entry((*src_broad_id, *tgt_broad_id))
                    .or_insert_with(Vec::new)
                    .push(engine_rule);
            }
        }

        Ok(Self { grouped_rules })
    }
}
#[derive(Debug)]
pub struct SpatialMetadata {
    pub x: f32,
    pub y: f32,
    pub area: f32,
}

// pub struct BoundingBox {
//     pub min_x: f32,
//     pub max_x: f32,
//     pub min_y: f32,
//     pub max_y: f32,
// }

pub fn extract_spatial_data(parquet_path: &str) -> anyhow::Result<(Vec<SpatialMetadata>, generator::types::BoundingBox)> {
    println!("📖 Reading Spatial Parquet...");
    let file = std::fs::File::open(parquet_path)?;
    let df = ParquetReader::new(file).finish()?;

    // Safely cast columns to f32
    let x_series = df.column("x_centroid")?.cast(&DataType::Float32)?;
    let y_series = df.column("y_centroid")?.cast(&DataType::Float32)?;
    let area_series = df.column("cell_area")?.cast(&DataType::Float32)?;

    // ==========================================
    // 🔥 VECTORIZED BOUNDING BOX
    // ==========================================
    // Polars does this instantly across millions of rows
    let min_x = x_series.f32()?.min().unwrap_or(0.0);
    let max_x = x_series.f32()?.max().unwrap_or(0.0);
    let min_y = y_series.f32()?.min().unwrap_or(0.0);
    let max_y = y_series.f32()?.max().unwrap_or(0.0);

    // Get down to the raw arrays for iteration
    let x_array = x_series.f32()?;
    let y_array = y_series.f32()?;
    let area_array = area_series.f32()?;

    let mut spatial_data = Vec::with_capacity(df.height());

    // Iterate and zip them together
    for i in 0..df.height() {
        spatial_data.push(SpatialMetadata {
            x: x_array.get(i).unwrap_or(0.0),
            y: y_array.get(i).unwrap_or(0.0),
            area: area_array.get(i).unwrap_or(10.0),
        });
    }

    Ok((spatial_data, generator::types::BoundingBox{
        min_x, max_x, min_y, max_y
    }))
}


// 1. Define the MapBook struct to hold the dictionaries
pub struct MapBook {
    pub mechanical: HashMap<String, u8>,
    pub granular: HashMap<String, u16>,
}

// 2. Build both maps using your streaming reader technique
pub fn build_maps(broad_path: &str, granular_path: &str) -> anyhow::Result<MapBook> {
    println!("📖 Loading MapBook dictionaries...");

    // Load Broad Map (Mechanical)
    let b_file = std::fs::File::open(broad_path)?;
    let b_reader = std::io::BufReader::new(b_file);
    let broad_map: HashMap<String, u8> = serde_json::from_reader(b_reader)?;

    // Load Granular Map
    let g_file = std::fs::File::open(granular_path)?;
    let g_reader = std::io::BufReader::new(g_file);
    let granular_map: HashMap<String, u16> = serde_json::from_reader(g_reader)?;

    println!("✅ MapBook loaded: {} Broad classes, {} Granular classes.",
             broad_map.len(), granular_map.len());

    Ok(MapBook {
        mechanical: broad_map,
        granular: granular_map,
    })
}

// ==========================================
// 6. GLOBAL HELPER: Binary File I/O
// ==========================================
pub fn save_tensor_to_bin(file_path: &str, batched_tensor: &[Vec<f32>]) -> anyhow::Result<()> {
    let mut file = File::create(file_path)?;

    let mut total_floats = 0;
    for tile in batched_tensor {
        // Cast each tile individually and write it to the same file handle
        let byte_data: &[u8] = bytemuck::cast_slice(tile);
        file.write_all(byte_data)?;
        total_floats += tile.len();
    }

    println!("💾 Saved {} tensor floats (batched) to {}", total_floats, file_path);
    Ok(())
}

pub fn load_tensor_from_bin(file_path: &str) -> anyhow::Result<Vec<f32>> {
    let byte_data = std::fs::read(file_path)?;
    // cast back from u8 to f32
    let float_data: &[f32] = bytemuck::cast_slice(&byte_data);
    Ok(float_data.to_vec())
}

/// A multipurpose utility to load ANY JSON file into ANY standard Rust type.
pub fn load_json<T: DeserializeOwned>(path: &str) -> anyhow::Result<T> {
    println!("📖 Loading JSON from {}...", path);
    let file = fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    // Serde magically infers the type T based on what you assign it to later!
    let data: T = serde_json::from_reader(reader)?;
    Ok(data)
}