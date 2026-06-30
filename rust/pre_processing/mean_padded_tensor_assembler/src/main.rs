use std::collections::HashMap;

// 1. Loaded once when the Wasm app starts
let model_genes: Vec<String> = load_model_genes(); // length: 768
let model_means: Vec<f32> = load_model_means();    // length: 768

// 2. Build the "Hardware to Model" map when the user drops the H5 file
// This maps the H5 column index (e.g., 0 to 478) to the Burn Tensor index (0 to 767)
let mut h5_to_tensor_map: HashMap<usize, usize> = HashMap::new();
let hardware_genes = read_h5_gene_names(&h5_file); // length: 479

for (h5_idx, gene) in hardware_genes.iter().enumerate() {
if let Some(tensor_idx) = model_genes.iter().position(|g| g == gene) {
h5_to_tensor_map.insert(h5_idx, tensor_idx);
}
}

// 3. THE STREAMING LOOP (Chunk size = 50,000 cells)
for local_idx in 0..current_chunk_size {

// Create the base array filled with the "safe" means, NOT zeros
let mut cell_tensor = model_means.clone();

// Unpack the CSR counts for this specific cell
for ptr in start_ptr..end_ptr {
let h5_idx = indices[ptr] as usize;
let count = counts[ptr] as f32; // Assuming you apply log1p normalization here if needed

// If this hardware gene exists in our universal model, overwrite the mean
if let Some(&tensor_idx) = h5_to_tensor_map.get(&h5_idx) {
cell_tensor[tensor_idx] = count;
}
}

// cell_tensor is now exactly 768 elements long!
// 479 slots contain real Xenium data.
// 289 slots contain the safe training mean.

// Push to the flat chunk buffer to send to Burn via WebGPU
chunk_buffer.extend_from_slice(&cell_tensor);
}