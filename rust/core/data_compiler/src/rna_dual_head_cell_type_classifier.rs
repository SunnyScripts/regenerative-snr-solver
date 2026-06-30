use std::fs;
use ndarray::ArrayView2;
use ndarray::Array2;
use ort::{
    execution_providers::CoreMLExecutionProvider,
    session::Session,
};
use ort::value::Value;

#[derive(Debug, Clone)]
pub struct DualPrediction {
    pub broad_class: u32,
    pub broad_confidence: f32,
    pub granular_class: u32,
    pub granular_confidence: f32,
}

#[derive(Debug)]
pub struct CellClassifier {
    session: Session,
    input_name: String,
    // output_name: String,
}

impl CellClassifier {
    pub fn load(onnx_path: &str) -> anyhow::Result<Self>
    {
        let absolute_path = fs::canonicalize(onnx_path)
            .map_err(|e| anyhow::anyhow!("Failed to resolve absolute path for model: {}. Error: {}", onnx_path, e))?;

        println!("Absolute Path:{}", absolute_path.display());

        // 1. Initialize the global environment
        // The .ok() ignores the Result, preventing a panic if you
        // instantiate the classifier more than once in the same app lifecycle.
        let _ = ort::init()
            .with_name("BioelectricVision")
            .commit();

        // 2. Build the session using rc.12 syntax
        // CoreML is now passed as an array to `with_execution_providers`
        let session = (|| {
            // Everything inside here can use ? normally
            Session::builder()?
                .with_execution_providers([
                    CoreMLExecutionProvider::default().build()
                ])?
                .commit_from_file(absolute_path)
        })()
            .map_err(|e| anyhow::anyhow!("ONNX Session Error: {}", e))?;

        println!("🔍 Inspecting ONNX Outputs:");
        for (i, out) in session.outputs().iter().enumerate() {
            println!("  -> Output [{}]: '{}'", i, out.name());
        }

        // 3. Dynamically extract the node names
        let input_name = session.inputs()[0].name().to_string();
        // let output_name = session.outputs()[0].name().to_string();

        // let output_name = session.outputs().iter()
        //     .find(|o| o.name() == "linear_3")
        //     .map(|o| o.name().clone().to_string())
        //     .unwrap_or_else(|| session.outputs()[1].name().clone().to_string());

        Ok(Self {
            session,
            input_name,
            // output_name,
        })
    }

    pub fn predict_batch(&mut self, input: Array2<f32>) -> anyhow::Result<Vec<DualPrediction>> {
        // --- DEBUG 1: Input Shape & Memory Layout ---
        println!("🐛 DEBUG: Input shape is {:?}", input.shape());

        if !input.is_standard_layout() {
            println!("⚠️ WARNING: Input array is not C-contiguous! ONNX will read this scrambled.");
        }

        // Check the actual gene values for Cell 0 (first 5 genes)
        let first_cell_genes: Vec<f32> = input.row(0).iter().take(5).copied().collect();
        println!("🐛 DEBUG: Cell 0, First 5 genes: {:?}", first_cell_genes);

        // Force standard layout (row-major) to guarantee ONNX reads it correctly
        let input_contiguous = input.as_standard_layout().into_owned();
        let input_tensor = ort::value::Value::from_array(input_contiguous)?;

        // Run inference
        let outputs = self.session.run(ort::inputs![self.input_name.as_str() => input_tensor])?;

        // Extract Broad Tensor
        let broad_val = outputs.get("logits_broad")
            .ok_or_else(|| anyhow::anyhow!("logits_broad output node missing"))?;
        let (b_shape, b_data) = broad_val.try_extract_tensor::<f32>()?;
        let broad_logits = ArrayView2::from_shape((b_shape[0] as usize, b_shape[1] as usize), b_data)?;

        // Extract Granular Tensor
        let gran_val = outputs.get("logits_granular")
            .ok_or_else(|| anyhow::anyhow!("logits_granular output node missing"))?;
        let (g_shape, g_data) = gran_val.try_extract_tensor::<f32>()?;
        let gran_logits = ArrayView2::from_shape((g_shape[0] as usize, g_shape[1] as usize), g_data)?;

        // --- DEBUG 2: Raw Logits Before Softmax ---
        let b_row_0: Vec<f32> = broad_logits.row(0).iter().copied().collect();
        println!("🐛 DEBUG: Cell 0 Raw Broad Logits: {:?}", b_row_0);

        let g_row_0: Vec<f32> = gran_logits.row(0).iter().take(5).copied().collect();
        println!("🐛 DEBUG: Cell 0 Raw Granular Logits (First 5): {:?}", g_row_0);

        // Apply Softmax & Prepare Outputs
        let mut batch_results = Vec::with_capacity(b_shape[0] as usize);

        for i in 0..b_shape[0] as usize {
            let b_row: Vec<f32> = broad_logits.row(i).iter().copied().collect();
            let g_row: Vec<f32> = gran_logits.row(i).iter().copied().collect();

            let (b_class, b_conf) = Self::softmax_and_argmax(&b_row);
            let (g_class, g_conf) = Self::softmax_and_argmax(&g_row);

            // --- DEBUG 3: Show the Softmax outcome for Cell 0 ---
            if i == 0 {
                println!("🐛 DEBUG: Cell 0 Softmax -> Broad Class {}, Conf: {:.2}%", b_class, b_conf * 100.0);
            }

            batch_results.push(DualPrediction {
                broad_class: b_class,
                broad_confidence: b_conf,
                granular_class: g_class,
                granular_confidence: g_conf,
            });
        }

        Ok(batch_results)
    }
    // --- Softmax Math Helper ---
    fn softmax_and_argmax(logits: &[f32]) -> (u32, f32) {
        // Find max to prevent numerical explosion (NaNs) when running exp()
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();

        let mut best_class = 0;
        let mut best_prob = 0.0;

        for (idx, &exp_val) in exps.iter().enumerate() {
            let prob = exp_val / sum;
            if prob > best_prob {
                best_prob = prob;
                best_class = idx as u32;
            }
        }

        (best_class, best_prob)
    }
}