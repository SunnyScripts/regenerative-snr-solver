use candle_core::{Tensor, Module};
use candle_nn::{Conv2d, Conv2dConfig, Linear, VarBuilder, sequential};

pub struct MorphologyBrain {
    // The Shared Spine uses Conv2d to see spatial relationships
    spine: candle_nn::Sequential,

    // Actor Heads (Per-Tile Predictions) use 1x1 Convolutions
    opcode_head: Conv2d,
    param_head: Conv2d,
    cell_type_head: Conv2d,

    // Critic Head (Global Organ Evaluation) uses a Linear layer
    value_head: Linear,
}

impl MorphologyBrain
{
    pub fn new(vb: VarBuilder) -> Self {
        // We need padding=1 so a 3x3 kernel doesn't shrink the 20x24 grid!
        let conv_cfg = Conv2dConfig { padding: 1, stride: 1, ..Default::default() };
        let point_cfg = Conv2dConfig { padding: 0, stride: 1, ..Default::default() };

        // 1. The Spatial Spine: Sees the neighbors!
        let spine = sequential()
            .add(candle_nn::conv2d(189, 128, 3, conv_cfg, vb.pp("spine_1")).unwrap())
            .add(candle_nn::Activation::Relu)
            .add(candle_nn::conv2d(128, 64, 3, conv_cfg, vb.pp("spine_2")).unwrap())
            .add(candle_nn::Activation::Relu);

        // 2. The Actor Heads: 1x1 Convolutions (acts like a Linear layer applied per-pixel)
        let opcode_head = candle_nn::conv2d(64, 8, 1, point_cfg, vb.pp("opcode_head")).unwrap();
        let param_head = candle_nn::conv2d(64, 4, 1, point_cfg, vb.pp("param_head")).unwrap();
        let cell_type_head = candle_nn::conv2d(64, 4, 1, point_cfg, vb.pp("type_head")).unwrap();

        // 3. The Critic Head: Evaluates the WHOLE organ.
        let value_head = candle_nn::linear(64, 1, vb.pp("value_head")).unwrap();

        Self { spine, opcode_head, param_head, cell_type_head, value_head }
    }

    // Forward expects: [Batch(1), Channels(189), Height(20), Width(24)]
    pub fn forward(&self, state: &Tensor) -> candle_core::Result<(Tensor, Tensor, Tensor, Tensor)> {
        let b_sz = state.dim(0)?; // Should be 1 (single organ)

        // Pass the image through the spine: Output is [1, 64, 20, 24]
        let thoughts = self.spine.forward(state)?;

        // --- ACTOR HEADS ---
        // Outputs are [1, OutChannels, 20, 24]
        let raw_logits = self.opcode_head.forward(&thoughts)?;
        let raw_params = self.param_head.forward(&thoughts)?;
        let raw_type_logits = self.cell_type_head.forward(&thoughts)?;

        // To make these compatible with your existing WGPU code, we permute and flatten them
        // back into [480, OutChannels]
        // Permute to [Batch, Height, Width, Channels], then reshape.
        let logits = raw_logits.permute((0, 2, 3, 1))?.reshape((480, 8))?;

        let params_flat = raw_params.permute((0, 2, 3, 1))?.reshape((480, 4))?;
        let safe_params = params_flat.tanh()?;

        let type_flat = raw_type_logits.permute((0, 2, 3, 1))?.reshape((480, 4))?;
        let type_ratios = candle_nn::ops::softmax(&type_flat, 1)?;

        // --- CRITIC HEAD (Grid-Size Independent) ---
        // 'thoughts' shape is currently: [Batch(1), Channels(64), Height, Width]

        // 1. Average across the Height (dimension 2)
        let pooled_h = thoughts.mean(2)?;
        // 'pooled_h' shape is now: [Batch, Channels, Width]

        // 2. Average across the Width (now dimension 2 again)
        let pooled_hw = pooled_h.mean(2)?;
        // 'pooled_hw' shape is now exactly: [Batch, 64]

        // 3. Pass to the Linear Critic
        let state_value = self.value_head.forward(&pooled_hw)?.squeeze(1)?;

        Ok((logits, safe_params, type_ratios, state_value))
    }
}