use crate::graphics_engine::NativeWgpuEngine;

//ToDo add loading and saving from old function
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


// --- F. SAVING CHECKPOINTS ---
//         if epoch % 50 == 0 {
//             println!("💾 Saving Checkpoint (Epoch {})...", epoch);
//             varmap.save(weights_path)?;
//
//             // Save the sidecar state
//             let state_json = serde_json::to_string_pretty(&state)?;
//             fs::write(state_path, state_json)?;
//         }



//observation space: 20 ripley, 36 (6x6 compare) enrich, 128 halo (16 for each of 8 neighbor tiles), 1 count of cells, 1 age step, 1 medgemma latent vector, vec3 slice normal
// 16 floats of neighbor halo data
//Float 0 (Mass): Total cell count within a 10µm strip of the shared border.
// Float 1-4 (Chemistry): The average concentration of the Morphogen grid (Growth, Repulsion, Adhesion, etc.) spilling across that border.
// Float 5-10 (Geometry): A heavily down-sampled Ripley's L curve or Neighborhood Enrichment score calculated only for the cells touching the border. (Tells the brain if the incoming tissue is a dense duct or scattered stroma).
// Float 11-13 (Vector/Velocity): The average $X, Y, Z$ movement vector of the cells on that border. (Tells the brain if the neighbor is physically invading or retreating).
// Float 14-15 (Tension): The average spring tension (distance vs adhesion) of the MembraneInterface edges crossing the boundary.

#[derive(Clone, Copy, PartialEq)]
pub enum TargetOrientation {
    Vertical, // Standard H&E cross-section (Normal along Y)
    EnFace,   // Horizontal slice (Normal along Z)
}


use rand::Rng;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AgentHyperparams {
    pub learning_rate: f64,
    pub local_weight: f32,
    pub global_weight: f32,
    // The Morphological weights
    pub count_weight: f32,
    pub ripley_weight: f32,
    pub enrich_weight: f32,
    pub halo_weight: f32,
    // pub vlm_weight: f32,
}

impl AgentHyperparams {
    // Starts with a standard "Kindergarten" baseline
    pub fn default_baseline() -> Self {
        Self {
            learning_rate: 1e-4,
            local_weight: 1.0,
            global_weight: 0.0,
            count_weight: 0.8,
            ripley_weight: 0.2,
            enrich_weight: 0.0,
            halo_weight: 0.0,
            // vlm_weight: 0.0,
        }
    }

    // THE EXPLORE MECHANIC: Mutate parameters by +/- 20%
    pub fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        let mutate_factor = || if rng.random_bool(0.5) { 1.2 } else { 0.8 };

        if rng.random_bool(0.3) { self.learning_rate *= mutate_factor() as f64; }

        // Mutate MARL balance, ensuring they always sum to ~1.0
        if rng.random_bool(0.3) {
            self.global_weight *= mutate_factor();
            self.global_weight = self.global_weight.clamp(0.0, 0.5); // Cap global care at 50%
            self.local_weight = 1.0 - self.global_weight;
        }

        // Add similar mutations for morphology weights here, clamping them to safe ranges
    }
}

pub struct PbtAgent {
    pub id: usize,
    pub varmap: VarMap,
    pub brain: MorphologyBrain,
    pub optimizer: AdamW,
    pub dna: AgentHyperparams,
    pub rolling_score: f32,
    pub wgpu_env: NativeWgpuEngine, // <-- Give the agent its own permanent universe
}

impl PbtAgent {
    pub fn new(id: usize, device: &Device) -> anyhow::Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let brain = MorphologyBrain::new(vb);

        // Initialize with default DNA
        let dna = AgentHyperparams::default_baseline();
        let optimizer = AdamW::new(brain.parameters(), dna.learning_rate)?;

        Ok(Self { id, varmap, brain, optimizer, dna, rolling_score: -999.0 })
    }

    // THE EXPLOIT MECHANIC: Overwrite this brain with a winner's brain
    pub fn clone_from_winner(&mut self, winner: &PbtAgent, device: &Device) -> anyhow::Result<()> {
        // Deep copy the DNA
        self.dna = winner.dna.clone();

        // Deep copy the neural weights via safetensors in memory
        let bytes = winner.varmap.save_to_vec()?;
        self.varmap = VarMap::new(); // Reset
        self.varmap.load_from_slice(&bytes)?;

        // Rebuild Brain and Optimizer with the winning weights and new DNA
        let vb = VarBuilder::from_varmap(&self.varmap, DType::F32, device);
        self.brain = MorphologyBrain::new(vb);
        self.optimizer = AdamW::new(self.brain.parameters(), self.dna.learning_rate)?;

        Ok(())
    }
}

pub async fn run_pbt_pipeline() -> anyhow::Result<()> {
    let candle_device = Device::new_metal(0).unwrap_or(Device::Cpu);

    let pop_size = 4; // Perfect for M4 Pro unified memory bandwidth
    let epochs_per_generation = 500;
    let max_generations = 200; // 100,000 total epochs

    // 1. Initialize Population
    let mut population = Vec::new();
    for i in 0..pop_size {
        population.push(PbtAgent::new(i, &candle_device)?);
    }

    let target_files = get_all_target_files("./target_bank/");

    // ==========================================
    // THE GENERATION LOOP (The Macro Wrapper)
    // ==========================================
    for gen in 0..max_generations {
        println!("\n🧬 --- STARTING GENERATION {} --- 🧬", gen);

        // 1. EVALUATE: Train every agent for a set number of epochs
        for agent in population.iter_mut() {
            println!("🤖 Training Agent {}...", agent.id);
            // Notice we pass the agent in mutably so it can update its own weights
            agent.rolling_score = train_agent_for_generation(
                agent,
                epochs_per_generation,
                &target_files,
                &candle_device
            ).await?;
        }

        // 2. RANK: Sort population by their rolling scores (highest to lowest)
        population.sort_by(|a, b| b.rolling_score.partial_cmp(&a.rolling_score).unwrap());

        println!("🏆 Generation {} Leaderboard:", gen);
        for agent in &population {
            println!("  - Agent {}: Score {:.3} | LR: {:.2e} | Global Wt: {:.2}",
                     agent.id, agent.rolling_score, agent.dna.learning_rate, agent.dna.global_weight);
        }

        // 3. CHECKPOINT: Only save the Alpha (Index 0) to disk
        let alpha = &population[0];
        alpha.varmap.save(format!("alpha_brain_gen_{}.safetensors", gen))?;
        fs::write("alpha_dna.json", serde_json::to_string_pretty(&alpha.dna)?)?;

        // 4. EXPLOIT & EXPLORE (If we haven't reached the end)
        if gen < max_generations - 1 {
            // Split population in half. Top 2 live, Bottom 2 die.
            let (winners, losers) = population.split_at_mut(pop_size / 2);

            for (i, loser) in losers.iter_mut().enumerate() {
                let parent_winner = &winners[i % winners.len()];

                // Exploit: Overwrite loser with winner's brain
                loser.clone_from_winner(parent_winner, &candle_device)?;

                // Explore: Mutate the newly cloned DNA
                loser.dna.mutate();
                println!("  🔬 Agent {} cloned from Agent {} and mutated its DNA.", loser.id, parent_winner.id);
            }
        }
    }

    Ok(())
}

// Now returns an f32 (the average score over this generation)
use rand::seq::SliceRandom;
use rand::Rng;
use candle_core::{Tensor, Device};
// Make sure you have your TargetOrientation enum imported here!

pub async fn train_agent_for_generation(
    agent: &mut PbtAgent,
    epochs_to_run: usize,
    targets: &Vec<(String, TargetOrientation)>, // Cleaned up signature
    sim_params: &SimParams,                     // Need this for dynamic bounds!
    candle_device: &Device,
) -> anyhow::Result<f32>
{
    let mut generation_rewards = Vec::new();
    let mut rng = rand::thread_rng();

    for epoch in 0..epochs_to_run
    {

        // 1. Randomly select a target AND its physical orientation
        let (active_target, orientation) = targets.choose(&mut rng).unwrap();

        agent.wgpu_env.load_target_buffer(active_target);
        let target_data = agent.wgpu_env.read_target_buffer().await;
        agent.wgpu_env.reset_tissue_buffer();

        // ==========================================
        // DYNAMIC BOUNDS CALCULATION
        // ==========================================
        let max_x = sim_params.grid_width as f32 * sim_params.bucket_size;
        let max_y = sim_params.grid_height as f32 * sim_params.bucket_size;
        let max_z = sim_params.grid_depth as f32 * sim_params.bucket_size;

        let center_x = max_x / 2.0;
        let center_y = max_y / 2.0;

        let margin_x = max_x * 0.2;
        let margin_y = max_y * 0.2;
        let margin_z = max_z * 0.2;

        // 2. Drop the Zygote exactly in the center of the XY plane, at the bottom of Z
        let global_target_count = target_data[0][184];
        if global_target_count > 0.0 {
            let stem_broad_id = 0;
            let stem_granular_id = 5;

            // Zygote injected at Z=0.0 (basal layer starting point)
            agent.wgpu_env.inject_zygote(
                center_x, center_y, 0.0,
                [0.0, 0.0, 1.0], // Apical-Basal polarity points UP
                stem_broad_id, stem_granular_id
            );
        }

        // ==========================================
        // 3. THE DETERMINISTIC MICROTOME
        // ==========================================
        let requested_normal: [f32; 3];
        let mut blade_center = [0.0, 0.0, 0.0];

        match orientation {
            TargetOrientation::Vertical => {
                // Look at XZ plane. Normal faces Y.
                requested_normal = [0.0, 1.0, 0.0];

                // Slide blade along Y, within margins
                let random_y = rng.gen_range(margin_y..(max_y - margin_y));
                blade_center = [center_x, random_y, max_z / 2.0];
            },
            TargetOrientation::EnFace => {
                // Look at XY plane. Normal faces Z.
                requested_normal = [0.0, 0.0, 1.0];

                // Slide blade along Z, within margins
                let random_z = rng.gen_range(margin_z..(max_z - margin_z));
                blade_center = [center_x, center_y, random_z];
            }
        }

        agent.wgpu_env.dispatch_observation_shader();
        let mut current_state = agent.wgpu_env.read_observation_buffer().await;

        let mut episode_log_probs = Vec::new();
        let mut episode_values = Vec::new();
        let mut episode_rewards = Vec::new();

        let mut step = 0;
        let mut done = false;
        let max_steps = 20;

        while !done && step < max_steps {
            let current_age = step as f32 / max_steps as f32;

            // Expand tensor capacity to 189 (185 + 1 Age + 3 Normal)
            let mut batched_delta_state = Vec::with_capacity(sim_params.num_tiles() * 189);

            for tile_idx in 0..sim_params.num_tiles() {
                for float_idx in 0..185 {
                    batched_delta_state.push(target_data[tile_idx][float_idx] - current_state[tile_idx][float_idx]);
                }
                batched_delta_state.push(current_age);
                batched_delta_state.push(requested_normal[0]);
                batched_delta_state.push(requested_normal[1]);
                batched_delta_state.push(requested_normal[2]);
            }

            let state_tensor = Tensor::from_slice(&batched_delta_state, [sim_params.num_tiles(), 189], &candle_device)?
                .reshape((1, 20, 24, 189))?
                .permute((0, 3, 1, 2))? // Moves Channels to dimension 1: [1, 189, 20, 24]
                .contiguous()?;

            // 1. Unpack all THREE heads from the Master Brain
            let (logits, safe_params, type_ratios, state_value) = agent.brain.forward(&state_tensor)?;

            episode_values.push(state_value);

            // 2. Sample the categorical action (Geometry)
            let action_probs = candle_nn::ops::softmax(&logits, 1)?;
            let (cmd_ids, log_probs) = sample_categorical_batched(&action_probs);

            // Save the log probabilities for the PPO ratio
            episode_log_probs.push(log_probs);

            // 3. Format ALL outputs for the WGPU engine
            let gpu_opcodes = format_batch_for_wgpu(&cmd_ids, &safe_params, &type_ratios);

            // Ship it to Metal!
            agent.wgpu_env.queue.write_buffer(&agent.wgpu_env.opcodes, 0, bytemuck::cast_slice(&gpu_opcodes));

            // DISPATCH WITH DYNAMIC BLADE CENTER
            agent.wgpu_env.dispatch_generation_sequence(blade_center, 10.0, requested_normal);

            let slice_2d_cells: Vec<CellNode> = agent.wgpu_env.read_microtome_buffer().await;
            let (gpu_cells, offsets, counts) = NativeWgpuEngine::prepare_compute_buffers(&slice_2d_cells, &sim_params);

            agent.wgpu_env.write_buffers(&gpu_cells, &offsets, &counts);
            agent.wgpu_env.dispatch_observation_shader();

            let next_state = agent.wgpu_env.read_observation_buffer().await;

            // ==========================================
            // USE THE DYNAMIC DNA WEIGHTS HERE
            // ==========================================
            let mut batched_step_rewards = Vec::with_capacity(sim_params.num_tiles());
            let mut global_mse_sum = 0.0;

            for tile_idx in 0..sim_params.num_tiles() {
                // Pass agent.dna instead of cur_weights
                let local_reward = calculate_reward(&next_state[tile_idx], &target_data[tile_idx], &agent.dna);
                batched_step_rewards.push(local_reward);
                global_mse_sum += local_reward;
            }

            let global_reward = global_mse_sum / sim_params.num_tiles() as f32;
            let total_global_cells = extract_total_cells(&next_state[0]);

            for tile_idx in 0..sim_params.num_tiles() {
                // Use DNA for Local/Global blending
                let blended_reward = (batched_step_rewards[tile_idx] * agent.dna.local_weight)
                    + (global_reward * agent.dna.global_weight);
                batched_step_rewards[tile_idx] = blended_reward - 0.05;
            }

            episode_rewards.push(batched_step_rewards);

            if total_global_cells > 10_000 { done = true; }

            current_state = next_state;
            step += 1;
        }

        // 1. Calculate Advantages and the Target Returns for the Critic
        let (advantages, returns) = calculate_gae_advantages_batched(
            &episode_rewards,
            &episode_values,
            &candle_device
        )?;

        // 2. Calculate the combined Actor-Critic Loss
        let combined_loss = calculate_actor_critic_loss_batched(
            &episode_log_probs,
            &episode_values,
            &advantages,
            &returns
        )?;

        // 3. Backpropagate through the entire Brain (Actor + Critic) at once!
        agent.optimizer.backward_step(&combined_loss)?;

        generation_rewards.push(calculate_average_reward(&episode_rewards));
    }

    // Return the average score of the last 50 epochs as the agent's true "Fitness"
    let final_fitness = generation_rewards.iter().rev().take(50).sum::<f32>() / 50.0;
    Ok(final_fitness)
}

pub fn format_batch_for_wgpu(
    cmd_ids: &[u32],
    safe_params: &Tensor,
    type_ratios: &Tensor, // NEW
) -> Vec<GpuOpcode> {

    // Convert Candle Tensors to standard 2D Rust Vecs
    let params_vec = safe_params.to_vec2::<f32>().unwrap();
    let ratios_vec = type_ratios.to_vec2::<f32>().unwrap();

    let mut opcodes = Vec::with_capacity(cmd_ids.len());

    for i in 0..cmd_ids.len() {
        opcodes.push(GpuOpcode {
            cmd_id: cmd_ids[i],
            p1: params_vec[i][0],
            p2: params_vec[i][1],
            p3: params_vec[i][2],
            p4: params_vec[i][3],
            type_ratios: [
                ratios_vec[i][0],
                ratios_vec[i][1],
                ratios_vec[i][2],
                ratios_vec[i][3],
            ],
            _pad: [0; 3],
        });
    }

    opcodes
}


// ==========================================
// THE BIOLOGICAL REWARD FUNCTION
// ==========================================
pub fn calculate_reward(
    current_state: &[f32],
    target_state: &[f32],
    dna: &AgentHyperparams,
) -> f32 {
    // Helper closure to calculate Mean Squared Error (MSE) for a slice
    let mse = |start: usize, end: usize| -> f32 {
        let mut sum_sq = 0.0;
        for i in start..end {
            let diff = current_state[i] - target_state[i];
            sum_sq += diff * diff;
        }
        sum_sq / (end - start) as f32
    };

    // Based on your obs_dim = 184 (20 Ripley + 36 Enrich + 128 Halo)
    let ripley_mse = mse(0, 20);
    let enrich_mse = mse(20, 56);
    let halo_mse = mse(56, 184);

    // Index 184 is the global cell count.
    // We normalize the count difference as a percentage so a difference of
    // 500 cells doesn't completely overpower the topological metrics (which are usually 0.0 - 1.0).
    let target_count = target_state[184].max(1.0); // Prevent division by zero
    let count_diff_pct = (current_state[184] - target_state[184]) / target_count;
    let count_mse = count_diff_pct * count_diff_pct;

    // The agent's DNA determines what "matters" in this specific episode
    let total_penalty = (dna.ripley_weight * ripley_mse)
        + (dna.enrich_weight * enrich_mse)
        + (dna.halo_weight * halo_mse)
        + (dna.count_weight * count_mse);

    // In RL, we want to maximize reward, meaning we want to MINIMIZE the penalty.
    // We map the penalty (0 to +inf) to a stable reward between -1.0 and 0.0
    // using the exponential function. Perfect match = 0.0.
    (-total_penalty).exp() - 1.0
}

// ==========================================
// TINY HELPERS
// ==========================================
pub fn extract_total_cells(state_tensor: &[f32]) -> f32 {
    // The count is appended at the very end of the 184 metrics
    state_tensor[184]
}

pub fn calculate_average_reward(episode_rewards: &Vec<Vec<f32>>) -> f32 {
    if episode_rewards.is_empty() {
        return -1.0;
    }

    let mut total = 0.0;
    let mut count = 0;

    for step in episode_rewards {
        for &tile_reward in step {
            total += tile_reward;
            count += 1;
        }
    }

    total / count as f32
}

use rand::Rng;

//todo is random a good idea for this?
pub fn generate_random_normal() -> [f32; 3] {
    let mut rng = rand::thread_rng();

    // Generate a random z-value between -1 and 1
    let z: f32 = rng.gen_range(-1.0..1.0);

    // Generate a random angle around the z-axis
    let theta: f32 = rng.gen_range(0.0..std::f32::consts::TAU);

    // Calculate the corresponding x and y based on the sphere's radius at z
    let r = (1.0 - z * z).sqrt();
    let x = r * theta.cos();
    let y = r * theta.sin();

    [x, y, z]
}
