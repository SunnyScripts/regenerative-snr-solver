use candle_core::{Tensor, Device, DType};

pub fn calculate_gae_advantages_batched(
    rewards: &Vec<Vec<f32>>,
    values: &Vec<Tensor>,
    device: &Device,
) -> candle_core::Result<(Tensor, Tensor)> {
    let gamma = 0.99f32;  // Discount factor (how much we care about the future)
    let lambda = 0.95f32; // GAE smoothing parameter

    let steps = rewards.len();
    let num_tiles = rewards[0].len();

    // 1. Pull the Critic's value estimates from the GPU to the CPU
    let mut v_matrix = Vec::with_capacity(steps);
    for v in values {
        // Assumes values are squeezed to [num_tiles] in the forward pass
        v_matrix.push(v.to_vec1::<f32>()?);
    }

    let mut advantages = vec![vec![0.0f32; num_tiles]; steps];
    let mut returns = vec![vec![0.0f32; num_tiles]; steps];

    let mut next_adv = vec![0.0f32; num_tiles];
    let mut next_val = vec![0.0f32; num_tiles];

    // 2. Reverse accumulation (The core of GAE)
    for t in (0..steps).rev() {
        for i in 0..num_tiles {
            let delta = rewards[t][i] + gamma * next_val[i] - v_matrix[t][i];
            let adv = delta + gamma * lambda * next_adv[i];

            advantages[t][i] = adv;
            returns[t][i] = adv + v_matrix[t][i]; // Return = Advantage + Value

            next_val[i] = v_matrix[t][i];
            next_adv[i] = adv;
        }
    }

    // 3. Flatten the 2D arrays into 1D arrays for the GPU
    let mut flat_adv: Vec<f32> = advantages.into_iter().flatten().collect();
    let flat_ret: Vec<f32> = returns.into_iter().flatten().collect();

    // 4. Normalize the Advantages (Crucial for RL stability to avoid explosive gradients)
    let mean = flat_adv.iter().sum::<f32>() / flat_adv.len() as f32;
    let var_sum: f32 = flat_adv.iter().map(|&a| (a - mean).powi(2)).sum();
    let std = (var_sum / flat_adv.len() as f32).sqrt();

    for a in &mut flat_adv {
        *a = (*a - mean) / (std + 1e-8);
    }

    // 5. Push the calculated targets back to the Metal GPU
    let adv_tensor = Tensor::from_slice(&flat_adv, &[steps * num_tiles], device)?;
    let ret_tensor = Tensor::from_slice(&flat_ret, &[steps * num_tiles], device)?;

    Ok((adv_tensor, ret_tensor))
}

pub fn calculate_actor_critic_loss_batched(
    log_probs: &Vec<Tensor>,
    values: &Vec<Tensor>,
    advantages: &Tensor,
    returns: &Tensor,
) -> candle_core::Result<Tensor> {

    // 1. Stack the temporal sequence into flat tensors [steps * num_tiles]
    let stacked_log_probs = Tensor::stack(log_probs, 0)?.flatten_all()?;
    let stacked_values = Tensor::stack(values, 0)?.flatten_all()?;

    // 2. ACTOR LOSS (Policy Gradient)
    // Loss = - mean(log_probs * advantages)
    // If advantage is positive (good action), we push log_prob higher.
    let actor_loss = stacked_log_probs
        .broadcast_mul(advantages)?
        .mean_all()?
        .neg()?;

    // 3. CRITIC LOSS (Mean Squared Error)
    // Loss = MSE(predicted_value, actual_return)
    let critic_loss = stacked_values
        .broadcast_sub(returns)?
        .sqr()?
        .mean_all()?;

    // 4. COMBINED LOSS
    // Standard practice is to weight the critic loss by 0.5
    let total_loss = (actor_loss + (critic_loss * 0.5)?)?;

    Ok(total_loss)
}