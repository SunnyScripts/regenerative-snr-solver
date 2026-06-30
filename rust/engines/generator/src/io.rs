use super::core::NativeWgpuEngine;
use super::types::*;
use std::collections::HashMap;
use glam::Vec4;
use wgpu::util::DeviceExt;
use tokio;

impl NativeWgpuEngine
{
    pub fn load_ground_truth(&mut self, nodes: &[CellNode], bounds: &BoundingBox)
    {
        // ==========================================
        // 1. AUTO-FRAMING THE CAMERA
        // ==========================================
        let center_x = bounds.min_x + (bounds.max_x - bounds.min_x) / 2.0;
        let center_y = bounds.min_y + (bounds.max_y - bounds.min_y) / 2.0;

        // Find the largest dimension of the tissue
        let width = bounds.max_x - bounds.min_x;
        let height = bounds.max_y - bounds.min_y;
        let max_dim = width.max(height);

        // Calculate the perfect Z distance using trigonometry
        let fov_y_rads = self.camera.fovy.to_radians();
        let half_fov_tan = (fov_y_rads / 2.0).tan();

        // 1.2 is our padding multiplier so the tissue doesn't hug the window borders
        let perfect_z = ((max_dim / 2.0) / half_fov_tan) * 1.2;

        // Update the CPU Camera State
        self.camera.eye = glam::Vec3::new(center_x, center_y, perfect_z);
        self.camera.target = glam::Vec3::new(center_x, center_y, 0.0);

        // Dynamically push the far clipping plane out so deep datasets don't vanish
        self.camera.zfar = perfect_z * 2.0;

        // ==========================================
        // 2. PUSH CAMERA MATRIX TO VRAM
        // ==========================================
        let new_uniform = CameraUniform {
            view_proj: self.camera.build_view_projection_matrix(),
            cell_scale: 1.0,
            mouse_x: 0.0,
            mouse_y: 0.0,
            _pad: 0.0,
            tile_bounds: Vec4::ZERO,
        };

        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[new_uniform])
        );

        // ==========================================
        // 3. UPLOAD BIOLOGICAL DATA TO VRAM
        // ==========================================
        self.queue.write_buffer(
            &self.volume_cells,
            0,
            bytemuck::cast_slice(nodes)
        );

        // Update the CPU count for the Render Pass
        self.cpu_cell_count = nodes.len() as u32;
    }

    pub fn reset_arena(&self) {
        // A single 4-byte zero
        let zero: [u8; 4] = [0, 0, 0, 0];

        // Reset the atomic counters. The shaders will now ignore all old data.
        self.queue.write_buffer(&self.global_cell_count, 0, &zero);
        self.queue.write_buffer(&self.global_edge_count, 0, &zero);
        self.queue.write_buffer(&self.emitter_count, 0, &zero);

        // (Optional) If your microtome uses a counter, reset that too
        self.queue.write_buffer(&self.slice_count, 0, &zero);
    }

    pub fn fetch_generated_cell_count(&self) -> u32 {
        // 1. Ask the GPU to copy the atomic counter to our readable staging buffer
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder")
        });

        encoder.copy_buffer_to_buffer(
            &self.global_cell_count, 0, // Source: Your compute atomic counter
            &self.count_staging_buffer, 0,     // Destination: Our CPU-readable buffer
            4                                  // Size: 4 bytes (u32)
        );
        self.queue.submit(Some(encoder.finish()));

        // 2. Set up the async map request
        let buffer_slice = self.count_staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        // 3. Force the GPU to finish its current queue so we can read it immediately
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();

        // 4. Extract the u32 value from the raw bytes
        let data = buffer_slice.get_mapped_range();
        let new_count = bytemuck::from_bytes::<u32>(&data).clone();

        // 5. Cleanup so the buffer can be used again next frame
        drop(data);
        self.count_staging_buffer.unmap();

        new_count
    }

    // ==========================================
    // 2. BUFFER WRITING & BINDING
    // ==========================================
    // Note: Requires `&mut self` because we are storing the buffers and bind group
    pub fn write_buffers(&mut self, cells: &[CellNode], offsets: &[u32], counts: &[u32])
    {
        let cell_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cell Node Buffer"),
            contents: bytemuck::cast_slice(cells),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let offset_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tile Offsets"),
            contents: bytemuck::cast_slice(offsets),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let count_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tile Counts"),
            contents: bytemuck::cast_slice(counts),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // We only build the OBSERVATION bind group here, because the observation
        // cells change dynamically based on the 2D Microtome slice.
        let bind_group_layout = self.observation_pipeline.get_bind_group_layout(0);
        let obs_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Observation Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cell_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: offset_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.observation_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.param_buffer.as_entire_binding() },
            ],
        });

        self.cell_buffer = Some(cell_buffer);
        self.offset_buffer = Some(offset_buffer);
        self.count_buffer = Some(count_buffer);
        self.obs_bind_group = Some(obs_bind_group); // Fixed name!
    }

    // ==========================================
    // 5. STATIC HELPER: Bins and Offsets
    // ==========================================
    pub fn prepare_compute_buffers(
        nodes: &[CellNode],
        params: &SimParams
    ) -> (Vec<CellNode>, Vec<u32>, Vec<u32>) {
        let num_tiles = (params.grid_width * params.grid_height) as usize;

        let mut tile_offsets = vec![0u32; num_tiles];
        let mut tile_counts = vec![0u32; num_tiles];

        let mut grid_bins: HashMap<usize, Vec<CellNode>> = HashMap::new();

        for node in nodes
        {
            // FIXED: Using node.pos.x and node.pos.y
            let tx = (node.pos.x / params.tile_size).floor() as u32;
            let ty = (node.pos.y / params.tile_size).floor() as u32;

            if tx < params.grid_width && ty < params.grid_height {
                let tile_idx = (tx + (ty * params.grid_width)) as usize;
                grid_bins.entry(tile_idx).or_default().push(*node);
            }
        }

        let mut gpu_cells: Vec<CellNode> = Vec::with_capacity(nodes.len());
        let mut current_offset = 0u32;

        for i in 0..num_tiles {
            if let Some(cells_in_tile) = grid_bins.get(&i) {
                tile_counts[i] = cells_in_tile.len() as u32;
                tile_offsets[i] = current_offset;
                gpu_cells.extend(cells_in_tile);
                current_offset += cells_in_tile.len() as u32;
            } else {
                tile_counts[i] = 0;
                tile_offsets[i] = current_offset;
            }
        }

        (gpu_cells, tile_offsets, tile_counts)
    }

    pub async fn read_observation_buffer(&self) -> Vec<Vec<f32>> {
        // 1. Get a slice of the staging buffer (which has CPU read access)
        let buffer_slice = self.staging_buffer.slice(..);

        // 2. Create a channel to await the GPU's memory mapping callback
        let (sender, receiver) = tokio::sync::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // 3. Force the WGPU device to process pending commands (Crucial!)
        if let Err(e) = self.device.poll(wgpu::PollType::wait_indefinitely()) {
            panic!("🚨 GPU Poll Error: {:?}", e);
        }

        // 4. Await the callback to ensure memory is safe to read
        if let Ok(Ok(())) = receiver.await {
            let data = buffer_slice.get_mapped_range();

            // DEBUG PRINT 1: How many bytes did we actually get from the GPU?
            println!("🔍 Observation Buffer Debug: Received {} bytes from GPU", data.len());

            let flat_floats: &[f32] = bytemuck::cast_slice(&data);

            // DEBUG PRINT 2: How many floats is that?
            println!("🔍 Float Count: {} floats", flat_floats.len());

            let floats_per_tile = 185;
            let mut batched_data = Vec::with_capacity(self.num_tiles as usize);

            for chunk in flat_floats.chunks_exact(floats_per_tile) {
                batched_data.push(chunk.to_vec());
            }

            println!("🔍 Batched Tiles: {} tiles", batched_data.len());

            drop(data);
            self.staging_buffer.unmap();
            batched_data
        } else {
            panic!("🚨 GPU Panic: Failed to map the observation staging buffer!");
        }
    }
}