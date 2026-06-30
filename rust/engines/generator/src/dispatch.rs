use super::core::NativeWgpuEngine;
use super::types::{MicrotomeConfig, MAX_VOLUME_CELLS};

impl NativeWgpuEngine {
    pub fn dispatch_generation_sequence(&self, blade_center: [f32; 3], slice_thickness: f32, slice_normal: [f32;3])
    {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Graph Morphogenesis Sequence")
        });

        // Always dispatch the maximum volume. The shader's atomic boundary check
        // will instantly kill unused threads at zero cost.
        let max_workgroups = (MAX_VOLUME_CELLS as u32 + 63) / 64;

        // ==========================================
        // PASS 1: ECM
        // ==========================================
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.morphogenesis_pipeline);

            cpass.set_bind_group(0, &self.morphogenesis_bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        // ==========================================
        // PASS 2: MITOSIS
        // ==========================================
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.mitosis_pipeline);
            // FIXED: Using the specific bind group from core.rs
            cpass.set_bind_group(0, &self.mitosis_bind_group, &[]);
            cpass.dispatch_workgroups(max_workgroups, 1, 1);
        }

        // ==========================================
        // PASS 2.5: SPATIAL HASH REBUILD
        // ==========================================
        // Assuming a 64x64x64 grid (262,144 buckets)
        let total_buckets = 64 * 64 * 64;
        let bucket_workgroups = (total_buckets + 63) / 64;

        // A. Clear the grid counters
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.hash_clear_pipeline);
            cpass.set_bind_group(0, &self.hash_bind_group, &[]);
            cpass.dispatch_workgroups(bucket_workgroups, 1, 1);
        }

        // B. Count cells per bucket
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.hash_count_pipeline);
            cpass.set_bind_group(0, &self.hash_bind_group, &[]);
            cpass.dispatch_workgroups(max_workgroups, 1, 1);
        }

        // C. Prefix Sum (Convert counts to array offsets)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.hash_scan_pipeline);
            cpass.set_bind_group(0, &self.hash_bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        // D. Insert cells into the sorted array
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.hash_insert_pipeline);
            cpass.set_bind_group(0, &self.hash_bind_group, &[]);
            cpass.dispatch_workgroups(max_workgroups, 1, 1);
        }

        // ==========================================
        // PASS 3: TOPOLOGY
        // ==========================================
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.topology_pipeline);
            // FIXED: Using the specific bind group
            cpass.set_bind_group(0, &self.topology_bind_group, &[]);
            cpass.dispatch_workgroups(max_workgroups, 1, 1);
        }

        // ==========================================
        // PASS 4: PHYSICS
        // ==========================================
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.physics_pipeline);
            // FIXED: Using the specific bind group
            cpass.set_bind_group(0, &self.physics_bind_group, &[]);
            cpass.dispatch_workgroups(max_workgroups, 1, 1);
        }

        // ==========================================
        // PASS 5: MICROTOME
        // ==========================================

        // Explicitly pack the 3D vectors and the scalar into the 4D arrays
        // let microtome_config = MicrotomeConfig {
        //     // [X, Y, Z, Padding]
        //     blade_center: [blade_center[0], blade_center[1], blade_center[2], 0.0],
        //
        //     // [Normal X, Normal Y, Normal Z, Thickness]
        //     normal_thick: [slice_normal[0], slice_normal[1], slice_normal[2], slice_thickness],
        // };
        //
        // // 2. Upload the config to the GPU
        // self.queue.write_buffer(&self.microtome_config_buffer, 0, bytemuck::bytes_of(&microtome_config));
        //
        // // 3. Reset the slice output counter
        // self.queue.write_buffer(&self.slice_count, 0, bytemuck::bytes_of(&0u32));
        //
        // {
        //     let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        //     cpass.set_pipeline(&self.microtome_pipeline);
        //     cpass.set_bind_group(0, &self.microtome_bind_group, &[]);
        //     cpass.dispatch_workgroups(max_workgroups, 1, 1);
        // }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn dispatch_observation_shader(&self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.observation_pipeline);
            // FIXED: Updated to the correct name from io.rs
            cpass.set_bind_group(0, self.obs_bind_group.as_ref().unwrap(), &[]);

            let workgroups = ((self.num_tiles as u32) + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.observation_buffer, 0,
            &self.staging_buffer, 0,
            self.output_size_bytes
        );

        self.queue.submit(Some(encoder.finish()));
    }
}