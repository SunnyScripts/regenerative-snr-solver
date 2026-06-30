use shared_biology::Coord;
use crate::core::NativeWgpuEngine;
use crate::types::*;

impl NativeWgpuEngine
{
    pub fn reset_tissue_buffer(&self) {
        // The easiest way to clear the VRAM is just to zero out the atomic counters.
        // The shaders will ignore any old ghost data if the count is 0.
        self.queue.write_buffer(&self.global_cell_count, 0, bytemuck::bytes_of(&0u32));
        self.queue.write_buffer(&self.global_edge_count, 0, bytemuck::bytes_of(&0u32));
        self.queue.write_buffer(&self.emitter_count, 0, bytemuck::bytes_of(&0u32));
    }

    // pub fn set_microtome_normal(&self, normal: [f32; 3])
    // {
    //     // Construct the config. (We center the blade at 0,0,0 or the tissue center).
    //     let config = MicrotomeConfig {
    //         blade_center: [0.0, 0.0, 50.0], // Adjust 50.0 to your Z-depth baseline
    //         slice_thickness: 10.0,          // 10µm standard biopsy slice
    //         slice_normal: normal,
    //         _padding: 0,
    //     };
    //
    //     self.queue.write_buffer(
    //         &self.microtome_config_buffer,
    //         0,
    //         bytemuck::bytes_of(&config)
    //     );
    // }

    // Notice the two new u32 parameters!
    // pub fn inject_zygote(&self, x: f32, y: f32, z: f32, polarity: [f32; 3], broad_id: u32, granular_id: u32) {
    //
    //     // 1. Create the singular mother cell
    //     let zygote = CellNode {
    //         pos: Coord { x, y, z }, // Assuming Coord struct from earlier
    //         polarity: Coord { x: polarity[0], y: polarity[1], z: polarity[2] },
    //         broad_id,        // Injected dynamically!
    //         granular_id,     // Injected dynamically!
    //         //ToDo get these values from data so this is empirical
    //         area: 1.5,       // Resting radius
    //         v_mem: -70.0,    // A good resting potential
    //
    //         // This safely fills connected_neighbors, phase_buffers, ligand_pool, etc., with 0
    //         ..Default::default()
    //     };
    //
    //     // 2. Upload it to Index 0 of the volume buffer
    //     self.queue.write_buffer(&self.volume_cells, 0, bytemuck::bytes_of(&zygote));
    //
    //     // 3. Reset the global counters to exactly 1 cell and 0 edges/emitters
    //     self.queue.write_buffer(&self.global_cell_count, 0, bytemuck::bytes_of(&1u32));
    //     self.queue.write_buffer(&self.global_edge_count, 0, bytemuck::bytes_of(&0u32));
    //     self.queue.write_buffer(&self.emitter_count, 0, bytemuck::bytes_of(&0u32));
    // }
    //
    // pub fn inject_batched_zygotes(&self, broad_id: u32, granular_id: u32)
    // {
    //     let total_tiles = (self.sim_params.grid_width * self.sim_params.grid_height) as usize;
    //     let mut batched_cells = Vec::with_capacity(total_tiles);
    //
    //     for ty in 0..self.sim_params.grid_height {
    //         for tx in 0..self.sim_params.grid_width {
    //             // Find the dead center of this specific 150µm tile
    //             let x = (tx as f32 + 0.5) * self.sim_params.tile_size;
    //             let y = (ty as f32 + 0.5) * self.sim_params.tile_size;
    //
    //             batched_cells.push(CellNode {
    //                 pos: Coord { x, y, z: 0.0 },
    //                 polarity: Coord { x: 0.0, y: 0.0, z: 1.0 }, // UP
    //                 broad_id,
    //                 granular_id,
    //                 area: 400.0,
    //                 v_mem: -70.0,
    //                 ..Default::default()
    //             });
    //         }
    //     }
    //
    //     // Upload all 4,096 cells to the buffer
    //     self.queue.write_buffer(&self.volume_cells, 0, bytemuck::cast_slice(&batched_cells));
    //
    //     // Set global count to 4096!
    //     self.queue.write_buffer(&self.global_cell_count, 0, bytemuck::bytes_of(&(total_tiles as u32)));
    //     self.queue.write_buffer(&self.global_edge_count, 0, bytemuck::bytes_of(&0u32));
    //     self.queue.write_buffer(&self.emitter_count, 0, bytemuck::bytes_of(&0u32));
    // }

    pub fn frame_generation_volume(&mut self)
    {
        let max_x = self.sim_params.grid_width as f32 * self.sim_params.tile_size;
        let max_y = self.sim_params.grid_height as f32 * self.sim_params.tile_size;

        let center_x = max_x / 2.0;
        let center_y = max_y / 2.0;
        let max_dim = max_x.max(max_y);

        let fov_y_rads = self.camera.fovy.to_radians();
        let perfect_z = (max_dim / 2.0) / (fov_y_rads / 2.0).tan() * 1.2;

        self.camera.target = glam::Vec3::new(center_x, center_y, 0.0);
        self.camera.radius = perfect_z;

        // --- ADJUSTED FOR YOUR ENGINE'S SPHERICAL MATRIX MATH ---
        self.camera.theta = 0.0;
        self.camera.phi = std::f32::consts::FRAC_PI_2; // 90 degrees (looks straight down)
        // --------------------------------------------------------

        self.camera.zfar = perfect_z * 3.0;
        self.camera_needs_update = true;

        let new_uniform = CameraUniform {
            view_proj: self.camera.build_view_projection_matrix(),
            cell_scale: 1.0,
            mouse_x: 0.0, mouse_y: 0.0, _pad: 0.0,
            tile_bounds: glam::Vec4::ZERO,
        };

        self.queue.write_buffer(
            &self.camera_buffer, 0, bytemuck::cast_slice(&[new_uniform])
        );
    }
}