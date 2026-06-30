use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

pub const MAX_VOLUME_CELLS: usize = 500_000;

pub use shared_biology::{SimParams, MembraneInterface, CellNode};
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuOpcode {
    pub cmd_id: u32,
    pub p1: f32, pub p2: f32, pub p3: f32, pub p4: f32,
    pub _pad1: u32, pub _pad2: u32, pub _pad3: u32,
}

// Replace GenerationParams with this:
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MicrotomeConfig {
    pub blade_center: [f32; 4],
    pub normal_thick: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MorphogenEmitter {
    pub emitter_type: u32,
    pub pos: [f32; 3],
    pub dir: [f32; 3],
    pub strength: f32,
    pub params: [f32; 4],
    pub decay_rate: f32,
    pub _pad: [u32; 3], // Align to 16 bytes
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform
{
    pub view_proj: Mat4,    // 64 bytes
    pub cell_scale: f32,          // 4 bytes
    pub mouse_x: f32,             // 4 bytes
    pub mouse_y: f32,             // 4 bytes
    pub _pad: f32,                // 4 bytes (Gets us to 80 bytes)
    pub tile_bounds: Vec4,  // 16 bytes! (Total 96 bytes)
}

pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub radius: f32, // Replaces eye.z for zoom
    pub theta: f32,  // Yaw (Left/Right rotation)
    pub phi: f32,    // Pitch (Up/Down rotation)
    pub up: Vec3,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    // pub fn build_view_projection_matrix(&self) -> Mat4 {
    //     let view = Mat4::look_at_rh(self.eye, self.target, self.up);
    //     let proj = Mat4::perspective_rh(self.fovy.to_radians(), self.aspect, self.znear, self.zfar);
    //     proj * view
    // }
    pub fn get_eye_position(&self) -> glam::Vec3 {
        // Spherical to Cartesian coordinate conversion
        let x = self.radius * self.phi.sin() * self.theta.sin();
        let y = self.radius * self.phi.cos();
        let z = self.radius * self.phi.sin() * self.theta.cos();

        glam::Vec3::new(x, y, z) + self.target
    }

    pub fn build_view_projection_matrix(&self) -> glam::Mat4 {
        let view = glam::Mat4::look_at_rh(
            self.get_eye_position(),
            self.target,
            glam::Vec3::Y, // Up vector is Y
        );
        let proj = glam::Mat4::perspective_rh(self.fovy.to_radians(), 1.0, 0.1, self.zfar);
        proj * view
    }
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY,
            cell_scale: 1.0,
            mouse_x: 0.0,
            mouse_y: 0.0,
            _pad: 0.0,
            tile_bounds: Vec4::ZERO,
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            eye: (0.0, 5.0, 10.0).into(),    // Elevated and back
            target: Vec3::ZERO,             // Looking at the center
            radius: 0.0,
            theta: 0.0,
            phi: 0.0,
            up: Vec3::Y,                    // Y-axis is "up"
            aspect: 16.0 / 9.0,             // Standard widescreen
            fovy: 45.0,                     // 45 degree field of view
            znear: 0.1,                     // Don't clip close objects
            zfar: 100.0,                    // Draw distance
        }
    }
}

// impl From<&Camera> for CameraUniform {
//     fn from(camera: &Camera) -> Self {
//         Self {
//             view_proj: camera.build_view_projection_matrix(),
//         }
//     }
// }


pub struct EngineResources {
    // pub instance: Option<wgpu::Instance>,
    // pub adapter: Option<wgpu::Adapter>,
    pub device: Option<wgpu::Device>,
    pub queue: Option<wgpu::Queue>,
}

// 2. Implement Default so you can call it with empty values
impl Default for EngineResources {
    fn default() -> Self {
        Self {
            // instance: None,
            // adapter: None,
            device: None,
            queue: None,
        }
    }
}

pub struct BoundingBox {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
}

// #[repr(C)]
// #[derive(Copy, Clone, Debug, Pod, Zeroable)]
// pub struct GenerationParams {
//     pub pass_color: u32, // 0 = Red, 1 = Black
//     pub offset_50: u32,  // 0 = False, 1 = True
//     pub slice_z_center: f32,
//     pub slice_thickness: f32,
// }

// #[repr(C)]
// #[derive(Copy, Clone, Debug, Pod, Zeroable)]
// pub struct GpuCursor {
//     pub x: f32,
//     pub y: f32,
//     pub z: f32,
//     pub dir_x: f32,
//     pub dir_y: f32,
//     pub dir_z: f32,
//     pub active_radius: f32,
//     pub _pad: u32, // 16-byte alignment requirement
// }