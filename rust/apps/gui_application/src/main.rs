use slint::{GraphicsAPI, RenderingState, ModelRc, SharedString, VecModel};
use std::rc::Rc;
use std::cell::RefCell;
use std::sync::mpsc;
use std::thread;
use glam::Vec4;

use data_compiler::gt_environment::GroundTruthEnvironment;
use data_compiler::rna_dual_head_cell_type_classifier::CellClassifier;
use generator::NativeWgpuEngine;
use generator::types::{CameraUniform, EngineResources};
use shared_biology::{BrainOpcode, SimParams};

// If you unified your wgpu versions in Cargo.toml to "28.0",
// you can just use `use wgpu;`. Otherwise, keep this:
use slint::wgpu_28::wgpu;

slint::include_modules!();

const BROAD: [&str; 5] = ["Endothelial", "Epithelial", "Immune/Fluid", "Mesenchymal", "Unknown"];
const GRANULAR: [&str; 25] = ["Arterial Endothelial Cell", "B Cell", "Basal Keratinocyte", "Capillary Endothelial Cell", "Cornified Keratinocyte", "Dendritic Cell", "Fibroblast", "Granular Keratinocyte", "Hair Follicle Cell", "Lymphatic Endothelial Cell", "Macrophage", "Mast Cell", "Merkel Cell", "Monocyte", "NK Cell", "Neutrophil", "Proliferating Keratinocyte", "Sebaceous Gland Cell", "Skeletal Muscle Cell", "Smooth Muscle Cell", "Spinous Keratinocyte", "Sweat Gland Cell", "T Cell", "Venous Endothelial Cell", "Unknown"];

fn main() -> anyhow::Result<()>
{
    unsafe { std::env::set_var("SLINT_BACKEND", "winit-wgpu"); }
    let ui = AppWindow::new()?;
    let ui_handle = ui.as_weak();

    // 1. Send a Tuple: The Graphics Engine AND the Raw Data
    let (tx, rx) = mpsc::channel::<(NativeWgpuEngine, GroundTruthEnvironment, Vec<Vec<f32>>)>();
    // 2. Create a separate storage for the biological data on the main thread
    let tensor_data: Rc<RefCell<Vec<Vec<f32>>>> = Rc::new(RefCell::new(Vec::new()));

    let engine_handle: Rc<RefCell<Option<NativeWgpuEngine>>> = Rc::new(RefCell::new(None));
    let app_env: Rc<RefCell<Option<GroundTruthEnvironment>>> = Rc::new(RefCell::new(None));
    let viewport_view: Rc<RefCell<Option<wgpu::TextureView>>> = Rc::new(RefCell::new(None));

    let view_clone = viewport_view.clone();
    let pan_zoom_engine = engine_handle.clone();

    // ==========================================
    // INTERACTIVITY CALLBACKS
    // ==========================================

    let dispatch_engine = engine_handle.clone();
    let ui_weak = ui.as_weak();

    ui.on_trigger_morphogenesis(move |cmd, p1, p2, p3, p4, endo, epi, imm, mes| {
        // if let Some(engine) = dispatch_engine.borrow_mut().as_mut() {
        //
        //     let total_tiles = (engine.sim_params.grid_width * engine.sim_params.grid_height) as usize;
        //
        //     // 1. INJECT ZYGOTE(S)
        //     engine.inject_batched_zygotes(0, 5);
        //
        //     // 2. SET THE GROWTH OPCODE
        //     let mut opcodes = vec![BrainOpcode::default(); total_tiles];
        //     for i in 0..total_tiles {
        //         opcodes[i] = BrainOpcode {
        //             cmd_id: cmd as u32,
        //             p1, p2, p3, p4,
        //             type_ratios: [epi, endo, mes, imm],
        //             _pad: [0.0; 3],
        //         };
        //     }
        //     engine.queue.write_buffer(&engine.opcodes, 0, bytemuck::cast_slice(&opcodes));
        //
        //     // Fixed the print statement to be dynamic!
        //     println!("🧬 Simulating {} Tiles: Cmd {}, P1: {:.2}, P2: {:.2}", total_tiles, cmd, p1, p2);
        //
        //     // 3. THE GROWTH PHASE (4 Steps of Mitosis)
        //     for current_step in 0..4 {
        //         engine.sim_params.epoch = current_step as u32;
        //
        //         // LOCK THE BOUNDARY FOR THIS TICK
        //         engine.sim_params.current_cell_count = engine.cpu_cell_count;
        //
        //         engine.queue.write_buffer(&engine.param_buffer, 0, bytemuck::bytes_of(&engine.sim_params));
        //         engine.dispatch_generation_sequence([0.0, 0.0, 0.0], 10.0, [0.0, 0.0, 1.0]);
        //
        //         // FETCH THE NEW COUNT FOR THE NEXT TICK
        //         engine.cpu_cell_count = engine.fetch_generated_cell_count();
        //     }
        //
        //     // 4. THE RELAXATION PHASE (20 Steps of pure physics)
        //     // Overwrite the opcodes with Command 7 (Halt) so no more cells spawn
        //     for i in 0..total_tiles { opcodes[i].cmd_id = 7; }
        //     engine.queue.write_buffer(&engine.opcodes, 0, bytemuck::cast_slice(&opcodes));
        //
        //     for _ in 0..50 {
        //         // Keep updating the boundary even during physics relaxation
        //         engine.sim_params.current_cell_count = engine.cpu_cell_count;
        //         engine.queue.write_buffer(&engine.param_buffer, 0, bytemuck::bytes_of(&engine.sim_params));
        //
        //         engine.dispatch_generation_sequence([0.0, 0.0, 0.0], 10.0, [0.0, 0.0, 1.0]);
        //     }
        //
        //     // 5. READBACK
        //     let new_cell_count = engine.fetch_generated_cell_count();
        //     println!("Final cell count: {}", new_cell_count);
        //
        //     engine.cpu_cell_count = new_cell_count;
        //
        //     // Frame the camera based on the tile grid
        //     engine.frame_generation_volume();
        //
        //     // If it's a 1x1 grid, zoom the camera way in so the clump fills the screen!
        //     // if total_tiles == 1 {
        //     //     engine.camera.radius = 40.0;
        //     // }
        //
        //     engine.camera_needs_update = true;
        //     if let Some(ui_instance) = ui_weak.upgrade() {
        //
        //         // Update the Slint text element!
        //         ui_instance.set_active_cell_count(new_cell_count as i32);
        //
        //         // Tell Slint to trigger the wgpu render pass
        //         ui_instance.window().request_redraw();
        //     }
        // }
    });

    let hover_env = app_env.clone();
    let hover_ui_handle = ui_handle.clone();

    ui.on_mouse_hover(move |mouse_pct_x, mouse_pct_y| {
        if let Some(engine) = pan_zoom_engine.borrow_mut().as_mut()
        {
            // 1. We already have 0.0-1.0 from Slint! Just convert to NDC (-1.0 to 1.0)
            let ndc_x = mouse_pct_x * 2.0 - 1.0;
            let ndc_y = 1.0 - (mouse_pct_y * 2.0); // UI Y is down, Math Y is up

            // 2. Bulletproof Near/Far Unprojection
            let inv_vp = engine.camera.build_view_projection_matrix().inverse();

            // WGPU Depth is 0.0 (near) to 1.0 (far)
            let near_clip = Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
            let far_clip = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

            let near_world = inv_vp * near_clip;
            let far_world = inv_vp * far_clip;

            let near_pos = near_world.truncate() / near_world.w;
            let far_pos = far_world.truncate() / far_world.w;

            // 3. Shoot the ray and hit the Z=0 plane
            let ray_dir = (far_pos - near_pos).normalize();
            let t = -near_pos.z / ray_dir.z;
            let world_hit = near_pos + ray_dir * t;

            // 4. Update Uniform
            engine.mouse_world_x = world_hit.x;
            engine.mouse_world_y = world_hit.y;
            engine.camera_needs_update = true;


            // ONLY RUN SEARCH IF THE ENVIRONMENT IS LOADED
            if let Some(env) = hover_env.borrow().as_ref()
            {
                let mut closest_cell = None;
                let mut min_dist_sq = f32::MAX;

                let bucket_size = env.sim_params.tile_size;
                let grid_width = env.sim_params.grid_width;

                let mouse_bx = (world_hit.x / bucket_size).floor() as i32;
                let mouse_by = (world_hit.y / bucket_size).floor() as i32;

                for offset_x in -1..=1 {
                    for offset_y in -1..=1 {
                        let check_bx = mouse_bx + offset_x;
                        let check_by = mouse_by + offset_y;

                        if check_bx < 0 || check_by < 0 { continue; }

                        let bucket_idx = (check_bx as u32) + (check_by as u32 * grid_width);

                        if let Some(cell_indices) = env.spatial_grid.get(&bucket_idx) {
                            for &idx in cell_indices {
                                let cell = &env.gpu_nodes[idx]; // Read straight from the source!

                                let dx = cell.pos.x - world_hit.x;
                                let dy = cell.pos.y - world_hit.y;
                                let dist_sq = dx * dx + dy * dy;

                                if dist_sq < min_dist_sq {
                                    min_dist_sq = dist_sq;
                                    closest_cell = Some(cell);
                                }
                            }
                        }
                    }
                }

                if min_dist_sq < (80.0 * 80.0) {
                    if let Some(cell) = closest_cell {
                        let info_string = format!(
                            "Cell Details:\nBroad ID: {}\nGranular ID: {}\nArea: {:.1}",
                            BROAD[cell.broad_id as usize], GRANULAR[cell.granular_id as usize], cell.area
                        );
                        if let Some(slint_ui) = hover_ui_handle.upgrade() {
                            slint_ui.set_hovered_cell_info(slint::SharedString::from(info_string));
                        }
                    }
                }
            }
        }
    });

    let scroll_engine = engine_handle.clone();
    let scroll_ui = ui_handle.clone();
    ui.on_mouse_wheel(move |dx, dy| {
        if let Some(engine) = scroll_engine.borrow_mut().as_mut() {
            engine.camera.radius -= dy * (engine.camera.radius * 0.005);
            if engine.camera.radius < 10.0 { engine.camera.radius = 10.0; }

            // Request exactly ONE redraw because the camera moved
            if let Some(ui) = scroll_ui.upgrade() { ui.window().request_redraw(); }
        }
    });

    let drag_engine = engine_handle.clone();
    let drag_ui = ui_handle.clone();
    ui.on_mouse_drag(move |dx, dy, shift_pressed| {
        if let Some(engine) = drag_engine.borrow_mut().as_mut() {

            if shift_pressed {
                // --- PANNING LOGIC ---
                let view_proj = engine.camera.build_view_projection_matrix();

                // Extract local coordinate axes from the view matrix rows
                let camera_right = view_proj.row(0).truncate().normalize();
                let camera_up = view_proj.row(1).truncate().normalize();

                // Scale panning sensitivity based on how far zoomed out you are
                let pan_speed = engine.camera.radius * 0.5;

                // Move the camera target
                engine.camera.target -= camera_right * dx * pan_speed;
                engine.camera.target += camera_up * dy * pan_speed;
            } else {
                // --- ROTATION LOGIC ---
                engine.camera.theta -= dx * 4.0;
                engine.camera.phi -= dy * 4.0;
                engine.camera.phi = engine.camera.phi.clamp(0.01, std::f32::consts::PI - 0.01);
            }

            engine.camera_needs_update = true;

            // Request exactly ONE redraw
            if let Some(ui) = drag_ui.upgrade() { ui.window().request_redraw(); }
        }
    });

    let click_engine = engine_handle.clone();
    let click_tensors = tensor_data.clone();
    let click_ui_handle = ui_handle.clone();

    let mut clickCount = 0;
    ui.on_viewport_clicked(move |mouse_pct_x, mouse_pct_y| {
        if let Some(engine) = click_engine.borrow_mut().as_mut()
        {
            // clickCount += 1;
            // println!("Clicked on {:?}", clickCount);

            // 1. Raycast (Same math as the hover callback!)
            let ndc_x = mouse_pct_x * 2.0 - 1.0;
            let ndc_y = 1.0 - (mouse_pct_y * 2.0);
            let inv_matrix = engine.camera.build_view_projection_matrix().inverse();

            let ray_clip = glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
            let ray_world = inv_matrix * ray_clip;
            let ray_pos = ray_world.truncate() / ray_world.w;
            let ray_dir = (ray_pos - engine.camera.eye).normalize();
            let t = -engine.camera.eye.z / ray_dir.z;
            let world_hit = engine.camera.eye + ray_dir * t;

            // 2. THE TILE MATH
            let tile_size = 150.0;
            let grid_width = 10; // Match this to your sim_params.grid_width!

            let tx = (world_hit.x / tile_size).floor() as i32;
            let ty = (world_hit.y / tile_size).floor() as i32;

            if tx >= 0 && ty >= 0 {
                let tile_idx = (tx + (ty * grid_width)) as usize;

                // 3. Set the Shader Bounding Box
                let min_x = tx as f32 * tile_size;
                let min_y = ty as f32 * tile_size;
                engine.selected_tile_bounds = glam::Vec4::new(min_x, min_y, min_x + tile_size, min_y + tile_size);
                engine.camera_needs_update = true;

                // 4. Update the UI with the Tensor Data!
                if let Some(tile_tensor) = click_tensors.borrow().get(tile_idx) {
                    let cell_count = tile_tensor[184];

                    // Update the UI text
                    println!("--- 🔬 SELECTED TILE {} ({} cells) ---", tile_idx, cell_count);

                    // Generate the image in memory and hand it to Slint
                    if cell_count > 0.0 {
                        let chart_image = render_ripley_chart(tile_tensor);

                        // UPGRADE THE WEAK HANDLE TO ACCESS SLINT!
                        if let Some(slint_ui) = click_ui_handle.upgrade() {
                            slint_ui.set_ripley_chart(chart_image);
                        }
                    }
                }
            }
        }
    });

    // ==========================================
    // RENDER LOOP
    // ==========================================
    ui.window().set_rendering_notifier(move |state, graphics_api| {
        match state {
            RenderingState::RenderingSetup => {
                let slint::GraphicsAPI::WGPU28 { device, queue, .. } = graphics_api else {
                    eprintln!("CRITICAL ERROR: Slint is NOT using the WGPU backend! It is using: {:?}", graphics_api);
                    return;
                };

                if let Some(ui) = ui_handle.upgrade() {
                    let size = wgpu::Extent3d { width: 800, height: 800, depth_or_array_layers: 1 };
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("Bio-SDE VRAM Texture"),
                        size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });

                    *viewport_view.borrow_mut() = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
                    let slint_image = slint::Image::try_from(texture).expect("Failed to map texture");
                    ui.set_viewport_texture(slint_image);
                }

                let device_clone = device.clone();
                let queue_clone = queue.clone();
                let tx_clone = tx.clone();

                // SPAWN BACKGROUND LOADER
                thread::spawn(move || {
                    println!("Background thread: Starting massive data load...");

                    let mut environment = GroundTruthEnvironment::new(
                        "../../../cellular_data/Spatial/Manchester/back/cell_feature_matrix.h5",
                        "../../../cellular_data/Spatial/Manchester/back/cells.parquet",
                        150.0
                    ).expect("Failed to load environment data");

                    let mut classifier = CellClassifier::load(
                        "../../../models/rna_cell_classifier/Dualv2/dual_head_classifier.onnx"
                    ).expect("Failed to load ONNX model");

                    environment.identify_broad_cell_types(
                        &mut classifier,
                        "../../../python/cell_type_model_builder/symbol2entrez.json",
                        "../../../python/cell_type_model_builder/true_entrez_list.json"
                    ).expect("Failed to identify cell types");

                    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();

                    let mut wgpu_engine = rt.block_on(async {
                        NativeWgpuEngine::new(
                            environment.sim_params,
                            EngineResources { device: Some(device_clone), queue: Some(queue_clone) }
                        ).await
                    });



                    // let volume_params = SimParams {
                    //     grid_width: 1,  // 3D Volume Width
                    //     grid_height: 1, // 3D Volume Height
                    //     grid_depth: 1,
                    //     tile_size: 150.0,
                    //     epoch: 0,
                    //     current_cell_count: 0,
                    //     _pad: [0; 6],
                    // };
                    //
                    // let mut wgpu_engine = rt.block_on(async {
                    //     NativeWgpuEngine::new(
                    //         volume_params,
                    //         EngineResources { device: Some(device_clone), queue: Some(queue_clone) }
                    //     ).await
                    // });


                    wgpu_engine.load_ground_truth(&environment.gpu_nodes, &environment.bounding_box);
                    wgpu_engine.cpu_cell_count = environment.gpu_nodes.len() as u32;

                    environment.build_spatial_hash();

                    let observations = rt.block_on(async {
                        environment.extract_observation_tensor().await.unwrap()
                    });

                    // PACK THE STATS INTO THE ENGINE BEFORE SENDING!
                    wgpu_engine.class_counts = environment.class_counts;
                    wgpu_engine.total_cells = environment.gpu_nodes.len();

                    let observations: Vec<Vec<f32>> = Vec::new();

                    println!("Background thread: Engine compiled and VRAM loaded!");
                    tx_clone.send((wgpu_engine, environment, observations)).unwrap();
                });
            }

            RenderingState::BeforeRendering =>
            {
                let slint::GraphicsAPI::WGPU28 { device, queue, .. } = graphics_api else { return; };
                let mut engine_lock = engine_handle.borrow_mut();

                if engine_lock.is_none()
                {
                    if let Ok((mut finished_engine, environment, observations)) = rx.try_recv()
                    {
                        *tensor_data.borrow_mut() = observations;

                        if let Some(ui) = ui_handle.upgrade()
                        {
                            ui.set_active_cell_count(finished_engine.cpu_cell_count as i32);

                            // UNPACK STATS AND BUILD THE UI ARRAY
                            let total = finished_engine.total_cells as f32;
                            let stats_vector = vec![
                                BroadStat {
                                    name: SharedString::from("Endothelial"),
                                    count: finished_engine.class_counts[0] as i32,
                                    pct: (finished_engine.class_counts[0] as f32 / total) * 100.0,
                                    color: slint::Color::from_rgb_u8(0, 122, 255)
                                },
                                BroadStat {
                                    name: SharedString::from("Epithelial"),
                                    count: finished_engine.class_counts[1] as i32,
                                    pct: (finished_engine.class_counts[1] as f32 / total) * 100.0,
                                    color: slint::Color::from_rgb_u8(255, 59, 48)
                                },
                                BroadStat {
                                    name: SharedString::from("Immune/Fluid"),
                                    count: finished_engine.class_counts[2] as i32,
                                    pct: (finished_engine.class_counts[2] as f32 / total) * 100.0,
                                    color: slint::Color::from_rgb_u8(255, 255, 255)
                                },
                                BroadStat {
                                    name: SharedString::from("Mesenchymal"),
                                    count: finished_engine.class_counts[3] as i32,
                                    pct: (finished_engine.class_counts[3] as f32 / total) * 100.0,
                                    color: slint::Color::from_rgb_u8(255, 204, 0)
                                },
                            ];

                            let stats_model = Rc::new(VecModel::from(stats_vector));
                            ui.set_global_stats(ModelRc::from(stats_model.clone()));
                        }

                        // Frame the camera perfectly around the newly loaded grid!
                        finished_engine.frame_generation_volume();

                        *app_env.borrow_mut() = Some(environment);
                        *engine_lock = Some(finished_engine);
                    } else {
                        if let Some(ui) = ui_handle.upgrade() { ui.window().request_redraw(); }
                        return;
                    }
                }

                let engine_ref = engine_lock.as_mut().unwrap();

                let mut current_scale = 1.0;
                if let Some(ui) = ui_handle.upgrade() {
                    current_scale = ui.get_z_slice() * 15.0;
                }

                // UPDATE UNIFORM WITH MOUSE COORDS (No more array padding!)
                let new_uniform = CameraUniform {
                    view_proj: engine_ref.camera.build_view_projection_matrix(),
                    cell_scale: current_scale,
                    mouse_x: engine_ref.mouse_world_x,
                    mouse_y: engine_ref.mouse_world_y,
                    _pad: 0.0,
                    tile_bounds: engine_ref.selected_tile_bounds,
                };

                queue.write_buffer(
                    &engine_ref.camera_buffer,
                    0,
                    bytemuck::cast_slice(&[new_uniform])
                );

                let view_borrow = view_clone.borrow();
                let Some(target_view) = view_borrow.as_ref() else { return; };

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Cell Drawing Encoder"),
                });

                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Instanced Cells Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: target_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.1, b: 0.1, a: 1.0 }),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });

                    render_pass.set_pipeline(&engine_ref.render_pipeline);
                    render_pass.set_bind_group(0, &engine_ref.cell_bind_group, &[]);
                    render_pass.set_bind_group(1, &engine_ref.camera_bind_group, &[]);
                    render_pass.draw(0..4, 0..engine_ref.cpu_cell_count);
                }

                queue.submit(std::iter::once(encoder.finish()));

                // if let Some(ui) = ui_handle.upgrade() {
                //     ui.window().request_redraw();
                // }
            }
            _ => {}
        }
    }).unwrap();

    ui.run()?;
    Ok(())
}

use plotters::prelude::*;
// 1. Change the import at the top
use slint::{SharedPixelBuffer, Image, Rgb8Pixel}; // <-- Change to Rgb8Pixel

fn render_ripley_chart(tensor: &[f32]) -> Image {
    let ripley_data = &tensor[0..20];

    // 1. DYNAMIC SCALING: Find the true range of your data
    let mut min_val = ripley_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let mut max_val = ripley_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Add a 10% "breathing room" margin so the line doesn't touch the edges
    let margin = (max_val - min_val).abs() * 0.1;
    if margin == 0.0 { // Handle flat data
        min_val -= 1.0;
        max_val += 1.0;
    } else {
        min_val -= margin;
        max_val += margin;
    }

    let mut pixel_buffer = SharedPixelBuffer::<Rgb8Pixel>::new(500, 300);

    {
        let root = BitMapBackend::with_buffer(
            pixel_buffer.make_mut_bytes(),
            (500, 300),
        ).into_drawing_area();

        root.fill(&RGBColor(18, 18, 18)).unwrap();

        let mut chart = ChartBuilder::on(&root)
            .margin(15)
            .x_label_area_size(20)
            .y_label_area_size(35)
            // 2. USE THE CALCULATED BOUNDS
            .build_cartesian_2d(0f32..19f32, min_val..max_val)
            .unwrap();

        chart.configure_mesh()
            .axis_style(WHITE.mix(0.3))
            .label_style(("sans-serif", 12).into_font().color(&WHITE))
            .draw()
            .unwrap();

        chart.draw_series(LineSeries::new(
            ripley_data.iter().enumerate().map(|(x, y)| (x as f32, *y)),
            &CYAN,
        )).unwrap();

        root.present().unwrap();
    }

    Image::from_rgb8(pixel_buffer)
}