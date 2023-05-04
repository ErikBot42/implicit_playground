use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use std::time::Instant;
use std::{
    fs::read_to_string,
    mem::{replace, size_of},
};

use cgmath::prelude::*;

// `~/.cargo/bin/wasm-pack build`
// # for plain html:
// `~/.cargo/bin/wasm-pack build --target web`

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        window.set_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }

    let mut state = State::new(window).await;
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() => match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    },
                ..
            } => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(physical_size) => {
                state.resize(*physical_size);
            }
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                // new_inner_size is &&mut so we have to dereference it twice
                state.resize(**new_inner_size);
            }
            _ => {
                state.input(event);
            }
        },
        Event::RedrawRequested(window_id) if window_id == state.window().id() => {
            state.update();
            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                // The system is out of memory, we should probably quit
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            // RedrawRequested will only trigger once, unless we manually
            // request it.
            state.window().request_redraw();
        }
        _ => {}
    });
}

// lib.rs
use winit::window::Window;

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,

    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,

    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    compute_bind_group: wgpu::BindGroup,
    camera_controller: CameraController,

    last_frame_print: Instant,
    frames: usize,

    player: PlayerController,

    last_update: Instant,

    active_gamepad: Option<gilrs::GamepadId>,
    gilrs_context: gilrs::Gilrs,

    shadow_compute_pipeline: wgpu::ComputePipeline,
    shadow_width: u32,
    shadow_height: u32,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window) -> Self {
        let gpu_get_device_timer_start = Instant::now();

        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .enumerate_adapters(wgpu::Backends::all())
            .filter(|adapter| {
                // Check if this adapter supports our surface
                adapter.is_surface_supported(&surface)
            })
            .next()
            .unwrap();
      
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();
        let gpu_get_device_timer_elapsed = gpu_get_device_timer_start.elapsed();

        let gpu_pipeline_setup_timer_start = Instant::now();

        let surface_caps = surface.get_capabilities(&adapter);

        // shader assumes srgb
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0], // for example vsync/not vsync
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        let camera = Camera {
            // position the camera one unit up and 2 units back
            // +z is out of the screen
            eye: (0.0, 1.0, 2.0).into(),
            // have it look at the origin
            target: (0.0, 0.0, 0.0).into(),
            // which way is "up"
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            forward: 0.0,
            turn: 0.0,
            last_update: Instant::now(),
        };
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shadow_height = 100;
        let shadow_width = 100;
        //let shadow_elements = shadow_width * shadow_height;
        //let shadow: Vec<f32> = (0..shadow_elements).map(|_| 0_f32).collect();
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Texture"),
            size: wgpu::Extent3d {
                width: shadow_width,
                height: shadow_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float, // Red channel only. 32 bit float per channel. Float in shader.
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let shadow_texture_view =
            shadow_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let pre_march_height = 72; //  9 * 8
        let pre_march_width = 128; // 16 * 8

        let pre_march_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pre march texture"),
            size: wgpu::Extent3d {
                width: pre_march_width,
                height: pre_march_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let pre_march_texture_view =
            pre_march_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let pre_march_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        //let camera_buffer_binding = camera_buffer.as_entire_binding();

        let (bind_group, bind_group_layout) = {
            make_bind_group(
                "Render",
                &device,
                [
                    (
                        wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        camera_buffer.as_entire_binding(),
                    ),
                    (
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        wgpu::BindingResource::TextureView(&shadow_texture_view),
                    ),
                    (
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        wgpu::BindingResource::Sampler(&shadow_sampler),
                    ),
                ],
            )
        };
        let (compute_bind_group, compute_bind_group_layout) = {
            make_bind_group(
                "Shadow",
                &device,
                [
                    (
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        wgpu::BindingResource::TextureView(&shadow_texture_view),
                    ),
                    (
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        camera_buffer.as_entire_binding(),
                    ),
                ],
            )
        };
        let (pre_march_bind_group, pre_march_bind_group_layout) = make_bind_group(
            "Shadow",
            &device,
            [
                (
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rg32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    wgpu::BindingResource::TextureView(&pre_march_texture_view),
                ),
                (
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    camera_buffer.as_entire_binding(),
                ),
            ],
        );

        surface.configure(&device, &config);

        let fragment_vertex_source =
            read_to_string("src/sdf.wgsl").unwrap() + &read_to_string("src/shader.wgsl").unwrap();
        let compute_source =
            read_to_string("src/sdf.wgsl").unwrap() + &read_to_string("src/compute.wgsl").unwrap();
        let pre_march_source = read_to_string("src/sdf.wgsl").unwrap()
            + &read_to_string("src/pre_march.wgsl").unwrap();

        println!("{compute_source}");

        let fragment_vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment and vertex shader"),
            source: wgpu::ShaderSource::Wgsl(fragment_vertex_source.into()),
        });
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute shader"),
            source: wgpu::ShaderSource::Wgsl(compute_source.into()),
        });
        let pre_march_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pre march shader"),
            source: wgpu::ShaderSource::Wgsl(pre_march_source.into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &fragment_vertex_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_vertex_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None, // 5.
        });
        let shadow_compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shadow compute pipeline layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });
        let shadow_compute_pipeline: wgpu::ComputePipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Shadow compute Pipeline"),
                layout: Some(&shadow_compute_pipeline_layout),
                module: &compute_shader,
                entry_point: "compute_main",
            });
        let pre_march_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pre march pipeline layout"),
                bind_group_layouts: &[&pre_march_bind_group_layout],
                push_constant_ranges: &[],
            });
        let pre_march_pipeline: wgpu::ComputePipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Pre march pipeline"),
                layout: Some(&pre_march_pipeline_layout),
                module: &pre_march_shader,
                entry_point: "compute_main",
            });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let num_vertices = VERTICES.len() as u32;

        let camera_controller = CameraController::new(0.2);
        let last_frame = Instant::now();
        let player = PlayerController::new();

        let last_update = Instant::now();

        let active_gamepad = None;
        let gilrs_context = gilrs::Gilrs::new().unwrap();

        println!(
            "Getting device took {:?}\nOther setup took {:?}",
            gpu_get_device_timer_elapsed,
            gpu_pipeline_setup_timer_start.elapsed()
        );
        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            num_vertices,
            camera,
            camera_uniform,
            camera_buffer,
            bind_group,
            camera_controller,
            last_frame_print: last_frame,
            frames: 0,
            player,
            last_update,
            active_gamepad,
            gilrs_context,
            shadow_compute_pipeline,
            shadow_width,
            shadow_height,
            compute_bind_group,
        }
    }

    //fn init_shadows(&mut self) {
    //}

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    // was an input fully processed?
    fn input(&mut self, event: &WindowEvent) -> bool {
        self.player.process_events(event) && self.camera_controller.process_events(event)
    }

    fn update(&mut self) {
        let dt = replace(&mut self.last_update, Instant::now())
            .elapsed()
            .as_secs_f32();
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.player
            .update(dt, &mut self.active_gamepad, &mut self.gilrs_context);

        let uniform = self.player.get_uniform();

        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            //bytemuck::cast_slice(&[self.camera_uniform]),
            bytemuck::cast_slice(&[uniform]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let print_freq: u32 = 120;
        self.frames = (self.frames + 1) % print_freq as usize;
        if self.frames == 0 {
            println!(
                "{:?}",
                replace(&mut self.last_frame_print, Instant::now()).elapsed() / print_freq
            );
        }
        /*
            [src/lib.rs:544] self.surface.get_current_texture() = Ok(
            SurfaceTexture {
                texture: Texture {
                    context: Context {
                        type: "Native",
                    },
                    id: ObjectId {
                        id: Some(
                            2305844009941073922,
                        ),
                    },
                    data: Any { .. },
                    owned: false,
                    descriptor: TextureDescriptor {
                        label: None,
                        size: Extent3d {
                            width: 958,
                            height: 1030,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: D2,
                        format: Bgra8UnormSrgb,
                        usage: TextureUsages(
                            RENDER_ATTACHMENT,
                        ),
                        view_formats: [],
                    },
                },
                suboptimal: false,
                presented: false,
                detail: Any { .. },
            },
        )
        */

        let output: wgpu::SurfaceTexture = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder: wgpu::CommandEncoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compute Encoder"),
                });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Shadow Compute pass"),
            });
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.set_pipeline(&self.shadow_compute_pipeline);
            cpass.dispatch_workgroups(self.shadow_width, self.shadow_height, 1);
        }

        //let mut encoder: wgpu::CommandEncoder =
        //    self.device
        //        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
        //            label: Some("Render Encoder"),
        //        });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // skip clearing buffer
                        load: wgpu::LoadOp::Load,
                        //load: wgpu::LoadOp::Clear(wgpu::Color {
                        //    r: 0.1,
                        //    g: 0.2,
                        //    b: 0.3,
                        //    a: 1.0,
                        //}),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..self.num_vertices, 0..1);
        }

        // submit will accept anything that implements IntoIter
        self.queue
            //.submit([compute_encoder.finish(), encoder.finish()]);
            .submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn make_bind_group<const N: usize>(
    name: &str,
    device: &wgpu::Device,
    data: [(wgpu::ShaderStages, wgpu::BindingType, wgpu::BindingResource); N],
) -> (wgpu::BindGroup, wgpu::BindGroupLayout) {
    let bind_group_layout: wgpu::BindGroupLayout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &data
                .iter()
                .enumerate()
                .map(
                    |(binding, &(visibility, ty, _))| wgpu::BindGroupLayoutEntry {
                        binding: binding as u32,
                        visibility,
                        ty,
                        count: None,
                    },
                )
                .collect::<Vec<_>>(),
            label: Some(&format!("{name}: bind_group_layout")),
        });
    (
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &data
                .into_iter()
                .enumerate()
                .map(|(binding, (_, _, resource))| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource,
                })
                .collect::<Vec<_>>(),
            label: Some(&format!("{name}: bind_group")),
        }),
        bind_group_layout,
    )
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
}
impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x2];
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

// A    D
//
// C    B
#[rustfmt::skip]
const VERTICES: &[Vertex] = {
    //TODO: remove z coord in input data
    const A: Vertex = Vertex { position: [ 1.0,  1.0]};
    const B: Vertex = Vertex { position: [-1.0, -1.0]};
    const C: Vertex = Vertex { position: [ 1.0, -1.0]};
    const D: Vertex = Vertex { position: [-1.0,  1.0]};
    &[
        A, B, C, 
        A, D, B,
    ]
};

#[derive(Debug)]
struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
    forward: f32,
    turn: f32,
    last_update: Instant,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        cgmath::Matrix4::from_translation(cgmath::Vector3::new(0.0, 0.0, self.forward))
            * cgmath::Matrix4::from_angle_y(cgmath::Deg(self.turn))
            * cgmath::Matrix4::from_angle_x(cgmath::Deg(0.0))

        //cgmath::Matrix4::from_translation(cgmath::Vector3::new(0.0, 0.0, 0.3))
        //    * cgmath::Matrix4::from_angle_y(cgmath::Deg(45.0))
        //    * cgmath::Matrix4::from_angle_x(cgmath::Deg(45.0))

        //OPENGL_TO_WGPU_MATRIX;// * proj * view;
    }
}
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}
impl CameraUniform {
    fn from_mat(mat: cgmath::Matrix4<f32>) -> Self {
        Self {
            view_proj: mat.into(),
        }
    }
    fn new() -> Self {
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }
    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

struct PlayerController {
    velocity: f32, // forward speed, also controls rotational speed.
    key_up: bool,
    key_down: bool,
    key_left: bool,
    key_right: bool,
    key_forward: bool,
    key_back: bool,

    state: cgmath::Matrix4<f32>,
}
impl PlayerController {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            velocity: 0.0,
            key_up: false,
            key_down: false,
            key_left: false,
            key_right: false,
            key_forward: false,
            key_back: false,
            state: cgmath::Matrix4::from_translation(cgmath::Vector3::new(0.0, 0.0, 0.5)),
        }
    }
    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let s = *state == ElementState::Pressed;
                use VirtualKeyCode::*;
                match keycode {
                    Up | W => Some(&mut self.key_forward),
                    Down | S => Some(&mut self.key_back),
                    Left | A => Some(&mut self.key_left),
                    Right | D => Some(&mut self.key_right),
                    Space => Some(&mut self.key_up),
                    LControl | LShift => Some(&mut self.key_down),
                    _ => None,
                }
                .map(|k| *k = s)
                .is_some()
            }
            _ => false,
        }
    }
    fn update(
        &mut self,
        dt: f32,
        active_gamepad: &mut Option<gilrs::GamepadId>,
        gilrs_context: &mut gilrs::Gilrs,
    ) {
        let dt = 0.2;
        let max_acceleration = 0.02; //todo make dynamic based on distance to object/centre
        let turn_factor = 0.3;

        let p = self.state.w.truncate();
        let s = 0.1; //sdf::sdf(p);

        let fac = s;

        //let decay = 0.2; (todo) // depend on acceleration?

        let range = |a, b| (a as i32 - b as i32) as f32;

        let mut translation_input: cgmath::Vector3<f32> = cgmath::Vector3::zero();
        let mut rotation_input: cgmath::Vector3<f32> = cgmath::Vector3::zero();

        let mut right = range(self.key_right, self.key_left);
        let mut forward = range(self.key_forward, self.key_back);
        let mut up = range(self.key_up, self.key_down);

        translation_input += cgmath::Vector3::new(right, up, forward);

        {
            //use gilrs::{Button, Event, Gilrs};
            //let mut gilrs = Gilrs::new().unwrap();
            while let Some(gilrs::Event { id, event, time }) = gilrs_context.next_event() {
                //println!("{:?} {}: {:?}", time, id, event);
            }

            fn value_with_deadzone(
                gamepad: gilrs::Gamepad,
                deadzone: f32,
                axis: gilrs::ev::Axis,
            ) -> f32 {
                gamepad.value(axis)
            }

            for (_id, gamepad) in gilrs_context.gamepads() {
                //println!("{} is {:?}", gamepad.name(), gamepad.power_info());
                use gilrs::ev::Axis::*;
                use gilrs::ev::Button::*;
                translation_input += cgmath::Vector3::new(
                    gamepad.value(LeftStickX),
                    //gamepad.value(RightStickY),
                    //
                    (gamepad
                        .button_data(RightTrigger2)
                        .map(|b| b.value())
                        .unwrap_or(0.0)
                        - gamepad
                            .button_data(LeftTrigger2)
                            .map(|b| b.value())
                            .unwrap_or(0.0))
                        + (gamepad
                            .button_data(RightTrigger)
                            .map(|b| b.value())
                            .unwrap_or(0.0)
                            - gamepad
                                .button_data(LeftTrigger)
                                .map(|b| b.value())
                                .unwrap_or(0.0)),
                    gamepad.value(LeftStickY),
                    //gamepad.value(RightZ), // - gamepad.value(LeftZ),
                );

                rotation_input += cgmath::Vector3::new(
                    gamepad.value(RightStickX),
                    gamepad.value(RightStickY),
                    0.0,
                )

                //println!(
                //    "{}\n",
                //    [
                //        LeftStickX,
                //        LeftStickY,
                //        LeftZ,
                //        RightStickX,
                //        RightStickY,
                //        RightZ,
                //        DPadX,
                //        DPadY
                //    ].into_iter()
                //    .map(|a| format!("{:?} {:?}\n", a, gamepad.value(a)))
                //    .fold(String::new(), |a, b| a + &b),
                //);

                //right += gamepad.value(LeftStickX);
                //forward += gamepad.value(RightStickY);
                //up += gamepad.value(LeftStickY);
            }
        }

        fn deadzone(a: f32) -> f32 {
            if a.abs() < 0.1 {
                0.0
            } else {
                a
            }
        }

        rotation_input = rotation_input.map(deadzone);
        translation_input = translation_input.map(deadzone);

        let da = dt * max_acceleration * forward;
        self.velocity += da;
        self.velocity *= 0.99;

        let rotation = cgmath::Matrix4::from_angle_y(cgmath::Rad(-right * dt * turn_factor))
            * cgmath::Matrix4::from_angle_x(cgmath::Rad(up * dt * turn_factor));

        let rotation =
            cgmath::Matrix4::from_angle_y(cgmath::Rad(-rotation_input.x * dt * turn_factor))
                * cgmath::Matrix4::from_angle_x(cgmath::Rad(rotation_input.y * dt * turn_factor));

        translation_input.z *= -1.0;
        let translation = cgmath::Matrix4::from_translation(
            /*(-self.state.z.truncate()) * */
            (self.state * translation_input.extend(0.0)).truncate() * dt * fac * p.magnitude(),
            //(-self.state.z.truncate()) * forward * dt * fac * p.magnitude(),
        );

        //let total = trans * rot;

        self.state = translation * self.state * rotation;
        //self.state = total;

        //cgmath::Matrix4::from_translation(cgmath::Vector3::new(0.0, 0.0, self.forward))
        //    * cgmath::Matrix4::from_angle_y(cgmath::Deg(self.turn))
        //    * cgmath::Matrix4::from_angle_x(cgmath::Deg(0.0))
    }
    fn get_uniform(&self) -> CameraUniform {
        CameraUniform::from_mat(self.state)
    }
}

#[derive(Debug)]
struct CameraController {
    speed: f32,
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
}
impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            forward: false,
            backward: false,
            left: false,
            right: false,
        }
    }
    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                use VirtualKeyCode::*;
                match keycode {
                    W | Up => {
                        self.forward = is_pressed;
                        true
                    }
                    A | Left => {
                        self.left = is_pressed;
                        true
                    }
                    S | Down => {
                        self.backward = is_pressed;
                        true
                    }
                    D | Right => {
                        self.right = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn update_camera(&self, camera: &mut Camera) {
        //use cgmath::InnerSpace;

        let dt = replace(&mut camera.last_update, Instant::now())
            .elapsed()
            .as_secs_f32();

        if self.forward {
            camera.forward += dt;
        }
        if self.backward {
            camera.forward -= dt;
        }
        if self.right {
            camera.turn += 180.0 * dt
        }
        if self.left {
            camera.turn -= 180.0 * dt
        }
    }
}

mod sdf {
    use cgmath::*;
    fn sd_sphere(pos: Vector3<f32>, r: f32) -> f32 {
        pos.magnitude() - r
    }

    fn mod_space(v: f32, r: f32) -> f32 {
        (v + r).abs() % (2.0 * r) - r
    }

    fn sd_scale<F: Fn(Vector3<f32>) -> f32>(p: Vector3<f32>, f: F, s: f32) -> f32 {
        f(p / s) * s
    }

    fn sphere_field(p: Vector3<f32>) -> f32 {
        let r = 0.5;
        sd_sphere(
            Vector3::new(
                mod_space(p.x + r, r),
                mod_space(p.y + r, r),
                mod_space(p.z + r, r),
            ),
            0.7 * r,
        )
    }

    pub(crate) fn sdf(p: Vector3<f32>) -> f32 {
        //sphere_field(p)
        //sd_menger(p)
        sd_menger_recursive(p)
    }

    fn sd_menger_recursive(p: Vector3<f32>) -> f32 {
        let s: f32 = 0.25;

        let d: f32 = p.magnitude();

        let x = d.log(s).floor();
        return sd_menger_single(p / s.powf(x))
            * s.powf(x)
                .min(sd_menger_single(p / s.powf(x + 1.0)) * s.powf(x + 1.0));
    }

    fn sd_cross(p: Vector3<f32>) -> f32 {
        let r = 1.0;
        let px = p.x.abs();
        let py = p.y.abs();
        let pz = p.z.abs();

        let da = px.max(py);
        let db = py.max(pz);
        let dc = pz.max(px);
        return da.min(db).min(dc) - r;
    }

    fn sd_menger_single(p: Vector3<f32>) -> f32 {
        let d = sd_box(p, Vector3::new(1.0, 1.0, 1.0));
        let c = sd_cross(p * 3.0) / 3.0;
        d.max(-c)
    }

    fn sd_box(p: Vector3<f32>, b: Vector3<f32>) -> f32 {
        let q = p.map(f32::abs) - b;
        q.map(|a| a.max(0.0)).magnitude() + q.x.max(q.y).max(q.z).min(0.0)
    }
}
