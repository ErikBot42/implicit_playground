use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

//use std::time::Instant;
use wasm_timer::Instant;

use std::borrow::Cow;
use std::{
    fs::read_to_string,
    mem::{replace, size_of},
};

use cgmath::prelude::*;

const NUM_SDF: usize = 4;

// `~/.cargo/bin/wasm-pack build`
// # for plain html:
// `~/.cargo/bin/wasm-pack build --target web`

fn print_universal(s: impl Into<String>) {
    let s: String = s.into();

    #[cfg(target_arch = "wasm32")]
    {
        web_sys::console::log_1(&s.into());
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("{}", String::from(s));
    }
}

macro_rules! dbgu {
    ($($tts:tt)*) => {
        let s = format!("{:#?}", $($tts)*);
        print_universal(s);
    }
}

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
    let window = WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize {
            width: 960,
            height: 540,
        })
        .build(&event_loop)
        .unwrap();

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

    render_pipelines: Vec<wgpu::RenderPipeline>,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,

    camera_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    compute_bind_group: wgpu::BindGroup,

    start_time: Instant,
    last_frame_print: Instant,
    frames_since_start: usize,
    frames: usize,

    player: PlayerController,

    last_update: Instant,

    active_gamepad: Option<gilrs::GamepadId>,
    gilrs_context: gilrs::Gilrs,

    shadow_compute_pipelines: Vec<wgpu::ComputePipeline>,
    shadow_width: u32,
    shadow_height: u32,

    pipeline_index: usize,
    pre_march_pipelines: Vec<wgpu::ComputePipeline>,
    pre_march_bind_group: wgpu::BindGroup,
    pre_march_width: u32,
    pre_march_height: u32,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window) -> Self {
        // ****************************************************************
        // Init surface/device
        // ****************************************************************
        print_universal("This should print to both wasm and stdout");

        let (size, surface, device, queue, gpu_get_device_timer_elapsed, config) =
            prepare_surface(&window).await;

        // ****************************************************************
        // Init other
        // ****************************************************************

        let shadow_height = 128;
        let shadow_width = 128;
        let pre_march_height = 9 * 8; //72; //  9 * 8
        let pre_march_width = 16 * 8; //128; // 16 * 8

        let gpu_pipeline_setup_timer_start = Instant::now();

        let player = PlayerController::new();
        let camera_uniform = player.get_uniform();
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shadow_texture: wgpu::Texture = device.create_texture(&wgpu::TextureDescriptor {
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
        let shadow_texture_view: wgpu::TextureView =
            shadow_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_sampler: wgpu::Sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            //mag_filter: wgpu::FilterMode::Nearest,
            //min_filter: wgpu::FilterMode::Nearest,
            //mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

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

        let (bind_group, bind_group_layout): (wgpu::BindGroup, wgpu::BindGroupLayout) = {
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
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        wgpu::BindingResource::TextureView(&shadow_texture_view),
                    ),
                    (
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        wgpu::BindingResource::Sampler(&shadow_sampler),
                    ),
                    (
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        wgpu::BindingResource::TextureView(&pre_march_texture_view),
                    ),
                    (
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        wgpu::BindingResource::Sampler(&pre_march_sampler),
                    ),
                ],
            )
        };
        let (compute_bind_group, compute_bind_group_layout): (
            wgpu::BindGroup,
            wgpu::BindGroupLayout,
        ) = {
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
            "pre march",
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

        let pre_march_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pre march pipeline layout"),
                bind_group_layouts: &[&pre_march_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline_layout: wgpu::PipelineLayout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let shadow_compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shadow compute pipeline layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let (render_pipelines, shadow_compute_pipelines, pre_march_pipelines) = {
            macro_rules! load_file {
                ($a:tt) => {
                    //include_str!($a).to_string()
                    read_to_string(concat!("src/", $a)).unwrap()
                };
            }

            let sdf_lib_source = load_file!("sdf.wgsl");
            let fragment_vertex_source = load_file!("shader.wgsl");
            let compute_source = load_file!("compute.wgsl");
            let pre_march_source = load_file!("pre_march.wgsl");

            itertools::multiunzip(sdf_wgsl_gen(NUM_SDF).iter().map(|sdf_prefix| {
                let fragment_vertex_source: String =
                    sdf_prefix.clone() + &sdf_lib_source + &fragment_vertex_source;
                let compute_source: String = sdf_prefix.clone() + &sdf_lib_source + &compute_source;
                let pre_march_source: String =
                    sdf_prefix.clone() + &sdf_lib_source + &pre_march_source;
                let render_pipeline = make_render_pipeline(
                    &device,
                    fragment_vertex_source,
                    &render_pipeline_layout,
                    &config,
                );
                let shadow_compute_pipeline =
                    make_compute_pipeline(&device, compute_source, &shadow_compute_pipeline_layout);
                let pre_march_pipeline =
                    make_compute_pipeline(&device, pre_march_source, &pre_march_pipeline_layout);
                (render_pipeline, shadow_compute_pipeline, pre_march_pipeline)
            }))
            //.unzip()
        };
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let num_vertices = VERTICES.len() as u32;

        let active_gamepad = None;
        let gilrs_context = gilrs::Gilrs::new().unwrap();

        println!(
            "Getting device took {:?}\nOther setup took {:?}",
            gpu_get_device_timer_elapsed,
            gpu_pipeline_setup_timer_start.elapsed()
        );
        let now = Instant::now();
        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipelines,
            vertex_buffer,
            num_vertices,
            camera_buffer,
            bind_group,
            last_frame_print: now,
            frames: 0,
            player,
            last_update: now,
            active_gamepad,
            gilrs_context,
            shadow_compute_pipelines,
            shadow_width,
            shadow_height,
            compute_bind_group,
            pipeline_index: 0,
            pre_march_pipelines,
            pre_march_bind_group,
            pre_march_width,
            pre_march_height,
            start_time: now,
            frames_since_start: 0,
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
        self.player.process_events(event)
    }

    fn update(&mut self) {
        let dt = replace(&mut self.last_update, Instant::now())
            .elapsed()
            .as_secs_f32();
        self.player.update(
            dt,
            &mut self.pipeline_index,
            &mut self.active_gamepad,
            &mut self.gilrs_context,
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let uniform = self.player.get_uniform();
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[uniform]));

        let print_freq: u32 = 12;
        self.frames = (self.frames + 1) % print_freq as usize;
        self.frames_since_start += 1;

        if self.frames == 0 {
            let frame_period =
                replace(&mut self.last_frame_print, Instant::now()).elapsed() / print_freq;
            let fps = 1.0 / frame_period.as_secs_f32();

            let total_frame_period = self.start_time.elapsed() / self.frames_since_start as _;
            let total_fps = 1.0 / total_frame_period.as_secs_f32();

            println!(
                "fps: {fps:.1}, ({frame_period:?}), fps: {total_fps:.1}, ({total_frame_period:?})",
            );
        }

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
            cpass.set_pipeline(&self.shadow_compute_pipelines[self.pipeline_index]);
            cpass.dispatch_workgroups(self.shadow_width, self.shadow_height, 1);
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Pre march pass"),
            });
            cpass.set_bind_group(0, &self.pre_march_bind_group, &[]);
            cpass.set_pipeline(&self.pre_march_pipelines[self.pipeline_index]);
            //cpass.dispatch_workgroups(self.pre_march_width, self.pre_march_height, 1);
            cpass.dispatch_workgroups(self.pre_march_width, self.pre_march_height, 1);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // skip clearing buffer
                        //load: wgpu::LoadOp::Load,
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.5,
                            g: 0.5,
                            b: 0.5,
                            a: 0.5,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipelines[self.pipeline_index]);
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

async fn prepare_surface(
    window: &Window,
) -> (
    winit::dpi::PhysicalSize<u32>,
    wgpu::Surface,
    wgpu::Device,
    wgpu::Queue,
    std::time::Duration,
    wgpu::SurfaceConfiguration,
) {
    let gpu_get_device_timer_start = Instant::now();

    let size: winit::dpi::PhysicalSize<u32> = window.inner_size();

    let enabled_backends = wgpu::Backends::all();
    // & (!wgpu::Backends::GL);

    // The instance is a handle to our GPU
    // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
    let instance: wgpu::Instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: enabled_backends,
        dx12_shader_compiler: Default::default(),
    });

    // # Safety
    //
    // The surface needs to live as long as the window that created it.
    // State owns the window so this should be safe.
    let surface: wgpu::Surface = unsafe { instance.create_surface(window) }.unwrap();

    //instance.enumerate_adapters(wgpu::Backends::all());
    //
    instance
        .enumerate_adapters(wgpu::Backends::all())
        .for_each(|adapter| {
            dbgu!(adapter);
            dbgu!(adapter.get_info());
        });

    dbgu!("selecting adapter...");
    let adapter: wgpu::Adapter = instance
        .enumerate_adapters(enabled_backends)
        .find(|adapter| {
            dbgu!(adapter);
            // Check if this adapter supports our surface
            adapter.is_surface_supported(&surface)
        })
        .unwrap();
    dbgu!("selected adapter");

    dbgu!(adapter.features());
    dbgu!(wgpu::Limits::downlevel_webgl2_defaults());

    let min_limit_needed = wgpu::Limits {
        max_uniform_buffers_per_shader_stage: 11,
        max_storage_buffers_per_shader_stage: 0,
        max_storage_textures_per_shader_stage: 1, // 0
        max_dynamic_storage_buffers_per_pipeline_layout: 0,
        max_storage_buffer_binding_size: 0,
        max_vertex_buffer_array_stride: 255,
        max_compute_workgroup_storage_size: 0,
        max_compute_invocations_per_workgroup: 1, // 0
        max_compute_workgroup_size_x: 1,           //1          // 0
        max_compute_workgroup_size_y: 1,           //1          // 0
        max_compute_workgroup_size_z: 1,           //1          // 0
        max_compute_workgroups_per_dimension: 1024, // 0
        ..wgpu::Limits::downlevel_defaults()
    };

    dbgu!("requesting device...");
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                //wgpu::Features::empty(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we'll have to disable some.
                limits: if cfg!(target_arch = "wasm32") {
                    //wgpu::Limits::downlevel_webgl2_defaults()
                    //wgpu::Limits::downlevel_defaults()
                    // attempt to use more resonable defaults
                    //wgpu::Limits::default()
                    min_limit_needed
                } else {
                    // use strict web limits
                    //wgpu::Limits::downlevel_webgl2_defaults()
                    //wgpu::Limits::downlevel_defaults() // <- ok
                    //wgpu::Limits::default()
                    min_limit_needed
                },
                label: None,
            },
            None, // Trace path
        )
        .await
        .unwrap();

    dbgu!(device.features());

    let gpu_get_device_timer_elapsed = gpu_get_device_timer_start.elapsed();

    let surface_caps = surface.get_capabilities(&adapter);
    // shader assumes srgb
    let surface_format: wgpu::TextureFormat = surface_caps
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
        present_mode: wgpu::PresentMode::AutoVsync, //AutoVsync, //Mailbox,//, //surface_caps.present_modes[0], // for example vsync/not vsync
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
    };
    (
        size,
        surface,
        device,
        queue,
        gpu_get_device_timer_elapsed,
        config,
    )
}

fn make_compute_pipeline<'a>(
    device: &wgpu::Device,
    compute_source: impl Into<Cow<'a, str>>,
    compute_pipeline_layout: &wgpu::PipelineLayout,
) -> wgpu::ComputePipeline {
    let compute_shader: wgpu::ShaderModule =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute shader"),
            source: wgpu::ShaderSource::Wgsl(compute_source.into()),
        });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute pipeline"),
        layout: Some(compute_pipeline_layout),
        module: &compute_shader,
        entry_point: "compute_main",
    })
}

fn make_render_pipeline(
    device: &wgpu::Device,
    fragment_vertex_source: String,
    render_pipeline_layout: &wgpu::PipelineLayout,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::RenderPipeline {
    let fragment_vertex_shader: wgpu::ShaderModule =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment and vertex shader"),
            source: wgpu::ShaderSource::Wgsl(fragment_vertex_source.into()),
        });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(render_pipeline_layout),
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
    })
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

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TestUniform {
    view_proj: [[f32; 4]; 4],
    random_norm: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    d1: [f32; 4],
}
impl CameraUniform {
    fn from_mat_data(mat: cgmath::Matrix4<f32>, d1: [f32; 4]) -> Self {
        Self {
            view_proj: mat.into(),
            d1,
        }
    }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

const ZERO_EULER: cgmath::Euler<cgmath::Rad<f32>> =
    cgmath::Euler::new(cgmath::Rad(0.0), cgmath::Rad(0.0), cgmath::Rad(0.0));
const ZERO_VECTOR3: cgmath::Vector3<f32> = cgmath::Vector3::new(0.0, 0.0, 0.0);
const ZERO_QUATERNION: cgmath::Quaternion<f32> = cgmath::Quaternion::from_sv(1.0, ZERO_VECTOR3);

struct PlayerController {
    time: f32,

    key_up: bool,
    key_down: bool,

    key_left: bool,
    key_right: bool,

    key_forward: bool,
    key_back: bool,

    key_turn_up: bool,
    key_turn_down: bool,

    key_turn_left: bool,
    key_turn_right: bool,

    state: cgmath::Matrix4<f32>,

    position: cgmath::Vector3<f32>,
    velocity: cgmath::Vector3<f32>,

    rotation: cgmath::Quaternion<f32>,
    rotation_velocity: cgmath::Quaternion<f32>,

    d1: [f32; 4],
}
impl PlayerController {
    fn new() -> Self {
        Self {
            time: 0.0,
            key_up: false,
            key_down: false,
            key_left: false,
            key_right: false,
            key_forward: false,
            key_back: false,
            key_turn_up: false,
            key_turn_down: false,
            key_turn_left: false,
            key_turn_right: false,
            state: cgmath::Matrix4::from_translation(cgmath::Vector3::new(0.0, 0.0, 0.5)),
            position: cgmath::Vector3::new(0.0, 0.0, 0.5),
            velocity: ZERO_VECTOR3,
            rotation: ZERO_QUATERNION,
            rotation_velocity: ZERO_QUATERNION,
            d1: [0.0; 4],
        }
    }
    fn process_events(&mut self, event: &WindowEvent) -> bool {
        if let WindowEvent::KeyboardInput {
            input:
                KeyboardInput {
                    state,
                    virtual_keycode: Some(keycode),
                    ..
                },
            ..
        } = event
        {
            let s = *state == ElementState::Pressed;
            use VirtualKeyCode::*;
            match keycode {
                Up | W => Some(&mut self.key_forward),
                Down | S => Some(&mut self.key_back),
                Left | A => Some(&mut self.key_left),
                Right | D => Some(&mut self.key_right),
                Space => Some(&mut self.key_up),
                LControl | LShift => Some(&mut self.key_down),
                I => Some(&mut self.key_turn_up),
                K => Some(&mut self.key_turn_down),
                J => Some(&mut self.key_turn_left),
                L => Some(&mut self.key_turn_right),
                _ => None,
            }
            .map(|k| *k = s)
            .is_some()
        } else {
            false
        }
    }
    fn update(
        &mut self,
        dt: f32,
        pipeline_index: &mut usize,
        _active_gamepad: &mut Option<gilrs::GamepadId>,
        gilrs_context: &mut gilrs::Gilrs,
    ) {
        fn tri_wave(a: f32) -> f32 {
            let a = a.rem_euclid(4.0);
            if a < 2.0 {
                a - 1.0 // -1 -> 1
            } else {
                -a + 3.0
            }
        }
        self.time += dt;

        let tri = {
            let f = 1024.0;
            self.time = self.time.rem_euclid(4.0 * f);
            tri_wave(self.time / f)
        };

        self.position = if self.position.magnitude() < 0.1 {
            *pipeline_index = (*pipeline_index + (NUM_SDF + 1)) % NUM_SDF;
            println!("level+=1 -> {pipeline_index}");
            self.position * 10.0
        } else if self.position.magnitude() > 1.0 {
            *pipeline_index = (*pipeline_index + (NUM_SDF - 1)) % NUM_SDF;
            println!("level-=1 -> {pipeline_index}");
            self.position / 10.0
        } else {
            self.position
        };

        let range = |a, b| (a as i32 - b as i32) as f32;
        let mut rotation_input = cgmath::Vector3::new(
            range(self.key_turn_right, self.key_turn_left),
            range(self.key_turn_up, self.key_turn_down), // + self.state.z.y * 0.3,
            -self.state.x.y * 1.0,
        );

        let mut translation_input = cgmath::Vector3::new(
            range(self.key_right, self.key_left),
            range(self.key_up, self.key_down),
            range(self.key_forward, self.key_back),
        );

        {
            while let Some(gilrs::Event {
                id: _,
                event: _,
                time: _,
            }) = gilrs_context.next_event()
            {
                //println!("{:?} {}: {:?}", time, id, event);
            }

            for (_id, gamepad) in gilrs_context.gamepads() {
                use gilrs::ev::Axis::*;
                use gilrs::ev::Button::*;
                translation_input += cgmath::Vector3::new(
                    gamepad.value(LeftStickX),
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
                );

                rotation_input += cgmath::Vector3::new(
                    gamepad.value(RightStickX),
                    gamepad.value(RightStickY),
                    0.0,
                )
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

        let rotation_euler = cgmath::Euler::new(
            cgmath::Rad(dt * 0.3 * rotation_input.y),
            cgmath::Rad(dt * 0.3 * -rotation_input.x),
            cgmath::Rad(dt * 0.3 * rotation_input.z),
        );

        self.rotation_velocity = self.rotation_velocity * cgmath::Quaternion::from(rotation_euler);

        self.rotation_velocity = self.rotation_velocity.slerp(
            cgmath::Quaternion::from(cgmath::Euler::new(
                cgmath::Rad(0.0),
                cgmath::Rad(0.0),
                cgmath::Rad(0.0),
            )),
            dt * 4.0,
        );

        let foo = cgmath::Decomposed {
            scale: 1.0,
            rot: cgmath::Quaternion::from(cgmath::Euler::new(
                cgmath::Rad(0.0),
                cgmath::Rad(0.0),
                cgmath::Rad(0.0),
            )),
            disp: cgmath::Vector3::new(0.0, 0.0, 0.5),
        };

        self.state = foo.into();

        translation_input.z *= -1.0;

        let translation_local_player =
            (self.state * translation_input.extend(0.0)).truncate() * dt * 0.1;

        self.rotation = self.rotation * self.rotation_velocity;
        self.velocity += self.rotation * translation_local_player;
        //+ (0.05 / sdf::sdf(self.position)).min(0.2)
        //    * sdf::sdf_normal(self.position, 0.001)
        //    * dt;
        self.velocity *= 0.5_f32.powf(dt);
        self.position += self.velocity * dt * self.position.magnitude();

        // manage floating point errors
        self.rotation /= self.rotation.magnitude();
        self.rotation_velocity /= self.rotation_velocity.magnitude();

        self.d1[3] = tri * 3.0 - 1.0;

        let d1_fac = 0.125 * 0.125 * 0.125;

        self.d1[0] = -0.5 + 0.5 * (0.1 + d1_fac * self.time * std::f32::consts::PI).sin();
        self.d1[1] = -0.5 + 0.5 * (0.2 + d1_fac * self.time * std::f32::consts::PI).cos();
        self.d1[2] = -0.5 + 0.5 * (0.3 + d1_fac * self.time * std::f32::consts::PI * 0.5).sin();

        let f = 1.0
            / (self.d1[0] * self.d1[0] + self.d1[1] * self.d1[1] + self.d1[2] * self.d1[2]).sqrt();

        self.d1[0] *= f;
        self.d1[1] *= f;
        self.d1[2] *= f;

        self.state = cgmath::Decomposed {
            scale: 1.0,
            rot: self.rotation,
            disp: self.position,
        }
        .into();
    }
    fn get_uniform(&self) -> CameraUniform {
        CameraUniform::from_mat_data(self.state, self.d1)
    }
}

mod sdf {
    #![allow(unused)]
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

    pub(crate) fn sdf_normal(p: Vector3<f32>, h: f32) -> Vector3<f32> {
        let x = 1.0;
        let a = -1.0;
        let v: Vector3<f32> = [
            Vector3::new(x, a, a),
            Vector3::new(a, x, a),
            Vector3::new(a, a, x),
            Vector3::new(x, x, x),
        ]
        .into_iter()
        .map(|a| a * sdf(p + a * h))
        .sum();
        v / v.magnitude()
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

fn sdf_wgsl_gen(max: usize) -> Vec<String> {
    (0..max)
        .map(|i| {
            let sdf0 = i;
            let sdf1 = (i + 1) % max;
            format!(
                //                "
                //fn sdf(p: vec3<f32>) -> f32 {{
                //    return sdf_outer(p);
                //}}
                //
                //fn sdf_outer(p: vec3<f32>) -> f32 {{
                //    return sdf{sdf0}(p);
                //}}
                //"
                "
fn sdf(p: vec3<f32>) -> f32 {{ 
    return min(sdf{sdf0}(p), sdf{sdf1}(p / 0.1) * 0.1); 
}}

fn sdf_outer(p: vec3<f32>) -> f32 {{
    return sdf{sdf0}(p);
}}
"
            )
            //format!("fn sdf(p: vec3<f32>) -> f32 {{ return sdf{sdf0}(p); }}\n")
        })
        .collect()
}
