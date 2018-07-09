#[macro_use]
extern crate vulkano;
extern crate winit;
// link between vulkano and winit
extern crate vulkano_win;
extern crate clock_ticks;
extern crate glsl_to_spirv;

use vulkano as vk;
use vulkano_win::VkSurfaceBuild;
use vulkano::sync::GpuFuture;

use std::sync::Arc;
use std::io::Read;
use std::borrow::Cow;
use std::ffi::CStr;
use vulkano::descriptor::descriptor_set::DescriptorSetDesc;

pub fn nanos_to_secs(ns: u64) -> f64 {
    (ns as f64) / 1_000_000_000.
}

fn main() {
    let mut extensions: vk::instance::RawInstanceExtensions = (&::vulkano_win::required_extensions()).into();
    // the VK_KHR_maintenance1 extension supports a viewport with lower left as origin
    // unfortunately it's not present on a mac
    // extensions.insert(std::ffi::CString::new("VK_KHR_maintenance1".as_bytes()).unwrap());

    // The Instance object is the API entry point. It is the first object you must create before
    // starting to use Vulkan.
    let instance = vk::instance::Instance::new(
        None,
        extensions,
        None,
    ).expect("no vulkano instance with surface extension (required to display in window)");

    // The PhysicalDevice object represents an implementation of Vulkan available on the system (eg. a graphics card, a software implementation, etc.). Physical devices can be enumerated from an instance with PhysicalDevice::enumerate().
    let physical_device = vk::instance::PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no physical vulkano device");

    println!("using physical device: {} (type: {:?})", physical_device.name(), physical_device.ty());

    let mut events_loop = ::winit::EventsLoop::new();
    let surface = ::winit::WindowBuilder::new()
        // whether to show buttons and title bar and borders
        .with_decorations(true)
        .with_title("minimal segfault example")
        .build_vk_surface(&events_loop, instance.clone())
        .expect("can't build vulkano surface");

    let graphical_queue_family = physical_device
        .queue_families()
        .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
        .expect("couldn't find a vulkano queue that supports graphics and is supported by surface");
    let device_extensions = vk::device::DeviceExtensions {
        khr_swapchain: true,
        ..vk::device::DeviceExtensions::none()
    };
    // The Device is the most important object of Vulkan, as it represents an open channel of communicaton with a physical device. You always need to have one before you can do interesting things with Vulkan.
    let (device, mut queues) = vk::device::Device::new(
        physical_device.clone(),
        physical_device.supported_features(),
        &device_extensions,
        [(graphical_queue_family, 0.5)].iter().cloned(),
    ).expect("failed to create vulkano device");

    // For the work to start, the command buffer must then be submitted to a Queue, which is obtained when you create the Device.
    let queue = queues.next().expect("vulkano device has no queues");

    let capabilities = surface
        .capabilities(device.physical_device())
        .expect("failure to get surface capabilities");
    let format = capabilities.supported_formats[0].0;
    let dimensions = capabilities.current_extent.unwrap_or([1024, 768]);
    let present = capabilities.present_modes.iter().next().unwrap();

    let (swapchain, images) = vk::swapchain::Swapchain::new(
        device.clone(),
        surface.clone(),
        capabilities.min_image_count,
        format,
        dimensions,
        1,
        capabilities.supported_usage_flags,
        &queue,
        vk::swapchain::SurfaceTransform::Identity,
        vk::swapchain::CompositeAlpha::Opaque,
        present,
        true,
        None,
    ).expect("failed to create vulkan swapchain");

    // A render pass describes the target which you are going to render to. It is a collection of
    // descriptions of one or more attachments (ie. image that are rendered to), and of one or
    // multiples subpasses. The render pass contains the format and number of samples of each
    // attachment, and the attachments that are attached to each subpass. They are represented in
    // vulkano with the RenderPass object.
    let render_pass = Arc::new(
        single_pass_renderpass!(
            device.clone(), attachments: {
                // `color` is a custom name we give to the first and only attachment.
                color: {
                    // `load: Clear` means that we ask the GPU to clear the content of this
                    // attachment at the start of the drawing.
                    load: Clear,
                    // `store: Store` means that we ask the GPU to store the output of the draw
                    // in the actual image. We could also ask it to discard the result.
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                // We use the attachment named `color` as the one and only color attachment.
                color: [color],
                // No depth-stencil attachment is indicated with empty brackets.
                depth_stencil: {}
            }
        ).unwrap(),
    );

    // This structure will tell Vulkan how input entries of our vertex shader
    // look like.
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    struct VertexShaderInput;
    unsafe impl vk::pipeline::shader::ShaderInterfaceDef for VertexShaderInput {
        type Iter = VertexShaderInputIter;

        fn elements(&self) -> VertexShaderInputIter {
            VertexShaderInputIter(0)
        }
    }
    #[derive(Debug, Copy, Clone)]
    struct VertexShaderInputIter(u16);
    impl Iterator for VertexShaderInputIter {
        type Item = vk::pipeline::shader::ShaderInterfaceDefEntry;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            // There are things to consider when giving out entries:
            // * There must be only one entry per one location, you can't have
            //   `color' and `position' entries both at 0..1 locations.  They also
            //   should not overlap.
            // * Format of each element must be no larger than 128 bits.
            if self.0 == 0 {
                self.0 += 1;
                return Some(vk::pipeline::shader::ShaderInterfaceDefEntry {
                    location: 0..1,
                    format: vk::format::Format::R32G32Sfloat,
                    name: Some(Cow::Borrowed("position"))
                })
            }
            None
        }
        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            // We must return exact number of entries left in iterator.
            let len = (1 - self.0) as usize;
            (len, Some(len))
        }
    }
    impl ExactSizeIterator for VertexShaderInputIter {
    }
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    struct VertexShaderOutput;
    unsafe impl vk::pipeline::shader::ShaderInterfaceDef for VertexShaderOutput {
        type Iter = VertexShaderOutputIter;

        fn elements(&self) -> VertexShaderOutputIter {
            VertexShaderOutputIter(0)
        }
    }
    // This structure will tell Vulkan how output entries (those passed to next
    // stage) of our vertex shader look like.
    #[derive(Debug, Copy, Clone)]
    struct VertexShaderOutputIter(u16);
    impl Iterator for VertexShaderOutputIter {
        type Item = vk::pipeline::shader::ShaderInterfaceDefEntry;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            None
        }
        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            (0, Some(0))
        }
    }
    impl ExactSizeIterator for VertexShaderOutputIter {
    }
    // This structure describes layout of this stage.
    #[derive(Debug, Copy, Clone)]
    struct VertexShaderLayout(vk::descriptor::descriptor::ShaderStages);
    unsafe impl vk::descriptor::pipeline_layout::PipelineLayoutDesc for VertexShaderLayout {
        // Number of descriptor sets it takes.
        fn num_sets(&self) -> usize { 0 }
        // Number of entries (bindings) in each set.
        fn num_bindings_in_set(&self, set: usize) -> Option<usize> { None }
        // Descriptor descriptions.
        fn descriptor(&self, set: usize, binding: usize) -> Option<vk::descriptor::descriptor::DescriptorDesc> { None }
        // Number of push constants ranges (think: number of push constants).
        fn num_push_constants_ranges(&self) -> usize { 0 }
        // Each push constant range in memory.
        fn push_constants_range(&self, num: usize) -> Option<vk::descriptor::pipeline_layout::PipelineLayoutDescPcRange> {
            if num != 0 || 0 == 0 { return None; }
            Some(vk::descriptor::pipeline_layout::PipelineLayoutDescPcRange { offset: 0,
                                             size: 0,
                                             stages: vk::descriptor::descriptor::ShaderStages::all() })
        }
    }

    let vertex_shader = {
        let mut glsl_file = ::std::fs::File::open("src/vertex_shader.glsl")
            .expect("can't open vertex shader glsl file");
        let mut glsl_string = String::new();
        glsl_file.read_to_string(&mut glsl_string).expect("failed to read vertex shader glsl file");
        let mut spirv_file = ::glsl_to_spirv::compile(&glsl_string, ::glsl_to_spirv::ShaderType::Vertex)
            .expect("failed to compile vertex shader from glsl to spirv");
        let mut spirv_bytes = vec![];
        spirv_file.read_to_end(&mut spirv_bytes).expect("failed to read vertex shader spirv file");
        // Create a ShaderModule on a device the same Shader::load does it.
        // NOTE: You will have to verify correctness of the data by yourself!
        unsafe { vk::pipeline::shader::ShaderModule::new(device.clone(), &spirv_bytes) }.unwrap()
    };

    // NOTE: ShaderModule::*_shader_entry_point calls do not do any error
    // checking and you have to verify correctness of what you are doing by
    // yourself.
    //
    // You must be extra careful to specify correct entry point, or program will
    // crash at runtime outside of rust and you will get NO meaningful error
    // information!
    let vertex_shader_main_entry_point = unsafe { vertex_shader.graphics_entry_point(
        CStr::from_bytes_with_nul_unchecked(b"main\0"),
        VertexShaderInput,
        VertexShaderOutput,
        VertexShaderLayout(vk::descriptor::descriptor::ShaderStages { vertex: true, ..vk::descriptor::descriptor::ShaderStages::none() }),
        vk::pipeline::shader::GraphicsShaderType::Vertex
    ) };

    let fragment_shader = {
        let mut glsl_file = ::std::fs::File::open("src/fragment_shader.glsl")
            .expect("can't open fragment shader glsl file");
        let mut glsl_string = String::new();
        glsl_file.read_to_string(&mut glsl_string).expect("failed to read fragment shader glsl file");
        let mut spirv_file = ::glsl_to_spirv::compile(&glsl_string, ::glsl_to_spirv::ShaderType::Fragment)
            .expect("failed to compile fragment shader from glsl to spirv");
        let mut spirv_bytes = vec![];
        spirv_file.read_to_end(&mut spirv_bytes).expect("failed to read fragment shader spirv file");
        // Create a ShaderModule on a device the same Shader::load does it.
        // NOTE: You will have to verify correctness of the data by yourself!
        unsafe { vk::pipeline::shader::ShaderModule::new(device.clone(), &spirv_bytes) }.unwrap()
    };

    // Same as with our vertex shader, but for fragment one instead.
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    struct FragmentShaderInput;
    unsafe impl vk::pipeline::shader::ShaderInterfaceDef for FragmentShaderInput {
        type Iter = FragmentShaderInputIter;

        fn elements(&self) -> FragmentShaderInputIter {
            FragmentShaderInputIter(0)
        }
    }
    #[derive(Debug, Copy, Clone)]
    struct FragmentShaderInputIter(u16);
    impl Iterator for FragmentShaderInputIter {
        type Item = vk::pipeline::shader::ShaderInterfaceDefEntry;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            None
        }
        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            (0, Some(0))
        }
    }
    impl ExactSizeIterator for FragmentShaderInputIter {
    }
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    struct FragmentShaderOutput;
    unsafe impl vk::pipeline::shader::ShaderInterfaceDef for FragmentShaderOutput {
        type Iter = FragmentShaderOutputIter;

        fn elements(&self) -> FragmentShaderOutputIter {
            FragmentShaderOutputIter(0)
        }
    }
    #[derive(Debug, Copy, Clone)]
    struct FragmentShaderOutputIter(u16);
    impl Iterator for FragmentShaderOutputIter {
        type Item = vk::pipeline::shader::ShaderInterfaceDefEntry;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            // Note that color fragment color entry will be determined
            // automatically by Vulkano.
            if self.0 == 0 {
                self.0 += 1;
                return Some(vk::pipeline::shader::ShaderInterfaceDefEntry {
                    location: 0..1,
                    format: vk::format::Format::R32G32B32A32Sfloat,
                    name: Some(Cow::Borrowed("fragColor"))
                })
            }
            None
        }
        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = (1 - self.0) as usize;
            (len, Some(len))
        }
    }
    impl ExactSizeIterator for FragmentShaderOutputIter {
    }
    // Layout same as with vertex shader.
    #[derive(Debug, Copy, Clone)]
    struct FragmentShaderLayout(vk::descriptor::descriptor::ShaderStages);
    unsafe impl vk::descriptor::pipeline_layout::PipelineLayoutDesc for FragmentShaderLayout {
        fn num_sets(&self) -> usize { 1 }
        fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
            match set {
                0 => Some(1),
                _ => None
            }
        }
        fn descriptor(&self, set: usize, binding: usize) -> Option<vk::descriptor::descriptor::DescriptorDesc> {
            match (set, binding) {
                (0, 0) => Some(vk::descriptor::descriptor::DescriptorDesc {
                    ty: vk::descriptor::descriptor::DescriptorDescTy::Buffer(
                        vk::descriptor::descriptor::DescriptorBufferDesc {
                            dynamic: Some(false),
                            storage: false
                        }
                    ),
                    array_count: 1,
                    stages: vk::descriptor::descriptor::ShaderStages::all(),
                    readonly: true,
                }),
                _ => None,
            }
        }
        fn num_push_constants_ranges(&self) -> usize { 1 }
        fn push_constants_range(&self, num: usize) -> Option<vk::descriptor::pipeline_layout::PipelineLayoutDescPcRange> {
            if num != 0 { return None; }
            // 3 floats
            let bytes = 4 + 4 + 4;
            Some(vk::descriptor::pipeline_layout::PipelineLayoutDescPcRange { offset: 0,
                                             size: bytes,
                                             stages: vk::descriptor::descriptor::ShaderStages::all() })
        }
    }

    let fragment_shader_main_entry_point = unsafe { fragment_shader.graphics_entry_point(
        CStr::from_bytes_with_nul_unchecked(b"main\0"),
        FragmentShaderInput,
        FragmentShaderOutput,
        FragmentShaderLayout(vk::descriptor::descriptor::ShaderStages { fragment: true, ..vk::descriptor::descriptor::ShaderStages::none() }),
        vk::pipeline::shader::GraphicsShaderType::Fragment
    ) };

    // In order to be able to add a compute operation or a graphics operation to a command buffer,
    // you need to have created a ComputePipeline or a GraphicsPipeline object that describes the
    // operation you want. These objects are usually created during your program's initialization.
    // Shaders are programs that the GPU will execute as part of a pipeline
    let pipeline: Arc<vk::pipeline::GraphicsPipeline<_, _, _>> = Arc::new(vk::pipeline::GraphicsPipeline::start()
        // We need to indicate the layout of the vertices.
        // The type `SingleBufferDefinition` actually contains a template parameter corresponding
        // to the type of each vertex. But in this code it is automatically inferred.
        // .vertex_input(vk::pipeline::vertex::SingleBufferDefinition::<vertex_shader::Vertex>::new())
        .vertex_input_single_buffer()
        // A Vulkan shader can in theory contain multiple entry points, so we have to specify
        // which one. The `main` word of `main_entry_point` actually corresponds to the name of
        // the entry point.
        .vertex_shader(vertex_shader_main_entry_point, ())
        // The content of the vertex buffer describes a list of triangles.
        .triangle_list()
        // Use a resizable viewport set to draw over the entire window
        .viewports_dynamic_scissors_irrelevant(1)
        // See `vertex_shader`.
        .fragment_shader(fragment_shader_main_entry_point, ())
        .viewports([
            vk::pipeline::viewport::Viewport {
                origin: [0.0, 0.0],
                depth_range: 0.0..1.0,
                dimensions: [images[0].dimensions()[0] as f32,
                             images[0].dimensions()[1] as f32],
            },
        ].iter().cloned())
        // We have to indicate which subpass of which render pass this pipeline is going to be used
        // in. The pipeline will only be usable from this particular subpass.
        .render_pass(vk::framebuffer::Subpass::from(render_pass.clone(), 0).unwrap())
        // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
        .build(device.clone())
        .expect("failed to start vulkan graphics pipeline"));

    // We now create a buffer that will store the shape of our 2 triangles
    let vertex_buffer = {
        #[derive(Debug, Clone)]
        struct Vertex { position: [f32; 2] }
        impl_vertex!(Vertex, position);

        vk::buffer::CpuAccessibleBuffer::from_iter(device.clone(), vk::buffer::BufferUsage::all(), [
            // first triangle

            // upper left
            Vertex { position: [-1.0,  1.0] },
            // upper right
            Vertex { position: [ 1.0,  1.0] },
            // lower left
            Vertex { position: [-1.0, -1.0] },

            // second triangle

            // upper right
            Vertex { position: [ 1.0,  1.0] },
            // lower right
            Vertex { position: [ 1.0, -1.0] },
            // lower left
            Vertex { position: [-1.0, -1.0] },
        ].iter().cloned()).expect("failed to create vertex buffer")
    };

    let autocorrelation_buffer = {
        vk::buffer::CpuAccessibleBuffer::from_iter(
            device.clone(),
            vk::buffer::BufferUsage::all(),
            [1. as f64; 480].iter()
        )
        .expect("failed to allocate autocorrelation_buffer ")
    };

    let descriptor_set_id = 0;
    let descriptor_set = Arc::new(vk::descriptor::descriptor_set::PersistentDescriptorSet::start(pipeline.clone(), descriptor_set_id)
        .add_buffer(autocorrelation_buffer.clone()).unwrap()
        .build().unwrap()
    );

    println!("descriptor description = {:?}", descriptor_set.descriptor(0));

    let framebuffers: Vec<_> = images
        .iter()
        .map(|image| Arc::new(
            vk::framebuffer::Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .build().unwrap(),
        ))
        .collect();

    let nanos_at_start = ::clock_ticks::precise_time_ns();

    loop {
        events_loop.poll_events(|_| ());

        let nanos_since_start = ::clock_ticks::precise_time_ns() - nanos_at_start;
        let secs_since_start = nanos_to_secs(nanos_since_start);

        let (image_num, acquire_future) =
            vk::swapchain::acquire_next_image(
                swapchain.clone(),
                None,
            ).expect("failed to acquire vulkano swapchain in time");

        // easiest and fastest but limited (128 bytes only) way to send data to shaders
        let push_constants = (secs_since_start as f32, dimensions[0] as f32, dimensions[1] as f32);
        // let push_constants = ();

        let descriptor_sets = (descriptor_set.clone());
        // let descriptor_sets = ();

        let command_buffer = vk::command_buffer::AutoCommandBufferBuilder::new(
                device.clone(),
                queue.family(),
            ).expect("failed to create auto command buffer builder")
            .begin_render_pass(
                framebuffers[image_num].clone(),
                false,
                vec![[0.0, 0.0, 0.0, 1.0].into()],
            ).unwrap()
            .draw(
                pipeline.clone(),
                vk::command_buffer::DynamicState::none(),
                vertex_buffer.clone(),
                descriptor_sets,
                push_constants,
            ).unwrap()
            .end_render_pass().expect("failed to end render pass")
            .build().unwrap();

        acquire_future
            .then_execute(queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();
    }
}
