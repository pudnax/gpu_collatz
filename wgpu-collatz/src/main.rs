use std::{convert::TryInto, str::FromStr};
use wgpu::util::DeviceExt;

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let numbers = if atty::isnt(atty::Stream::Stdin) {
        let mut buf = Default::default();
        std::io::stdin().read_line(&mut buf)?;
        buf.trim().split(' ').map(|s| u32::from_str(&s)).collect()
    } else if std::env::args().len() <= 1 {
        let default = vec![1, 2, 3, 4];
        println!("No numbers were provided, defauting to {:?}", default);
        Ok(default)
    } else {
        std::env::args()
            .skip(1)
            .map(|s| u32::from_str(&s))
            .collect()
    }?;

    let times = execute_gpu(numbers).await?;
    println!("Times: {:?}", times);
    Ok(())
}

async fn execute_gpu(numbers: Vec<u32>) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .ok_or_else::<Box<dyn std::error::Error>, _>(|| {
            String::from("Error requesting adapter.").into()
        })?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                shader_validation: true,
            },
            None,
        )
        .await?;

    let cs_module = device.create_shader_module(wgpu::include_spirv!("shader.comp.spv"));

    let slice_size = numbers.len() * std::mem::size_of::<u32>();
    let size = slice_size as wgpu::BufferAddress;

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
        mapped_at_creation: false,
    });

    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(&numbers),
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStage::COMPUTE,
            ty: wgpu::BindingType::StorageBuffer {
                dynamic: false,
                readonly: false,
                min_binding_size: wgpu::BufferSize::new(4),
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(storage_buffer.slice(..)),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &cs_module,
            entry_point: "main",
        },
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let mut cpass = encoder.begin_compute_pass();
    cpass.set_pipeline(&compute_pipeline);
    cpass.set_bind_group(0, &bind_group, &[]);
    cpass.dispatch(numbers.len() as u32, 1, 1);
    drop(cpass);

    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);

    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    device.poll(wgpu::Maintain::Wait);

    if let Ok(()) = buffer_future.await {
        let data = buffer_slice.get_mapped_range();
        let result = data
            .chunks_exact(4)
            .map(|x| Ok(u32::from_ne_bytes(x.try_into()?)))
            .collect();
        drop(data);
        staging_buffer.unmap();
        result
    } else {
        panic!("failed to run compute on gpu!")
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    wgpu_subscriber::initialize_default_subscriber(None);
    futures::executor::block_on(run())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_1() {
        let input = vec![1, 2, 3, 4];
        futures::executor::block_on(assert_execute_gpu(input, vec![0, 1, 7, 2]));
    }

    #[test]
    fn test_compute_2() {
        let input = vec![5, 23, 10, 9];
        futures::executor::block_on(assert_execute_gpu(input, vec![5, 15, 6, 19]));
    }

    #[test]
    fn test_multithreaded_compute() {
        use std::{sync::mpsc, thread, time::Duration};

        let thread_count = 8;

        let (tx, rx) = mpsc::channel();
        for _ in 0..thread_count {
            let tx = tx.clone();
            thread::spawn(move || {
                let input = vec![100, 100, 100];
                futures::executor::block_on(assert_execute_gpu(input, vec![25, 25, 25]));
                tx.send(true).unwrap();
            });
        }

        for _ in 0..thread_count {
            rx.recv_timeout(Duration::from_secs(10))
                .expect("A thread never completed.");
        }
    }

    async fn assert_execute_gpu(input: Vec<u32>, expected: Vec<u32>) {
        assert_eq!(execute_gpu(input).await.unwrap(), expected);
    }
}
