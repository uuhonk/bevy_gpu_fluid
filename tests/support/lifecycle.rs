use bevy::{
    prelude::*,
    render::{
        RenderApp,
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
    },
};

pub fn frames(app: &mut App, count: usize) {
    for _ in 0..count {
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(3));
    }
}
pub fn read_u32(app: &App, source: &Buffer) -> Vec<u32> {
    let world = app.sub_app(RenderApp).world();
    let device = world.resource::<RenderDevice>();
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("test lifecycle readback"),
        size: source.size(),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&default());
    encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, source.size());
    world.resource::<RenderQueue>().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    bytemuck::cast_slice(&buffer.slice(..).get_mapped_range().unwrap()).to_vec()
}
