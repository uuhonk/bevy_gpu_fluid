use bevy::{asset::RenderAssetUsages, prelude::*, render::render_resource::*};

/// Optional GPU evidence for visual QA; regular test runs write no files.
pub fn capture(name: &str, bytes: &[u8], linear: bool) {
    capture_sized(name, bytes, UVec2::splat(256), linear);
}

pub fn capture_sized(name: &str, bytes: &[u8], size: UVec2, linear: bool) {
    let Ok(directory) = std::env::var("BEVY_FLUID_CAPTURE_DIR") else {
        return;
    };
    let path = std::path::Path::new(&directory);
    std::fs::create_dir_all(path).unwrap();
    let mut bytes = bytes.to_vec();
    if linear {
        for pixel in bytes.chunks_exact_mut(4) {
            for channel in &mut pixel[..3] {
                let v = *channel as f32 / 255.0;
                let srgb = if v <= 0.0031308 {
                    v * 12.92
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                };
                *channel = (srgb * 255.0).round() as u8;
            }
        }
    }
    let image = Image::new(
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD,
    );
    image
        .try_into_dynamic()
        .unwrap()
        .save(path.join(name))
        .unwrap();
}
