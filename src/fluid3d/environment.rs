use super::{Fluid3dBounds, Fluid3dError, Fluid3dSettings};
use bevy::prelude::*;

/// Combined sphere/cuboid limit per simulation.
pub const MAX_COLLIDERS_3D: usize = 16;

/// Solid obstacles in simulation coordinates; oriented cuboids use a unit quaternion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fluid3dCollider {
    Cuboid(Fluid3dBounds),
    Sphere {
        center: Vec3,
        radius: f32,
    },
    /// Oriented box with positive half extents; rotation must be a unit quaternion.
    OrientedCuboid {
        center: Vec3,
        half_extents: Vec3,
        rotation: Quat,
    },
}
impl Fluid3dCollider {
    pub(crate) fn validate(&self) -> Result<(), Fluid3dError> {
        let valid = match self {
            Self::Cuboid(b) => {
                b.min.is_finite()
                    && b.max.is_finite()
                    && b.size().is_finite()
                    && b.size().min_element() > 0.0
            }
            Self::OrientedCuboid {
                center,
                half_extents,
                rotation,
            } => {
                center.is_finite()
                    && half_extents.is_finite()
                    && half_extents.min_element() > 0.0
                    && (*center + Vec3::splat(half_extents.length())).is_finite()
                    && (*center - Vec3::splat(half_extents.length())).is_finite()
                    && rotation.is_finite()
                    && (rotation.length_squared() - 1.0).abs() <= 0.001
            }
            Self::Sphere { center, radius } => {
                center.is_finite()
                    && radius.is_finite()
                    && *radius > 0.0
                    && (*center + Vec3::splat(*radius)).is_finite()
                    && (*center - Vec3::splat(*radius)).is_finite()
            }
        };
        if valid {
            Ok(())
        } else {
            Err(Fluid3dError(
                "colliders require finite positions, positive extents and unit rotations",
            ))
        }
    }
}

/// Finite rectangular outlet in the horizontal XZ plane, with arbitrary initial velocity.
/// Switching, moving or resizing never replenishes the fixed particle budget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fluid3dFaucet {
    pub position: Vec3,
    pub width: f32,
    pub depth: f32,
    pub is_open: bool,
    pub velocity: Vec3,
    pub particle_spacing: f32,
    /// At most one layer per fixed step; closing/pausing does not accumulate backlog.
    pub layers_per_second: f32,
}
impl Default for Fluid3dFaucet {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 2.3, 0.0),
            width: 0.8,
            depth: 0.8,
            is_open: true,
            velocity: Vec3::new(0.0, -2.0, 0.0),
            particle_spacing: 0.16,
            layers_per_second: 20.0,
        }
    }
}
impl Fluid3dFaucet {
    pub(crate) fn validate(&self, s: &Fluid3dSettings) -> Result<(), Fluid3dError> {
        let values = [
            self.width,
            self.depth,
            self.particle_spacing,
            self.layers_per_second,
        ];
        if !self.position.is_finite()
            || !self.velocity.is_finite()
            || values.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || self.layers_per_second as f64 * s.time_step as f64 > 1.0 + 1e-6
        {
            return Err(Fluid3dError(
                "invalid faucet geometry, velocity or rate (maximum one layer per step)",
            ));
        }
        let lanes_x = (self.width as f64 / self.particle_spacing as f64)
            .floor()
            .max(1.0);
        let lanes_z = (self.depth as f64 / self.particle_spacing as f64)
            .floor()
            .max(1.0);
        if lanes_x * lanes_z >= u32::MAX as f64 {
            return Err(Fluid3dError("too many particles in a faucet layer"));
        }
        let extent = Vec3::new(self.width * 0.5, 0.0, self.depth * 0.5);
        let min = s.bounds.min + Vec3::splat(s.particle_radius);
        let max = s.bounds.max - Vec3::splat(s.particle_radius);
        if (self.position - extent).cmplt(min).any() || (self.position + extent).cmpgt(max).any() {
            return Err(Fluid3dError(
                "faucet outlet must lie inside the collision bounds",
            ));
        }
        Ok(())
    }
    pub(crate) fn lanes(&self) -> UVec2 {
        UVec2::new(
            (self.width as f64 / self.particle_spacing as f64)
                .floor()
                .max(1.0) as u32,
            (self.depth as f64 / self.particle_spacing as f64)
                .floor()
                .max(1.0) as u32,
        )
    }
}
