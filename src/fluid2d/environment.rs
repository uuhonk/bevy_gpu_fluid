use bevy::prelude::*;

use super::{Fluid2dError, Fluid2dSettings};

/// Maximum total number of rectangle and circle colliders in one simulation.
pub const MAX_COLLIDERS_2D: usize = 16;

/// Solid obstacle in simulation coordinates. Rotations are relative to the center.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fluid2dCollider {
    Rectangle(Rect),
    Circle {
        center: Vec2,
        radius: f32,
    },
    /// Oriented box with positive half extents; rotation is in radians.
    OrientedRectangle {
        center: Vec2,
        half_extents: Vec2,
        rotation: f32,
    },
}

impl Fluid2dCollider {
    pub(crate) fn validate(&self) -> Result<(), Fluid2dError> {
        let valid = match self {
            Self::Rectangle(r) => {
                r.min.is_finite()
                    && r.max.is_finite()
                    && r.size().is_finite()
                    && r.size().min_element() > 0.0
            }
            Self::OrientedRectangle {
                center,
                half_extents,
                rotation,
            } => {
                center.is_finite()
                    && half_extents.is_finite()
                    && half_extents.min_element() > 0.0
                    && (*center + Vec2::splat(half_extents.length())).is_finite()
                    && (*center - Vec2::splat(half_extents.length())).is_finite()
                    && rotation.is_finite()
            }
            Self::Circle { center, radius } => {
                center.is_finite()
                    && radius.is_finite()
                    && *radius > 0.0
                    && (*center + Vec2::splat(*radius)).is_finite()
                    && (*center - Vec2::splat(*radius)).is_finite()
            }
        };
        if valid {
            Ok(())
        } else {
            Err(Fluid2dError(
                "colliders require finite positions, positive extents and finite rotations",
            ))
        }
    }
}

/// Horizontal faucet releasing a finite particle budget, one row at a time.
/// Closing preserves remaining particles. Reopening never replenishes the budget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fluid2dFaucet {
    pub position: Vec2,
    pub width: f32,
    pub is_open: bool,
    pub velocity: Vec2,
    /// Desired minimum spacing within a row. Narrow outlets emit one particle.
    pub particle_spacing: f32,
    /// At most one row per fixed simulation step.
    pub rows_per_second: f32,
}

impl Default for Fluid2dFaucet {
    fn default() -> Self {
        Self {
            position: Vec2::new(0.0, 3.5),
            width: 1.6,
            is_open: true,
            velocity: Vec2::new(0.0, -2.0),
            particle_spacing: 0.16,
            rows_per_second: 20.0,
        }
    }
}

impl Fluid2dFaucet {
    pub(crate) fn validate(&self, settings: &Fluid2dSettings) -> Result<(), Fluid2dError> {
        if !self.position.is_finite()
            || !self.velocity.is_finite()
            || [self.width, self.particle_spacing, self.rows_per_second]
                .iter()
                .any(|x| !x.is_finite() || *x <= 0.0)
            || self.rows_per_second as f64 * settings.time_step as f64 > 1.0 + 1e-6
            || self.width as f64 / self.particle_spacing as f64 > u32::MAX as f64
        {
            return Err(Fluid2dError(
                "invalid faucet position, velocity, width, spacing or rate (maximum one row per time step)",
            ));
        }
        let min = settings.bounds.min + Vec2::splat(settings.particle_radius);
        let max = settings.bounds.max - Vec2::splat(settings.particle_radius);
        if self.position.y < min.y
            || self.position.y > max.y
            || self.position.x - self.width * 0.5 < min.x
            || self.position.x + self.width * 0.5 > max.x
        {
            return Err(Fluid2dError(
                "faucet outlet must be inside the collision bounds",
            ));
        }
        Ok(())
    }

    pub(crate) fn lanes(&self) -> u32 {
        (self.width / self.particle_spacing).floor().max(1.0) as u32
    }
}
