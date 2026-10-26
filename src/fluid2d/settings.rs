use std::{fmt, sync::Arc};

use super::{Fluid2dCollider, Fluid2dFaucet, MAX_COLLIDERS_2D};
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

/// GPU particle ABI: position at byte 0, velocity at byte 8, stride 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct FluidParticle2d {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
}

impl FluidParticle2d {
    pub fn new(position: Vec2) -> Self {
        Self {
            position: position.to_array(),
            velocity: [0.0; 2],
        }
    }

    /// A rectangular seed in row-major order; `origin` is the first particle center.
    pub fn lattice(origin: Vec2, size: UVec2, spacing: f32) -> Vec<Self> {
        (0..size.y)
            .flat_map(|y| {
                (0..size.x)
                    .map(move |x| Self::new(origin + Vec2::new(x as f32, y as f32) * spacing))
            })
            .collect()
    }
}

/// Weakly compressible SPH settings, in simulation/world units.
#[derive(Clone, Debug, PartialEq)]
pub struct Fluid2dSettings {
    pub bounds: Rect,
    /// Kernel support radius, also the grid cell width.
    pub smoothing_radius: f32,
    /// Wall collision radius. Independent from the visual particle radius.
    pub particle_radius: f32,
    pub particle_mass: f32,
    pub rest_density: f32,
    /// Linear equation of state: p = stiffness * max(density - rest_density, 0).
    pub stiffness: f32,
    pub viscosity: f32,
    pub gravity: Vec2,
    /// Normal velocity retained on wall collision, in [0, 1].
    pub restitution: f32,
    pub time_step: f32,
    /// Maximum fixed substeps per rendered frame. Excess time is dropped.
    pub max_substeps: u32,
    /// Numerical safety caps; these do not replace a stable time step.
    pub max_speed: f32,
    pub max_acceleration: f32,
}

impl Default for Fluid2dSettings {
    fn default() -> Self {
        Self {
            bounds: Rect::new(-8.0, -4.5, 8.0, 4.5),
            smoothing_radius: 0.32,
            particle_radius: 0.07,
            particle_mass: 25.6,
            rest_density: 1000.0,
            stiffness: 200.0,
            viscosity: 0.1,
            gravity: Vec2::new(0.0, -9.81),
            restitution: 0.25,
            time_step: 1.0 / 240.0,
            max_substeps: 8,
            max_speed: 50.0,
            max_acceleration: 10_000.0,
        }
    }
}

/// Invalid configuration or seed data. GPU-specific limits are checked on allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fluid2dError(pub(crate) &'static str);

impl fmt::Display for Fluid2dError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for Fluid2dError {}

impl Fluid2dSettings {
    pub fn validate(&self) -> Result<(), Fluid2dError> {
        let positive = [
            self.smoothing_radius,
            self.particle_radius,
            self.particle_mass,
            self.rest_density,
            self.time_step,
            self.max_speed,
            self.max_acceleration,
        ];
        if positive.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(Fluid2dError(
                "radii, mass, density, time step and safety caps must be finite and positive",
            ));
        }
        if !self.bounds.min.is_finite()
            || !self.bounds.max.is_finite()
            || !self.bounds.size().is_finite()
            || self.bounds.size().min_element() <= 2.0 * self.particle_radius
        {
            return Err(Fluid2dError(
                "bounds must be finite and wider than two collision radii",
            ));
        }
        if !self.gravity.is_finite()
            || !self.stiffness.is_finite()
            || self.stiffness < 0.0
            || !self.viscosity.is_finite()
            || self.viscosity < 0.0
            || !self.restitution.is_finite()
            || !(0.0..=1.0).contains(&self.restitution)
            || !(1..=64).contains(&self.max_substeps)
        {
            return Err(Fluid2dError(
                "invalid gravity, stiffness, viscosity, restitution or substep count (1..=64)",
            ));
        }
        let cells = (self.bounds.size() / self.smoothing_radius)
            .ceil()
            .max(Vec2::ONE);
        // Bound dense-grid memory as well as float-to-integer conversion.
        if !cells.is_finite()
            || cells.max_element() > 1_048_576.0
            || cells.x as u64 * cells.y as u64 > 16_777_216
        {
            return Err(Fluid2dError(
                "grid exceeds 16M cells; increase smoothing radius or reduce bounds",
            ));
        }
        Ok(())
    }

    pub(crate) fn grid_size(&self) -> UVec2 {
        (self.bounds.size() / self.smoothing_radius)
            .ceil()
            .max(Vec2::ONE)
            .as_uvec2()
    }
}

/// One independent simulation. Multiple entities are supported.
///
/// Seed particles are uploaded only on creation/reset, never read back each frame.
/// Change settings through [`Self::set_settings`], pause with [`Self::set_paused`].
#[derive(Component, Clone)]
pub struct Fluid2d {
    pub(crate) settings: Fluid2dSettings,
    pub(crate) seed: Arc<[FluidParticle2d]>,
    pub(crate) revision: u64,
    pub(crate) paused: bool,
    pub(crate) capacity: u32,
    pub(crate) faucet: Option<Fluid2dFaucet>,
    pub(crate) colliders: Arc<[Fluid2dCollider]>,
}

impl Fluid2d {
    pub fn new(
        settings: Fluid2dSettings,
        particles: Vec<FluidParticle2d>,
    ) -> Result<Self, Fluid2dError> {
        settings.validate()?;
        validate_particles(&settings, &particles)?;
        Ok(Self {
            settings,
            capacity: particles.len() as u32,
            seed: particles.into(),
            revision: 0,
            paused: false,
            faucet: None,
            colliders: Arc::from([]),
        })
    }
    /// Starts empty and allocates exactly `particle_limit` GPU slots. Each slot
    /// is released at most once until `reset()`; particles are never recycled.
    pub fn from_faucet(
        settings: Fluid2dSettings,
        particle_limit: u32,
        faucet: Fluid2dFaucet,
    ) -> Result<Self, Fluid2dError> {
        settings.validate()?;
        faucet.validate(&settings)?;
        if particle_limit == u32::MAX {
            return Err(Fluid2dError(
                "particle limit reserves u32::MAX for linked-list sentinel",
            ));
        }
        Ok(Self {
            settings,
            capacity: particle_limit,
            seed: Arc::from([]),
            faucet: Some(faucet),
            colliders: Arc::from([]),
            revision: 0,
            paused: false,
        })
    }
    pub fn settings(&self) -> &Fluid2dSettings {
        &self.settings
    }
    /// Total particle budget. Active count is available from render-world GpuFluid2d.
    pub fn particle_count(&self) -> usize {
        self.capacity as usize
    }
    pub fn faucet(&self) -> Option<&Fluid2dFaucet> {
        self.faucet.as_ref()
    }
    /// Updates/removes the outlet without resetting particles or refilling its budget.
    pub fn set_faucet(&mut self, faucet: Option<Fluid2dFaucet>) -> Result<(), Fluid2dError> {
        if let Some(f) = faucet {
            f.validate(&self.settings)?;
        }
        self.faucet = faucet;
        Ok(())
    }
    pub fn set_faucet_open(&mut self, open: bool) -> Result<(), Fluid2dError> {
        self.faucet
            .as_mut()
            .ok_or(Fluid2dError("simulation has no faucet"))?
            .is_open = open;
        Ok(())
    }
    pub fn colliders(&self) -> &[Fluid2dCollider] {
        &self.colliders
    }
    /// Replaces all solid obstacles transactionally. Shapes share the limit of 16.
    pub fn set_colliders(&mut self, colliders: Vec<Fluid2dCollider>) -> Result<(), Fluid2dError> {
        if colliders.len() > MAX_COLLIDERS_2D {
            return Err(Fluid2dError("at most 16 colliders per simulation"));
        }
        for c in &colliders {
            c.validate()?;
        }
        self.colliders = colliders.into();
        Ok(())
    }
    pub fn is_paused(&self) -> bool {
        self.paused
    }
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Applies settings without resetting positions. A new grid is allocated if needed;
    /// moved walls clamp particles on the next integration step.
    pub fn set_settings(&mut self, settings: Fluid2dSettings) -> Result<(), Fluid2dError> {
        settings.validate()?;
        if let Some(f) = &self.faucet {
            f.validate(&settings)?;
        }
        self.settings = settings;
        Ok(())
    }
    /// Reset to the original seed and clear the accumulator, preserving pause state.
    pub fn reset(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    /// Replaces the seed and requests a GPU reset. Empty seeds are supported.
    pub fn reset_with_particles(
        &mut self,
        particles: Vec<FluidParticle2d>,
    ) -> Result<(), Fluid2dError> {
        validate_particles(&self.settings, &particles)?;
        self.capacity = particles.len() as u32;
        self.seed = particles.into();
        self.reset();
        Ok(())
    }
}

fn validate_particles(
    settings: &Fluid2dSettings,
    particles: &[FluidParticle2d],
) -> Result<(), Fluid2dError> {
    if particles.len() > u32::MAX as usize - 1 {
        return Err(Fluid2dError(
            "too many particles for u32 linked-list indices",
        ));
    }
    let min = settings.bounds.min + Vec2::splat(settings.particle_radius);
    let max = settings.bounds.max - Vec2::splat(settings.particle_radius);
    if particles.iter().any(|p| {
        let pos = Vec2::from_array(p.position);
        !pos.is_finite()
            || !Vec2::from_array(p.velocity).is_finite()
            || pos.cmplt(min).any()
            || pos.cmpgt(max).any()
    }) {
        return Err(Fluid2dError(
            "seed particles must be finite and inside the collision bounds",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_grid_and_seed() {
        let mut s = Fluid2dSettings::default();
        assert!(s.validate().is_ok());
        assert_eq!(s.grid_size(), UVec2::new(50, 29));
        assert!(Fluid2d::new(s.clone(), vec![]).is_ok());
        assert!(Fluid2d::new(s.clone(), vec![FluidParticle2d::new(s.bounds.min)]).is_err());
        s.smoothing_radius = f32::NAN;
        assert!(s.validate().is_err());
        s.smoothing_radius = 1e-8;
        assert!(s.validate().is_err());
    }
    #[test]
    fn reset_is_explicit_and_settings_are_transactional() {
        let mut f = Fluid2d::new(
            Fluid2dSettings::default(),
            vec![FluidParticle2d::new(Vec2::ZERO)],
        )
        .unwrap();
        let original = f.settings().clone();
        let mut invalid = original.clone();
        invalid.time_step = 0.0;
        assert!(f.set_settings(invalid).is_err());
        assert_eq!(f.settings(), &original);
        f.reset_with_particles(vec![]).unwrap();
        assert_eq!(f.revision, 1);
        assert_eq!(f.particle_count(), 0);
    }
    #[test]
    fn environment_is_validated_transactionally() {
        let mut f = Fluid2d::from_faucet(Fluid2dSettings::default(), 129, Fluid2dFaucet::default())
            .unwrap();
        assert_eq!(f.particle_count(), 129);
        assert!(f.seed.is_empty());
        let circle = Fluid2dCollider::Circle {
            center: Vec2::ZERO,
            radius: 0.5,
        };
        f.set_colliders(vec![circle; MAX_COLLIDERS_2D]).unwrap();
        assert!(f.set_colliders(vec![circle; MAX_COLLIDERS_2D + 1]).is_err());
        assert_eq!(f.colliders().len(), 16);
        assert!(
            f.set_colliders(vec![Fluid2dCollider::Rectangle(Rect::new(
                0.0, 0.0, 0.0, 1.0
            ))])
            .is_err()
        );
        assert!(
            f.set_colliders(vec![Fluid2dCollider::Circle {
                center: Vec2::ZERO,
                radius: f32::NAN
            }])
            .is_err()
        );
        let original = *f.faucet().unwrap();
        assert!(
            f.set_faucet(Some(Fluid2dFaucet {
                width: -1.0,
                ..original
            }))
            .is_err()
        );
        assert_eq!(f.faucet(), Some(&original));
        let mut settings = f.settings().clone();
        settings.bounds = Rect::new(-1.0, -1.0, 1.0, 1.0);
        assert!(f.set_settings(settings).is_err());
        f.set_faucet_open(false).unwrap();
        f.reset();
        assert!(!f.faucet().unwrap().is_open);
        assert_eq!(f.particle_count(), 129);
    }
}
