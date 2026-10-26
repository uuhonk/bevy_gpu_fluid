# bevy_gpu_fluid

面向 **Bevy 0.20** 的 GPU SPH 流体库，提供独立的 **2D / 3D** 模块。两者分别实现核函数、网格、碰撞、GPU ABI 和默认渲染，可以在同一个 App 中使用。

**旋转碰撞、自定义计算 pass 与可选区域回收见 [docs/compute.md](docs/compute.md)**。

下方包含 2D / 3D 接入示例，完整的 **3D API 说明见 [docs/3d.md](docs/3d.md)**。

## 运行

需要 Rust **1.97.1**（仓库已配置 `rust-toolchain.toml`），以及支持 compute shader 和 storage buffer 的 GPU。支持 Bevy 的原生 wgpu 后端；当前未验证 WebGPU，不支持 WebGL2。

```sh
cargo run --example dam_break
cargo run --example custom_pass
cargo run --example faucet
cargo run --example dam_break_3d
cargo run --example faucet_3d
cargo run --example custom_pass_3d
```

- `dam_break`：1920 粒子的溃坝，默认渲染。空格暂停/继续，R 重置，方向键改变重力。
- `custom_pass`：只加载计算插件，应用自己创建 render pipeline、bind group 和 render pass；根据速度拉伸粒子并着色。
- `faucet`：1920 粒子总量的有限出水口，混合圆形和矩形障碍物。空格开关，R 重置/补满，P 暂停；A/D 移动出水口，W/S 调整宽度。

3D 示例：`dam_break_3d` 为 6144 粒子溃坝；`faucet_3d` 为 8000 粒子有限水龙头与水槽；`custom_pass_3d` 由应用自己实现相机 render pass。均支持方向键环绕相机；出水口用空格开关、P 暂停、R 重置、A/D 移动 X、Q/E 移动 Z、W/S 改变尺寸。

水槽由 4 块槽壁、4 块底板组成，底板中间留有排水孔，外围另有 4 块桌面，共使用 12 个长方体碰撞体。下方的大回收盒同时回收孔洞排水和桌边落水；示例默认安装回收插件，传入 `-- --no-recycle` 可关闭。移动出水口到桌面上可观察桌边落水，粒子总预算保持不变，流完后按 R 重新填充。

## 3D 作为库接入

```toml
[dependencies]
bevy = { version = "0.20", default-features = false, features = ["3d"] }
bevy_gpu_fluid = { path = "path/to/bevy_gpu_fluid" }
```

安装 `Fluid3dPlugins`，创建相机，再生成带 `Fluid3dRender` 的流体实体即可显示。内部 shader 随库嵌入，计算 pass 和默认渲染 pass 由库处理，不需要复制 assets 或手动调度。

```rust,no_run
use bevy::prelude::*;
use bevy_gpu_fluid::prelude::*;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, Fluid3dPlugins))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(8.0, 5.5, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    let particles = FluidParticle3d::lattice(
        Vec3::new(-3.6, -2.5, -1.2), UVec3::new(16, 24, 16), 0.16,
    );
    let fluid = Fluid3d::new(Fluid3dSettings::default(), particles).unwrap();
    commands.spawn((fluid, Fluid3dRender::default()));
}
```

默认效果为略透明的纯色液体表面，不需要灯光。模拟边界、重力、粒子半径等由 `Fluid3dSettings` 配置；`Fluid3dRender` 可调整颜色、融合阈值和显示尺寸。碰撞体使用世界坐标，应用可自行添加可视网格。

有限出水口可替换初始化方式，并按需添加碰撞体和回收区域：

```rust
use bevy::prelude::*;
use bevy_gpu_fluid::prelude::*;

// 在创建相机的 setup 中调用此函数；App 额外安装 Fluid3dRecyclingPlugin。
fn spawn_faucet(commands: &mut Commands) -> Result<(), Fluid3dError> {
    let faucet = Fluid3dFaucet {
        position: Vec3::new(0.0, 2.3, 0.0),
        width: 0.8,
        depth: 0.8,
        is_open: true,
        ..default()
    };
    let mut fluid = Fluid3d::from_faucet(Fluid3dSettings::default(), 8000, faucet)?;
    fluid.set_colliders(vec![Fluid3dCollider::Sphere {
        center: Vec3::ZERO,
        radius: 0.65,
    }])?;
    let recycling = Fluid3dRecycling::new(vec![Fluid3dCollider::Cuboid(
        Fluid3dBounds::new(Vec3::new(-4.0, -3.0, -3.0), Vec3::new(4.0, -2.7, 3.0)),
    )])?;
    commands.spawn((fluid, Fluid3dRender::default(), recycling));
    Ok(())
}
```

`Fluid3dPlugins` 包含计算插件和默认渲染插件；回收插件需要单独安装：`.add_plugins((DefaultPlugins, Fluid3dPlugins, Fluid3dRecyclingPlugin))`。回收不补充 8000 粒子的有限预算，`set_faucet_open` 控制开关，`reset` 重置预算。自定义渲染只安装 `Fluid3dPlugin`，从 render world 的 `GpuFluids3d` 读取粒子和活动标记；自定义计算通过 `Fluid3dComputePass` 扩展。完整的接入、碰撞、生命周期和扩展接口见 [3D 文档](docs/3d.md)，水槽示例见 `examples/faucet_3d.rs`。

## 2D 作为库接入

```toml
[dependencies]
bevy = { version = "0.20", default-features = false, features = ["2d"] }
bevy_gpu_fluid = { path = "path/to/bevy_gpu_fluid" }
```

```rust,no_run
use bevy::{camera::ScalingMode, prelude::*};
use bevy_gpu_fluid::prelude::*;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, Fluid2dPlugins))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    commands.spawn((Camera2d, Projection::Orthographic(OrthographicProjection {
        scaling_mode: ScalingMode::FixedVertical { viewport_height: 10.0 },
        ..OrthographicProjection::default_2d()
    })));
    let particles = FluidParticle2d::lattice(
        Vec2::new(-7.5, -3.9), UVec2::new(40, 48), 0.16,
    );
    let fluid = Fluid2d::new(Fluid2dSettings::default(), particles).unwrap();
    commands.spawn(
        Fluid2dBundle::new(fluid, &mut images, UVec2::new(1280, 720)).unwrap()
    );
}
```

`Fluid2dPlugins` = `Fluid2dPlugin`（计算）+ `Fluid2dRenderPlugin`（默认渲染）。必须放在 `DefaultPlugins` 后面。库不会创建相机或窗口。可生成多个实体，各自持有独立的粒子、网格、时钟及输出纹理。

`Fluid2dBundle` 包含模拟组件、输出组件、Sprite 和 Transform。也可以只生成 `Fluid2d` 做计算，或者手动组合 `Fluid2d` 与 `Fluid2dSurface`，把 `surface.image` 交给其他 Sprite、UI 或材质。输出纹理是线性颜色的 `Rgba8Unorm`，背景透明，内部渲染发生在 Bevy 相机之前。默认 renderer 先用 instanced quads 累加密度核到 `Rgba16Float` 场，再用全屏 pass 提取等值面并着色；中间纹理由库缓存，无逐粒子 ECS 实体和每帧 CPU 回读。

默认效果是 **metaball 液体表面**：相邻粒子的密度贡献相加，靠近时形成连续轮廓，内部直接使用 `color` 纯色填充，仅边缘保留 alpha 抗锯齿。`Fluid2dSurface::style` 提供颜色、显示半径和 `surface_threshold`（默认 `0.18`，越低越容易融合）、`smoothing_factor`（默认 `2.5`，控制核支撑范围）。详见 [渲染说明与调参](docs/rendering.md)。这是独立于 SPH 物理密度的视觉场，不改变粒子运动；不同流体实例独立重建，不相互融合。Transform 只改变显示，不改变物理空间。修改模拟 bounds 后，若想让显示保持世界坐标一致，需要同步更新 Sprite 的 `custom_size` 和 Transform。

## 有限出水口与碰撞体

```rust
use bevy::prelude::*;
use bevy_gpu_fluid::prelude::*;

let faucet = Fluid2dFaucet {
    position: Vec2::new(0.0, 3.5),
    width: 1.6,
    is_open: true,
    ..default()
};
let mut fluid = Fluid2d::from_faucet(Fluid2dSettings::default(), 1920, faucet)?;
fluid.set_colliders(vec![
    Fluid2dCollider::Rectangle(Rect::new(-2.0, -1.0, 2.0, -0.7)),
    Fluid2dCollider::Circle { center: Vec2::new(0.0, 1.0), radius: 0.6 },
])?;
fluid.set_faucet_open(false)?;
// 修改位置、宽度和开关状态不会重置已释放的粒子。
fluid.set_faucet(Some(Fluid2dFaucet { position: Vec2::new(2.0, 3.5), width: 0.8, ..faucet }))?;
# Ok::<(), Fluid2dError>(())
```

`from_faucet` 一开始没有活动粒子，但预先分配固定上限的 GPU 缓冲区。开启后按固定子步释放水平粒子排，`position` 是出口中心，`width` 是横向宽度。`particle_spacing` 决定每排数量（`floor(width / spacing)`，至少一个），`rows_per_second` 决定流量，`velocity` 是初速度。每个固定子步最多释放一排，流量不能超过模拟步频；末排只释放剩余数量。

关闭只停止新粒子的释放，已释放粒子继续运动；重新开启继续使用剩余预算。耗尽后停止释放，不自动补充、不循环，不因切换开关、移动出口或修改宽度而增加总数。只有显式 `reset()` 才会清空活动粒子、恢复原来的总预算，保留出口配置和开关状态。暂停模拟也会暂停释放，关闭或暂停期间不累计补发。`set_faucet(None)` 移除出口但保留现有粒子及预算。

主世界 `Fluid2d::particle_count()` 表示总预算；render world 的 `GpuFluid2d::particle_count()` 表示已释放槽位前缀（开启回收后包含回收槽，实际活动数见 GPU `live_count_buffer()`），`particle_capacity()` 表示总上限，`is_exhausted()` 表示是否已全部释放。自定义 render pass 遍历前 `particle_count()` 个槽位，并检查 `particle_activity_buffer()` 中的活动标记。缓冲区容量及 16-byte 粒子 ABI 保持固定，开关和释放不会替换粒子缓冲区。

碰撞体使用模拟坐标，矩形支持轴对齐实心 `Rect` 和 `OrientedRectangle { center, half_extents, rotation }`（弧度），圆形由中心和半径定义。所有形状合计每个流体实例最多 **16** 个，超过上限或无效尺寸会返回错误并保持原配置。`set_colliders` 可在运行中更新整个列表，不会重置粒子。碰撞使用物理 `particle_radius` 和 `restitution`，包含圆角矩形及圆形的连续扫掠，避免高速穿过薄障碍物；粒子已处于实体内部时会进行推出。每步最多求解四次碰撞和四轮穿透修正；彼此重叠或与边界挤成不可容纳空间的实体不保证全部约束同时满足。旋转矩形在局部坐标执行相同的圆角连续扫掠；碰撞体不传递线/角速度，没有刚体反馈。默认 render pass 只画流体表面；示例中用独立 Sprite / Mesh 显示障碍物。

## 自定义 render pass

添加 `Fluid2dPlugin`，不添加 `Fluid2dRenderPlugin`。在 `RenderApp` 的 `RenderGraph` schedule 中注册自己的系统，放入 `Fluid2dSystems::Render`，或显式排在 `Fluid2dSystems::Simulate` 后、`camera_driver` 前。

```rust,ignore
app.sub_app_mut(RenderApp)
    .add_systems(RenderGraph, my_render_pass.in_set(Fluid2dSystems::Render));
```

计算插件会建立这些 set 的顺序。`GpuFluids2d` 是 render world 中公开的资源，键是 **main world 的 Entity**：

```rust,ignore
let fluid = gpu_fluids.get(main_world_entity).unwrap();
let particles = fluid.particle_buffer(); // read-only use by applications
let count = fluid.particle_count();
let bounds = fluid.settings().bounds;
```

缓冲区布局固定为 `FluidParticle2d`：`position: vec2<f32>`（offset 0）、`velocity: vec2<f32>`（offset 8），stride **16 bytes**；支持 `STORAGE | VERTEX | COPY_SRC`。不向外暴露计算用的内部 bind group。渲染阶段只读粒子及活动标记，自己决定 storage binding 或 instanced vertex binding；计算扩展可在子步回调中写入，见 [计算扩展说明](docs/compute.md)。

初始化和重置会更换粒子缓冲区；自定义 renderer 应在 `Buffer::id()` 改变时重建 bind group，或者像示例一样每帧构建。所有 pass 编码、资源上传和 pipeline cache 操作都在 render world。`GpuFluids2d::error(entity)` 可读取设备限制造成的分配失败，错误也会写日志。

完整示例见 [examples/custom_pass.rs](examples/custom_pass.rs) 和 [应用 shader](examples/shaders/custom.wgsl)。该示例无需复制任何 SPH 或网格实现。

## 2D 算法与调度

采用基础 weakly compressible SPH：二维归一化 poly6 密度核、二维 spiky 压力梯度、viscosity Laplacian、线性状态方程 `p = stiffness * max(rho - rest_density, 0)`、半隐式 Euler 积分、矩形边界及矩形/圆形实心障碍物碰撞。

网格使用 **dense uniform grid + atomic linked list**，cell width 等于 smoothing radius。每个格子一个 atomic 表头，每个粒子一个 next 索引。构建时 `atomicExchange` 插入，查询邻近 3×3 个格子。不需要排序，也没有固定的每格容量，不会因格子过满丢粒子。内存是 O(粒子数 + 格子数)，正常分布下邻域查询接近线性；粒子全部挤在一个格子时仍可能退化到 O(N²)。极稀疏、巨大空间后续可考虑排序或稀疏哈希。

每个固定子步由库内部顺序执行以下 compute pass：

1. 出水口释放新增粒子（没有新粒子时跳过）。
2. 清空 grid heads。
3. 为活动粒子构建链表。
4. 计算密度和压力（包含自身密度贡献）。
5. 计算加速度，写入独立缓冲区。
6. 更新速度、位置，处理障碍物和墙壁碰撞。

密度、加速度和积分之间存在全局 GPU pass 边界；不会在同一个并发邻居遍历中修改粒子速度。最后一个子步完成后再运行渲染。链表插入次序依赖 GPU 调度，因此浮点累加不保证逐位确定性。

时间使用 Bevy `Time`，默认 `dt = 1/240 s`，正常 60 FPS 大约执行四个子步；不足一步的时间会累计。每帧最多 `max_substeps`（默认 8），长帧多出的时间丢弃，防止追帧失控。pipeline 尚未就绪时不积累时间；暂停时清空余量。

参数按模拟单位设置。默认间距 `0.16`、mass `25.6` 对应 `rest_density = 1000`；改变间距时可先按 `mass ≈ rest_density * spacing²` 调整。增大 stiffness、重力或初始速度可能需要缩小 time step。速度和加速度上限用于控制数值爆炸，不替代稳定时间步。粒子容量固定，支持直接初始化全部 seed 或从有限出水口逐步释放；不支持运行中无限发射、任意形状障碍物或刚体耦合。

## 更新、暂停和生命周期

```rust,ignore
fluid.set_paused(true);
fluid.reset(); // 恢复原 seed，不改变暂停状态
fluid.reset_with_particles(new_particles)?; // 用新的已激活 seed 重设总预算，支持空数组

let mut settings = fluid.settings().clone();
settings.gravity = Vec2::new(3.0, -9.81);
fluid.set_settings(settings)?;
```

设置、出水口、碰撞体和 seed 在主世界验证。日常修改设置不重新上传粒子；网格尺寸变化只替换网格及辅助缓冲区，保留当前粒子位置、活动数量和剩余预算。reset 会恢复初始 seed（出水口模式为空），并清空时间余量。设置 bounds 缩小后，粒子在下一个积分子步被夹回墙内；出水口仍需处于合法的碰撞边界内。

despawn 或移除 `Fluid2d` 后，下次 extraction 会释放对应的内部 GPU 资源；移除 `Fluid2dSurface` 会停止默认渲染。应用自己持有的 `Handle<Image>` 和 buffer clone 仍遵循 Bevy 引用计数生命周期。

## Shader 资源与验证

内部 WGSL shader 用 `embedded_asset!` 编译进库，经 `embedded://bevy_gpu_fluid/...` 和 Bevy `AssetServer` 加载，保留强句柄。消费者无需复制 assets、配置相对路径或依赖运行目录。没有使用 naga_oil 预处理指令；Bevy 0.20 支持这种纯 WGSL 文件，参见 [官方迁移指南](https://bevy.org/learn/migration-guides/0-19-to-0-20/#wesl-shaders)。

```sh
cargo check --all-targets
cargo test
cargo clippy --all-targets -- -D warnings
# 下面两项需要真实 GPU，可在无显示器环境运行
cargo test --lib -- --include-ignored
cargo test --test bevy_smoke -- --ignored
cargo test --test bevy_3d_smoke -- --ignored
```

普通测试验证配置、固定步长、开关时钟、碰撞体总量及所有 WGSL 的 Naga 校验。GPU 测试使用独立 CPU O(N²) 实现对照密度、速度和位置，检查 129 粒子（工作组尾部）、单粒子、空流体、同格 129 粒子无截断、链表无环及 600 子步后位置有限且在边界内；验证高速扫掠、圆角、内部推出和多次反弹。Bevy 集成测试覆盖内部 assets 加载、默认及外部自定义 pass 的实际纹理像素输出、多实例、暂停、reset 后 renderer 重新绑定、网格调整、有限释放/关闭/移动/耗尽/重置和 despawn 清理；还验证相邻粒子融合、阈值调高后分离、远粒子不融合和空流体清屏。设置 `BEVY_FLUID_CAPTURE_DIR` 环境变量可将融合/分离测试的实际 GPU 输出保存为 PNG。

## 许可

MIT OR Apache-2.0。

## 开发与发布

开发时按需启用 `dev` feature（动态链接 Bevy，缩短链接时间，不改变模拟行为）：

```sh
cargo check --all-targets
cargo test
cargo clippy --all-targets -- -D warnings
```

发布构建不要启用 `dev` 或 `--all-features`：`cargo build --release --locked`。

`.cargo/` 属于本机环境配置，不纳入版本库；需要共享编译缓存等设置时请自行创建，不要提交。
