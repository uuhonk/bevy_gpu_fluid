# 旋转碰撞与自定义计算 pass

2D / 3D 分别提供独立的碰撞、GPU 活动状态和扩展接口。旋转碰撞内置于模拟；区域回收通过可选插件实现，默认插件组不包含回收。

## 旋转碰撞体

```rust
use bevy::prelude::*;
use bevy_gpu_fluid::prelude::*;

let slope = Fluid2dCollider::OrientedRectangle {
    center: Vec2::new(-2.0, -0.8),
    half_extents: Vec2::new(1.4, 0.15),
    rotation: -0.28, // 弧度，逆时针为正
};
let platform = Fluid3dCollider::OrientedCuboid {
    center: Vec3::new(1.9, -1.25, 0.0),
    half_extents: Vec3::new(1.1, 0.15, 1.5),
    rotation: Quat::from_euler(EulerRot::XYZ, 0.12, 0.18, -0.22),
};
```

将它们交给各自的 `fluid.set_colliders(...)`。原来的 `Rectangle(Rect)` / `Cuboid(Fluid3dBounds)` / 圆 / 球继续支持，每个实例所有实心碰撞体合计最多 16 个。半尺寸必须为有限正数；2D 角度必须有限，3D quaternion 必须为单位旋转（长度平方误差容许 `0.001`，上传时归一化）。验证失败保留原配置。

粒子位置、位移转到碰撞体局部坐标，执行现有的圆角矩形或圆角长方体连续扫掠，再将法线转回世界坐标。包括面、边、角和内部推出，不采用外包 AABB 代替旋转几何。运行时更新位置/旋转不重置粒子；每个子步把碰撞体视为静止，不传递障碍物线速度或角速度。可视 Sprite / Mesh 的旋转需由应用同步。

## 可选区域回收插件

```rust,no_run
use bevy::prelude::*;
use bevy_gpu_fluid::prelude::*;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, Fluid2dPlugins, Fluid2dRecyclingPlugin))
        .add_systems(Startup, setup)
        .run();
}
fn setup(mut commands: Commands) {
    let fluid = Fluid2d::from_faucet(
        Fluid2dSettings::default(), 1920, Fluid2dFaucet::default(),
    ).unwrap();
    let drain = Fluid2dRecycling::new(vec![
        Fluid2dCollider::OrientedRectangle {
            center: Vec2::new(0.0, -3.8),
            half_extents: Vec2::new(6.0, 0.2),
            rotation: 0.05,
        },
    ]).unwrap();
    // 这里只生成计算组件。需要默认显示时可把 drain 附加到 Fluid2dBundle。
    commands.spawn((fluid, drain));
}
```

3D 对应 `Fluid3dRecyclingPlugin`、`Fluid3dRecycling::new(Vec<Fluid3dCollider>)`。这两个插件独立选择安装，必须放在各自模拟插件后；只计算也能使用，不依赖默认渲染插件。内部 WGSL 全部内嵌。

回收区域复用碰撞体几何，但**独立于实心碰撞列表**：不用把 drain 再加到 `set_colliders`，否则实心响应可能先把粒子弹开。每个实例至多 16 个回收区域，支持圆/球、轴对齐和旋转矩形/长方体。`set_regions` 验证后替换，空列表关闭回收；`enabled = false` 或移除组件也会停止回收。没有安装插件时，这个组件不产生任何效果。

插件在每个固定子步执行：

1. 发射后、建网格前记录位置，回收已经在区域内的粒子。
2. 积分后检测本子步起点到终点的线段，回收穿过区域的粒子，即使两端都在区域外。

触发依据粒子中心，不扩张物理/显示半径；区域边界计入内部。线段表示子步净位移，不重建实心障碍物多次反弹的完整折线路径。暂停同步暂停回收；移动区域会在下一子步检测当前粒子，不模拟区域自身的扫掠。

回收粒子通过 GPU 活动标记从网格、SPH、积分和默认/示例渲染中移除。多个区域重复命中只减少一次活动计数。**回收不补充出水预算，也不自动重生**：总容量固定，流完即止；只有显式 reset 恢复 seed / 原预算。关闭插件不会复活已回收粒子。

运行示例：

```sh
# 2D：带旋转障碍物，回收插件未安装
cargo run --example faucet
# 2D：安装回收插件；绿色区域会回收粒子
cargo run --example faucet -- --recycle
# 3D：水龙头与水槽，默认安装回收插件
cargo run --example faucet_3d
# 3D：不安装回收插件，以便比较
cargo run --example faucet_3d -- --no-recycle
```

## GPU 粒子生命周期接口

粒子 ABI 仍为 2D 16 bytes、3D 32 bytes，索引稳定，不压缩粒子数组，也不把回收状态藏进 position / velocity。

| Render-world 方法 | 含义 |
| --- | --- |
| `particle_capacity()` | 固定池容量 |
| `particle_count()` | 已释放槽位前缀上界，包含回收槽；没有回收时等于活动数量 |
| `particle_activity_buffer()` | 每槽一个 `u32`，`1` 活动、`0` 未释放/已回收；计算 shader 使用 `atomic<u32>`，渲染只读 |
| `live_count_buffer()` | 单个 GPU `atomic<u32>`，精确活动数量，支持 `STORAGE | COPY_SRC` |
| `is_exhausted()` | 有限出水预算已全部释放，不代表池内仍有活动粒子 |

主世界的 `Fluid2d/3d::particle_count()` 仍表示总预算。正常运行没有 CPU 回读；需要 UI 统计时由应用异步读取 `live_count_buffer`。回收后的槽位仍占固定池内存，遍历/绘制以前缀上界 dispatch，并由标记跳过。

reset 更换粒子、标记、活动计数缓冲区；网格尺寸变化保留三者以及剩余预算。自定义 renderer 必须同时绑定粒子与 activity，跳过 `activity[instance] == 0u`；两个 `custom_pass` 示例已演示。不要只拿 `particle_count()` 当成精确活动数量。

## 接入自定义计算 pass

| 阶段 | 可用状态与典型用途 |
| --- | --- |
| `FluidComputeStage::BeforeSubstep` | 新发射粒子已激活，尚未建网格；修改位置/速度、标记回收 |
| `BeforeIntegrate` | SPH 密度、压力、加速度已完成；增加力、修改加速度 |
| `AfterSubstep` | 积分/实心碰撞已完成；触发器、回收、约束 |

实现 `Fluid2dComputePass` / `Fluid3dComputePass`，将拥有 shader pipeline 和 bind group 的对象插入 render-world 的 `Fluid2dComputePasses` / `Fluid3dComputePasses`。无需复制 SPH shader 或访问私有 bind group。

```rust,ignore
struct MyForcePass { /* CachedComputePipelineId、各流体的 BindGroup */ }
impl Fluid2dComputePass for MyForcePass {
    fn is_ready(&self, entity: Entity, cache: &PipelineCache) -> bool {
        // 对该实例所需的所有 pipeline 检查就绪状态。
        true
    }
    fn encode(&self, stage: FluidComputeStage, ctx: &mut Fluid2dComputeContext<'_>) {
        if stage != FluidComputeStage::BeforeIntegrate { return; }
        // ctx.entity 是主世界 Entity。
        // ctx.particle_count 是当前子步的释放前缀，而非整帧最终前缀。
        // ctx.fluid.acceleration_buffer() 可读写，activity 必须检查。
        // 在 ctx.encoder 中创建自己的 compute pass，按 ctx.particle_count dispatch。
    }
}
```

注册/更新扩展的系统放在 **Render schedule** 的 `Fluid2dSystems::PrepareCompute` 或 3D 对应 set 中：此时库已分配 GPU 缓冲区，尚未计划发射和固定步长。默认 renderer 在 `Plan` 后准备，能看到整帧的新释放前缀。`insert(pass)` 每种 Rust 类型只有一项，重复插入更新准备状态并保留原执行顺序；`remove::<MyForcePass>()` 移除。每个阶段按注册顺序执行；多个准备系统存在依赖时，应用需显式排序它们。

`is_ready` 为 false 时只暂停相关实例的子步计划，不释放新粒子，也不累积之后补跑的时间。`encode` 每个固定子步调用三次，而非每帧一次；`ctx.substep_index` 为本实例自 reset 起的子步编号，reset 后从 0 开始。每个阶段都有独立 GPU pass 边界。

可用缓冲区还包括 `density_pressure_buffer()`（每槽 `vec2<f32>`，只读）和 `acceleration_buffer()`（2D `vec2<f32>`，3D `vec4<f32>` 的 xyz）。只在子步扩展中写粒子/加速度；不重排槽位、不激活未释放或已回收的槽位。回收时需 `atomicExchange(activity[i], 0u)`，仅在旧值为 `1` 时 `atomicSub(live_count, 1u)`。普通渲染阶段只读。

不要用同帧重复 `RenderQueue::write_buffer` 覆盖同一个子步 uniform；使用独立 uniform 区段/动态偏移，或在 command encoder 中编码有序数据复制。重置可能替换所有 buffer；网格调整也可能替换密度/加速度缓冲区，即使粒子 buffer 没换。缓存 bind group 时检查所有绑定 buffer 的 `Buffer::id()`；也可以像外部渲染示例一样重新建立。

实现参考：库内的 [2D recycling.rs](../src/fluid2d/recycling.rs) / [3D recycling.rs](../src/fluid3d/recycling.rs) 完全使用上述公共扩展接口；集成测试 `application_compute_passes_gate_readiness_run_each_substep_and_add_forces` 则从应用侧独立创建 WGSL 力场，验证就绪门控、三个阶段、多个 pass 顺序和动态移除。
