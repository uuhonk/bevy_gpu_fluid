# 默认液体渲染

2D / 3D 的默认 renderer 都从粒子构造视觉密度场，让附近粒子连接成连续液体表面。已有示例直接使用新效果，应用无需增加系统或复制 shader。所有新 WGSL 仍经 `embedded_asset!` 随库打包加载，多 pass 和中间纹理由库管理；外部自定义 renderer 的 GPU buffer 接口不变。

## 密度与融合

视觉核为 `w(r) = max(1 - r² / R², 0)³`，其中 `R = particle_radius × smoothing_factor`。同一实例中粒子的贡献相加，超过 `surface_threshold` 的区域作为液体内部，因此两个独立圆形粒子靠近时会长出连接的液桥。3D 在屏幕空间累加投影核，并剔除离最近表面超过 `particle_radius × max(3, smoothing_factor)` 的深度层。

这里的密度是无量纲的**渲染场**，不直接使用 SPH 的 `rest_density`、质量或计算密度。修改视觉参数不改变模拟稳定性、碰撞半径或粒子运动，也不会加入物理表面张力。每个流体实例独立重建，跨实例不会融合。

## 2D

内部两步：将粒子核累加到 `Rgba16Float` 场，再提取密度等值线，以 `color.rgb` 纯色填充，仅边缘使用 alpha 抗锯齿。最终输出仍是透明背景、直通 alpha、线性颜色的 `Fluid2dSurface::image`（`Rgba8Unorm`），可以继续用于 Sprite / UI / 自定义材质。

```rust,ignore
// 在创建 bundle 后或运行时修改 surface.style。
bundle.surface.style.surface_threshold = 0.18;
bundle.surface.style.smoothing_factor = 2.5;
```

| `Fluid2dStyle` 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `particle_radius` | `0.095` | 显示尺寸，独立于物理碰撞半径 |
| `smoothing_factor` | `2.5` | 核支撑半径倍数，范围 `[1, 8]`；越大越容易连成片 |
| `surface_threshold` | `0.18` | 等值阈值，范围 `(0, 1)`；越低越容易融合 |
| `color` | 蓝色 | 线性 RGBA，alpha 控制最终透明度 |
| `speed_color_scale` | `0.0` | 仅为源码兼容保留，默认 renderer 忽略 |

2D 精度由 `Fluid2dSurface::new` 的纹理尺寸决定。更高分辨率能改善轮廓，同时增加密度场填充与着色成本。

## 3D

内部五步：有效表面核心深度、横向 bilateral 深度滤波及连接处补齐、纵向滤波及补齐、前表面附近的投影密度累加、最终表面合成。不可见的核外围不参与最近深度选择，避免裁掉后方有效液体、出现环形空洞。滤波保留深度边缘，合成时直接以 `color.rgb` 纯色填充，不按速度、厚度、法线或视角改变颜色。最终连续采样密度场，在轮廓像素内以 4×4 子采样积分覆盖率，以预乘 alpha 混入相机颜色；不开 MSAA 也有轮廓抗锯齿，降低内部场分辨率时也不会直接放大方块像素。密度始终高于阈值的内部区域保持统一颜色与透明度，不因深度层间的密度跳变画出暗色圆圈。深度插值忽略空场像素，核心深度使用与密度阈值一致的半径，减少与场景交界处的错误裁边。支持 RenderLayers、MSAA、HDR 和相机视口。

```rust,ignore
commands.spawn((fluid, Fluid3dRender {
    surface_threshold: 0.18,
    smoothing_factor: 2.5,
    blur_radius: 3.0,
    resolution_scale: 1.0,
    ..default()
}));
```

| `Fluid3dRender` 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `particle_radius` | `0.09` | 显示半径与表面深度尺度 |
| `smoothing_factor` | `2.5` | 投影核支撑半径倍数，范围 `[1, 8]` |
| `surface_threshold` | `0.18` | 融合阈值，范围 `(0, 1)` |
| `blur_radius` | `3.0` | 深度场像素中的滤波半径，范围 `[0, 8]`，按整数取整；`0` 关闭核心深度平滑（连接处深度补齐仍保留） |
| `resolution_scale` | `1.0` | 中间场相对视口的分辨率，范围 `[0.25, 1]` |
| `color` | `(0.04, 0.36, 0.56, 0.78)` | 统一的线性 RGBA 水色；alpha 控制透明度 |
| `enabled` | `true` | 运行时开关 |

`speed_color_scale`、`absorption`、`roughness` 仅为源码兼容保留，不影响默认纯色渲染。

先调 `particle_radius` 适配粒子间距，再调 `smoothing_factor` / `surface_threshold` 控制连接；滤波平滑核心深度，并在每个方向最多补齐 8 个场像素的连接区深度；最终轮廓仍由密度阈值控制。默认 `resolution_scale = 1.0` 优先保证轮廓精度；减小分辨率或 `blur_radius` 可降低 GPU 成本。每个可见相机/实例组合持有五张场纹理，约 `24 × 场宽 × 场高` bytes；尺寸不变时复用纹理和 bind group。

3D 默认是略透明的纯色水面，alpha 在每个实例的重建表面上混合一次，不会因粒子层数增加而反复叠色。它检测场景不透明深度，但不写场景深度，避免半透明边缘硬挡住后面的绘制。`color.a = 1` 可设为不透明，`0` 隐藏。多个流体实例按模拟 bounds 中心近似从远到近合成；不与 Bevy 的透明 Mesh 逐物体交错排序，互相穿插的透明物体需要自定义 renderer。

3D 是屏幕空间近似，不做光照、高光、反射或折射，不会生成网格。较近的不同深度层仍可能连接；极近相机、粒子穿越近裁面和很大的显示核也会受到 billboard 近似的限制。

## 技术参考

- [GPU Gems 3: Point-Based Visualization of Metaballs on a GPU](https://developer.nvidia.com/gpugems/gpugems3/part-i-geometry/chapter-7-point-based-visualization-metaballs-gpu)：核叠加与隐式表面。
- [NVIDIA: Screen Space Fluid Rendering for Games](https://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf)：屏幕空间深度与滤波思路。本库采用简化的纯色实现。
