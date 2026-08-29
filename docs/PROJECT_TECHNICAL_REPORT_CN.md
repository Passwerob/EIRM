# Project 技术报告总结

更新时间：2026-08-29
报告范围：旋转棱镜 AMI 事件相机模拟器、CUDA 加速链路及可选事件后处理

> **命名说明**：本文将内部项目代号统一写作 `Project`，可选后处理层写作
> `Project-Lite`。仓库目录、Python 包名、命令名和历史产物仍保留原有名称，
> 这是为了兼容已有脚本和保证结果可复现，不代表对外使用内部代号。

## 1. 结论摘要

Project 实现了一条物理顺序明确的旋转棱镜 AMI 事件模拟链路：

```text
带时间戳的 RGB
    -> 任意时间点 VFI
    -> 线性亮度
    -> 旋转棱镜成像 warp
    -> DVS 事件生成
    -> 按事件时间逆补偿到 reference plane
    -> 原始/补偿事件、指标和可视化
```

核心结论如下：

1. 棱镜成像必须发生在 DVS 事件生成之前；生成事件后再整体旋转坐标不能
   表达棱镜运动与自然图像运动在传感器成像阶段的共同作用。
2. 在 8 个 DL3DV 场景、统一 40→480 帧协议下，AMI 平均事件数是无棱镜
   baseline 的 `2.083x`，补偿后 20 ms reference-plane edge coverage 为
   `0.730 +/- 0.148`。
3. 在匹配的 `5.428 ms` 短窗口中，补偿相对 raw AMI 将 edge distance 从
   `1.165 px` 降至 `0.898 px`，Edge F1 从 `0.583` 提升至 `0.592`。
4. 在独立的 `8.142 ms` 下游评估中，补偿 AMI 的平均 event-flow EPE 为
   `1.502 px`、tracking EPE 为 `0.792 px`、1 px 跟踪成功率为 `0.837`，
   均优于 raw AMI 和无棱镜 DVS baseline 的对应结果（edge distance 除外）。
5. Custom CUDA DVS 在 A100 上相对官方 PyTorch CUDA 路径达到约 `3.1x`
   （256x256）和 `1.68x`（720x720）；960x540 完整纯计算链路达到 `91.3 FPS`。
6. VFI 误差已被单独量化。RIFE Full/Lite 的事件数分别保留真实高帧率 RGB
   事件数的 `96.38%`/`96.52%`，空间 occupancy IoU 约 `0.998`；因此不能
   宣称 VFI 完全无伪影，但现有证据不支持把 AMI 增益归因于明显的 VFI 空间伪影。

当前最稳妥的表述是：Project 在给定的简化棱镜模型、DVS 参数和 VFI 设置下，
提供物理顺序正确、可复现且可扩展的 AMI 事件模拟与坐标补偿；它尚未完成真实
硬件标定、完整光学模型或自适应旋转的闭环验证。

## 2. 系统组成与实现

### 2.1 输入、VFI 与亮度

- 输入为 RGB 帧、单调递增时间戳和相机内参；DL3DV 焦距从
  `transforms.json` 读取。
- `DSECRGBSequence` 统一分辨率、时间单位和 calibration，并拒绝非递增时间戳。
- Practical-RIFE 使用直接 timestep API，可按同一时间轴生成 240/480/960 Hz，
  不依赖隐藏的递归 2x 插帧。正式协议使用 40 个源帧生成 480 帧。
- RGB 先从 sRGB 转线性空间，再计算
  `Y = 0.2126 R + 0.7152 G + 0.0722 B`（此处 RGB 已为 linear）。
- Full RIFE 用作论文 reference；4.25 Lite 用于速度优先的演示。

### 2.2 棱镜模型与旋转

- 默认 `PaperSimplifiedPrismModel` 是 calibrated paraxial/ray-deflection
  approximation，根据相机射线、内参和有效偏折角计算像素映射。
- 主实验参数：有效偏折角 `0.5°`、`theta0=0`、固定转速 `720 RPM`。
- 固定旋转使用解析轨迹：

  ```text
  theta(t) = theta0 + omega * (t - t0)
  omega = RPM * 2*pi / 60
  ```

- `PrismWarper` 使用 inverse sampling 和双线性采样，支持
  `border`/`reflection`/`zeros` padding，并保存全角度 `valid_mask`。

### 2.3 DVS 事件生成

提供三条可复现路径：

| 路径 | 用途 |
| --- | --- |
| 官方 reference DVS | 论文主结果和数值回归 |
| Torch CUDA DVS | 官方模型的 GPU-resident 实现 |
| Custom CUDA fast | 性能路径；逐帧内并行像素递归并预分配事件 buffer |

统一事件格式为 `x, y, t, p`，其中 `t` 是秒制 `float64`，`p` 为 `{-1,+1}`。
Custom kernel 内部使用 `int32[timestamp_us, x, y, polarity]`，仅在 API/HDF5
边界转换为标准格式。GPU 与 CPU 使用不同随机数流，因此质量判定采用统计门，
不要求逐事件 bitwise 相同。

### 2.4 逐事件逆补偿

- 每个事件使用其真实时间 `t` 计算棱镜角度，而不是整帧中心角度。
- 固定旋转使用解析角度；已排序的 DVS 输出跳过重复稳定排序。
- 逆 Rodrigues 映射采用展开公式并缓存固定三角常量；大事件流按 chunk 处理。
- 输出同时保留原始整数坐标和补偿后的 `x_float/y_float`。

### 2.5 Project-Lite 后处理

Project-Lite 是可选过滤层，不是 VFI 模型变体。默认顺序为：

1. all-orientation valid ROI；
2. 每像素 `50 us` refractory；
3. `1 ms` 时空活动窗口，空间半径 1 像素。

科学实验使用未过滤的原始事件；Project-Lite 面向部署和下游输入，并在 HDF5
属性中记录源 stage、过滤参数、各阶段事件数和最终保留数。

## 3. 主实验结果

### 3.1 8 场景批量实验

协议固定为 8 个 DL3DV 场景、40 个源帧、同一约 1.3 s 时间范围输出 480 帧、
Practical-RIFE Full fp16、`paper_simplified`、`0.5°`、`720 RPM`、DVS346、
seed=0。全部运行通过有限值、边界、时间戳、极性、排序和补偿检查。

| 指标（20 ms 窗口） | 均值 | 场景标准差 |
| --- | ---: | ---: |
| 无棱镜 / sparse direct 事件数 | `1.706x` | `0.344` |
| AMI / sparse direct 事件数 | `3.379x` | `0.855` |
| AMI / 无棱镜事件数 | `2.083x` | `0.725` |
| Compensated edge coverage | `0.730` | `0.148` |
| AMI-only recovered edge coverage | `0.247` | `0.214` |
| Compensated edge distance | `2.635 px` | `0.797` |
| 有效旋转区域比例 | `0.9865` | `0.00001` |

这说明固定棱镜旋转在不同场景中稳定增加观测事件，并可通过补偿回到统一
reference plane；增益大小随场景纹理、自然运动和边缘方向变化，不应只用总事件数解释。

### 3.2 匹配短窗口主表

主场景 `03780d41` 使用两个 480-frame 输出间隔，即 `5.428 ms`。Raw AMI 与
Compensated AMI 的事件数量、时间和极性完全相同，区别仅为坐标平面。

| 方法 | 坐标平面 | 事件数/窗 | Valid coverage | Edge F1 | Edge distance |
| --- | --- | ---: | ---: | ---: | ---: |
| DVS 480-frame，无棱镜 | reference | `19,306` | `0.022` | `0.323` | `1.156 px` |
| Raw AMI | prism sensor | `62,815` | `0.065` | `0.583` | `1.165 px` |
| Compensated AMI | reference | `62,815` | `0.103` | `0.592` | `0.898 px` |

统一 RGB-flow oracle 仅作为 photometric diagnostic：无棱镜 DVS 为
`36.78 dB / 0.965 SSIM`，补偿 AMI 为 `36.04 dB / 0.963 SSIM`。因此报告中
不应声称 Project 在所有 PSNR 指标上优于无棱镜 baseline；核心贡献是坐标对齐、
有效覆盖和下游几何指标。

### 3.3 逆补偿闭环

- 合成静态线和角点测试：mean edge distance 从 `1.487 px` 降至 `0.142 px`，
  P90 从 `2.865 px` 降至 `0.955 px`。
- 主场景 raw AMI 的 mean/P90 为 `0.662/1.910 px`，补偿后为
  `0.326/0.955 px`。
- 补偿不改变事件数、时间和极性，目标是消除棱镜诱导坐标位移。

### 3.4 下游任务与 simulator baseline

8 场景、统一 `8.142 ms` 窗口；预测阶段只使用事件流，RGB 只用于生成参考光流。
ESIM 行是官方核心逻辑的 NumPy reference port，不是原生 ROS/C++ 构建。

| 方法 | Edge F1 ↑ | Flow EPE ↓ | Outlier >1 px ↓ | Tracking EPE ↓ | Track <=1 px ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| DVS-Voltmeter，无棱镜 | `0.403 +/- 0.213` | `2.578 +/- 2.506` | `0.439 +/- 0.262` | `1.551 +/- 1.277` | `0.708 +/- 0.268` |
| V2E clean，无棱镜 | `0.426 +/- 0.158` | `1.673 +/- 1.401` | `0.304 +/- 0.150` | `0.900 +/- 0.611` | `0.805 +/- 0.182` |
| ESIM reference，无棱镜 | `0.320 +/- 0.182` | `2.409 +/- 1.933` | `0.472 +/- 0.260` | `1.697 +/- 1.402` | `0.658 +/- 0.286` |
| Raw AMI | `0.620 +/- 0.149` | `2.016 +/- 1.408` | `0.625 +/- 0.102` | `1.359 +/- 0.511` | `0.430 +/- 0.121` |
| **Compensated AMI** | **`0.636 +/- 0.169`** | **`1.502 +/- 1.479`** | **`0.286 +/- 0.212`** | **`0.792 +/- 0.594`** | **`0.837 +/- 0.148`** |

相对 Raw AMI，补偿后的 Edge F1 提升 `0.016`，flow EPE 降低 `0.514 px`，
outlier 降低 `0.338`，tracking EPE 降低 `0.568 px`，1 px 跟踪成功率提升
`0.407`。无棱镜 DVS 的 edge distance 更低是预期现象，因为它没有经历棱镜位移；
这不构成对补偿效果的反例。

### 3.5 VFI 质量与事件影响

DL3DV `03780d41` 使用 80 帧留帧实验（偶数帧输入、奇数帧为目标，960x540）：

| 方法 | PSNR | SSIM | 事件数 | 相对真实 RGB |
| --- | ---: | ---: | ---: | ---: |
| 线性插值 | `21.93 dB` | `0.636` | `9,997,243` | `77.05%` |
| RIFE Lite | `25.89 dB` | `0.816` | `12,522,533` | `96.52%` |
| RIFE Full | `26.04 dB` | `0.819` | `12,504,891` | `96.38%` |
| 真实 RGB（参考） | - | - | `12,974,474` | `100%` |

RIFE Lite/Full 的空间 occupancy IoU 分别为 `0.9989/0.9981`，极性比例差为
`0.114%/0.124%`；线性插值的事件数明显偏低。结论是 VFI 影响了精确时间和局部
事件数量，但 RIFE 的整体空间统计接近真实高帧率 RGB，不能写成“VFI 完全没有伪影”。

## 4. CUDA 性能与正确性

### 4.1 性能

以下为 A100 warmed compute-only 结果，不包含 HDF5 写盘：

| 测试 | 结果 |
| --- | ---: |
| Custom DVS，256x256、32 帧 | `80.0 帧/s` |
| 官方 Torch CUDA DVS，256x256、32 帧 | `25.4 帧/s`，Custom `3.1x` |
| Custom DVS，720x720、32 帧 | `9.64 帧/s`，相对官方 `1.68x` |
| RIFE Lite fp16，960x540 | `361 帧/s` |
| RIFE Full fp16，960x540 | `303 帧/s` |
| 完整 40→480 纯计算链路，960x540 | `5.257 s`，等效 `91.3 帧/s` |
| 其中逆补偿阶段 | `0.856 s`，等效 `560.8 帧/s` |
| Project-Lite，31.99M events | CPU `5.809 s` → A100 `0.095 s`，`60.9x` |

### 4.2 统计一致性质量门

DL3DV `03780d41` 的官方 reference 与 Custom CUDA 对比：

| 检查项 | 差异/结果 |
| --- | ---: |
| 事件数相对差 | `9.65e-6` |
| 正极性比例差 | `9.9e-7` |
| 时间直方图 L1 差 | `1.00e-4` |
| 空间 occupancy IoU | `0.99796` |
| 有限值、边界、排序、极性 | 全部通过 |

三个 256x256 合成运动模式（移动方块、正弦运动、随机纹理平移）也全部通过；
最大事件数相对差 `3.34e-5`，空间分布 TV 不超过 `9.05e-4`。Project-Lite
CPU 与 CUDA 的保留索引集合 Jaccard 为 `1.0`（相同整数微秒分桶）。

## 5. 当前边界与待办

以下项目尚未完成，不应在报告中提前宣称：

- `PaperSimplifiedPrismModel` 不是完整玻璃棱镜光学模型，真实装配误差和折射率
  仍需真机标定。
- 没有真实事件相机数据来校准 threshold、噪声、hot pixel、不应期和时间同步偏移。
- Adaptive rotation 目前只有基于外部运动量的规则策略；CLI 尚未自动估计运动量，
  也没有统一的事件预算、覆盖率、延迟和能耗闭环目标。
- 仍需高帧率 RGB/Blender ground truth，以把 VFI 误差与棱镜观测增益完全分离，
  并覆盖低纹理、快速运动、亮度范围和边缘方向等条件。
- 官方 DVS reference 仍沿时间维度递归；Custom CUDA 的显存上限受
  `height * width * max_events_per_pixel` 约束，I/O 也未计入上述吞吐。

后续优先级：先补齐合成真值和棱镜/RPM/时间延迟敏感性，再做真实数据标定，最后
评估自适应旋转。所有新增实验应沿用匹配时间窗口，并同时报告事件数、coverage、
Edge F1、edge distance、flow 和 tracking 指标。

## 6. 复现入口与数据索引

EIRM 是此最小迁移包的实现目录。第三方模型和结果大文件不包含在仓库中；安装方式、
外部组件和迁移边界分别见 `README.md`、`DEPENDENCIES.md` 和 `MIGRATION_SCOPE.md`。

```bash
cd EIRM
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest

PYTHONPATH=. python scripts/benchmark_cuda.py \
  --device cuda --frames 64 --height 720 --width 720 \
  --warp-batch-size 32 --transfer-chunk-frames 16 --dvs-precision float64
```

关键结果资产：

- [8 场景汇总 JSON](../results/dl3dv_8scene_summary.json)
- [8 场景汇总 CSV](../results/dl3dv_8scene_summary.csv)
- [匹配短窗口主表](../results/final_table.csv)
- [全基线下游聚合](../results/downstream_aggregate.json)
- [全基线聚合 CSV](../results/baseline_aggregate.csv)

Project-Lite 代码和其完整结果不包含在本最小迁移包中；本报告仅保留其已验证的
聚合结果与配置说明。完整事件流、图像和过滤后 HDF5 均被有意排除。

正式论文结果建议使用官方 DVS CUDA reference；Custom CUDA 用于性能展示并报告
统计质量门，不要求逐事件 bitwise 一致。所有数字均应注明场景数、时间窗口、
VFI 版本、棱镜参数、DVS 配置和是否包含 I/O。
