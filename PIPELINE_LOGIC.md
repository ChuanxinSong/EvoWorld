# EvoWorld 数据流与推理逻辑说明

## 1. 项目概览

EvoWorld 是基于 **Stable Video Diffusion (SVD)** 的全景视频生成系统，支持相机轨迹条件生成与自回归长视频推理。

- **Base Model**: `stabilityai/stable-video-diffusion-img2vid-xt-1-1`
- **训练框架**: HuggingFace Accelerate + DeepSpeed ZeRO Stage 1, fp16 混合精度
- **核心特性**: 以重投影（reprojection）帧作为几何记忆（memory），指导未来帧生成

---

## 2. 数据集结构

每个 episode 包含以下内容：

```
episode_XXXX/
├── panorama/
│   ├── 001.png   ← 第一帧（原始全景图）
│   ├── 002.png
│   └── ...       ← 约 100 帧
├── rendered_panorama_vggt_open3d_camera_aligned_new_code/
│   ├── 00.png    ← 重投影帧（固定 24 帧）
│   ├── 01.png
│   └── ...
│   └── 23.png
└── camera_poses.txt  ← 每帧 6-DOF 相机位姿
```

---

## 3. 训练数据加载逻辑（`CameraTrajDataset`）

### 3.1 关键参数

| 参数 | 说明 |
|------|------|
| `sequence_length` | 当前片段帧数，必须等于 `num_frames` |
| `last_segment_length` | 最后一段的帧数，必须等于 `num_frames` |
| `num_frames` | 训练时每次输入的帧数（如 10） |

> ⚠️ `sequence_length` 和 `last_segment_length` 必须与 `args.num_frames` 保持一致，否则会导致文件索引越界。

### 3.2 `__getitem__` 数据加载流程

```
episode_length = 100 (举例)
num_frames = 10

valid_range_start_idx = episode_length - last_segment_length + 1 = 91
start_idx ∈ [1, 91]
end_idx = start_idx + sequence_length = start_idx + 10

current_images = panorama/[start_idx].png ... panorama/[end_idx-1].png  (10帧)
```

### 3.3 `load_reprojection()` 逻辑

```python
# 重投影文件夹中最多有 24 张图（00.png ~ 23.png）
# 但实际加载数量被限制为 last_segment_length - 1

reprojection_length = min(文件夹中图片数, last_segment_length - 1)
# → 当 num_frames=10 时，reprojection_length = 9

# 加载重投影帧
images = [repro/00.png, repro/01.png, ..., repro/08.png]  # 9 帧

# 加载第一帧
first_frame = panorama/001.png

# 插入到最前面
images = [panorama/001.png, repro/00.png, ..., repro/08.png]  # 共 10 帧
```

最终返回结构（以 `num_frames=10` 为例）：

| 索引 | 内容 | 来源 |
|------|------|------|
| `[0]` | 第一帧 | `panorama/001.png` |
| `[1]` | 重投影帧 0 | `repro/00.png` |
| `[2]` | 重投影帧 1 | `repro/01.png` |
| ... | ... | ... |
| `[9]` | 重投影帧 8 | `repro/08.png` |

---

## 4. Pipeline 条件构建逻辑（`pipeline_evoworld.py`）

输入 `image` 的形状为 `[B, num_cond, C, H, W]`，其中：
- `image[:, 0]` = 第一帧（`panorama/001.png`）
- `image[:, 1:]` = 重投影帧

### 4.1 VAE 编码

```python
image_latents = vae.encode(image)  # [B, num_cond, 4, h, w]
```

### 4.2 `mask_mem` 控制

```python
if mask_mem:
    # 将所有重投影帧的 latent 置零（不使用几何记忆）
    image_latents[:, 1:] = torch.zeros_like(image_latents[:, 1:])
```

### 4.3 最终 condition 拼接

```python
# 第一帧 latent 重复 num_frames 次
conditional_latents_first_frame = image_latents[:, 0:1].repeat(1, num_frames, 1, 1, 1)

# 拼接：[第一帧重复, 重投影latent(或zeros), plucker相机位姿]
conditional_latents = torch.cat([
    conditional_latents_first_frame,  # 第一帧外观条件（始终有效）
    image_latents[:, 1:],             # 几何记忆（mask_mem=True 时为全零）
    plucker_embedding                 # 相机轨迹条件（始终有效）
], dim=2)
```

### 4.4 两种模式对比

| 模式 | `mask_mem` | 第一帧条件 | 重投影条件 | 相机位姿 |
|------|-----------|-----------|-----------|---------|
| Clip 1（无记忆） | `True` | ✅ 有效 | ❌ 全零 | ✅ 有效 |
| Clip 2+（有记忆）| `False` | ✅ 有效 | ✅ 真实重投影 latent | ✅ 有效 |

---

## 5. 推理自回归生成逻辑（`navigator_evoworld.py`）

### 5.1 轨迹分段

```python
# split_curve_into_segments 将完整轨迹切分为 25 帧滑动窗口，1 帧重叠
segments = split_curve_into_segments(camera_curve, window=25, overlap=1)
```

### 5.2 自回归生成流程

```
Clip 1:
  - current_image = 用户提供的起始帧
  - use_memory = False  →  mask_mem = True（无重投影prior）
  - 生成 25 帧视频

Clip 2:
  - current_image = Clip 1 最后一帧（已生成）
  - use_memory = True   →  mask_mem = False（使用重投影几何）
  - 生成 25 帧视频

Clip 3, 4, ...:
  - 同 Clip 2，持续自回归
```

### 5.3 关键逻辑示意图

```
[起始帧] → Clip 1 (mask_mem=True)  → [帧_25]
                                         ↓
                   Clip 2 (mask_mem=False, reprojection) → [帧_49]
                                                              ↓
                                  Clip 3 (mask_mem=False, reprojection) → ...
```

---

## 6. Validation 推理逻辑（`train_evoworld.py`）

### 6.1 触发条件

```python
if (global_step % args.validation_steps == 0) or (global_step == 1):
    # 每隔 validation_steps 步执行一次，或在第 1 步时执行
```

### 6.2 数据来源

Validation 数据来自 `val_loader`（`val_dataset`），加载逻辑与训练集完全相同：

```python
images = batch["pixel_values"]           # [B, num_frames, C, H, W]
first_frame = images[:, 0, :, :, :]     # 取 clip 的第 0 帧作为生成起点
memorized_pixel_values = batch["memorized_pixel_values"]  # 重投影 memory
```

**第一帧来源**：`pixel_values[:, 0]` = `panorama/{valid_range_start_idx}.png`

以 episode 长度 100、`num_frames=10` 为例：`start_idx = 100 - 10 + 1 = 91`，即 `panorama/091.png`。

> ⚠️ 注意：`first_frame`（clip 实际第一帧）与 `memorized_pixel_values[0]`（`load_reprojection` 中硬编码的 `001.png`）**不是同一张图**，存在不一致。

### 6.3 Pipeline 输入构成

```python
# 相机轨迹转换为 3x4 矩阵（相对于第一帧）
camera_traj[i] = xyz_euler_to_three_by_four_matrix_batch(
    camera_traj_raw[i], relative=True
)  # [num_frames, 3, 4]

# 计算 Plucker embedding（相机射线方向编码）
plucker_embedding[i] = ray_c2w_to_plucker(rays, camera_traj[i])
# [num_frames, 6, H//8, W//8]

# 调用 pipeline 生成视频
video_frames = pipeline(
    first_frame,                          # [B, C, H, W] clip 第一帧
    plucker_embedding=plucker_embedding,  # 相机轨迹条件
    memorized_pixel_values=memorized_pixel_values,  # 重投影 memory
    num_frames=num_frames,
    ...
)
```

### 6.4 输出保存

```
输出路径: {output_dir}/validation_images/step_{global_step}_val_img_{val_step}.mp4
格式: GIF（上半：GT 帧，下半：预测帧 拼接对比）
帧率: 7 fps
```

对比方式：每一帧将 **GT（真实帧）** 和 **预测帧** 垂直拼接，方便直观对比生成质量。

### 6.5 各数据对应关系总结

| 变量 | 实际内容 | 来源 |
|------|---------|------|
| `first_frame` | clip 起始帧，送入 SVD 作为外观条件 | `pixel_values[:, 0]` = `panorama/091.png`（举例） |
| `memorized_pixel_values[0]` | memory 通道的第一帧 | `load_reprojection` 硬编码 `panorama/001.png` |
| `memorized_pixel_values[1:]` | 重投影帧 | `repro/00.png` ~ `repro/08.png` |
| `plucker_embedding` | 相机轨迹编码（相对坐标） | 由 `cam_traj` 计算得到 |

---

## 7. 训练配置（`train.sh`）

```bash
BASE_FOLDER="/data2/songcx/dataset/evoworld/unity_curve"
REPROJ_NAME="rendered_panorama_vggt_open3d_camera_aligned_new_code"
GPU_IDS="5,7"
CONFIG_NAME="deepspeed_o1_4gpu"
MASTER_PORT=49507        # 多进程通信端口
NUM_FRAMES=10            # 每次训练输入帧数
PRETRAIN_MODEL="stabilityai/stable-video-diffusion-img2vid-xt-1-1"
LR="1e-5"
STEP=30000
BATCH_SIZE_PER_GPU=1
GRAD_ACCUM_STEP=8
```

---

## 8. Bug 修复记录

### Bug 1：重投影帧数量与 `num_frames` 不匹配

- **问题**: 重投影文件夹固定有 24 张图，当 `num_frames=10` 时，`load_reprojection` 仍加载 24 帧，导致帧数超出预期。
- **位置**: `dataset/CameraTrajDataset.py` → `load_reprojection()`
- **修复**:
  ```python
  max_reprojection_frames = self.last_segment_length - 1
  reprojection_length = min(reprojection_length, max_reprojection_frames)
  ```

### Bug 2：`sequence_length` 未与 `num_frames` 同步

- **问题**: `CameraTrajDataset.__init__` 中 `sequence_length` 默认为 25，`train_evoworld.py` 未传入该参数。当 `last_segment_length=10` 时：
  ```
  start_idx = episode_length - last_segment_length + 1 = 91
  end_idx = start_idx + sequence_length = 91 + 25 = 116  # 文件不存在！
  ```
- **位置**: `evoworld/trainer/train_evoworld.py`
- **修复**:
  ```python
  train_dataset = CameraTrajDataset(
      ...,
      sequence_length=args.num_frames,  # ← 新增
      last_segment_length=args.num_frames,
  )
  ```
