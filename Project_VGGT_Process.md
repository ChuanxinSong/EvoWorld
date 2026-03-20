# EvoWorld：VGGT 输出与 GT 位姿对齐并重投影流程说明

## 1. 目标与结论

本文记录 EvoWorld 当前“重投影数据生成”流程：  
1. 先把全景图转换为透视图并生成 `look_at_center` 相机位姿；  
2. 用 VGGT 对透视图序列做几何重建（深度反投影点云 + 预测相机）；  
3. 用 GT 相机轨迹与 VGGT 预测轨迹估计相似变换（尺度+旋转+平移）；  
4. 将 GT 的目标渲染位姿映射到 VGGT 坐标系；  
5. 用 Open3D 渲染 cubemap 并转换成全景图，输出固定 24 帧重投影结果。

---

## 2. 流程入口与脚本链路

### 2.1 顶层启动脚本（单卡）

文件：`scripts/reprojection/launch_reprojection_for_train.sh`  
关键点：
- 固定 GPU：`export CUDA_VISIBLE_DEVICES=4`（`scripts/reprojection/launch_reprojection_for_train.sh:2`）
- 调用下一级脚本：`SCRIPT_PATH="scripts/reprojection/reproject_vggt_open3d_for_train.sh"`（`scripts/reprojection/launch_reprojection_for_train.sh:6`）
- 传入数据目录、chunk 信息（`scripts/reprojection/launch_reprojection_for_train.sh:9`, `scripts/reprojection/launch_reprojection_for_train.sh:10`, `scripts/reprojection/launch_reprojection_for_train.sh:12`, `scripts/reprojection/launch_reprojection_for_train.sh:14`）

### 2.2 重投影执行脚本

文件：`scripts/reprojection/reproject_vggt_open3d_for_train.sh`  
关键配置：
- 输入图像目录：`SUB=perspective_look_at_center`（`scripts/reprojection/reproject_vggt_open3d_for_train.sh:8`）
- 相机文件：`CAM=camera_poses_look_at_center.txt`（`scripts/reprojection/reproject_vggt_open3d_for_train.sh:9`）
- 输出目录：`OUTPUT_SUBDIR=rendered_panorama_vggt_open3d_camera_aligned_new_code`（`scripts/reprojection/reproject_vggt_open3d_for_train.sh:10`）
- 调用主程序：`python -m evoworld.reprojection.reproject_vggt_open3d`（`scripts/reprojection/reproject_vggt_open3d_for_train.sh:12`）
- 关键参数：
  - `--prediction_mode depth_unproject`（`scripts/reprojection/reproject_vggt_open3d_for_train.sh:16`）
  - `--only_render_last_24_frame`（`scripts/reprojection/reproject_vggt_open3d_for_train.sh:21`）
  - `--no_mask_sky`（`scripts/reprojection/reproject_vggt_open3d_for_train.sh:15`）

---

## 3. 前置数据准备：全景到透视 + look_at_center 位姿

### 3.1 调用入口

文件：`scripts/reprojection/pano_to_pers_for_train.sh`  
- 逐 episode 调用 `python -m evoworld.reprojection.pano_to_pers`（`scripts/reprojection/pano_to_pers_for_train.sh:22`）
- 生成 `camera_poses_look_at_center.txt`（`scripts/reprojection/pano_to_pers_for_train.sh:19`）

### 3.2 透视转换与位姿构造逻辑

文件：`evoworld/reprojection/pano_to_pers.py`

关键步骤：
1. 读取 `camera_poses.txt` 并做 Unity -> OpenCV(RDF) 轴系转换  
   - `UNITY_TO_OPENCV = [1, -1, 1, -1, 1, -1]`（`evoworld/reprojection/pano_to_pers.py:10`）
   - 转换入口：`read_camera_file_and_convert_to_rdf`（`evoworld/reprojection/pano_to_pers.py:49`, `evoworld/reprojection/pano_to_pers.py:59`）
2. 使用 `Equi2Pers` 生成透视图序列（输出 `frame_XXX.png`）  
   - 输出写入：`evoworld/reprojection/pano_to_pers.py:123`
3. 重新计算 yaw（使视角更符合 look-at-center 设定）  
   - 常规段：`calculate_target_yaw`（`evoworld/reprojection/pano_to_pers.py:79`）
   - 最后段：`calculate_target_yaw_last_segment`（`evoworld/reprojection/pano_to_pers.py:71`）
4. 用更新后的 yaw 回写 `camera_poses_look_at_center.txt`  
   - `camera_params[:, 4] = target_yaw`（`evoworld/reprojection/pano_to_pers.py:209`）
   - `write_camera_file`（`evoworld/reprojection/pano_to_pers.py:63`, `evoworld/reprojection/pano_to_pers.py:212`）

---

## 4. 主程序：VGGT 推理 + 对齐 + 重投影

文件：`evoworld/reprojection/reproject_vggt_open3d.py`  
工具函数文件：`evoworld/reprojection/reproject_vggt_open3d_utils.py`

### 4.1 数据分片与样本选择

- chunk 切分：`get_target_directories`（`evoworld/reprojection/reproject_vggt_open3d.py:177`）
- 切分计算：`chunk_size/start/end`（`evoworld/reprojection/reproject_vggt_open3d.py:191`, `evoworld/reprojection/reproject_vggt_open3d.py:192`, `evoworld/reprojection/reproject_vggt_open3d.py:193`）
- 若输出目录已存在且 png 数量等于目标视角数（默认24）则跳过：`should_skip_processing`（`evoworld/reprojection/reproject_vggt_open3d.py:203`）

### 4.2 VGGT 推理与深度反投影

`VGGTProcessor.run_inference`（`evoworld/reprojection/reproject_vggt_open3d.py:57`）：

1. 读取透视图序列  
   - 图像读取：`image_names = glob.glob(...)`（`evoworld/reprojection/reproject_vggt_open3d.py:72`）
   - 若 `only_render_last_24_frame=True`，推理时去掉最后 24 帧：`image_names = image_names[:-24]`（`evoworld/reprojection/reproject_vggt_open3d.py:76`）
2. VGGT 前向推理，得到 `pose_enc/depth/...`
3. `pose_enc -> extrinsic/intrinsic`  
   - `pose_encoding_to_extri_intri`（`evoworld/reprojection/reproject_vggt_open3d.py:93`）
4. 用深度 + 相机参数反投影得到 3D 点  
   - `unproject_depth_map_to_point_map`（`evoworld/reprojection/reproject_vggt_open3d.py:107`）
   - 保存到 `predictions["world_points_from_depth"]`

### 4.3 GT 位姿读取方式

- 读取 `camera_poses_look_at_center.txt`：`load_camera_poses`（`evoworld/reprojection/reproject_vggt_open3d.py:213`）
- 文本行格式为：`frame_id x y z rx ry rz`
- 调用 `xyz_euler_to_four_by_four_matrix_batch(..., relative=True)` 转为 4x4（`evoworld/reprojection/reproject_vggt_open3d.py:220`）
- 该函数定义在：`utils/geometry.py:5`，`relative=True` 表示以首帧为相对坐标系

---

## 5. 对齐核心：如何把 GT 位姿映射到 VGGT 坐标系

### 5.1 对齐函数入口

文件：`evoworld/reprojection/reproject_vggt_open3d_utils.py`  
函数：`SceneBuilder.align_extrinsics`（`evoworld/reprojection/reproject_vggt_open3d_utils.py:472`）

### 5.2 时间切片策略（关键）

在 `only_render_last_24_frame=True` 时：
- `target_start_id = -num_target_view`（`evoworld/reprojection/reproject_vggt_open3d_utils.py:494`）
- 用 `camera_pose[:target_start_id]` 作为对齐参考轨迹（`evoworld/reprojection/reproject_vggt_open3d_utils.py:496`）
- 用 `camera_pose[target_start_id:]` 作为目标渲染 GT 位姿（`evoworld/reprojection/reproject_vggt_open3d_utils.py:500`）

这意味着：  
- 前 N-24 帧只用于估计对齐变换；  
- 最后 24 帧用于最终渲染。

### 5.3 变换估计方式

`align_extrinsics` 内部调用：
- `align_first_and_last_points(A, B)`（`evoworld/reprojection/reproject_vggt_open3d_utils.py:504`, 定义在 `evoworld/reprojection/reproject_vggt_open3d_utils.py:1176`）

其中：
- `A` 为 GT 参考段相机中心轨迹；
- `B` 为 VGGT 预测参考段相机中心轨迹（由预测外参求逆得到相机中心）；
- 估计相似变换参数 `(s, R, t)`，并组装：
  - `transform_mat[:3, :3] = s * R`（`evoworld/reprojection/reproject_vggt_open3d_utils.py:509`）
  - `transform_mat[:3, 3] = t`
- 对 GT 目标段位姿左乘该变换得到 `new_target_extrinsic`（`evoworld/reprojection/reproject_vggt_open3d_utils.py:515`, `evoworld/reprojection/reproject_vggt_open3d_utils.py:516`）

> 注意：当前算法使用“首尾两点”估计，不是全轨迹最小二乘，因此对参考段质量较敏感。

---

## 6. 重投影渲染：点云 -> cubemap -> 全景

### 6.1 选点与建场景

- 点云筛选入口：`PointCloudProcessor.filter_predictions`（`evoworld/reprojection/reproject_vggt_open3d_utils.py:174`）
- 在 `prediction_mode=depth_unproject` 下，使用 `world_points_from_depth`（`evoworld/reprojection/reproject_vggt_open3d_utils.py:241`）
- Open3D 离屏场景构建：`build_open3d_scene`（`evoworld/reprojection/reproject_vggt_open3d_utils.py:457`）

### 6.2 目标位姿渲染

- 主渲染入口：`predictions_to_target_view`（`evoworld/reprojection/reproject_vggt_open3d_utils.py:1216`）
- 对齐后位姿来自 `align_extrinsics`（`evoworld/reprojection/reproject_vggt_open3d_utils.py:472`）
- 渲染函数：`render_cubemaps_to_panoramas`（`evoworld/reprojection/reproject_vggt_open3d_utils.py:668`）
  - 每个目标位姿渲染六个面
  - 调 `cube_to_equirectangular_cuda` 合成全景（`evoworld/reprojection/reproject_vggt_open3d_utils.py:705`）
  - 保存 `00.png ~ 23.png`（`evoworld/reprojection/reproject_vggt_open3d_utils.py:708`）

---

## 7. 主程序执行顺序（端到端）

文件：`evoworld/reprojection/reproject_vggt_open3d.py`

1. 解析参数：`parse_arguments`
2. `args.mask_sky = not args.no_mask_sky`（`evoworld/reprojection/reproject_vggt_open3d.py:298`）
3. 获取当前 chunk 的 episode 列表：`get_target_directories`
4. 初始化 VGGT：`VGGTProcessor()`
5. 对每个 episode：
   - 若输出已有 24 张图则跳过：`should_skip_processing`
   - 读 GT 位姿：`load_camera_poses`
   - `perform_reconstruction`：
     - 跑 VGGT 推理（去掉最后24帧）
     - 调 `predictions_to_target_view` 对齐并渲染最后24帧目标视角

---

## 8. 输入输出约定

### 8.1 每个 episode 关键输入

- 透视图：`perspective_look_at_center/*.png`
- GT 位姿：`camera_poses_look_at_center.txt`

### 8.2 每个 episode 输出

- 重投影全景：`rendered_panorama_vggt_open3d_camera_aligned_new_code/00.png ... 23.png`

---

## 9. 为什么这套方法有效

- VGGT 重建出的点云在“VGGT自身坐标系”中；
- 训练期又希望使用“GT 未来轨迹视角”来渲染 memory；
- 因此必须先把 GT 轨迹映射到 VGGT 坐标系，再做渲染；
- 该实现通过 `(s,R,t)` 实现了跨坐标系对齐，保证目标位姿与点云处于同一坐标系后再投影。

---

## 10. 当前实现的已知注意点

1. 对齐使用 `align_first_and_last_points`（首尾点法），对异常轨迹鲁棒性一般。  
2. `align_extrinsics` 中 `segment_id` 依赖 `outdir` 名字末尾下划线字段，解析失败时默认 `segment_id=1`，在 `only_render_last_24_frame=True` 场景影响较小，但仍建议留意。  
3. `only_render_last_24_frame=True` 的语义是“VGGT只看前 N-24 帧、渲染最后24帧”，不要误解成“只渲染输入的最后24帧图像”。

---

## 11. 相关文件清单

- `scripts/reprojection/launch_reprojection_for_train.sh`
- `scripts/reprojection/reproject_vggt_open3d_for_train.sh`
- `scripts/reprojection/pano_to_pers_for_train.sh`
- `evoworld/reprojection/pano_to_pers.py`
- `evoworld/reprojection/reproject_vggt_open3d.py`
- `evoworld/reprojection/reproject_vggt_open3d_utils.py`
- `utils/geometry.py`
- `PIPELINE_LOGIC.md`