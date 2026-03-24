import torch
import numpy as np
from einops import rearrange
import equilib
import math
from typing import List, Dict, Union
from equilib import equi2equi

def equirectangular_to_cubemap(equi, mode="nearest"):
    """
    将等距投影(equirectangular)格式的张量或数组转换为立方体贴图(cubemap)格式。
    同时支持 4D (bsz, C, H, W) 和 5D (bsz, num_frames, C, H, W) 输入。
    同时支持 torch.Tensor 和 numpy.ndarray 类型。

    参数:
        equi (torch.Tensor 或 numpy.ndarray): 
            输入张量或数组，形状为 (bsz, C, H, W) 或 (bsz, num_frames, C, H, W)。
            H 和 W 为等距投影的高度和宽度（通常 W=2*H）。
        mode (str): 插值模式，默认 "nearest"。

    返回:
        torch.Tensor 或 numpy.ndarray: 
            转换后的张量或数组，类型与输入一致。
            如果输入为 4D，输出形状为 (bsz, 6, C, face_w, face_w)。
            如果输入为 5D，输出形状为 (bsz, num_frames, 6, C, face_w, face_w)。
            其中 face_w = H // 2。
    """
    
    # 1. 确定后端 (torch 或 numpy)
    is_torch = isinstance(equi, torch.Tensor)
    backend = torch if is_torch else np
    original_dtype = equi.dtype

    # 辅助函数，用于统一类型转换
    def to_float32(x):
        if is_torch:
            return x.to(torch.float32)
        return x.astype(np.float32)

    def to_original_dtype(x):
        if is_torch:
            return x.to(original_dtype)
        return x.astype(original_dtype)

    # 2. 检查输入维度并统一为 (B, C, H, W) 格式
    if equi.ndim == 5:
        # 5D: (bsz, num_frames, C, H, W)
        is_video = True
        bsz, num_frames, C, equi_h, equi_w = equi.shape
        batch_frames = bsz * num_frames
        
        # 展平 batch 和 time 维度
        equi_input = rearrange(equi, 'b t c h w -> (b t) c h w')
        
    elif equi.ndim == 4:
        # 4D: (bsz, C, H, W)
        is_video = False
        bsz, C, equi_h, equi_w = equi.shape
        batch_frames = bsz
        equi_input = equi # 已经是 (B, C, H, W) 格式
        
    else:
        raise ValueError(f"输入维度必须是 4 (b c h w) 或 5 (b t c h w)，但得到了 {equi.ndim}")

    # 3. 验证 ERP 格式
    if not equi_w == 2 * equi_h:
        print(f"警告: 输入似乎不是标准等距投影 (宽度 W={equi_w} != 2 * 高度 H={equi_h})")
        # 假设 H 是正确的高度
        
    M = 6  # 立方体面数
    face_w = equi_h // 2
    
    # 4. 准备 equilib 参数
    default_rots_list = [{'roll': 0., 'pitch': 0., 'yaw': 0.}] * batch_frames
    
    # 5. 调用 equilib.equi2cube (它同时支持 torch 和 numpy)
    list_of_face_lists = equilib.equi2cube(
        equi=to_float32(equi_input), # 转换为 float32 以便插值
        rots=default_rots_list,
        w_face=face_w,
        cube_format='list',
        mode=mode
    )
    
    # 6. 将所有面堆叠成一个大的张量
    all_faces_tensor_list = []
    for face_list in list_of_face_lists:
        # 每个 face_list 包含 6 个 (C, H, W) 张量
        all_faces_tensor_list.extend(face_list)
        
    # (B*6, C, face_w, face_w)
    output_cube = backend.stack(all_faces_tensor_list, axis=0) 
    output_cube = to_original_dtype(output_cube) # 转换回原始类型
    
    # 7. 恢复为目标形状
    if is_video:
        # (b*t*m, c, h, w) -> (b, t, m, c, h, w)
        output_cube = rearrange(
            output_cube, '(b t m) c h w -> b t m c h w',
            b=bsz, m=M
        )
    else:
        # (b*m, c, h, w) -> (b, m, c, h, w)
        output_cube = rearrange(
            output_cube, '(b m) c h w -> b m c h w',
            b=bsz, m=M
        )
        
    return output_cube


def safe_cube2equi_wrapper(
    cubemap: Union[torch.Tensor, np.ndarray],
    height: int,
    width: int,
    mode: str = 'nearest', # 对于特征图，强烈建议使用 'nearest'
    **kwargs 
) -> Union[torch.Tensor, np.ndarray]:
    """
    一个健壮的 cube2equi 包装器，专门用于 'horizon' 格式的立方体贴图。
    
    - 它可以处理非8倍数的宽高。
    - 它通过先 padding 到8的倍数调用 cube2equi，然后再用 equi2equi 精确缩放回目标尺寸。
    - 支持 torch.Tensor 和 numpy.ndarray。
    - 支持高维输入 (如 5D/4D/3D)，自动展平和恢复。
    
    参数:
        cubemap (torch.Tensor 或 np.ndarray): 'horizon' 格式的立方体贴图输入。
            - 5D: (B, T, C, F, 6*F) - 视频
            - 4D: (B, C, F, 6*F) - 批处理
            - 3D: (C, F, 6*F) - 单个
        height: 目标等距投影高度
        width: 目标等距投影宽度
        mode: 插值模式，默认 'nearest'
    
    返回:
        转换后的等距投影张量/数组，维度与输入匹配
        (B, T, C, H, W), (B, C, H, W) 或 (C, H, W)
    """
    
    is_video = False
    input_for_equilib = cubemap
    bsz = 1
    original_dtype = cubemap.dtype

    # 1. 展平高维输入 (B, T, ...)
    if cubemap.ndim == 5: # (B, T, C, F, 6*F)
        is_video = True
        bsz, num_frames = cubemap.shape[:2]
        input_for_equilib = rearrange(cubemap, 'b t c h w -> (b t) c h w')
    elif cubemap.ndim == 4: # (B, C, F, 6*F)
        bsz = cubemap.shape[0]
        input_for_equilib = cubemap
    elif cubemap.ndim == 3: # (C, F, 6*F)
        unbatched_output = True
        input_for_equilib = rearrange(cubemap, 'c h w -> 1 c h w')
    else:
        raise ValueError(f"输入必须是 'horizon' 格式 (3D, 4D 或 5D)，但得到了 {cubemap.ndim}D")
        
    # 2. 检查是否需要处理 (您的原始逻辑)
    # import pdb; pdb.set_trace()
    is_multiple_of_8 = (height % 8 == 0) and (width % 8 == 0)

    if isinstance(input_for_equilib, torch.Tensor):
        input_for_equilib = input_for_equilib.to(torch.float32)
    else: # 假设是 np.ndarray
        input_for_equilib = input_for_equilib.astype(np.float32)
    
    if is_multiple_of_8:
        # 如果尺寸合法，直接调用原始函数
        equi_final = equilib.cube2equi(
            cubemap=input_for_equilib,
            height=height,
            width=width,
            cube_format='horizon', # 硬编码
            mode=mode,
            **kwargs
        )
    
    else:
        # --- 尺寸不合法，执行 "Padding -> Resize" 流程 ---
        
        # 3. 计算最小的合法（8的倍数）填充尺寸
        padded_height = math.ceil(height / 8) * 8
        padded_width = padded_height * 2
        
        # 4. 使用 'padded' 尺寸调用 cube2equi
        # 这是第一次采样
        equi_padded = equilib.cube2equi(
            cubemap=input_for_equilib,
            height=padded_height,
            width=padded_width,
            cube_format='horizon', # 硬编码
            mode=mode,
            **kwargs
        )
        
        # 5. 使用 equi2equi 进行几何准确的缩放
        
        # equi2equi 需要一个 rots 参数。对于缩放，我们使用0旋转。
        # equi_padded 此时的形状是 (C,H,W) 或 (B,C,H,W)
        rots: Union[Dict, List[Dict]]
        if len(equi_padded.shape) == 3:
            # (C, H, W) - 单个特征图 (理论上此时应是批处理过的)
            rots = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        else:
            # (B, C, H, W) - 批处理
            batch_size = equi_padded.shape[0]
            rots = [{"roll": 0.0, "pitch": 0.0, "yaw": 0.0}] * batch_size

        # 这是第二次采样，使用相同的模式 (例如 'nearest')
        equi_final = equilib.equi2equi(
            src=equi_padded,
            rots=rots,
            height=height,   # 目标高度
            width=width,     # 目标宽度
            mode=mode,
            **kwargs
        )
    
    # 6. 恢复原始维度 (B, T, ...)
    # --- 核心修复：补回可能被 equilib 自动 squeeze 掉的 batch 维度 ---
    if equi_final.ndim == 3:
        if isinstance(equi_final, torch.Tensor):
            equi_final = equi_final.unsqueeze(0)
        else:
            equi_final = equi_final[np.newaxis, ...]
    output = equi_final
    if is_video:
        output = rearrange(equi_final, '(b t) c h w -> b t c h w', b=bsz)
    # elif unbatched_output:
    #     if output.ndim == 4 and output.shape[0] == 1:
    #         # (1, C, H, W) -> (C, H, W)
    #         output = output[0]
    if isinstance(output, torch.Tensor):
        output = output.to(original_dtype)
    else: # 假设是 np.ndarray
        output = output.astype(original_dtype)
    
    return output


def safe_equi2equi_resize(
    equi_img: Union[torch.Tensor, np.ndarray],
    height: int,
    width: int,
    mode: str = 'nearest', 
    **kwargs 
) -> Union[torch.Tensor, np.ndarray]:
    """
    一个健壮的 equi2equi 包装器，专门用于几何感知的尺寸缩放 (仅支持批处理)。
    
    - 它通过自动处理 0 旋转来简化 equi2equi 的调用。
    - 支持 torch.Tensor 和 numpy.ndarray。
    - 支持 5D (视频) 和 4D (批处理) 输入，自动展平和恢复。
    
    参数:
        equi_img (torch.Tensor 或 np.ndarray): 等距投影图输入。
            - 5D: (B, T, C, H_in, W_in) - 视频
            - 4D: (B, C, H_in, W_in) - 批处理
        height: 目标高度 (H_out)
        width: 目标宽度 (W_out)
        mode: 插值模式，默认 'bilinear'。如果缩放特征图，可使用 'nearest'。
    
    返回:
        缩放后的等距投影张量/数组，维度与输入匹配
        (B, T, C, H_out, W_out) 或 (B, C, H_out, W_out)
    """
    
    is_video = False
    input_for_equilib = equi_img
    bsz = 1
    original_dtype = equi_img.dtype

    # 1. 展平高维输入 (B, T, ...)
    if equi_img.ndim == 5: # (B, T, C, H, W)
        is_video = True
        bsz, num_frames = equi_img.shape[:2]
        # 展平为 (B*T, C, H, W)
        input_for_equilib = rearrange(equi_img, 'b t c h w -> (b t) c h w')
    elif equi_img.ndim == 4: # (B, C, H, W)
        bsz = equi_img.shape[0]
        input_for_equilib = equi_img
    else:
        # 移除了 3D 支持
        raise ValueError(f"输入必须是 4D (B,C,H,W) 或 5D (B,T,C,H,W)，但得到了 {equi_img.ndim}D")
        
    # 2. 转换 Dtype 为 float32 以便 equilib 处理
    if isinstance(input_for_equilib, torch.Tensor):
        input_for_equilib = input_for_equilib.to(torch.float32)
    else: # 假设是 np.ndarray
        input_for_equilib = input_for_equilib.astype(np.float32)
    
    # 3. 准备 rots 参数 (0 旋转)
    # input_for_equilib 此时一定是 4D: (B*T 或 B, C, H, W)
    batch_size = input_for_equilib.shape[0]
    rots = [{"roll": 0.0, "pitch": 0.0, "yaw": 0.0}] * batch_size

    # 4. 调用 equi2equi 进行几何感知的缩放
    equi_final = equilib.equi2equi(
        src=input_for_equilib,
        rots=rots,
        height=height,   # 目标高度
        width=width,     # 目标宽度
        mode=mode,
        **kwargs
    )
    
    # 5. 恢复原始维度 (B, T, ...)
    output = equi_final
    if is_video:
        # 从 (B*T, C, H_out, W_out) 恢复为 (B, T, C, H_out, W_out)
        output = rearrange(equi_final, '(b t) c h w -> b t c h w', b=bsz)
    # 如果输入是 4D，equi_final (即 output) 已经是 (B, C, H_out, W_out)，无需操作
            
    # 6. 恢复原始 Dtype
    if isinstance(output, torch.Tensor):
        output = output.to(original_dtype)
    else: # 假设是 np.ndarray
        output = output.astype(original_dtype)
    
    return output

def rotate_equirect(
    img,
    yaw_deg=0.0,
    pitch_deg=0.0,
    roll_deg=0.0,
    mode="bilinear",
):
    """
    Rotate an equirectangular image using equilib.

    Args:
        img: np.ndarray or torch.Tensor
            shape:
                numpy: (H,W,C) or (C,H,W)
                torch: (C,H,W) or (B,C,H,W)
        yaw_deg, pitch_deg, roll_deg: rotation in degrees
        mode: interpolation mode ("bilinear" recommended)

    Returns:
        rotated image with same type/shape as input
    """

    is_torch = isinstance(img, torch.Tensor)
    device = img.device if is_torch else None

    # ---------- 转 numpy ----------
    if is_torch:
        img_np = img.detach().cpu().numpy()
    else:
        img_np = img

    # ---------- 统一 shape ----------
    added_batch = False

    if img_np.ndim == 3:
        # (H,W,C) → (C,H,W)
        if img_np.shape[-1] == 3:
            img_np = np.transpose(img_np, (2, 0, 1))
        img_np = img_np[None]  # → (1,C,H,W)
        added_batch = True

    elif img_np.ndim == 4:
        # (B,H,W,C) → (B,C,H,W)
        if img_np.shape[-1] == 3:
            img_np = np.transpose(img_np, (0, 3, 1, 2))

    else:
        raise ValueError("Unsupported input shape")

    # ---------- 构造 rotation ----------
    rot = {
        "roll": np.deg2rad(roll_deg),
        "pitch": np.deg2rad(pitch_deg),
        "yaw": np.deg2rad(yaw_deg),
    }
    rots = [rot] * img_np.shape[0]

    # ---------- 执行旋转 ----------
    out = equi2equi(
        src=img_np,
        rots=rots,
        mode=mode,
    )

    # ---------- 恢复 shape ----------
    if added_batch:
        out = out[0]

    # (C,H,W) → (H,W,C)（如果原来是这样）
    if not is_torch and img.ndim == 3 and img.shape[-1] == 3:
        out = np.transpose(out, (1, 2, 0))

    if not is_torch and img.ndim == 4 and img.shape[-1] == 3:
        out = np.transpose(out, (0, 2, 3, 1))

    # ---------- 转回 torch ----------
    if is_torch:
        out = torch.from_numpy(out).to(device)

    return out



# --- 示例用法 ---
if __name__ == '__main__':
    
    # 1. 测试 5D Torch 输入
    print("测试 5D Torch...")
    bsz, T, C, H, W = 2, 3, 4, 256, 512
    face_w = H // 2
    dummy_5d_torch = torch.randn(bsz, T, C, H, W, dtype=torch.float16)
    cubemap_5d_torch = equirectangular_to_cubemap(dummy_5d_torch)
    print(f"输入形状: {dummy_5d_torch.shape}, 类型: {dummy_5d_torch.dtype}")
    print(f"输出形状: {cubemap_5d_torch.shape}, 类型: {cubemap_5d_torch.dtype}")
    assert cubemap_5d_torch.shape == (bsz, T, 6, C, face_w, face_w)
    assert cubemap_5d_torch.dtype == torch.float16
    print("5D Torch 测试通过!\n")

    # 2. 测试 4D Torch 输入
    print("测试 4D Torch...")
    bsz, C, H, W = 2, 4, 256, 512
    face_w = H // 2
    dummy_4d_torch = torch.randn(bsz, C, H, W)
    cubemap_4d_torch = equirectangular_to_cubemap(dummy_4d_torch)
    print(f"输入形状: {dummy_4d_torch.shape}, 类型: {dummy_4d_torch.dtype}")
    print(f"输出形状: {cubemap_4d_torch.shape}, 类型: {cubemap_4d_torch.dtype}")
    assert cubemap_4d_torch.shape == (bsz, 6, C, face_w, face_w)
    assert cubemap_4d_torch.dtype == torch.float32 # 默认为 float32
    print("4D Torch 测试通过!\n")

    # 3. 测试 5D Numpy 输入
    print("测试 5D Numpy...")
    bsz, T, C, H, W = 2, 3, 4, 256, 512
    face_w = H // 2
    dummy_5d_numpy = np.random.rand(bsz, T, C, H, W).astype(np.float64)
    cubemap_5d_numpy = equirectangular_to_cubemap(dummy_5d_numpy)
    print(f"输入形状: {dummy_5d_numpy.shape}, 类型: {dummy_5d_numpy.dtype}")
    print(f"输出形状: {cubemap_5d_numpy.shape}, 类型: {cubemap_5d_numpy.dtype}")
    assert cubemap_5d_numpy.shape == (bsz, T, 6, C, face_w, face_w)
    assert cubemap_5d_numpy.dtype == np.float64
    print("5D Numpy 测试通过!\n")

    # 4. 测试 4D Numpy 输入
    print("测试 4D Numpy...")
    bsz, C, H, W = 2, 4, 256, 512
    face_w = H // 2
    dummy_4d_numpy = np.random.rand(bsz, C, H, W).astype(np.float16)
    cubemap_4d_numpy = equirectangular_to_cubemap(dummy_4d_numpy)
    print(f"输入形状: {dummy_4d_numpy.shape}, 类型: {dummy_4d_numpy.dtype}")
    print(f"输出形状: {cubemap_4d_numpy.shape}, 类型: {cubemap_4d_numpy.dtype}")
    assert cubemap_4d_numpy.shape == (bsz, 6, C, face_w, face_w)
    assert cubemap_4d_numpy.dtype == np.float16
    print("4D Numpy 测试通过!\n")
    
    # 5. 测试错误维度
    print("测试错误维度...")
    try:
        dummy_3d = torch.randn(2, 2, 2)
        equirectangular_to_cubemap(dummy_3d)
    except ValueError as e:
        print(f"捕获到预期错误: {e}")
    print("错误维度测试通过!\n")



    H, W = 250, 500 # 非8倍数
    F = 128
    C = 4
    bsz = 2
    T = 3
    
    # 1. 测试 4D Torch 'horizon' (B, C, F, 6*F)
    print("测试 4D Torch 'horizon' (Batch)...")
    dummy_4d_horizon = torch.randn(bsz, C, F, 6 * F)
    equi_4d_h = safe_cube2equi_wrapper(dummy_4d_horizon, H, W)
    print(f"输入形状: {dummy_4d_horizon.shape}")
    print(f"输出形状: {equi_4d_h.shape}")
    assert equi_4d_h.shape == (bsz, C, H, W)
    print("4D 'horizon' (Batch) 测试通过!\n")

    # 2. 测试 5D Torch 'horizon' (B, T, C, F, 6*F)
    print("测试 5D Torch 'horizon' (Video)...")
    dummy_5d_horizon = torch.randn(bsz, T, C, F, 6 * F)
    equi_5d_h = safe_cube2equi_wrapper(dummy_5d_horizon, H, W)
    print(f"输入形状: {dummy_5d_horizon.shape}")
    print(f"输出形状: {equi_5d_h.shape}")
    assert equi_5d_h.shape == (bsz, T, C, H, W)
    print("5D 'horizon' (Video) 测试通过!\n")

    # 3. 测试 3D Numpy 'horizon' (C, F, 6*F)
    print("测试 3D Numpy 'horizon' (Single)...")
    dummy_3d_numpy = np.random.rand(C, F, 6 * F).astype(np.float32)
    equi_3d_np = safe_cube2equi_wrapper(dummy_3d_numpy, H, W)
    print(f"输入形状: {dummy_3d_numpy.shape}")
    print(f"输出形状: {equi_3d_np.shape}")
    assert equi_3d_np.shape == (C, H, W)
    print("3D 'horizon' (Single) 测试通过!\n")
    
    # 4. 测试合法尺寸 (直接调用)
    print("测试合法尺寸 (H=256, W=512)...")
    H_legal, W_legal = 256, 512
    dummy_4d_horizon = torch.randn(bsz, C, F, 6 * F)
    equi_4d_legal = safe_cube2equi_wrapper(dummy_4d_horizon, H_legal, W_legal)
    print(f"输入形状: {dummy_4d_horizon.shape}")
    print(f"输出形状: {equi_4d_legal.shape}")
    assert equi_4d_legal.shape == (bsz, C, H_legal, W_legal)
    print("合法尺寸测试通过!\n")

    import torchvision.transforms.functional as TF
    import os
    from PIL import Image

    # 请确保此路径正确
    IMAGE_PATH = "/workspace1/songcx/dataset/pvdepth/town01/town01_path1_rz_4.0_fov120/panorama_rgb/007970.png"
    OUTPUT_PATH = "reconstructed_panorama.png"
    
    # 为 'nearest'，以匹配您的 wrapper 默认值
    # 如果是RGB图像用于可视化，也可以使用 'bilinear'
    INTERPOLATION_MODE = 'bilinear' 

    # 检查图像文件是否存在
    if not os.path.exists(IMAGE_PATH):
        print(f"错误: 找不到输入图像文件: {IMAGE_PATH}")
        print("请检查路径是否正确。")
    else:
        print(f"--- 开始验证流程 ---")
        
        # --- 2. 加载全景图 (Equirectangular) ---
        print(f"步骤 1: 加载全景图: {IMAGE_PATH}")
        pil_img = Image.open(IMAGE_PATH).convert('RGB')
        original_width, original_height = pil_img.size
        
        # 转换为 (C, H, W) 格式的 Torch 张量，数值范围 [0.0, 1.0]
        equi_tensor_in = TF.to_tensor(pil_img)
        C, H, W = equi_tensor_in.shape
        print(f"加载成功。 形状: ({C}, {H}, {W}), 类型: {equi_tensor_in.dtype}")
        
        
        # --- 3. 全景图 -> 立方体贴图 (Cubemap) ---
        print(f"\n步骤 2: 转换 Equirectangular -> Cubemap (格式: 'list')")
        # equirectangular_to_cubemap 接受 (C,H,W), (B,C,H,W) 或 (B,T,C,H,W)
        # 输入 (C, H, W)，将输出 (1, 6, C, F, F)
        cubemap_tensor = equirectangular_to_cubemap(
            equi_tensor_in.unsqueeze(0), 
            mode=INTERPOLATION_MODE
        )
        face_w = H // 2
        print(f"转换成功。 立方体贴图形状: {cubemap_tensor.shape}")
        assert cubemap_tensor.shape == (1, 6, C, face_w, face_w)

        # --- 4. 立方体贴图 -> Horizon 格式 ---
        print(f"\n步骤 3: Rearrange Cubemap -> 'Horizon' 格式")
        # 输入: (B, M, C, H_face, W_face) -> (1, 6, C, F, F)
        # 目标: (B, C, H_face, M*W_face) -> (1, C, F, 6*F)
        horizon_tensor = rearrange(
            cubemap_tensor, 
            'b m c h w -> b c h (m w)', 
            m=6
        )
        print(f"Rearrange 成功。 Horizon 形状: {horizon_tensor.shape}")
        assert horizon_tensor.shape == (1, C, face_w, 6 * face_w)

        # --- 5. Horizon 格式 -> 全景图 ---
        print(f"\n步骤 4: 转换 'Horizon' -> Equirectangular (使用 wrapper)")
        # 目标尺寸为原始图像尺寸
        reconstructed_equi_tensor = safe_cube2equi_wrapper(
            cubemap=horizon_tensor,
            height=original_height,
            width=original_width,
            mode=INTERPOLATION_MODE
        )
        print(f"重建成功。 重建全景图形状: {reconstructed_equi_tensor.shape}")
        assert reconstructed_equi_tensor.shape == (C, H, W)

        # --- 6. 保存重建的图像 ---
        print(f"\n步骤 5: 保存重建图像到: {OUTPUT_PATH}")
        
        # 移除 batch 维度: (1, C, H, W) -> (C, H, W)
        reconstructed_img_squeezed = reconstructed_equi_tensor.squeeze(0)
        
        # 将 (C, H, W) 格式的张量 (范围 [0, 1]) 转换回 PIL Image (范围 [0, 255])
        pil_img_out = TF.to_pil_image(reconstructed_img_squeezed)
        
        pil_img_out.save(OUTPUT_PATH)
        print(f"--- 验证流程完毕 ---")
