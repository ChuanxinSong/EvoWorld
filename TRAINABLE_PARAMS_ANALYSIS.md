# EvoWorld 可训练参数分析报告

## 概述
根据代码分析，EvoWorld 使用了**参数高效微调 (Parameter-Efficient Fine-Tuning)** 策略，只训练模型的一小部分关键参数，而冻结大部分预训练权重。

## 训练的参数类型

### 1. **Temporal Transformer Blocks** (时序变换模块)
- **作用**: 处理视频帧之间的时序关系，这是视频生成的核心
- **关键字**: `temporal_transformer_block`
- **重要性**: ⭐⭐⭐⭐⭐ (最重要)
- **说明**: 这些模块负责学习相机运动和场景变化的时序一致性

### 2. **Conv_in** (输入卷积层)
- **作用**: 处理输入特征，将输入映射到UNet的特征空间
- **关键字**: `conv_in`
- **重要性**: ⭐⭐⭐⭐
- **说明**: 由于添加了 Plucker 坐标嵌入，需要训练此层以适应新的输入维度

### 3. **Conv_out** (输出卷积层)
- **作用**: 生成最终的去噪结果
- **关键字**: `conv_out`
- **重要性**: ⭐⭐⭐⭐
- **说明**: 输出层的训练可以更好地适应全景图像生成任务

### 4. **Normalization Layers** (归一化层)
- **作用**: 稳定训练过程，加速收敛
- **关键字**: `norm` 或 `Norm`
- **重要性**: ⭐⭐⭐
- **说明**: 包括 LayerNorm, GroupNorm 等各种归一化层

## 冻结的参数

### 1. **VAE (变分自编码器)**
- 用于图像的编码和解码
- 保持预训练权重不变

### 2. **Image Encoder (CLIP 图像编码器)**
- 用于提取图像特征作为条件
- 保持预训练权重不变

### 3. **UNet 的空间注意力模块**
- 处理单帧内的空间特征
- 保持预训练知识

### 4. **UNet 的 MLP 层**
- 特征变换层
- 保持预训练知识

## 代码位置

### 主要训练代码
**文件**: `evoworld/trainer/train_evoworld.py`
**关键代码段** (第 302-314 行):

```python
parameters_list = []

# Customize the parameters that need to be trained;
for name, param in unet.named_parameters():
    if (
        "temporal_transformer_block" in name
        or "conv_in" in name
        or "conv_out" in name
        or "norm" in name
        or "Norm" in name
    ):
        parameters_list.append(param)
        param.requires_grad = True
    else:
        param.requires_grad = False
logger.info(f"Trainable params num: {len(parameters_list)}")
```

### 模型定义
**文件**: `evoworld/trainer/unet_plucker.py`
- 定义了 `UNetSpatioTemporalConditionModel` 类
- 这是一个时空条件UNet，专门用于视频生成

### 训练配置
**文件**: `train.sh`
- 配置所有训练超参数
- 学习率: `1e-5`
- 优化器: AdamW
- 混合精度训练: `bf16`

### 冻结模型代码
**文件**: `evoworld/trainer/train_evoworld.py` (第 204-207 行):

```python
# Freeze vae and image_encoder
vae.requires_grad_(False)
image_encoder.requires_grad_(False)
unet.requires_grad_(False)  # 先全部冻结，然后选择性解冻
```

## 训练策略分析

### 为什么这样设计？

1. **保留预训练知识**
   - Stable Video Diffusion 已经在大规模视频数据上预训练
   - 空间特征提取能力无需重新学习

2. **专注时序建模**
   - 相机轨迹视频生成的核心是时序一致性
   - 集中训练时序相关的 temporal_transformer_block

3. **适应新输入**
   - Plucker 坐标嵌入是新加入的特征
   - 需要训练 conv_in 来处理这些新特征

4. **资源高效**
   - 只训练 <10% 的参数
   - 大幅减少显存需求和训练时间
   - 降低过拟合风险

### 参数量估计

基于 Stable Video Diffusion 的架构：
- **总参数**: 约 1.5B (15亿)
- **可训练参数**: 约 50-150M (5000万-1.5亿)
- **可训练比例**: 约 3-10%

## 相关配置文件

### Accelerate 配置
- `config/deepspeed_o1_4gpu.yaml` - DeepSpeed 优化配置
- `config/accelerate_config.yaml` - Accelerate 分布式训练配置

### 数据集配置
- `dataset/CameraTrajDataset.py` - 相机轨迹数据集实现
- 包含 Plucker 坐标的计算和处理

## 如何验证可训练参数

### 方法 1: 查看训练日志
训练开始时会打印：
```
Trainable params num: XXX
```

### 方法 2: 运行分析脚本
```bash
python analyze_trainable_params.py
```
这个脚本会详细列出所有可训练参数的名称和数量。

### 方法 3: 使用 PyTorch 检查
```python
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params}, Total: {total_params}")
```

## 训练命令

从 `train.sh` 脚本中，实际的训练命令是：

```bash
accelerate launch --config_file="config/deepspeed_o1_4gpu.yaml" \
    --num_processes=1 \
    --gpu_ids=4 \
    evoworld/trainer/train_evoworld.py \
    --add_plucker \
    --learning_rate=1e-5 \
    --gradient_checkpointing \
    ... (其他参数)
```

关键标志：
- `--add_plucker`: 启用 Plucker 坐标嵌入
- `--gradient_checkpointing`: 启用梯度检查点以节省显存

## 总结

EvoWorld 的训练策略非常明确：
1. ✅ **训练**: temporal_transformer_block, conv_in, conv_out, norm层
2. ❌ **冻结**: VAE, Image Encoder, UNet的空间注意力和MLP
3. 🎯 **目标**: 学习相机轨迹的时序一致性，同时保留预训练的图像生成能力
4. 📊 **效率**: 仅训练约 5-10% 的参数，实现参数高效微调

这种设计既保证了训练效率，又能充分利用预训练模型的能力，是一个很好的实践案例。
