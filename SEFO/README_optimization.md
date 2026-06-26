# 注意力分数计算优化版本

## 性能优化说明

原始代码存在以下性能瓶颈：
1. **嵌套循环计算**：对每个时间步和每层都进行循环，效率低下
2. **内存管理不当**：没有及时清理GPU内存
3. **生成token过多**：默认生成512个token，但实际可能不需要这么多
4. **缺乏进度显示**：无法知道处理进度
5. **错误处理不足**：单个样本出错会导致整个程序崩溃

## 优化版本对比

### 1. 基础优化版本 (`compute_attention_score_lora_optimized.py`)
主要优化：
- 批量处理注意力计算
- 添加进度条显示
- 优化内存管理
- 添加错误处理
- 支持批处理参数

### 2. 高速版本 (`compute_attention_score_lora_fast.py`)
进一步优化：
- 减少默认生成token数（从512降到128）
- 更高效的注意力计算算法
- 定期保存中间结果
- 详细的性能统计
- 更好的错误恢复机制

## 使用方法

### 基础优化版本
```bash
python compute_attention_score_lora_optimized.py \
    --input_path your_input.json \
    --output_path your_output.json \
    --model_path your_model_path \
    --base_model your_base_model_path \
    --max_new_tokens 256 \
    --verbose
```

### 高速版本
```bash
python compute_attention_score_lora_fast.py \
    --input_path your_input.json \
    --output_path your_output.json \
    --model_path your_model_path \
    --base_model your_base_model_path \
    --max_new_tokens 128 \
    --save_interval 50 \
    --verbose
```

## 参数说明

- `--max_new_tokens`: 最大生成token数（建议128-256）
- `--save_interval`: 每处理多少个样本保存一次中间结果
- `--verbose`: 显示详细输出
- `--batch_size`: 批处理大小（仅基础版本支持）

## 性能提升预期

1. **计算速度**：预计提升2-5倍
   - 减少生成token数：2-4倍提升
   - 优化注意力计算：1.5-2倍提升
   - 内存管理优化：1.2-1.5倍提升

2. **内存使用**：减少30-50%
   - 及时清理GPU内存
   - 更高效的数据结构

3. **稳定性**：显著提升
   - 错误处理机制
   - 定期保存中间结果
   - 进度监控

## 使用建议

1. **首次使用**：建议使用高速版本，设置较小的`max_new_tokens`（如64-128）
2. **大批量数据**：设置合适的`save_interval`，避免数据丢失
3. **调试模式**：使用`--verbose`参数查看详细输出
4. **内存不足**：进一步减少`max_new_tokens`或使用基础版本

## 注意事项

1. 确保安装了`tqdm`库：`pip install tqdm`
2. 如果遇到内存不足，可以进一步减少`max_new_tokens`
3. 中间结果文件会保存在`output_path_checkpoint_X.json`格式
4. 程序会自动处理单个样本的错误，不会中断整个处理过程 