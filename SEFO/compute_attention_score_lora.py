import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import argparse
from peft import PeftModel
import math
import gc

# --- 文件读写函数 (无变化) ---
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="/data/DFO_wang/data/Qwen2.5_7B_only_iteration_DFO_data.json")
parser.add_argument("--output_path", type=str, default="data/Qwen2.5_7B_only_iteration_DFO_data_with_attention_score_1.json")
parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
args = parser.parse_args()

# [核心修改] 定义一个可管理的批处理大小
# 您可以根据您的显存大小调整这个值。常见的选择有 8, 16, 32, 64。
# 从一个较小的值开始，如果没问题再逐渐增大。
BATCH_SIZE = 16

data = read_json(args.input_path)
model_path = args.model_path

with torch.no_grad():
    # 1. 加载模型和 Tokenizer (只需加载一次)
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # 加载LoRA模型
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()


    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    print("Model and tokenizer loaded.")

    # 准备好所有待处理的文本
    contexts = [item['conversations'][0]['value'] for item in data]
    total_samples = len(contexts)
    # 计算总批次数，用于显示进度
    total_batches = math.ceil(total_samples / BATCH_SIZE)

    # 2. [核心修改] 增加外层循环，按 BATCH_SIZE 分割数据进行处理
    for i in range(0, total_samples, BATCH_SIZE):
        # 获取当前批次的数据
        batch_contexts = contexts[i : i + BATCH_SIZE]
        current_batch_num = (i // BATCH_SIZE) + 1
        
        print(f"\n--- Processing Batch {current_batch_num}/{total_batches} (Samples {i+1}-{min(i+BATCH_SIZE, total_samples)}) ---")

        # 对当前小批次进行 Tokenize
        inputs = tokenizer(
            batch_contexts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=5096
        ).to(model.device)

        # 对当前小批次进行推理
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            output_attentions=True,
            return_dict_in_generate=True,
            temperature=0.1,
        )

        # 3. 解析当前小批次的结果
        context_lengths = inputs.attention_mask.sum(dim=1)
        batch_attentions = outputs.attentions
        
        batch_size_current = inputs.input_ids.shape[0]
        new_token_length = len(batch_attentions)
        num_layers = len(batch_attentions[0])
        num_heads = batch_attentions[0][0].shape[1]

        for b_idx in range(batch_size_current):
            context_len = context_lengths[b_idx].item()
            
            lookback_ratio = torch.zeros((num_layers, num_heads, new_token_length))

            for token_idx in range(new_token_length - 1):
                for layer_idx in range(num_layers):
                    current_attention_tensor = batch_attentions[token_idx+1][layer_idx][b_idx].clone()
                    attn_on_context = current_attention_tensor[:, -1, :context_len].mean(dim=-1)
                    attn_on_new_tokens = current_attention_tensor[:, -1, context_len:].mean(dim=-1)
                    epsilon = 1e-9
                    lookback_ratio[layer_idx, :, token_idx+1] = attn_on_context / (attn_on_context + attn_on_new_tokens + epsilon)
                    
                    # 删除临时计算的注意力张量以释放GPU内存
                    del current_attention_tensor, attn_on_context, attn_on_new_tokens

            current_ratio = lookback_ratio.mean().item()
            
            # [核心修改] 使用全局索引来更新原始的 data 列表
            global_idx = i + b_idx
            data[global_idx]['ratio'] = current_ratio
            
            # (可选) 打印单个样本的完成信息
            print(f"Sample {global_idx+1} processed with ratio: {current_ratio}")
            
            # 显式删除lookback_ratio以释放GPU内存
            del lookback_ratio

        # 每个batch处理完后，手动清理变量并释放显存
        del inputs, outputs, batch_attentions, context_lengths
        del batch_size_current, new_token_length, num_layers, num_heads
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # 同步GPU操作


# 4. 所有批次处理完成后，汇总结果并保存
print("\n--- All batches processed ---")
final_ratios = [item['ratio'] for item in data if 'ratio' in item]
print(f"Total samples with ratio: {len(final_ratios)}")
if final_ratios:
    print("Mean ratio:", np.mean(final_ratios))

write_json(data, args.output_path)
print(f"Results saved to {args.output_path}")