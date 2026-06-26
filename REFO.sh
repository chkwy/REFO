#!/bin/bash
# 设置默认参数
current_model_path=${1:-"meta-llama/Meta-Llama-3-8B-Instruct"}
base_model_path=${2:-"meta-llama/Meta-Llama-3-8B-Instruct"}
initial_model_name=${3:-"Llama_3_8B_Residual"}
model_path_step0=${4:-"/data/DFO_wang/saves/Llama_3_8B_only_iteration_base"}
template=${5:-"llama3"}   
num_iterations=${6:-5}  # 默认 5次迭代
gpu_num=${7:-8}
lambda_value=1.0
alpha_value=0.0
beta_value=0.1
lr=0.00002


filter_threshold_list=(0.2 0.3 0.3 0.3 0.4)
residual_threshold_list=(0.2 0.2 0.2 0.2 0.2)

# 数据相关变量
dataset_dir=${initial_model_name}_data  # 初始化数据集目录

# 导出环境变量
export ENTROPY_ALPHA="$alpha_value"
export LAMBDA="$lambda_value"
export NAME="$initial_model_name"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_TRUST_REMOTE_CODE=1

# 创建必要的目录
mkdir -p logs
mkdir -p saves
mkdir -p data

# 日志文件
run_log_file="logs/${initial_model_name}_run.log"
eval_metrics_log="logs/${initial_model_name}_eval_metrics.log"


create_data(){
    local g="$1"  # 迭代次数
    local model_path="$2"  # 模型路径
    local dataset_dir_input="$3"  # 数据集目录
    local threshold="$4"  # 阈值

    # 最终输出文件
    if [ -f "data/${dataset_dir_input}_${g}_step1_filtered.json" ]; then
        echo "已存在 data/${dataset_dir_input}_${g}_step1_filtered.json，跳过数据生成步骤" >> "$run_log_file"
        echo "${dataset_dir_input}_${g}_step1_filtered"
        return
    fi

    echo "创建数据 - 迭代${g}, 模型: $model_path" >> "$run_log_file"
    
    python create_ReST_data_lora.py \
        --model_name "${initial_model_name}_iter${g}_step1" \
        --model_path "$model_path" \
        --filter_data \
        --dataset_path "data/rag_truth/all_data_${g}.json" \
        --output_path "data/${dataset_dir_input}_${g}_step1.json" \
        --dataset_name "${dataset_dir_input}_${g}_step1" \
        --base_model "$base_model_path" \
        --gpu_num "$gpu_num" \
        >> "$run_log_file" 2>&1
    
    python SEFO/compute_hallucination_reward.py \
        --input_path "data/${dataset_dir_input}_${g}_step1.json" \
        --output_path "data/${dataset_dir_input}_${g}_step1_with_hallucination_reward.json" \
        >> "$run_log_file" 2>&1
    
    python SEFO/filtered_high_quality_samples.py \
        --input_path "data/${dataset_dir_input}_${g}_step1_with_hallucination_reward.json" \
        --output_path "data/${dataset_dir_input}_${g}_step1_filtered.json" \
        --threshold $threshold \
        --dataset_name "${dataset_dir_input}_${g}_step1_filtered" \
        >> "$run_log_file" 2>&1
    
    # 返回过滤后的数据集名称
    echo "${dataset_dir_input}_${g}_step1_filtered"
}

# 创建残差数据函数
create_residual_data(){
    local g="$1"  # 迭代次数
    local model_path="$2"  # 模型路径
    local dataset_dir_input="$3"  # 数据集目录
    local threshold="$4"  # 阈值
    local step="$5"  # 步骤

    # 最终输出文件
    if [ -f "data/${dataset_dir_input}_${g}_step${step}_filtered.json" ]; then
        echo "已存在 data/${dataset_dir_input}_${g}_step${step}_filtered.json，跳过残差数据生成步骤" >> "$run_log_file"
        echo "${dataset_dir_input}_${g}_step${step}_filtered"
        return
    fi

    echo "step${step}: 生成残差数据" >> "$run_log_file"
    python test_create_data_residual.py \
            --model_name "${initial_model_name}_iter${g}_step${step}" \
            --model_path "$model_path" \
            --filter_data \
            --dataset_path "data/${dataset_dir_input}_${g}_step$(($step-1)).json" \
            --output_path "data/${dataset_dir_input}_${g}_step${step}.json" \
            --dataset_name "${dataset_dir_input}_${g}_step${step}" \
            --base_model "$base_model_path" \
            --gpu_num "$gpu_num" \
            >> "$run_log_file" 2>&1


    python SEFO/compute_hallucination_reward.py \
        --input_path "data/${dataset_dir_input}_${g}_step${step}.json" \
        --output_path "data/${dataset_dir_input}_${g}_step${step}_with_hallucination_reward.json" \
        >> "$run_log_file" 2>&1
    
    python SEFO/filtered_high_quality_samples.py \
        --input_path "data/${dataset_dir_input}_${g}_step${step}_with_hallucination_reward.json" \
        --output_path "data/${dataset_dir_input}_${g}_step${step}_filtered.json" \
        --threshold $threshold \
        --dataset_name "${dataset_dir_input}_${g}_step${step}_filtered" \
        >> "$run_log_file" 2>&1
    
    # 返回过滤后的数据集名称
    echo "${dataset_dir_input}_${g}_step${step}_filtered"
}

# 训练模型函数
train_model(){
    local model_name_to_train="$1"
    local adapter_path="$2"
    local log_file="$3"
    local dataset_name="$4"
    
    if [ -d "/data/DFO_wang/saves/${model_name_to_train}" ]; then
        echo "模型目录 /data/DFO_wang/saves/${model_name_to_train} 已存在，跳过训练" >> "$log_file"
        return
    fi

    echo "开始训练模型: $model_name_to_train" >> "$log_file"
    echo "使用数据集: $dataset_name" >> "$log_file"
    
    # if [ "$adapter_path" == "None" ]; then
    #     llamafactory-cli train \
    #     --stage dpo \
    #     --do_train True \
    #     --model_name_or_path "$base_model_path" \
    #     --preprocessing_num_workers 16 \
    #     --finetuning_type lora \
    #     --template "$template" \
    #     --flash_attn auto \
    #     --dataset_dir data \
    #     --dataset "$dataset_name" \
    #     --cutoff_len 4096 \
    #     --learning_rate $lr \
    #     --num_train_epochs 1.0 \
    #     --max_samples 100000 \
    #     --per_device_train_batch_size 1 \
    #     --gradient_accumulation_steps 1 \
    #     --lr_scheduler_type cosine \
    #     --max_grad_norm 1.0 \
    #     --logging_steps 1 \
    #     --save_steps 1000 \
    #     --warmup_steps 0 \
    #     --packing False \
    #     --report_to wandb \
    #     --output_dir "/data/DFO_wang/saves/${model_name_to_train}" \
    #     --bf16 True \
    #     --plot_loss True \
    #     --trust_remote_code True \
    #     --ddp_timeout 180000000 \
    #     --optim adamw_torch \
    #     --lora_rank 8 \
    #     --lora_alpha 16 \
    #     --lora_dropout 0 \
    #     --lora_target all \
    #     --pref_beta "$beta_value" \
    #     --pref_ftx 0 \
    #     --pref_loss sigmoid \
    #     --deepspeed cache/ds_z3_config.json \
    #     >> "$log_file" 2>&1
    # else
    #     llamafactory-cli train \
    #     --stage dpo \
    #     --do_train True \
    #     --model_name_or_path "$base_model_path" \
    #     --preprocessing_num_workers 16 \
    #     --finetuning_type lora \
    #     --template "$template" \
    #     --flash_attn auto \
    #     --dataset_dir data \
    #     --dataset "$dataset_name" \
    #     --cutoff_len 4096 \
    #     --learning_rate $lr \
    #     --num_train_epochs 1.0 \
    #     --max_samples 100000 \
    #     --per_device_train_batch_size 1 \
    #     --gradient_accumulation_steps 1 \
    #     --lr_scheduler_type cosine \
    #     --max_grad_norm 1.0 \
    #     --logging_steps 1 \
    #     --save_steps 1000 \
    #     --warmup_steps 0 \
    #     --packing False \
    #     --report_to wandb \
    #     --output_dir "/data/DFO_wang/saves/${model_name_to_train}" \
    #     --bf16 True \
    #     --plot_loss True \
    #     --trust_remote_code True \
    #     --ddp_timeout 180000000 \
    #     --optim adamw_torch \
    #     --adapter_name_or_path "$adapter_path" \
    #     --lora_rank 8 \
    #     --lora_alpha 16 \
    #     --lora_dropout 0 \
    #     --lora_target all \
    #     --pref_beta "$beta_value" \
    #     --pref_ftx 0 \
    #     --pref_loss sigmoid \
    #     --deepspeed cache/ds_z3_config.json \
    #     --disable_shuffling True \
    #     >> "$log_file" 2>&1
    # fi
    llamafactory-cli train \
    --stage dpo \
    --do_train True \
    --model_name_or_path "$base_model_path" \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template "$template" \
    --flash_attn auto \
    --dataset_dir data \
    --dataset "$dataset_name" \
    --cutoff_len 4096 \
    --learning_rate $lr \
    --num_train_epochs 1.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --packing False \
    --report_to wandb \
    --output_dir "/data/DFO_wang/saves/${model_name_to_train}" \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --adapter_name_or_path "$adapter_path" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all \
    --pref_beta "$beta_value" \
    --pref_ftx 0 \
    --pref_loss sigmoid \
    --deepspeed cache/ds_z3_config.json \
    --disable_shuffling True \
    >> "$log_file" 2>&1
}


# 记录评估指标函数
log_eval_metrics() {
    local model_name="$1"
    local datasets=("nqswap" "nqopen" "memo-trap")
    
    # 确保logs目录和日志文件存在
    mkdir -p logs
    
    echo "模型评估: $model_name" >> "$eval_metrics_log"
    echo "----------------------------------------" >> "$eval_metrics_log"
    
    local total_score=0.0
    local valid_datasets=0
    
    for dataset in "${datasets[@]}"; do
        local metrics_file="eval/outputs/${dataset}_${model_name}_metrics_final.json"
        if [ -f "$metrics_file" ]; then
            # 提取所有可能的指标
            local subspan_em=$(cat "$metrics_file" | tr ',' '\n' | grep '"sub_Subspan_EM"' | cut -d':' -f2 | tr -d ' }' | awk '{printf "%.4f", $1}' 2>/dev/null || echo "N/A")

            echo "  数据集: $dataset" >> "$eval_metrics_log"
            echo "    Sub_Subspan_EM: $subspan_em" >> "$eval_metrics_log"
            
            # 累加主要指标用于平均分计算
            if [[ "$subspan_em" != "N/A" ]]; then
                total_score=$(python3 -c "print($total_score + float('$subspan_em'))")
                valid_datasets=$((valid_datasets + 1))
            fi
        else
            echo "  数据集: $dataset - 评估文件不存在" >> "$eval_metrics_log"
        fi
    done
    
    # 计算平均分数
    if [ $valid_datasets -gt 0 ]; then
        local avg_score=$(python3 -c "print('%.4f' % ($total_score / $valid_datasets))")
        echo "  平均分数: $avg_score" >> "$eval_metrics_log"
    else
        echo "  平均分数: N/A (无有效评估结果)" >> "$eval_metrics_log"
    fi
    
    echo "" >> "$eval_metrics_log"
}

# 评估模型函数
eval_model(){
    local model_name_to_train="$1"
    local model_to_train_path="/data/DFO_wang/saves/${model_name_to_train}"
    local iteration="$2"
    
    # 检查所有评估文件是否都存在
    if [ -f "eval/outputs/nqswap_${model_name_to_train}_metrics_final.json" ] && \
       [ -f "eval/outputs/nqopen_${model_name_to_train}_metrics_final.json" ] && \
       [ -f "eval/outputs/memo-trap_${model_name_to_train}_metrics_final.json" ]; then
        echo "所有评估文件已存在，跳过评估步骤" >> "$run_log_file"
        log_eval_metrics "$model_name_to_train" "迭代${iteration}"
        return
    fi

    echo "在评估数据集上评估模型性能: $model_name_to_train" >> "$run_log_file"
    
    cd eval
    model_to_train_score=0
    total_datasets=3
    
    for dataset in "nqswap" "nqopen" "memo-trap"; do
        if [ ! -f "outputs/${dataset}_${model_name_to_train}_metrics_final.json" ]; then
            echo "评估数据集: $dataset" >> "../$run_log_file"
            python main_lora.py --model-id "${model_to_train_path}" --eval-dataset "${dataset}" --base_model "${base_model_path}" --gpu_num "$gpu_num" >> "../$run_log_file" 2>&1
        fi
        dataset_score=$(cat outputs/${dataset}_${model_name_to_train}_metrics_final.json | \
                    tr ',' '\n' | grep '"sub_Subspan_EM"' | cut -d':' -f2 | tr -d ' }' | \
                    awk '{printf "%.4f", $1}' 2>/dev/null || echo "0.0")
        model_to_train_score=$(python3 -c "print($model_to_train_score + $dataset_score)")
    done
    
    model_to_train_score=$(python3 -c "print('%.4f' % ($model_to_train_score / $total_datasets))")
    cd ..
    
    log_eval_metrics "$model_name_to_train" "迭代${iteration}"
    echo "模型 $model_name_to_train 平均分数: $model_to_train_score" >> "$run_log_file"
}

# 主训练循环
echo "开始ReST训练循环..." >> "$run_log_file"

global_best_score=0.0  # 初始化全局最佳分数
# 需要有初始模型，初始模型需要有lora权重
# 初始模型路径

step_list=(1 2 3)
for i in $(seq 1 $num_iterations); do
    filter_threshold=${filter_threshold_list[$i-1]}
    residual_threshold=${residual_threshold_list[$i-1]}
    echo "=== 开始第 $i 次迭代 ===" >> "$run_log_file"
    
    # 第1步：用llama3-8b-instruct生成数据
    echo "步骤1: 使用llama3-8b-instruct生成数据" >> "$run_log_file"
    filtered_dataset_name_step1=$(create_data "$i" "$model_path_step0" "$dataset_dir" "$filter_threshold")

    # 第2步：训练模型
    echo "步骤2: 训练模型" >> "$run_log_file"
    model_name_step1="${initial_model_name}_iter${i}_step1"
    train_model "$model_name_step1" "$model_path_step0" "$run_log_file" "$filtered_dataset_name_step1"

    # 第3步：评估模型
    echo "步骤3: 评估模型" >> "$run_log_file"
    eval_model "$model_name_step1" "$i"

    # 使用for循环处理步骤2和3的残差训练
    model_path_prev="/data/DFO_wang/saves/${model_name_step1}"
    for step in "${step_list[@]:1}"; do  # 从步骤2开始
        # 生成残差数据
        echo "步骤$((step*2)): 使用新模型生成数据" >> "$run_log_file"
        filtered_dataset_name_step=$(create_residual_data "$i" "$model_path_prev" "$dataset_dir" "$residual_threshold" "$step")

        # 训练模型
        echo "步骤$((step*2+1)): 训练模型" >> "$run_log_file"
        model_name_step="${initial_model_name}_iter${i}_step${step}"
        train_model "$model_name_step" "$model_path_prev" "$run_log_file" "$filtered_dataset_name_step"

        # 评估模型
        echo "步骤$((step*2+2)): 评估模型" >> "$run_log_file"
        eval_model "$model_name_step" "$i"
        
        # 获取当前模型分数
        model_to_train_score=0.0
        for dataset in "nqswap" "nqopen" "memo-trap"; do
            score=$(cat eval/outputs/${dataset}_${model_name_step}_metrics_final.json 2>/dev/null | \
                tr ',' '\n' | grep '"sub_Subspan_EM"' | cut -d':' -f2 | tr -d ' }' | \
                awk '{printf "%.4f", $1}' 2>/dev/null || echo "0.0")
            model_to_train_score=$(python3 -c "print($model_to_train_score + float('$score'))")
        done
        model_to_train_score=$(python3 -c "print('%.4f' % ($model_to_train_score / 3))")

        # 早停判断
        if [ $step -gt 1 ]; then
            improvement_check=$(python3 -c "
prev_score = float('$global_best_score')
curr_score = float('$model_to_train_score')
improvement = curr_score - prev_score
print(f'{improvement:.6f}')
print('continue' if improvement > 0.001 else 'stop')
")
            improvement=$(echo "$improvement_check" | head -n1)
            should_continue=$(echo "$improvement_check" | tail -n1)
            if [ "$should_continue" = "stop" ]; then
                echo "性能下降($improvement)，触发早停机制" | tee -a "$run_log_file"
                break
            else
                echo "性能提升: $improvement" | tee -a "$run_log_file"
            fi
        fi
        # 更新global_best_score
        cmp=$(python3 -c "print('1' if float('$model_to_train_score') > float('$global_best_score') else '0')")
        if [ "$cmp" = "1" ]; then
            global_best_score=$model_to_train_score
        fi
        # 更新模型路径为当前训练的模型
        model_path_prev="/data/DFO_wang/saves/${model_name_step}"
    done
    
    # 更新当前模型路径为最终训练的模型
    model_path_step0="$model_path_prev"
done

echo "=== ReST训练完成 $(date) ===" >> "$run_log_file"
echo "最终模型路径: $current_model_path" >> "$run_log_file"
