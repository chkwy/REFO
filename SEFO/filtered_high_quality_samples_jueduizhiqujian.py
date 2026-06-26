#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高质量样本筛选脚本
根据Hallucination_Reward分数筛选高质量样本用于ReST算法训练
"""

import argparse
import json
import os
from typing import List, Dict, Any

def read_json(file_path: str) -> List[Dict[str, Any]]:
    """
    读取JSON文件
    
    Args:
        file_path (str): JSON文件路径
        
    Returns:
        List[Dict[str, Any]]: 数据列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    写入JSON文件
    
    Args:
        data: 要写入的数据
        file_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def filter_high_quality_samples(
    input_data: List[Dict[str, Any]], 
    threshold: float
) -> List[Dict[str, Any]]:
    """
    根据Hallucination_Reward分数筛选高质量样本
    
    Args:
        input_data: 包含原始数据及奖励分数的数据列表
        threshold: 奖励分数阈值
        
    Returns:
        List[Dict[str, Any]]: 筛选后的高质量样本列表
    """
    high_quality_samples = []
    
    for item in input_data:
        # 获取幻觉奖励分数
        hallucination_reward = item.get('Hallucination_Reward_chosen', 0.0)
        
        # 只保留分数大于阈值的样本
        if hallucination_reward < threshold and hallucination_reward > threshold-0.3:
            # 提取原始数据用于训练
            original_data = item.get('original_data', {})
            high_quality_samples.append(original_data)
    # 如果样本数量超过1000，只保留后1000个
    if len(high_quality_samples) > 800:
        # 如果样本数量超过1280，只保留前1280个样本
        high_quality_samples = high_quality_samples[:800]
    return high_quality_samples

def print_statistics(
    original_count: int, 
    filtered_count: int, 
    threshold: float,
    reward_scores: List[float]
) -> None:
    """
    打印筛选统计信息
    
    Args:
        original_count: 原始样本数量
        filtered_count: 筛选后样本数量  
        threshold: 使用的阈值
        reward_scores: 所有样本的奖励分数列表
    """
    print("=" * 50)
    print("高质量样本筛选统计")
    print("=" * 50)
    print(f"奖励阈值: {threshold:.4f}")
    print(f"原始样本数量: {original_count}")
    print(f"筛选后样本数量: {filtered_count}")
    print(f"筛选比例: {filtered_count/original_count*100:.2f}%")
    
    if reward_scores:
        print(f"奖励分数统计:")
        print(f"  最小值: {min(reward_scores):.4f}")
        print(f"  最大值: {max(reward_scores):.4f}")
        print(f"  平均值: {sum(reward_scores)/len(reward_scores):.4f}")
        
        # 计算高于阈值的分数统计
        high_scores = [score for score in reward_scores if score > threshold]
        if high_scores:
            print(f"高质量样本奖励分数统计:")
            print(f"  最小值: {min(high_scores):.4f}")
            print(f"  最大值: {max(high_scores):.4f}")
            print(f"  平均值: {sum(high_scores)/len(high_scores):.4f}")
    
    print("=" * 50)

def update_dataset_info(dataset_name):
    dataset_info = {
        f"{dataset_name}": {
            "file_name": f"{dataset_name}.json",
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected"
            }
        }
    }

    # Load existing dataset info from the file
    existing_data = read_json('data/dataset_info.json')

    # Append new dataset info
    existing_data.update(dataset_info)

    write_json(existing_data, 'data/dataset_info.json')


def main():
    parser = argparse.ArgumentParser(
        description='根据Hallucination_Reward分数筛选高质量样本'
    )
    parser.add_argument(
        '--input_path', 
        type=str, 
        required=True,
        help='输入文件路径（包含幻觉奖励分数的数据）'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        required=True,
        help='输出文件路径（筛选后的高质量样本）'
    )
    parser.add_argument(
        '--threshold', 
        type=float, 
        required=True,
        help='奖励分数阈值（只保留分数大于该值的样本）'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='是否显示详细信息'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help='数据集名称'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"输入文件不存在: {args.input_path}")
    
    print(f"正在读取输入文件: {args.input_path}")
    
    # 读取输入数据
    try:
        input_data = read_json(args.input_path)
    except Exception as e:
        raise RuntimeError(f"读取输入文件失败: {e}")
    
    if not input_data:
        print("警告: 输入数据为空")
        return
    
    # 提取所有奖励分数用于统计
    reward_scores = []
    for item in input_data:
        reward_score = item.get('score_distance', 0.0)
        reward_scores.append(reward_score)
    
    # 筛选高质量样本
    print(f"正在使用阈值 {args.threshold} 筛选高质量样本...")
    
    try:
        high_quality_samples = filter_high_quality_samples(input_data, args.threshold)
    except Exception as e:
        raise RuntimeError(f"筛选样本时出错: {e}")
    
    # 输出统计信息
    if args.verbose:
        print_statistics(
            len(input_data), 
            len(high_quality_samples), 
            args.threshold,
            reward_scores
        )
    else:
        print(f"原始样本数量: {len(input_data)}")
        print(f"筛选后样本数量: {len(high_quality_samples)}")
        print(f"筛选比例: {len(high_quality_samples)/len(input_data)*100:.2f}%")
    
    # 检查是否有样本通过筛选
    if not high_quality_samples:
        print(f"警告: 没有样本的奖励分数大于阈值 {args.threshold}")
        print("建议降低阈值或检查输入数据质量")
        # 仍然创建空文件以避免后续脚本出错
        write_json([], args.output_path)
        return
    
    # 保存筛选后的样本
    print(f"正在保存筛选结果到: {args.output_path}")
    
    try:
        write_json(high_quality_samples, args.output_path)
        print(f"筛选完成！高质量样本已保存到: {args.output_path}")
    except Exception as e:
        raise RuntimeError(f"保存输出文件失败: {e}")

    update_dataset_info(args.dataset_name)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        exit(1)