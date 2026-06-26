#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分层数据集创建脚本
根据Hallucination_Reward分数创建多个阈值的子数据集用于ReST算法训练
"""

import argparse
import json
import os
from typing import List, Dict, Any, Tuple

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

def filter_samples_by_threshold(
    input_data: List[Dict[str, Any]], 
    threshold: float
) -> List[Dict[str, Any]]:
    """
    根据Hallucination_Reward分数筛选样本
    
    Args:
        input_data: 包含原始数据及奖励分数的数据列表
        threshold: 奖励分数阈值
        
    Returns:
        List[Dict[str, Any]]: 筛选后的样本列表
    """
    filtered_samples = []
    
    for item in input_data:
        # 获取幻觉奖励分数
        hallucination_reward = item.get('score_distance', 0.0)
        
        # 只保留分数大于阈值的样本
        if hallucination_reward > threshold and hallucination_reward < threshold+0.1:
            # 提取原始数据用于训练
            original_data = item.get('original_data', {})
            filtered_samples.append(original_data)
    
    return filtered_samples

def generate_thresholds(start_threshold: float, end_threshold: float, step: float) -> List[float]:
    """
    生成阈值列表
    
    Args:
        start_threshold: 起始阈值（最高）
        end_threshold: 结束阈值（最低）
        step: 递减步长
        
    Returns:
        List[float]: 阈值列表
    """
    thresholds = []
    current = start_threshold
    while current >= end_threshold:
        thresholds.append(round(current, 2))
        current -= step
    return thresholds

def create_tiered_datasets(
    input_data: List[Dict[str, Any]], 
    thresholds: List[float],
    base_output_path: str,
    dataset_base_name: str
) -> List[Tuple[str, int, float]]:
    """
    创建分层数据集
    
    Args:
        input_data: 输入数据
        thresholds: 阈值列表
        base_output_path: 基础输出路径
        dataset_base_name: 数据集基础名称
        
    Returns:
        List[Tuple[str, int, float]]: (数据集名称, 样本数量, 阈值) 的列表
    """
    dataset_info_list = []
    
    for threshold in thresholds:
        # 筛选样本
        filtered_samples = filter_samples_by_threshold(input_data, threshold)
        
        if not filtered_samples:
            print(f"警告: 阈值 {threshold} 下没有符合条件的样本，跳过")
            continue
        
        # 生成输出文件名
        threshold_str = str(threshold).replace('.', '_')
        dataset_name = f"{dataset_base_name}_threshold_{threshold_str}"
        output_file = os.path.join(base_output_path, f"{dataset_name}.json")
        
        # 保存数据集
        write_json(filtered_samples, output_file)
        
        dataset_info_list.append((dataset_name, len(filtered_samples), threshold))
        print(f"创建数据集: {dataset_name}, 样本数量: {len(filtered_samples)}, 阈值: {threshold}")
    
    return dataset_info_list

def update_dataset_info(dataset_info_list: List[Tuple[str, int, float]]) -> None:
    """
    更新数据集信息文件
    
    Args:
        dataset_info_list: 数据集信息列表
    """
    # 读取现有的数据集信息
    dataset_info_path = 'data/dataset_info.json'
    try:
        existing_data = read_json(dataset_info_path)
    except FileNotFoundError:
        existing_data = {}
    
    # 添加新的数据集信息
    for dataset_name, sample_count, threshold in dataset_info_list:
        dataset_info = {
            "file_name": f"{dataset_name}.json",
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected"
            },
            "metadata": {
                "sample_count": sample_count,
                "threshold": threshold,
                "description": f"高质量样本数据集，奖励分数阈值: {threshold}"
            }
        }
        existing_data[dataset_name] = dataset_info
    
    # 保存更新后的数据集信息
    write_json(existing_data, dataset_info_path)
    print(f"已更新数据集信息文件: {dataset_info_path}")

def print_summary(dataset_info_list: List[Tuple[str, int, float]], total_samples: int) -> None:
    """
    打印汇总信息
    
    Args:
        dataset_info_list: 数据集信息列表
        total_samples: 总样本数量
    """
    print("\n" + "=" * 60)
    print("分层数据集创建汇总")
    print("=" * 60)
    print(f"原始总样本数量: {total_samples}")
    print(f"创建的数据集数量: {len(dataset_info_list)}")
    print("\n数据集详情:")
    print("-" * 60)
    print(f"{'数据集名称':<30} {'样本数量':<10} {'阈值':<10} {'比例':<10}")
    print("-" * 60)
    
    for dataset_name, sample_count, threshold in dataset_info_list:
        ratio = sample_count / total_samples * 100
        print(f"{dataset_name:<30} {sample_count:<10} {threshold:<10} {ratio:.2f}%")
    
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description='根据不同阈值创建分层数据集'
    )
    parser.add_argument(
        '--input_path', 
        type=str, 
        required=True,
        help='输入文件路径（包含幻觉奖励分数的数据）'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True,
        help='输出目录路径'
    )
    parser.add_argument(
        '--dataset_base_name',
        type=str,
        required=True,
        help='数据集基础名称'
    )
    parser.add_argument(
        '--start_threshold', 
        type=float, 
        default=0.9,
        help='起始阈值（默认: 0.9）'
    )
    parser.add_argument(
        '--end_threshold', 
        type=float, 
        default=0.1,
        help='结束阈值（默认: 0.1）'
    )
    parser.add_argument(
        '--step', 
        type=float, 
        default=0.1,
        help='阈值递减步长（默认: 0.1）'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='是否显示详细信息'
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
    
    # 生成阈值列表
    thresholds = generate_thresholds(args.start_threshold, args.end_threshold, args.step)
    print(f"生成的阈值列表: {thresholds}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建分层数据集
    print(f"\n开始创建分层数据集...")
    dataset_info_list = create_tiered_datasets(
        input_data, 
        thresholds, 
        args.output_dir, 
        args.dataset_base_name
    )
    
    if not dataset_info_list:
        print("错误: 没有创建任何数据集，请检查阈值设置和输入数据")
        return
    
    # 更新数据集信息
    try:
        update_dataset_info(dataset_info_list)
    except Exception as e:
        print(f"警告: 更新数据集信息时出错: {e}")
    
    # 打印汇总信息
    if args.verbose:
        print_summary(dataset_info_list, len(input_data))
    else:
        print(f"\n成功创建 {len(dataset_info_list)} 个数据集")
        print(f"输出目录: {args.output_dir}")
    
    print("\n分层数据集创建完成！")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        exit(1) 