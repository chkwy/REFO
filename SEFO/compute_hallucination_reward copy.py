# %%

import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch
from transformers import AutoModelForSequenceClassification
# need to set /data/miniconda3/lib/python3.9/site-packages/scale_score/utils.py 
from transformers import pipeline, AutoTokenizer
import nltk
import json  # 添加缺失的导入
import re    # 添加缺失的导入

tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

def sentencize(text):
    """
    将文本分割成句子列表。
    
    Args:
        text (str): 输入文本
        
    Returns:
        list: 句子列表
        
    Raises:
        AssertionError: 如果文本包含'\n210. '或分割后的句子不在原文本中
    """
    # assert '\n210. ' not in text
    ori_text = text
    
    # 替换编号格式，避免分句错误
    for i in range(1, 200):  # 假设最多有20个句子
        text = text.replace(f"\n{i}. ", f"\n{i}_PLACEHOLDER_")
        text = text.replace(f"\n {i}. ", f"\n {i}_PLACEHOLDER_")
    
    # 使用NLTK进行分句
    sentences = [tokenizer.tokenize(text)][0]
    
    # 恢复编号格式
    for i in range(1, 200):
        sentences = [sentence.replace(f"{i}_PLACEHOLDER_", f"{i}. ") 
                    for sentence in sentences]
    
    # 验证所有分割后的句子都在原文本中
    for s in sentences:
        if s not in ori_text:
            assert False, f"SENTENCIZE ERROR: {s} not in {ori_text}"
            
    return sentences

def write_json(data, file_path):
    """
    Writes data to a JSON file.
    """
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

def read_json(file_path):
    """
    Reads a JSON file and returns the data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_data(path):

    data = read_json(path)

    hypothesis = []
    premises_chosen = []
    premises_rejected = []

    for i in data:
        premises_chosen.append((i['chosen']['value']))
        # premises_chosen.append((i['output']))
        # premises_rejected.append((i['output']))
        premises_rejected.append((i['rejected']['value']))

        # hypothesis.append((output))
        hypothesis.append(i['conversations'][0]['value'])
        # hypothesis.append(i['prompt'])

    return data,premises_chosen,premises_rejected,hypothesis
        

def process_func(data,premises_chosen,premises_rejected,hypothesis,method):
        
    if method == 'HaluScore':
        pairs = [ (p,hs) for p,hs in zip(hypothesis,premises_chosen)]

        prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
        input_pairs = [prompt.format(text1=pair[0], text2=pair[1]) for pair in pairs]

        # Use text-classification pipeline to predict
        classifier = pipeline(
                    "text-classification",
                    model='vectara/hallucination_evaluation_model',
                    tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-base'),
                    trust_remote_code=True
                )
        full_scores = classifier(input_pairs, top_k=None) # List[List[Dict[str, float]]]

        # Optional: Extract the scores for the 'consistent' label
        simple_scores = [score_dict['score'] for score_for_both_labels in full_scores for score_dict in score_for_both_labels if score_dict['label'] == 'consistent']

                
        return simple_scores


if __name__ == '__main__':
    
    #* get score
    # path = 'output/single_stage-v2-batch-2.json'
    # data,premises,hypothesis,outputs = load_data_gpt(path)

    # path = 'data/RefGPT_en_sampled_human_resolved.json'

    parser = argparse.ArgumentParser(
        description='Compute hallucination reward'
    )
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
    args = parser.parse_args()

    method = 'HaluScore'

    path = args.input_path

    data,premises_chosen,premises_rejected,hypothesis = load_data(path)

    results_chosen = process_func(data,premises_chosen,premises_rejected,hypothesis,method=method)
    results_rejected = process_func(data,premises_rejected,premises_chosen,hypothesis,method=method)

    # 将结果写入新的JSON文件
    output_path = args.output_path
    
    # 创建包含原始数据和得分的输出结构
    output_data = []
    for i, (score_chosen, score_rejected) in enumerate(zip(results_chosen, results_rejected)):
        output_data.append({
            'original_data': data[i],
            'Hallucination_Reward_chosen': score_chosen,
            'Hallucination_Reward_rejected': score_rejected,
            "score_distance": score_chosen - score_rejected,
        })
        
    # 按Hallucination_Reward降序排序
    output_data.sort(key=lambda x: x['score_distance'], reverse=True)
    
    # 使用utils.py中的write_json函数写入文件

    write_json(output_data, output_path)
    print(f"结果已保存到: {output_path}")


