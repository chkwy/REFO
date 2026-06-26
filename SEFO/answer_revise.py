import json
import os
from vllm import LLM, SamplingParams
import re

def extract_question_from_prompt(prompt):
    """
    从llama3 chat template格式的prompt中提取用户问题内容
    """
    # 匹配passage部分：从source:开始到Answer the following question之前
    pattern_passage = r'source:\s*(.*?)\n\nAnswer the following question'
    match_passage = re.search(pattern_passage, prompt, re.DOTALL)
    
    # 匹配question部分：从"The answer needs to be just in a few words."后面开始到Answer:<之前
    pattern_question = r'The answer needs to be just in a few words\.\s*(.*?)\n\nAnswer:'
    match_question = re.search(pattern_question, prompt, re.DOTALL)
    
    if match_passage and match_question:
        passage = match_passage.group(1).strip()
        question = match_question.group(1).strip()
        return passage, question
    else:
        print("no match") 

def extract_revise_text(text):
    """
    从模型输出中提取修订后的文本
    """
    # 尝试匹配0-1之间的小数
    pattern = r'\b(?:0\.?\d*|1\.?0*)\b'
    matches = re.findall(pattern, text)
    
    for match in matches:
        try:
            score = float(match)
            if 0 <= score <= 1:
                return score
        except ValueError:
            continue
    
    # 如果没找到有效分数，返回None
    return None

def generate_revise_results(input_json_path, output_json_path, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    """
    读取原json，生成修订prompt，调用vLLM获得修订结果，保存结果
    """
    print("Loading model with vLLM...")
    llm = LLM(
        model=model_name,
        dtype="half",  # 使用float16
        gpu_memory_utilization=0.8,
        max_model_len=4096
    )
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=50,
        stop=None
    )
    
    print("Loading data...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 准备所有prompt
    revise_prompts = []
    for item in data:
        # prompt = item['prompt']
        output = item['output']
        
        # 从prompt中提取用户问题
        # passage,question = extract_question_from_prompt(prompt)
        
        # user_content = f"Please assess the correctness of the My Answer and provide your confidence score in My Answer to the Question based on the passage as a number between 0 and 1. Only output the score. Reference:{passage}\n Question: {question}\n\nHere is My Answer: {output}\n\n"
        
        user_content = f"Please revise my sentence to make it more concise and clear. Here is my sentence: {output}\n\n Please directly output the revised sentence without any other words. "

        
        # 转换为聊天格式
        chat_messages = [{"role": "user", "content": user_content}]
        revise_prompts.append(chat_messages)
    
    print(f"Generating revise results for {len(revise_prompts)} items...")
    
    # 批量推理 - 使用chat格式
    responses = llm.chat(revise_prompts, sampling_params)
    
    results = []
    
    for i, (item, response) in enumerate(zip(data, responses)):
        print(f"Processing {i+1}/{len(data)}")
        
        output = item['output']
        revise_response = response.outputs[0].text.strip()
        
        # 保存结果
        result_item = item.copy()
        result_item['revise_prompt'] = revise_prompts[i]
        result_item['revise_response'] = revise_response
        
        print(f"Output: {output}...")
        print(f"Revise response: {revise_response}")
        print("-" * 50)
        
        results.append(result_item)
    
    # 保存结果
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_json_path}")
# 示例用法
if __name__ == "__main__":
    input_json = "/data/DFO/eval/outputs/nqopen_Meta-Llama-3-8B-Instruct_output_prompt_1.json"
    output_json = "/data/DFO/eval/outputs/nqopen_Meta-Llama-3-8B-Instruct_output_prompt_1_with_revise.json"
    
    # 直接生成修订结果
    generate_revise_results(input_json, output_json)
