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

def extract_confidence_score(text):
    """
    从模型输出中提取难度分数(1-5)
    """
    # 尝试匹配1-5之间的整数
    pattern = r'\b[1-5]\b'
    matches = re.findall(pattern, text)
    
    for match in matches:
        try:
            score = int(match)
            if 1 <= score <= 5:
                return score
        except ValueError:
            continue
    
    # 如果没找到有效分数，返回None
    return None

def generate_confidence_scores(input_json_path, output_json_path, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    """
    读取原json，生成置信度prompt，调用vLLM获得置信度分数，保存结果
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
        max_tokens=500,
        stop=None
    )
    
    print("Loading data...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 准备所有prompt
    confidence_prompts = []
    for item in data:
        prompt = item['prompt']
        output = item['revise_response']
        
        # 从prompt中提取用户问题
        passage,question = extract_question_from_prompt(prompt)
        
        # user_content = f"Please assess the correctness of the My Answer and provide your confidence score in My Answer to the Question based on the passage as a number between 0 and 1. Only output the score. Reference:{passage}\n Question: {question}\n\nHere is My Answer: {output}\n\n"
        
        # user_content = f"Please revise my sentence to make it more concise and clear. Here is my sentence: {output}\n\n Please directly output the revised sentence without any other words. "

        user_content = f"""Please assess the confidence of your answer to the following question based on the given passage. Provide a confidence score from 1-5, where 1 means very uncertain and 5 means very confident.

        Passage:
        {passage}

        Question:
        {question}
        
        Your Answer:
        {output}
        
        Please directly output the confidence score without any other words.
        """
        
        # 转换为聊天格式
        chat_messages = [{"role": "user", "content": user_content}]
        confidence_prompts.append(chat_messages)
    
    print(f"Generating confidence scores for {len(confidence_prompts)} items...")
    
    # 批量推理 - 使用chat格式
    responses = llm.chat(confidence_prompts, sampling_params)
    
    results = []
    
    for i, (item, response) in enumerate(zip(data, responses)):
        print(f"Processing {i+1}/{len(data)}")
        
        output = item['output']
        confidence_response = response.outputs[0].text.strip()
        
        # 提取置信度分数
        confidence_score = extract_confidence_score(confidence_response)
        
        # 保存结果
        result_item = item.copy()
        result_item['confidence_prompt'] = confidence_prompts[i]
        result_item['confidence_response'] = confidence_response
        result_item['confidence_score'] = confidence_score
        # 按confidence score降序排序
        results.sort(key=lambda x: x['confidence_score'] if x['confidence_score'] is not None else -1, reverse=True)
        results.append(result_item)
        
        print(f"Output: {output}...")
        print(f"Confidence response: {confidence_response}")
        print(f"Confidence score: {confidence_score}")
        print("-" * 50)
    
    # 保存结果
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_json_path}")
    
    # 统计
    valid_scores = [item['confidence_score'] for item in results if item['confidence_score'] is not None]
    if valid_scores:
        avg_confidence = sum(valid_scores) / len(valid_scores)
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Valid confidence scores: {len(valid_scores)}/{len(results)}")


# 示例用法
if __name__ == "__main__":
    input_json = "/data/DFO/eval/outputs/nqopen_Meta-Llama-3-8B-Instruct_output_prompt_1_with_revise.json"
    output_json = "/data/DFO/eval/outputs/nqopen_Meta-Llama-3-8B-Instruct_output_prompt_1_with_revise_with_confidence_difficulty_score.json"
    
    # 直接生成置信度分数
    generate_confidence_scores(input_json, output_json)
