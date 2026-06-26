import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def main(args):
    model_path = args.model_path
    data_path = args.data_path

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(data_path, 'r') as file:
        data = json.load(file)

    input_text = []
    label_text = []

    for item in data:
        for conversation in item.get("conversations", []):
            input_text.append(conversation.get("value", ""))

        if args.is_chosen:
            chosen_value = item.get("chosen", {}).get("value", "")
        else:
            chosen_value = item.get("rejected", {}).get("value", "")
        assert chosen_value, "没有找到chosen或rejected的值"

        label_text.append(chosen_value)

    probabilities = process_batches(input_text, label_text, model, tokenizer)
    average_probability = sum(probabilities) / len(probabilities)
    print(f"所有批次的平均生成概率: {average_probability:.6f}")

def process_batches(input_text, label_text, model, tokenizer):
    batch_size = 16
    probabilities = []
    for i in tqdm(range(0, len(input_text), batch_size), desc="Processing batches"):
        input_batch = input_text[i:i + batch_size]
        label_batch = label_text[i:i + batch_size]
        batch_probability = calculate_sequence_probability(input_batch, label_batch, model, tokenizer)
        probabilities.append(batch_probability)
    return probabilities

def calculate_sequence_probability(input_batch, label_batch, model, tokenizer):
    # 拼接输入和标签（根据任务需求调整格式）
    # full_text = torch.cat([input_text , label_text],dim=1)
    prompt_list = []
    labels = []
    for question,answer in zip(input_batch,label_batch):
        prompt = [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            },
        ]
        question =prompt[:2]
        question_tensor = tokenizer.apply_chat_template(
            question,
            tokenize=False,
            add_generation_prompt=True
        )
        question_tensor=tokenizer(question_tensor, return_tensors="pt")
        question_len=len(question_tensor.input_ids[0])
        prompt_tensor = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
       
        prompt_list.append(prompt_tensor)

        labels.append(question_len)
    prompt_tensor=tokenizer(prompt_list, return_tensors="pt",padding=True, return_attention_mask=True)
    # for i in range(len(labels)):
    #     prompt_tensor["attention_mask"][i,:labels[i]]=0
    input_ids = prompt_tensor.input_ids
    # print("input_ids done")
    with torch.no_grad():
        outputs = model(**prompt_tensor, return_dict=True, use_cache=False)
    logits = outputs.logits.to(torch.float32)
    # print(logits.shape)
    logits = logits[:, :-1, :]  # 去掉最后一个预测
    labels = (input_ids*prompt_tensor.attention_mask)[:,1:]
    per_token_logps = torch.gather(logits.softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    total_prob = per_token_logps.sum()/prompt_tensor.attention_mask[:,1:].sum()
    return total_prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model and data path arguments")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--is_chosen", action="store_true", help="Whether to use chosen data")
    args = parser.parse_args()
    main(args)
