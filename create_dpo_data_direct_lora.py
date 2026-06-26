from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import argparse
from utils import read_json, write_json
from vllm.lora.request import LoRARequest

prompt_template_context = """
Based on the following context:
Context: {reference}\nQuestion: {question}\n
Please directly answer the question without any explanation."""

prompt_template_direct = """Question: {question}\n
Please directly answer the question without any explanation."""

LLAMA3_TEMPLETE = '''
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{user_message} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
'''


def prepare_dataset(data, tokenizer, context=True):
    """
    Prepares the dataset by formatting prompts and tokenizing them.
    """
    prompts = []
    raw_prompts = []
    for k, v in data.items():
        reference = v['passages']
        question = v['question']

        # Format the prompt
        if context:
            formatted_prompt = prompt_template_context.format(
                reference=reference,
                question=question
            )
        else:
            formatted_prompt = prompt_template_direct.format(
                question=question
            )
        raw_prompts.append(formatted_prompt)

        formatted_prompt = LLAMA3_TEMPLETE.format(
            user_message=formatted_prompt)
        # formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": formatted_prompt}], tokenize=False)
        prompts.append(formatted_prompt)

    return prompts, raw_prompts


def inference(model, sampling_params, tokenizer, data, model_path):
    """
    Performs inference using the model to generate hallucination analysis.
    """
    # Prepare the inputs
    inputs_win, input_raw = prepare_dataset(data, tokenizer)
    # inputs_win = inputs_win[:10]
    # input_raw = input_raw[:10]
    inputs_loss, _ = prepare_dataset(data, tokenizer, context=False)
    # inputs_loss = inputs_loss[:10]

    y_win = model.generate(inputs_win, sampling_params,lora_request=LoRARequest("LLAMA_adapter", 1, model_path))
    y_win = [output.outputs[0].text for output in y_win]

    y_loss = model.generate(inputs_loss, sampling_params,lora_request=LoRARequest("LLAMA_adapter", 1, model_path))
    y_loss = [output.outputs[0].text for output in y_loss]

    return y_win, y_loss, input_raw


def postprocess(y_win, y_loss, inputs_win):
    new_win = []
    new_loss = []
    new_inputs = []
    for w, l, i in zip(y_win, y_loss, inputs_win):
            if "i don't know" in l.lower() or "don't know" in l.lower():
                continue
            new_win.append(w)
            new_loss.append(l)
            new_inputs.append(i)
    return new_win, new_loss, new_inputs


def make_DPO_data(y_win, y_loss, inputs_win):
    DPO_data = []
    for i, (w, l, in_w) in enumerate(zip(y_win, y_loss, inputs_win)):
        DPO_data.append({
            "conversations": [
                {
                    "from": "human",
                    "value": in_w
                }
            ],
            "chosen": {
                "from": "gpt",  
                "value": w
            },
            "rejected": {
                "from": "gpt",
                "value": l
            }
        })
    return DPO_data


def update_dataset_info(model_name):
    dataset_info = {
        f"{model_name}_DFO_data": {
            "file_name": f"{model_name}_DFO_data.json",
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
    
    parser = argparse.ArgumentParser(description="DPO data generation script")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to use")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model")
    parser.add_argument(
        "--filter_data",action="store_true",)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='data/rag_truth/all_data.json',
        help="Path to the dataset file")
    args = parser.parse_args()
    data = read_json(args.dataset_path)

    model_name = args.model_name
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Load the generation config
    try:
        generation_config = read_json(
            args.model_path + '/generation_config.json')
        allowed_keys = {"repetition_penalty", "temperature", "top_p", "top_k"}
        filtered_config = {
            key: generation_config[key] for key in allowed_keys if key in generation_config}
        print(filtered_config)
        sampling_params = SamplingParams(**filtered_config, max_tokens=1024)
    except BaseException:
        print(
            "Warning: generation_config.json not found. Using default sampling parameters.")
        sampling_params = SamplingParams(
            temperature=0.1, top_p=0.9, max_tokens=1024)

    model = LLM("meta-llama/Meta-Llama-3-8B-Instruct", 
                tokenizer=model_path,
                tensor_parallel_size=8,
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                dtype=torch.float16, enforce_eager=True,
                trust_remote_code=True,
                enable_lora=True)

    y_win, y_loss, inputs_win = inference(
        model, sampling_params, tokenizer, data, model_path)

    if args.filter_data:
        print(f"len before postprocess: {len(y_win)}")
        y_win, y_loss, inputs_win = postprocess(y_win, y_loss, inputs_win)
        print(f"len after filter: {len(y_win)}")
    else:
        print("not filtering data")
        model_name+="_no_filter"

    DPO_data = make_DPO_data(y_win, y_loss, inputs_win)

    # Save the augmented data to a new JSON file
    output_file_path = f'data/{model_name}_DFO_data.json'
    write_json(DPO_data, output_file_path)

    print(f"completed and saved to {output_file_path}")

    update_dataset_info(model_name)


if __name__ == "__main__":
    main()
