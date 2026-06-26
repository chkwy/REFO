import pandas as pd
import json
import glob

def log_to_excel(input_path, output_path):
    with open(input_path, 'r') as f:
        log_data = f.read()

    records = []
    blocks = log_data.strip().split('-----------------------------------')
    
    for block in blocks:
        if not block.strip():
            continue
        record = {}
        for line in block.strip().split('\n'):
            if ': ' in line:
                key, val = line.split(': ', 1)
                record[key.strip()] = val.strip()
        
        # 从 model_path 提取文件名并查找匹配的 JSON 文件
        if 'model_path' in record:
            model_filename = record['model_path'].split('/')[-1]
            json_pattern = f'/data/DFO/eval/outputs/*{model_filename}*_metrics*.json'
            json_files = glob.glob(json_pattern)
            
            if not json_files:  # 如果 json_files 是空列表
                print(f"Warning: No JSON files found for model {model_filename}")
            
            for json_path in json_files:
                try:
                    with open(json_path, 'r') as json_file:
                        json_data = json.load(json_file)
                        # 根据 JSON 文件名生成动态键名
                        json_prefix = json_path.split('/')[-1].split('_')[0]
                        key_name = f"{json_prefix}_sub_Subspan_EM"
                        if key_name=='eli5_sub_Subspan_EM':
                            record[key_name] = json_data.get('rouge-1', None).get('f', None)
                        else:
                            record[key_name] = json_data.get('sub_Subspan_EM', None)

                except FileNotFoundError:
                    continue
        
        records.append(record)
        
    df = pd.DataFrame(records)
    # 动态列名处理，确保所有动态键名都包含在 DataFrame 中
    all_columns = ['average_chosen_prob', 'average_reject_prob', 'model_path', 'dataset_path'] + \
                  [col for col in records[0].keys() if col.endswith('_sub_Subspan_EM')]
    df = df[all_columns]
    df.to_excel(output_path, index=False)

# 使用示例
log_to_excel('/data/DFO/logs/prob.txt', 'batch_prob.xlsx')