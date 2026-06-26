# <center>REFO: Reinforced Evolutionary Faithfulness Optimization for Large Language Models</center>

![Method Overview](figs/method.pdf)

This repository contains the official code and datasets for **REFO**, a novel iterative self-evolution framework designed to significantly enhance the faithfulness of Large Language Models (LLMs) in Retrieval-Augmented Generation (RAG). 

While RAG successfully enriches LLMs with external knowledge, it is often plagued by faithfulness hallucinations where the generated text contradicts the retrieved source information. Previous research has been hindered by prohibitive manual annotation costs and a dependency on static datasets. REFO solves this by autonomously generating high-quality preference data and leveraging it for continuous, lightweight self-optimization.

## 🌟 Key Features
* **Iterative Self-Evolution:** Achieves self-improvement by learning directly from its own generated responses, operating independently of pre-existing, human-annotated datasets.
* **Relative Preference Optimization:** Leverages the output differences between successive model generations (current vs. previous model) to establish a robust, self-sustaining evolutionary loop.
* **Attention-Guided Training:** Employs a novel Att-DPO loss function that implicitly scales optimization based on the model's "Lookback Ratio," successfully directing the model's generation process toward the provided context.
* **State-of-the-Art Performance:** Markedly outperforms post-training methods, decoding strategy-based methods, and other self-supervised approaches across diverse LLMs (LLaMA-3, Qwen-2.5, Mistral) and benchmarks.

---

## 🛠️ Requirements

Ensure your environment is set up with the following dependencies:
* `Llamafactory`
* `deepspeed`
* `vllm`
* `rouge_score`
* `sacrebleu`

*(Note: During experiments, LoRA was employed for efficient parameter tuning with an AdamW optimizer, and evaluation relies heavily on vLLM for rapid inference.)*

---

## 🚀 Usage

We provide automated bash scripts (`REFO.sh` and `Att-REFO.sh`) to seamlessly run the data preparation, training, and evaluation phases. The scripts require seven specific arguments to configure the run environment.

### Standard REFO Training
```shell
bash REFO.sh <model_path> <model_path> <experiment_name> <model_path_step0> <chat_template> <num_iterations> <gpu_num>
```

### Attention-Guided REFO (Att-REFO) Training
```shell
bash Att-REFO.sh <model_path> <model_path> <experiment_name> <model_path_step0> <chat_template> <num_iterations> <gpu_num>
```

### Arguments Explanation

| Argument | Description | Example |
| :--- | :--- | :--- |
| `model_path` | Path to the pre-trained base model. | `/models/llama-3-8b` |
| `experiment_name` | Custom name for the current training output. | `refo_run_01` |
| `model_path_step0` | Path to the initial seed model checkpoint. | `/models/seed_ckpt` |
| `chat_template` | Chat template used for data formatting. | `llama3` |
| `num_iterations` | Number of outer-loop training iterations. | `5` |
| `gpu_num` | Number of GPUs allocated for DeepSpeed/vLLM. | `8` |

---

## 🧠 Methodology Overview

The REFO framework employs a nested-loop architecture to recycle "waste" data and steer the model toward higher faithfulness:

1.  **Step 1 (Self-generated Preference Optimization):** The outer loop begins by generating a preference pair using a context-present prompt (winning response) and a context-absent prompt (losing response).
2.  **Step 2 (Iterative Relative Preference Optimization):** The inner loop compares responses from the *current* generation model against the *previous* generation model to optimize the relative difference in faithfulness.
3.  **Attention-DPO:**


## Data Setup
The `data` directory is currently empty. Please download the dataset from https://huggingface.co/datasets/chkwy/REFO_data and extract its contents into the `data` folder before running the scripts.