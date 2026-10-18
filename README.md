# SAID: Accelerating Diffusion-Based Language Models via Scaffold-Aware Iterative Decoding


[![Paper PDF](https://img.shields.io/badge/Paper-PDF-b31b1b)](https://arxiv.org/abs/2606.04974)
[![GitHub Code](https://img.shields.io/badge/GitHub-Code-181717?logo=github)](https://github.com/TH-AI-Lab-PKU/SAID)

SAID is a training-free two-stage framework that accelerates diffusion language models by decomposing generation into **interleaved-anchor scaffold construction** and **highly parallel detail completion**, maintaining competitive generation quality. SAID supports both full-sequence and block-wise generation.

We provide **token-level SAID** on LLaDA-8B-Instruct and **block-level SAID** on LLaDA 1.5. Both implementations use OpenCompass for evaluation. 

## Setup

### Environment

Run from the repository root:

```bash
conda create -n said python=3.10 -y
conda activate said

pip install torch==2.5.1
pip install transformers==4.49.0

cd SAID-v1/opencompass
pip install -e .
pip install setuptools==69.5.1
```

### Model Download

Download the two models used in our experiments:

```bash
hf download GSAI-ML/LLaDA-8B-Instruct
hf download GSAI-ML/LLaDA-1.5
```

> **Note:** Replace `path` in the following model configurations with the absolute path to your downloaded checkpoint.

| Model | Configuration files |
|---|---|
| LLaDA-8B-Instruct | `SAID-v1/opencompass/opencompass/configs/models/dllm/llada_instruct_8b.py` |
| LLaDA-8B-Instruct | `SAID-v1/opencompass/opencompass/configs/models/dllm/llada_instruct_said.py` |
| LLaDA 1.5 | `SAID-block/opencompass/opencompass/configs/models/dllm/llada_15_instruct_8b.py` |
| LLaDA 1.5 | `SAID-block/opencompass/opencompass/configs/models/dllm/llada_15_instruct_8b_said.py` |

## SAID-v1

Run the following commands from `SAID-v1/opencompass`.

### Run Baseline

```bash
python run.py examples/llada_instruct_gen_arcc_length512_block512.py
python run.py examples/llada_instruct_gen_gpqa_length64_block64_confidence.py
python run.py examples/llada_instruct_gen_gsm8k_length512_block512_confidence.py
python run.py examples/llada_instruct_gen_math_length512_block512_confidence.py
python run.py examples/llada_instruct_gen_mbpp_length256_block256_confidence.py
python run.py examples/llada_instruct_gen_mmlupro_length256_block256.py
```

### Run SAID

```bash
python run.py examples/llada_instruct_gsm8k.py
python run.py examples/llada_instruct_math.py
python run.py examples/llada_instruct_gpqa.py
python run.py examples/llada_instruct_arcc.py
python run.py examples/llada_instruct_mmlupro.py
```

> **MBPP note:** Set `num_transfer_hard` steps to `8` in `my_generate_said.py` before running:
> ```bash
> python run.py examples/llada_instruct_mbpp.py
> ```


## SAID-Block

Block-level SAID decodes successive blocks from left to right, constructing scaffold groups before completing the remaining details within each block.

### Prepare the Framework

Since both implementations use the **OpenCompass** framework, we do not include a duplicate copy in **SAID-block**. Simply copy the code files from `SAID-block` into the corresponding paths under `SAID-v1`.

### Run Baseline

```bash
python run.py examples/llada_1p5_gen_gpqa.py
python run.py examples/llada_1p5_gen_arcc.py
python run.py examples/llada_1p5_gen_mbpp.py
python run.py examples/llada_1p5_gen_math.py
```

### Run SAID

```bash
python run.py examples/llada_1p5_said_gpqa_16.py
python run.py examples/llada_1p5_said_arcc_16.py
python run.py examples/llada_1p5_said_mbpp_32.py
python run.py examples/llada_1p5_said_math_32.py
```

### Configurations

| Benchmark | Block length | Generation length | Baseline steps | SAID steps | `num_stages` | `rec_steps` |
|---|---:|---:|---:|---:|---:|---:|
| GPQA | 16 | 128 | 128 | 32 | 2 | 1 |
| ARC-C | 16 | 256 | 256 | 96 | 3 | 1 |
| MBPP | 32 | 256 | 256 | 136 | 2 | 1 |
| MATH | 32 | 256 | 256 | 128 | 2 | 1 |

## Ascend‑adapted

Our acceleration approach works effectively on **NVIDIA GPUs**. Furthermore, we provide the first‑of‑its‑kind native adaptation for **Ascend NPUs**, delivering full compatibility and optimized performance for LLM inference on Ascend hardware. If you are using Ascend NPUs, The following modifications are required:

For SAID-v1, find `SAID-v1/opencompass/opencompass/models/dllmsaid.py`. For SAID-block, find `SAID-block/opencompass/opencompass/models/dllmblock.py`, 
replace **auto** and **cuda** with **npu**。Such as SAID-v1：
```bash
import torch_npu
torch.cuda.synchronize() -->> torch.npu.synchronize()
model_kwargs: dict = dict(device_map='auto') -->> model_kwargs: dict = dict(device="npu")
```

