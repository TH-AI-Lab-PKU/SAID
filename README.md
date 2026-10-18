# SAID: Scaffold-Aware Iterative Decoding

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-b31b1b)](https://arxiv.org/abs/2606.04974)

SAID is a training-free two-stage framework that accelerates diffusion language models by decomposing generation into **interleaved-anchor scaffold construction** and **highly parallel detail completion**, maintaining competitive generation quality.

We provide **token-level SAID** on LLaDA-8B-Instruct and **block-level SAID** on LLaDA 1.5. Both implementations use OpenCompass for evaluation. The full framework is included in `SAID-v1`, while `SAID-block` provides the block-specific code and configurations.

## Setup

### Environment

Run from the repository root:

```bash
conda create -n said python=3.10 -y
conda activate said

# PyTorch 2.5.1 with CUDA 12.1
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.49.0

cd SAID-v1/opencompass
pip install -e .
pip install setuptools==69.5.1
cd ../..
```

### Model Download

Download the two models used in our experiments:

```bash
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir models/LLaDA-8B-Instruct
huggingface-cli download GSAI-ML/LLaDA-1.5 --local-dir models/LLaDA-1.5
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

### Visualization

Scripts in `SAID-v1/visualization/` visualize the generation process and produce HTML or PNG outputs. Update the model and output paths before running.

```bash
# From the repository root
cd SAID-v1/visualization
python my_generate_said.py
python visualization_paper.py
python html_to_png.py
```

## SAID-Block

Block-level SAID decodes successive blocks from left to right, constructing scaffold groups before completing the remaining details within each block.

### Prepare the Framework

Copy the OpenCompass framework from `SAID-v1/opencompass` into `SAID-block/opencompass`, preserving the block-specific files already present, and copy `SAID-v1/generate.py` into `SAID-block/`. Install the resulting framework with `pip install -e .` from `SAID-block/opencompass`; run the commands below from that directory.

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

> **File-name update:** In the current block implementation, the wrapper is `SAID-block/opencompass/opencompass/models/dllmblock.py`. References to the block-side `dllm.py` below refer to this file.

Our acceleration approach works effectively on **NVIDIA GPUs**. Furthermore, we provide the first‑of‑its‑kind native adaptation for **Ascend NPUs**, delivering full compatibility and optimized performance for LLM inference on Ascend hardware. If you are using Ascend NPUs, The following modifications are required:

For SAID-v1, find `SAID-v1/opencompass/opencompass/models/dllmsaid.py`. For SAID-block, find `SAID-block/opencompass/opencompass/models/dllm.py`, 
replace **auto** and **cuda** with **npu**。Such as SAID-v1：
```bash
import torch_npu
torch.cuda.synchronize() -->> torch.npu.synchronize()
model_kwargs: dict = dict(device_map='auto') -->> model_kwargs: dict = dict(device="npu")
```

## Acknowledgments

This project builds on [OpenCompass](https://github.com/open-compass/opencompass), [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct), and [LLaDA 1.5](https://huggingface.co/GSAI-ML/LLaDA-1.5). We thank their authors for making the code and models available.
