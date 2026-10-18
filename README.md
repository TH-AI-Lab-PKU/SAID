# SAID: Scaffold-Aware Iterative Decoding

**A training-free two-stage acceleration framework for diffusion language models.**

📄 [Paper: Accelerating Diffusion-Based Language Models via Scaffold-Aware Iterative Decoding](https://arxiv.org/abs/2606.04974)

SAID decomposes generation into **interleaved-anchor scaffold construction** and **highly parallel detail completion**. Non-adjacent anchors distributed throughout the sequence establish a global semantic scaffold, which guides the completion of the remaining tokens.

Within token-level detail completion, **Confidence-Hierarchical Layered Generation (CHLG)** allocates computation according to predictive confidence: high-confidence tokens are committed together, while uncertain positions are completed iteratively using the enriched context. The decoding budgets are configurable rather than fixed to a universal number of steps.

This repository provides:

- **SAID-v1:** token-level SAID on **LLaDA-8B-Instruct**.
- **SAID-Block:** block-level SAID on **LLaDA 1.5**, applying scaffold construction and detail completion within successive blocks.

Both implementations use **OpenCompass** for evaluation. The full evaluation framework is included only in `SAID-v1`; `SAID-block` contains the additional generation code, model wrapper, and configurations needed for block-level evaluation.

## Repository Structure

```text
SAID-main/
├── README.md
├── SAID-v1/
│   ├── generate.py                    # Baseline LLaDA generation
│   ├── my_generate_said.py            # Token-level SAID with CHLG
│   ├── eval.sh
│   ├── opencompass/                   # Shared evaluation framework
│   │   ├── run.py
│   │   ├── examples/                  # Baseline and token-level SAID configs
│   │   └── opencompass/
│   │       ├── models/
│   │       │   ├── dllm.py            # Baseline wrapper
│   │       │   └── dllmsaid.py        # Token-level SAID wrapper
│   │       └── configs/models/dllm/
│   ├── others/                       # Anchor-selection ablations
│   │   ├── generate_random.py
│   │   ├── my_generate_said_duihuan.py
│   │   └── no_checkboard.py
│   └── visualization/
│       ├── my_generate_said.py
│       ├── visualization_paper.py
│       ├── visualization_zhihu.py
│       └── html_to_png.py
└── SAID-block/
    ├── generate_said.py               # Block-level SAID generation
    └── opencompass/                   # Block-specific files only
        ├── examples/                  # Four baseline and four SAID configs
        └── opencompass/
            ├── models/
            │   ├── __init__.py
            │   └── dllmblock.py       # Baseline and block-level SAID wrappers
            └── configs/models/dllm/
                ├── llada_15_instruct_8b.py
                └── llada_15_instruct_8b_said.py
```

## Setup

### 1. Create the environment

The following commands use Bash on Linux. Start from the repository root.

```bash
conda create -n said python=3.10 -y
conda activate said

# CUDA 12.1 build of PyTorch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.49.0

cd SAID-v1/opencompass
pip install -e .
pip install setuptools==69.5.1
cd ../..
```

Use a compatible NVIDIA driver for the CUDA build above. Other PyTorch 2.5.1 CUDA builds are available in the [official installation instructions](https://pytorch.org/get-started/previous-versions/).

If dependency installation changes the requested package versions, restore them before running experiments:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.49.0 setuptools==69.5.1
```

### 2. Download the models

Download the official checkpoints from Hugging Face:

- [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
- [LLaDA 1.5](https://huggingface.co/GSAI-ML/LLaDA-1.5)

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='GSAI-ML/LLaDA-8B-Instruct', local_dir='models/LLaDA-8B-Instruct')"

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='GSAI-ML/LLaDA-1.5', local_dir='models/LLaDA-1.5')"
```

> **Model paths:** Replace the existing `path` value in each configuration below with the absolute path to your downloaded checkpoint.

| Implementation | Model configuration files |
|---|---|
| SAID-v1 | `SAID-v1/opencompass/opencompass/configs/models/dllm/llada_instruct_8b.py` |
| SAID-v1 | `SAID-v1/opencompass/opencompass/configs/models/dllm/llada_instruct_said.py` |
| SAID-Block | `SAID-block/opencompass/opencompass/configs/models/dllm/llada_15_instruct_8b.py` |
| SAID-Block | `SAID-block/opencompass/opencompass/configs/models/dllm/llada_15_instruct_8b_said.py` |

For example:

```python
path='/absolute/path/to/SAID-main/models/LLaDA-8B-Instruct'
```

Prepare any dataset files or access credentials required by the corresponding OpenCompass dataset configurations before evaluation.

## SAID-v1

Token-level SAID builds an interleaved scaffold over the response and then completes the remaining positions with confidence-guided computation allocation.

Run the following commands from `SAID-v1/opencompass`.

### Run the baseline

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

**MBPP:** In `SAID-v1/my_generate_said.py`, change the low-confidence decoding budget to `8` before running MBPP:

```python
num_transfer_hard = get_num_transfer_tokens(hard_mask, steps=8)
```

Then run:

```bash
python run.py examples/llada_instruct_mbpp.py
```

Restore this budget to `3` when using the other supplied token-level configurations. The current confidence threshold is `CONF_THRESH = 0.8`.

### Visualization

The scripts in `SAID-v1/visualization/` visualize the decoding process and produce HTML or PNG outputs. Adjust the model and output paths in the scripts before use.

```bash
# From the repository root
cd SAID-v1/visualization
python my_generate_said.py
python visualization_paper.py
python html_to_png.py
```

## SAID-Block

Block-level SAID processes response blocks from left to right. Within each block, it first resolves distributed scaffold groups and then completes the final detail group. Completed groups and earlier blocks remain available as context.

### Prepare the shared framework

`SAID-block/opencompass` contains only block-specific files. Populate the missing framework files from `SAID-v1`, retaining the supplied block-specific files.

Run once from the repository root:

```bash
# Copy missing framework files without overwriting block-specific files.
cp -rn SAID-v1/opencompass/. SAID-block/opencompass/

# The block wrapper also imports the baseline generation function.
cp -n SAID-v1/generate.py SAID-block/generate.py

# Activate the block-level OpenCompass checkout.
cd SAID-block/opencompass
pip install -e .
pip install transformers==4.49.0 setuptools==69.5.1
```

> Keep the block-specific `opencompass/models/__init__.py`, which imports `LLaDAModel` and `LLaDASAIDModel` from `dllmblock.py`. Do not import both `dllm.py` and `dllmblock.py` into the same model registry: both define `LLaDAModel`.

The commands below run from `SAID-block/opencompass`.

### Run the baseline

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

The following values are taken from the supplied example configurations.

| Benchmark | Block length | Generation length | Baseline steps | SAID steps | `num_stages` | `rec_steps` |
|---|---:|---:|---:|---:|---:|---:|
| GPQA | 16 | 128 | 128 | 32 | 2 | 1 |
| ARC-C | 16 | 256 | 256 | 96 | 3 | 1 |
| MBPP | 32 | 256 | 256 | 136 | 2 | 1 |
| MATH | 32 | 256 | 256 | 128 | 2 | 1 |

| Parameter | Meaning |
|---|---|
| `gen_length` | Number of response tokens to generate. |
| `gen_blocksize` | Number of tokens per block; must divide `gen_length`. |
| `gen_steps` | Total allocated decoding budget, divided across blocks. |
| `num_stages` | Number of groups within each block, including the final detail group. |
| `rec_steps` | Step budget for the final detail group in each block. |
| `batch_size`, `batch_size_` | Evaluation batching settings; both are `1` in the supplied configs. |

For `B = gen_length / gen_blocksize` blocks, each block receives `gen_steps // B` steps. The final detail group receives `rec_steps`; the remaining budget is assigned to the scaffold groups. Keep enough steps for every nonempty group.

The example configurations use multiple evaluation workers. Adjust `num_worker` and `max_num_workers` to match the available hardware.

### Switch back to SAID-v1

When returning to token-level evaluation, reactivate its framework checkout:

```bash
# From the repository root
cd SAID-v1/opencompass
pip install -e .
```

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
