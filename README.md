# SAID: Scaffold-Aware Iterative Decoding

**Accelerating Diffusion-Based Language Models via Scaffold-Aware Iterative Decoding**

SAID is a training-free acceleration method for masked diffusion language models (e.g., LLaDA). The core idea is a two-stage decode: first generate a sparse "scaffold" of tokens at even positions, then use the scaffold as context to rapidly reconstruct the remaining odd-position tokens — splitting reconstruction into high-confidence (1 step) and low-confidence (3 steps) subgroups based on a confidence threshold.

## Method

Standard LLaDA iteratively unmasks all positions over many diffusion steps. SAID splits generation into two stages:

1. **Scaffold stage**: Run standard masked diffusion on even-indexed positions for `steps` iterations to produce a set of high-quality scaffold tokens.
2. **Reconstruction stage**: Given the scaffold, fill in odd-indexed positions in one forward pass, then split by confidence:
   - **Easy tokens** (confidence ≥ `CONF_THRESH`): resolved in 1 step.
   - **Hard tokens** (confidence < `CONF_THRESH`): refined over 3 additional steps.


## Repository Structure

```
SAID/
├── SAID-v1/                   # Core SAID method + evaluation
│   ├── generate.py              # Standard LLaDA block-AR generation (baseline)
│   ├── my_generate_said.py    # SAID generation (scaffold + confidence reconstruction)
│   ├── eval.sh                  # Evaluation commands (baseline vs. SAID)
│   ├── opencompass/
│   │   ├── examples/            # SAID eval configs (our method)
│   ├── others/                  # Experimental generation variants
│   │   ├── generate_random.py          # Random-order generation ablation
│   │   ├── my_generate_said_duihuan.py  # SAID dialogue variant
│   │   └── no_checkboard.py            # Non-checkerboard ablation
│   └── visualization/           # Generation process visualization
│       ├── my_generate_said.py
│       ├── visualization_paper.py
│       ├── visualization_zhihu.py
│       └── html_to_png.py
│
└── SAID-block/                  # Block-AR + SAID acceleration variant
    ├── generate_said.py          # SAID sampling
    └── opencompass/
        ├── examples/            # OpenCompass eval configs
        ├── opencompass/
        │   ├── models/dllm.py   # LLaDAModel and LLaDASAIDModel (SAID) wrappers
        │   └── configs/models/dllm/
        └── summarize_perf.py    # Parse [Perf] logs and print throughput table
```

---

## SAID-v1

### Setup

```bash
cd SAID-v1/opencompass
pip install -e .
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct
```

Update the model path in `opencompass/opencompass/configs/models/dllm/llada_instruct_8b.py`.

### Run Baseline (standard block-AR)

```bash
cd SAID-v1/opencompass
python run.py examples/llada_instruct_gen_arcc_length512_block512.py
python run.py examples/llada_instruct_gen_gpqa_length64_block64_confidence.py
python run.py examples/llada_instruct_gen_gsm8k_length512_block512_confidence.py
python run.py examples/llada_instruct_gen_math_length512_block512_confidence.py
python run.py examples/llada_instruct_gen_mbpp_length256_block256_confidence.py
python run.py examples/llada_instruct_gen_mmlupro_length256_block256.py
```

### Run SAID (our method)

```bash
cd SAID-v1/opencompass
python run.py examples/llada_instruct_gsm8k.py
python run.py examples/llada_instruct_math.py
python run.py examples/llada_instruct_gpqa.py
python run.py examples/llada_instruct_arcc.py
python run.py examples/llada_instruct_mmlupro.py
```

> **MBPP note**: set `num_transfer_hard` steps to 8 in `my_generate_said.py` before running:
> ```bash
> python run.py examples/llada_instruct_mbpp.py
> ```

### Visualization

Scripts in `SAID-v1/visualization/` animate the generation process step by step, producing HTML or PNG figures for papers or blog posts.

```bash
cd SAID-v1/visualization
python my_generate_said.py
python visualization_paper.py
python html_to_png.py
```

---

## SAID-Block: Block-AR + SAID

An alternative acceleration approach combining semi-autoregressive block decoding with hierarchical checkerboard sampling.

**Block-AR**: Decode left-to-right one block at a time. Each block is fully resolved before the next starts, enabling KV-cache reuse.

**SAID**: Within each block, apply a checkerboard stage partition. Generation stages handle maximally-spaced positions with full step budgets. The reconstruction stage fills the remaining ~50% of positions in as few as 1–2 steps, since they are surrounded by already-generated context.

### Generation Script

**`generate_said.py`** — Block-AR + intra-block SAID. `block_length` controls the AR granularity; set `block_length=gen_length` for flat (no block-AR) decoding.

```python
from generate_said import generate_said

# Block-AR + intra-block SAID (block_length < gen_length)
out = generate_said(
    model, input_ids, attention_mask,
    steps=128, gen_length=256, block_length=32,
    num_stages=3,   # K=3: 2 generation stages + 1 reconstruction stage
    rec_steps=2,
)

# Flat SAID, no block-AR (block_length == gen_length)
out = generate_said(
    model, input_ids, attention_mask,
    steps=128, gen_length=128, block_length=128,
    num_stages=3, rec_steps=2,
)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `block_length` | Tokens per AR block (`gen_length` must be divisible) |
| `num_stages` | SAID stages K. K=1 → vanilla LLaDA, K=2 → basic SAID, K=3 → extra sub-stage |
| `rec_steps` | Steps for the reconstruction stage; as low as 1 for maximum speedup |
| `diff_confidence_eos_eot_inf` | Mask EOS/EOT confidence to prevent premature stopping |

**Speedup intuition**: with `num_stages=3, rec_steps=2, steps=128` — stages use ~63 / ~63 / 2 steps for 25% / 25% / 50% of tokens. To get wall-clock speedup, reduce total steps: `steps=64, rec_steps=1` halves model calls with comparable quality.

### OpenCompass Evaluation

Model wrappers in `SAID-block/opencompass/opencompass/models/dllm.py`:

| Class | Generation |
|-------|-----------|
| `LLaDAModel` | Standard block-AR (baseline) |
| `LLaDASAIDModel` | Block-AR + SAID (accelerated) |

**LLaDA 1.5 SAID configs**

| Config | Benchmark | Block | Length |
|--------|-----------|-------|--------|
| `llada_1p5_said_arcc_length512_block512.py` | ARC-Challenge | 512 | 512 |
| `llada_1p5_said_gpqa_length256_block16.py` | GPQA | 16 | 256 |
| `llada_1p5_said_gsm8k_length256_block16.py` | GSM8K | 16 | 256 |
| `llada_1p5_said_humaneval_length512_block32.py` | HumanEval | 32 | 512 |
| `llada_1p5_said_ifeval_length256_block16.py` | IFEval | 16 | 256 |
| `llada_1p5_said_math_length1024_block128.py` | MATH | 128 | 1024 |
| `llada_1p5_said_mbpp_length512_block32.py` | MBPP | 32 | 512 |

**LLaDA 1.5 baseline configs**

| Config | Benchmark | Block | Length |
|--------|-----------|-------|--------|
| `llada_1p5_gen_arcc_length512_block32.py` | ARC-Challenge | 32 | 512 |
| `llada_1p5_gen_arcc_length512_block64.py` | ARC-Challenge | 64 | 512 |
| `llada_1p5_gen_gpqa_length256_block16.py` | GPQA | 16 | 256 |
| `llada_1p5_gen_gpqa_length256_block16_server.py` | GPQA (server) | 16 | 256 |
| `llada_1p5_gen_mbpp_length512_block32_confidence.py` | MBPP (confidence EOS) | 32 | 512 |

```bash
cd SAID-block/opencompass
python run.py examples/<config>.py

# Summarize throughput from logs
python summarize_perf.py outputs/default/<run_timestamp>
```


## Ascend‑adapted

The project code is implemented based on **NVIDIA**, with concurrent support for **Huawei Ascend**. The following modifications are required:

For SAID-v1, find `SAID-v1/opencompass/opencompass/models/dllmsaid.py`. For SAID-block, find `SAID-block/opencompass/opencompass/models/dllm.py`, 
replace `auto` and `cuda` with `npu`。Such as SAID-v1：





