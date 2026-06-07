# SAID-Block

Block-based generation with GTR (Generation-then-Reconstruction) acceleration for LLaDA, integrated into OpenCompass evaluation.

## Background

Standard LLaDA generates all tokens simultaneously over many diffusion steps. This module improves throughput with two complementary ideas:

**Block-AR**: Decode left-to-right one block at a time (semi-autoregressive). Each block is fully resolved before the next starts, enabling KV-cache reuse and reducing interference between distant tokens.

**GTR**: Within each block, apply a checkerboard stage partition. "Generation" stages handle maximally-spaced positions with full step budgets (establishing global structure). The "reconstruction" stage fills the remaining ~50% of positions in as few as 1–2 steps — since they are surrounded by already-generated context, few steps suffice. This is **training-free**.

## Repository Structure

```
SAID-block/
├── generate_gtr.py          # GTR sampling (flat, no block-AR)
├── generate_block_gtr.py    # Block-AR + intra-block GTR
└── opencompass/
    ├── examples/            # OpenCompass eval configs
    ├── opencompass/
    │   ├── models/dllm.py   # LLaDAModel and LLaDAGTRModel wrappers
    │   └── configs/models/dllm/   # Model configs
    └── summarize_perf.py    # Parse [Perf] logs and print throughput table
```

## Generation Scripts

### `generate_gtr.py` — Flat GTR

Applies GTR across the entire generation length at once.

```python
from generate_gtr import generate_gtr

out = generate_gtr(
    model, input_ids, attention_mask,
    steps=128,
    gen_length=128,
    block_length=128,   # set equal to gen_length for flat (no block-AR)
    num_stages=3,       # K=3: 2 generation stages + 1 reconstruction stage
    rec_steps=2,        # reconstruction uses only 2 steps
)
```

### `generate_block_gtr.py` — Block-AR + Intra-Block GTR

Combines semi-autoregressive block decoding with GTR inside each block.

```python
from generate_block_gtr import generate_block_gtr

out = generate_block_gtr(
    model, input_ids, attention_mask,
    steps=256,
    gen_length=256,
    block_length=32,    # 8 blocks of 32 tokens each
    num_stages=2,       # 1 generation stage + 1 reconstruction stage per block
    rec_steps=1,
    confidence_eos_eot_inf=True,  # recommended for LLaDA 1.5
)
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `gen_length` | Total response tokens to generate |
| `block_length` | Tokens per AR block (`gen_length` must be divisible) |
| `steps` | Total denoising step budget (split evenly across blocks) |
| `num_stages` | GTR stages K per block. K=1 → vanilla LLaDA, K=2 → basic GTR, K=3 → extra sub-stage |
| `rec_steps` | Steps for the reconstruction stage; as low as 1 for maximum speedup |
| `diff_confidence_eos_eot_inf` | Mask EOS/EOT token confidence to prevent premature stopping (recommended for open-ended tasks) |

### Speedup intuition

With `num_stages=3, rec_steps=2, steps=128` per block:
- Stage 1 (~25% tokens): ~63 steps
- Stage 2 (~25% tokens): ~63 steps
- Stage 3 (~50% tokens): 2 steps

Total model calls = 128, but half the tokens are decoded in 2 calls instead of 64. To get wall-clock speedup, reduce total steps: `steps=64, rec_steps=1` halves model calls with comparable quality.

## OpenCompass Evaluation

### Model Classes (`opencompass/opencompass/models/dllm.py`)

| Class | Generation | Use for |
|-------|-----------|---------|
| `LLaDAModel` | Standard block-AR (`generate`) | Baseline |
| `LLaDAGTRModel` | Block-AR + GTR (`generate_gtr`) | Accelerated eval |

Both log per-batch and cumulative `[Perf]` / `[Perf-GTR]` throughput to stdout.

### Eval Configs (`opencompass/examples/`)

**LLaDA 1.5 with GTR**

| Config | Benchmark | Block | Length |
|--------|-----------|-------|--------|
| `llada_1p5_gtr_arcc_length512_block512.py` | ARC-Challenge | 512 | 512 |
| `llada_1p5_gtr_gpqa_length256_block16.py` | GPQA | 16 | 256 |
| `llada_1p5_gtr_gsm8k_length256_block16.py` | GSM8K | 16 | 256 |
| `llada_1p5_gtr_humaneval_length512_block32.py` | HumanEval | 32 | 512 |
| `llada_1p5_gtr_ifeval_length256_block16.py` | IFEval | 16 | 256 |
| `llada_1p5_gtr_math_length1024_block128.py` | MATH | 128 | 1024 |
| `llada_1p5_gtr_mbpp_length512_block32.py` | MBPP | 32 | 512 |

**LLaDA Instruct with GTR**

| Config | Benchmark | Block | Length |
|--------|-----------|-------|--------|
| `llada_instruct_gtr_arcc_length512_block512.py` | ARC-Challenge | 512 | 512 |
| `llada_instruct_gtr_gpqa_length128_block64.py` | GPQA | 64 | 128 |
| `llada_instruct_gtr_gsm8k_length512_block512.py` | GSM8K | 512 | 512 |
| `llada_instruct_gtr_humaneval_length512_block32.py` | HumanEval | 32 | 512 |
| `llada_instruct_gtr_mmlu_length256_block256.py` | MMLU | 256 | 256 |
| `llada_instruct_gtr_mmlupro_length256_block256.py` | MMLU-Pro | 256 | 256 |
| `llada_instruct_gen_gsm8k_length512_block512_confidence.py` | GSM8K (confidence EOS) | 512 | 512 |

### Running Evaluation

```bash
cd SAID-block/opencompass
python run.py examples/<config>.py
```

### Summarizing Throughput

After evaluation, parse `[Perf]` entries from logs:

```bash
python summarize_perf.py outputs/default/<run_timestamp>
```

Prints per-task and total throughput (samples, tokens, time, tokens/s), plus cross-model speedup comparison if multiple models were evaluated together.
