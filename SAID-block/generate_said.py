"""
SAID inference for LLaDA.

Adapted from the paper:
  "Generation Then Reconstruction: Accelerating Masked Autoregressive Models
   via Two-Stage Sampling"  (Yan et al.)

Core idea (adapted from 2D vision to 1D text):
  - Stage Partitioning: Divide the gen_length positions into K disjoint subsets
    using a 1D analogue of the checkerboard pattern:
      * Iteration k: positions where  pos mod 2^k == 2^(k-1)  go into subset U_k
      * The remaining positions (pos mod 2^k == 0) are carried to the next iteration
    This ensures earlier stages contain maximally-spaced positions (global structure),
    while later stages fill in the gaps (local detail).

  - Generation Stage (stages 1..K-1): Generate tokens slowly with many steps,
    establishing the "semantic scaffold". Steps decrease linearly from step_max to step_min.

  - Reconstruction Stage (stage K): Generate the remaining ~50% tokens in very few
    steps (as few as 1-2), since they are surrounded by already-generated context.

  - Within each stage the standard LLaDA masked-diffusion loop is used
    (predict all masks → remasking by confidence / random → repeat).

This is **training-free** — it only changes the sampling order of the existing LLaDA model.
"""

import torch
import numpy as np
import torch.nn.functional as F
import time
from transformers import AutoTokenizer, AutoModel


# ──────────────────────────────────────────────
# Utility functions (same as original generate.py)
# ──────────────────────────────────────────────

def add_gumbel_noise(logits, temperature):
    """Gumbel-Max sampling with float64 precision."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Pre-compute how many tokens to unmask at each step (linear schedule)."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


# ──────────────────────────────────────────────
# SAID Stage Partitioning  (Algorithm 1, 1D version)
# ──────────────────────────────────────────────

def said_stage_partition(gen_length, num_stages):
    """
    Partition positions [0, gen_length) into `num_stages` disjoint subsets
    using the 1D checkerboard hierarchy (Algorithm 1 from the SAID paper).

    In 1D the rule  (i+j) mod 2^k  reduces to  pos mod 2^k.

    Returns:
        stages: list of K sorted position lists, stages[0] is generated first.
    """
    R = set(range(gen_length))
    stack = []  # LIFO

    K = num_stages
    for k in range(1, K):
        power = 2 ** k
        half = 2 ** (k - 1)
        U_k = {pos for pos in R if pos % power == half}
        stack.append(sorted(U_k))
        R = R - U_k  # remaining: pos % power == 0

    stack.append(sorted(R))  # last remainder pushed on top

    # Pop in LIFO order → first popped = last pushed = coarsest grid
    stages = []
    while stack:
        stages.append(stack.pop())

    return stages


def compute_stage_steps(num_stages, total_gen_steps, rec_steps=1):
    """
    Stage-aware step scheduling.

    Generation stages (0 .. K-2): linearly decreasing steps from step_max to step_min,
    where total generation budget = total_gen_steps - rec_steps.
    Reconstruction stage (K-1): fixed rec_steps.

    Returns:
        list of int, one per stage.
    """
    K = num_stages
    if K == 1:
        return [total_gen_steps]

    gen_budget = total_gen_steps - rec_steps
    num_gen_stages = K - 1

    if num_gen_stages == 1:
        return [gen_budget, rec_steps]

    # Linearly decreasing: s_max, ..., s_min  over num_gen_stages stages
    # Total = num_gen_stages * (s_max + s_min) / 2 = gen_budget
    # We fix s_min = max(1, gen_budget // (num_gen_stages * 2)) to ensure minimum work
    s_min = max(1, gen_budget // (num_gen_stages * 2))
    s_max = max(s_min, (2 * gen_budget) // num_gen_stages - s_min)

    # Generate linearly spaced values, round to ints, adjust to match budget exactly
    raw = np.linspace(s_max, s_min, num_gen_stages)
    stage_steps = np.round(raw).astype(int)
    stage_steps = np.maximum(stage_steps, 1)

    # Adjust to hit budget exactly
    diff = gen_budget - stage_steps.sum()
    for i in range(abs(diff)):
        idx = i % num_gen_stages
        stage_steps[idx] += 1 if diff > 0 else -1
    stage_steps = np.maximum(stage_steps, 1)

    result = stage_steps.tolist() + [rec_steps]
    return result


# ──────────────────────────────────────────────
# Core: one intra-stage masked-diffusion loop
# ──────────────────────────────────────────────

def _run_stage(
    model, x, prompt_index, stage_positions, steps,
    attention_mask=None, temperature=0., cfg_scale=0.,
    remasking='low_confidence', mask_id=126336,
    logits_eos_inf=False, confidence_eos_eot_inf=False,
):
    """
    Run the standard LLaDA masked-diffusion loop, but ONLY unmask tokens
    whose absolute positions are in `stage_positions`.

    Args:
        x:               (B, L) current token sequence (may contain mask_id).
        prompt_index:     (B, L) bool, True for prompt positions.
        stage_positions:  set of absolute positions (in x) that this stage should unmask.
        steps:            number of denoising steps for this stage.
    Returns:
        x: updated sequence.
    """
    if steps == 0 or len(stage_positions) == 0:
        return x

    B, L = x.shape
    device = x.device

    # Build a boolean mask of positions this stage is responsible for
    stage_mask = torch.zeros(L, dtype=torch.bool, device=device)
    stage_pos_tensor = torch.tensor(sorted(stage_positions), dtype=torch.long, device=device)
    stage_mask[stage_pos_tensor] = True
    stage_mask = stage_mask.unsqueeze(0).expand(B, -1)  # (B, L)

    # Count how many tokens in this stage are still masked
    still_masked = (x == mask_id) & stage_mask  # (B, L)
    num_transfer_tokens = get_num_transfer_tokens(still_masked, steps)

    for i in range(steps):
        mask_index = (x == mask_id)

        # ── Forward pass (with optional CFG) ──
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            if attention_mask is not None:
                attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
            else:
                attention_mask_ = None
            logits = model(x_, attention_mask=attention_mask_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x, attention_mask=attention_mask).logits

        if logits_eos_inf:
            logits[:, :, 126081] = -torch.inf

        # ── Sample x0 ──
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L)

        if confidence_eos_eot_inf:
            logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

        # ── Confidence for remasking ──
        if remasking == 'low_confidence':
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
            )  # (B, L)
        elif remasking == 'random':
            x0_p = torch.rand(x0.shape, device=device)
        else:
            raise NotImplementedError(remasking)

        # Only consider positions belonging to THIS stage (ignore future stages)
        x0_p[~stage_mask] = -np.inf

        # Only fill in masked positions
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        # ── Top-k selection ──
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
        for j in range(B):
            k = num_transfer_tokens[j, i].item()
            if k > 0:
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True

        x[transfer_index] = x0[transfer_index]

    return x


# ──────────────────────────────────────────────
# Main entry: generate_said
# ──────────────────────────────────────────────

@torch.no_grad()
def generate_said(
    model, prompt, attention_mask=None,
    steps=128, gen_length=128, block_length=128,
    temperature=0., cfg_scale=0.,
    remasking='low_confidence', mask_id=126336,
    logits_eos_inf=False, confidence_eos_eot_inf=False,
    # ─── SAID-specific parameters ───
    num_stages=3,
    rec_steps=2,
):
    """
    SAID-accelerated generation for LLaDA.

    Compared to the original generate(), this adds:
      - num_stages:  K in the paper. Number of hierarchical stages.
                     Stage 1..K-1 = generation (slow), stage K = reconstruction (fast).
                     K=1 degrades to the original LLaDA sampling.
                     K=2 is the basic two-stage SAID.
                     K=3 (default) adds one extra sub-stage for early global structure.
      - rec_steps:   Number of denoising steps for the reconstruction stage (default 2).
                     Can be as low as 1 for maximum speedup.

    All other parameters are identical to the original generate().

    Speedup comes from:
      1. The reconstruction stage (covering ~50% of tokens) uses very few steps.
      2. Earlier generation stages work on fewer, maximally-spaced positions,
         so each model call is still full-sequence but more tokens get finalized early.
    """
    B = prompt.shape[0]
    device = prompt.device

    # ── Initialize: prompt + [MASK] * gen_length ──
    x = torch.full((B, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((B, gen_length), dtype=attention_mask.dtype, device=device)
        ], dim=-1)

    prompt_index = (x != mask_id)
    prompt_len = prompt.shape[1]

    # ── Block-level loop (semi-autoregressive, same as original) ──
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length
        current_block_length = block_end - block_start

        # ── SAID stage partitioning within this block ──
        # Positions are relative to the block; convert to absolute later
        stages_rel = said_stage_partition(current_block_length, num_stages)
        stage_steps_list = compute_stage_steps(num_stages, steps // num_blocks, rec_steps)

        for stage_idx, (rel_positions, stage_steps) in enumerate(
            zip(stages_rel, stage_steps_list)
        ):
            # Convert relative block positions → absolute positions in x
            abs_positions = set(block_start + p for p in rel_positions)

            is_rec = (stage_idx == len(stages_rel) - 1)

            x = _run_stage(
                model, x, prompt_index, abs_positions, stage_steps,
                attention_mask=attention_mask,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                mask_id=mask_id,
                logits_eos_inf=logits_eos_inf,
                confidence_eos_eot_inf=confidence_eos_eot_inf,
            )

    return x


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────

def main():
    device = 'cuda'

    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-1.5', trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-1.5', trust_remote_code=True
    )

    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    assert tokenizer.pad_token_id != 126336

    prompts = [
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
        "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
        "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?",
    ]

    messages = [{"role": "user", "content": p} for p in prompts]
    prompts_str = [
        tokenizer.apply_chat_template([m], add_generation_prompt=True, tokenize=False)
        for m in messages
    ]

    encoded = tokenizer(prompts_str, add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    gen_length = 128
    block_length = 128
    total_steps = 128

    print("=" * 60)
    print("Original LLaDA sampling")
    print("=" * 60)
    from generate import generate as generate_original

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_orig = generate_original(
        model, input_ids, attention_mask,
        steps=total_steps, gen_length=gen_length, block_length=block_length,
        temperature=0., cfg_scale=0., remasking='low_confidence',
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed_orig = t1 - t0
    total_tokens_orig = input_ids.shape[0] * gen_length
    print(f'[Perf-Original] batch={input_ids.shape[0]}, gen_length={gen_length}, '
          f'total_tokens={total_tokens_orig}, '
          f'time={elapsed_orig:.3f}s, '
          f'tokens/s={total_tokens_orig/elapsed_orig:.2f}, '
          f'latency/sample={elapsed_orig/input_ids.shape[0]:.3f}s')

    for o in tokenizer.batch_decode(out_orig[:, input_ids.shape[1]:], skip_special_tokens=True):
        print(o)
        print("-" * 50)

    # ── SAID: same total steps budget, but reconstruction stage uses only 2 steps ──
    # With num_stages=3, rec_steps=2:
    #   Stage 1 (~25% tokens): ~63 steps  (slow, global structure)
    #   Stage 2 (~25% tokens): ~63 steps  (slow, fill gaps)
    #   Stage 3 (~50% tokens):   2 steps  (fast reconstruction)
    # Total model calls ≈ 63 + 63 + 2 = 128  (same budget, but 50% tokens done in 2 calls)
    #
    # For actual speedup, reduce total_steps:
    #   e.g. total_steps=64, num_stages=3, rec_steps=2
    #   → ~31 + ~31 + 2 = 64 model calls (vs original 128), ~2x speedup

    print("\n" + "=" * 60)
    print("SAID sampling (same budget, num_stages=3, rec_steps=2)")
    print("=" * 60)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_said = generate_said(
        model, input_ids, attention_mask,
        steps=total_steps, gen_length=gen_length, block_length=block_length,
        temperature=0., cfg_scale=0., remasking='low_confidence',
        num_stages=3, rec_steps=2,
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed_said = t1 - t0
    print(f'[Perf-SAID] batch={input_ids.shape[0]}, gen_length={gen_length}, '
          f'total_tokens={total_tokens_orig}, '
          f'time={elapsed_said:.3f}s, '
          f'tokens/s={total_tokens_orig/elapsed_said:.2f}, '
          f'latency/sample={elapsed_said/input_ids.shape[0]:.3f}s, '
          f'speedup={elapsed_orig/elapsed_said:.2f}x')

    for o in tokenizer.batch_decode(out_said[:, input_ids.shape[1]:], skip_special_tokens=True):
        print(o)
        print("-" * 50)

    # ── SAID with actual speedup: fewer total steps ──
    print("\n" + "=" * 60)
    print("SAID sampling (speedup mode: steps=64, num_stages=3, rec_steps=1)")
    print("=" * 60)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_said_fast = generate_said(
        model, input_ids, attention_mask,
        steps=64, gen_length=gen_length, block_length=block_length,
        temperature=0., cfg_scale=0., remasking='low_confidence',
        num_stages=3, rec_steps=1,
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed_said_fast = t1 - t0
    print(f'[Perf-SAID-Fast] batch={input_ids.shape[0]}, gen_length={gen_length}, '
          f'total_tokens={total_tokens_orig}, '
          f'time={elapsed_said_fast:.3f}s, '
          f'tokens/s={total_tokens_orig/elapsed_said_fast:.2f}, '
          f'latency/sample={elapsed_said_fast/input_ids.shape[0]:.3f}s, '
          f'speedup={elapsed_orig/elapsed_said_fast:.2f}x')

    for o in tokenizer.batch_decode(out_said_fast[:, input_ids.shape[1]:], skip_special_tokens=True):
        print(o)
        print("-" * 50)

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"{'Method':<25} {'Time(s)':<10} {'Tokens/s':<12} {'Speedup':<10}")
    print(f"{'Original LLaDA':<25} {elapsed_orig:<10.3f} {total_tokens_orig/elapsed_orig:<12.2f} {'1.00x':<10}")
    print(f"{'SAID (same budget)':<25} {elapsed_said:<10.3f} {total_tokens_orig/elapsed_said:<12.2f} {elapsed_orig/elapsed_said:<10.2f}x")
    print(f"{'SAID (fast, steps=64)':<25} {elapsed_said_fast:<10.3f} {total_tokens_orig/elapsed_said_fast:<12.2f} {elapsed_orig/elapsed_said_fast:<10.2f}x")
    print("=" * 60)


if __name__ == '__main__':
    main()
