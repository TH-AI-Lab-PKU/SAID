#my: SAID+block
import torch
import numpy as np
import torch.nn.functional as F
import time
from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


def said_stage_partition(gen_length, num_stages):

    R = set(range(gen_length))
    stack = [] 

    K = num_stages
    for k in range(1, K):
        power = 2 ** k
        half = 2 ** (k - 1)
        U_k = {pos for pos in R if pos % power == half}
        stack.append(sorted(U_k))
        R = R - U_k  

    stack.append(sorted(R)) 
    stages = []
    while stack:
        stages.append(stack.pop())

    return stages


def compute_stage_steps(num_stages, total_gen_steps, rec_steps=1):

    K = num_stages
    if K == 1:
        return [total_gen_steps]

    gen_budget = total_gen_steps - rec_steps
    num_gen_stages = K - 1

    if num_gen_stages == 1:
        return [gen_budget, rec_steps]
    s_min = max(1, gen_budget // (num_gen_stages * 2))
    s_max = max(s_min, (2 * gen_budget) // num_gen_stages - s_min)

    raw = np.linspace(s_max, s_min, num_gen_stages)
    stage_steps = np.round(raw).astype(int)
    stage_steps = np.maximum(stage_steps, 1)

    diff = gen_budget - stage_steps.sum()
    for i in range(abs(diff)):
        idx = i % num_gen_stages
        stage_steps[idx] += 1 if diff > 0 else -1
    stage_steps = np.maximum(stage_steps, 1)

    result = stage_steps.tolist() + [rec_steps]
    return result


def _run_stage(
    model, x, prompt_index, stage_positions, steps,
    attention_mask=None, temperature=0., cfg_scale=0.,
    remasking='low_confidence', mask_id=126336,
    logits_eos_inf=False, confidence_eos_eot_inf=False,
):

    if steps == 0 or len(stage_positions) == 0:
        return x

    B, L = x.shape
    device = x.device

    stage_mask = torch.zeros(L, dtype=torch.bool, device=device)
    stage_pos_tensor = torch.tensor(sorted(stage_positions), dtype=torch.long, device=device)
    stage_mask[stage_pos_tensor] = True
    stage_mask = stage_mask.unsqueeze(0).expand(B, -1) 

    still_masked = (x == mask_id) & stage_mask  
    num_transfer_tokens = get_num_transfer_tokens(still_masked, steps)

    for i in range(steps):
        mask_index = (x == mask_id)

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

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1) 

        if confidence_eos_eot_inf:
            logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

        if remasking == 'low_confidence':
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
            )  # (B, L)
        elif remasking == 'random':
            x0_p = torch.rand(x0.shape, device=device)
        else:
            raise NotImplementedError(remasking)

        x0_p[~stage_mask] = -np.inf

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
        for j in range(B):
            k = num_transfer_tokens[j, i].item()
            if k > 0:
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True

        x[transfer_index] = x0[transfer_index]

    return x


@torch.no_grad()
def generate_said(
    model, prompt, attention_mask=None,
    steps=128, gen_length=128, block_length=128,
    temperature=0., cfg_scale=0.,
    remasking='low_confidence', mask_id=126336,
    logits_eos_inf=False, confidence_eos_eot_inf=False,
    num_stages=3,
    rec_steps=2,
):

    B = prompt.shape[0]
    device = prompt.device

    x = torch.full((B, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((B, gen_length), dtype=attention_mask.dtype, device=device)
        ], dim=-1)

    prompt_index = (x != mask_id)
    prompt_len = prompt.shape[1]

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length
        current_block_length = block_end - block_start

        stages_rel = said_stage_partition(current_block_length, num_stages)
        stage_steps_list = compute_stage_steps(num_stages, steps // num_blocks, rec_steps)

        for stage_idx, (rel_positions, stage_steps) in enumerate(
            zip(stages_rel, stage_steps_list)
        ):
            abs_positions = set(block_start + p for p in rel_positions)


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
