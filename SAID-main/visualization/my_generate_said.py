import torch
import numpy as np
import torch.nn.functional as F
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
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate(model, prompt, tokenizer, attention_mask=None, steps=128, gen_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    
    prompt_len = prompt.shape[1] 
    x_len = x.shape[1]
    even = torch.arange(x_len, device=x.device) % 2 == 0
    even_mask = torch.zeros_like(x, dtype=torch.bool)
    even_mask[:, prompt_len:] = even[prompt_len:] 
    
    even_index = (x == mask_id) & even_mask
    num_transfer_tokens_even = get_num_transfer_tokens(even_index, steps)

    print_i = 0
 
    for i in range(steps):
        mask_index = (x == mask_id) & even_mask
        logits = model(x, attention_mask=attention_mask).logits 
        if logits_eos_inf:
             logits[:, :, 126081] = -torch.inf

        logits_with_noise = add_gumbel_noise(logits, temperature=0.0)
        
        x0 = torch.argmax(logits_with_noise, dim=-1) 
        if confidence_eos_eot_inf:
            logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

        p = F.softmax(logits, dim=-1) 
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) 
        
        x0 = torch.where(mask_index, x0, x) 
        confidence = torch.where(mask_index, x0_p, -np.inf) 
        
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device) 
        
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens_even[j, i])
            transfer_index[j, select_index] = True
        x[transfer_index] = x0[transfer_index]


        print_i += 1   
        generated_token_ids = x[0, prompt.shape[1]:]
        formatted_output = []
        for token_id in generated_token_ids:
            decoded_token = tokenizer.decode(token_id).replace("\n", " ").replace("<|eot_id|>", " ").replace("<|endoftext|>", " ")
            formatted_token = f"*{decoded_token}&"
            formatted_output.append(formatted_token)
        final_output = "".join(formatted_output).strip()
        print(f"{print_i}, {final_output}", file=open("sample_process.txt", "a"))
    
    
    odd = torch.arange(x_len, device=x.device) % 2 == 1
    odd_mask = torch.zeros_like(x, dtype=torch.bool)
    odd_mask[:, prompt_len:] = odd[prompt_len:]
    
    odd_index = (x == mask_id) & odd_mask
    num_transfer_tokens_odd = get_num_transfer_tokens(odd_index, steps=1)
    
    mask_index = (x == mask_id) & odd_mask
    logits = model(x, attention_mask=attention_mask).logits
    if logits_eos_inf:
        logits[:, :, 126081] = -torch.inf

    logits_with_noise = add_gumbel_noise(logits, temperature=1)
    
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if confidence_eos_eot_inf:
        logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

    
    p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) 
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)
    
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens_odd[j, 0])
        transfer_index[j, select_index] = True
    x[transfer_index] = x0[transfer_index]

    print_i += 1    
    generated_token_ids = x[0, prompt.shape[1]:]
    formatted_output = []
    for token_id in generated_token_ids:
        decoded_token = tokenizer.decode(token_id).replace("\n", " ").replace("<|eot_id|>", " ").replace("<|endoftext|>", " ")
        formatted_token = f"*{decoded_token}&"
        formatted_output.append(formatted_token)
    final_output = "".join(formatted_output).strip()
    print(f"{print_i}, {final_output}", file=open("sample_process.txt", "a"))
    
    return x

def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('/home/nvme04/worldf/wm_mem/model', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('/home/nvme04/worldf/wm_mem/model', trust_remote_code=True)

    prompt = "Explain what artificial intelligence is."

    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, tokenizer, steps=32, gen_length=64, cfg_scale=0., remasking='low_confidence')


if __name__ == '__main__':
    main()

