import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def get_num_transfer_tokens(mask_index, steps): 
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    
    prompt_len = prompt.shape[1] 
    x_len = x.shape[1]

    # even = torch.arange(x_len, device=x.device) % 2 == 0
    # even_mask = torch.zeros_like(x, dtype=torch.bool)
    # even_mask[:, prompt_len:] = even[prompt_len:] 
    #even_index = (x == mask_id) & even_mask
    #num_transfer_tokens_even = get_num_transfer_tokens(even_index, steps)
    
    odd = torch.arange(x_len, device=x.device) % 2 == 1
    odd_mask = torch.zeros_like(x, dtype=torch.bool)
    odd_mask[:, prompt_len:] = odd[prompt_len:]
    odd_index = (x == mask_id) & odd_mask
    num_transfer_tokens_odd = get_num_transfer_tokens(odd_index, steps)



    for i in range(steps):
        mask_index = (x == mask_id) & odd_mask
        logits = model(x, attention_mask=attention_mask).logits 
        if logits_eos_inf:
             logits[:, :, 126081] = -torch.inf
            
        x0 = torch.argmax(logits, dim=-1) 
        if confidence_eos_eot_inf:
            logits[:, :, 126081] = logits[:, :, 126348] = -torch.inf
        
        p = F.softmax(logits, dim=-1) 
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) 
        
        x0 = torch.where(mask_index, x0, x) 
        confidence = torch.where(mask_index, x0_p, -np.inf) 
        
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device) 
        
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens_odd[j, i])
            transfer_index[j, select_index] = True
        x[transfer_index] = x0[transfer_index] 
    
    
    # odd = torch.arange(x_len, device=x.device) % 2 == 1
    # odd_mask = torch.zeros_like(x, dtype=torch.bool)
    # odd_mask[:, prompt_len:] = odd[prompt_len:]
    # odd_index = (x == mask_id) & odd_mask
    # num_transfer_tokens_odd = get_num_transfer_tokens(odd_index, steps=1)

    even = torch.arange(x_len, device=x.device) % 2 == 0
    even_mask = torch.zeros_like(x, dtype=torch.bool)
    even_mask[:, prompt_len:] = even[prompt_len:]
    
    even_index = (x == mask_id) & even_mask
    num_transfer_tokens_even = get_num_transfer_tokens(even_index, steps=1)


    
    mask_index = (x == mask_id) & even_mask
    logits = model(x, attention_mask=attention_mask).logits
    if logits_eos_inf:
        logits[:, :, 126081] = -torch.inf

    x0 = torch.argmax(logits, dim=-1)
    if confidence_eos_eot_inf:
        logits[:, :, 126081] = logits[:, :, 126348] = -torch.inf

    
    p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) 
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)
    
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens_even[j, 0])
        transfer_index[j, select_index] = True
    x[transfer_index] = x0[transfer_index]
    
    return x