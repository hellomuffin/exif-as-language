
    
    
from collections import defaultdict
import torch

def num_dict(def_value = 0):
    def def_num_value():
        return def_value
    return defaultdict(def_num_value)

def str_dict(def_value = ""):
    def def_str_value():
        return def_value
    return defaultdict(def_str_value)

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for _,p in model.named_parameters(): 
        p.data = p.data.float() 
        if type(p.grad)!=type(None): p.grad.data = p.grad.data.float()
        
        
def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 =  logit_scale * x1 @ x2.t()
    logits_per_x2 =  logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2