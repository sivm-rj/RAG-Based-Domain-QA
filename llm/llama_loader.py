
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_llama():
    name = "meta-llama/Llama-2-7b-chat-hf"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map="auto"
    )
    return tok, model
