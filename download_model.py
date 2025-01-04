from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
from huggingface_hub import login

login(token="your_huggingface_token")
model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.3"
model =  AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)