import sys
import os
sys.path.append(os.getcwd())

import torch
import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel

def load_model(
        base_model_name_or_path,lora_config_dir = "None"
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path,torch_dtype=torch.float16)
    if lora_config_dir != "None":
        model = PeftModel.from_pretrained(model, model_id = lora_config_dir)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "left"
    return model,tokenizer


def chat(
        prompt,model,tokenizer,device,max_new_tokens = 256
):

    """ get prompt tensor"""
    input = tokenizer(prompt, return_tensors="pt").to(device)
    model.to(device)

    """ inference """
    generated_ids = model.generate(**input, 
                                   max_new_tokens = max_new_tokens, 
                                   do_sample = False)
    
    """ analysis results """
    generate_language = tokenizer.batch_decode(generated_ids)
    print("generate language:\n",generate_language[0])
    return generate_language[0]

def main(args):
    """ device """
    device = args.device if torch.cuda.is_available() else "cpu"

    """ load model """
    model,tokenizer = load_model(base_model_name_or_path = args.base_model_name_or_path,
                                 lora_config_dir = args.lora_config_dir)
    
    """ chat with model """
    response = chat(prompt = args.prompt,
         model = model,
         tokenizer = tokenizer,
         device = device,
         max_new_tokens = args.max_new_tokens)
    
if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",
                        type = str,
                        default = "your prompt")
    
    parser.add_argument("--base_model_name_or_path",
                        type = str,
                        default = "mistralai/Mistral-7B-Instruct-v0.3")
    
    parser.add_argument("--lora_config_dir",
                        type = str,
                        default = "saved_model/finetune")
    
    parser.add_argument("--max_new_tokens",
                        type = int,
                        default = 256)
    
    parser.add_argument("--device",
                        type = str,
                        default = "cuda")
    
    args = parser.parse_args()
    main(args)
    

    
    

