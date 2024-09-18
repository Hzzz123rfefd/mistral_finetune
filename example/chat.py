import sys
import os
sys.path.append(os.getcwd())

import torch
import argparse

from llmai.model import Mistral

def main(args):
    """ device """
    device = args.device if torch.cuda.is_available() else "cpu"

    """ load model """
    model = Mistral(base_model_name_or_path = args.base_model_name_or_path,
                                    peft_config_dir = args.lora_config_dir)
    
    """ chat with model """
    model.chat_with_prompt(prompt = args.prompt,
                                                max_new_tokens = args.max_new_tokens,
                                                device = args.device)
    
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
    

    
    

