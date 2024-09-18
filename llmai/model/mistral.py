import json
import os
import sys
sys.path.append(os.getcwd())
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType,PeftModel
from torch import nn

class Mistral(nn.Module):
    def __init__(self,base_model_name_or_path,peft_config_dir = None):
        super(Mistral, self).__init__()
        self.model_name_or_path = base_model_name_or_path
        self.peft_config_dir = peft_config_dir
        if self.peft_config_dir == "None":
            self.peft_config_dir = None

        # load base model
        self.base_model =  AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16
        )
        if self.peft_config_dir:
            self.tokenizer = AutoTokenizer.from_pretrained(self.peft_config_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.tokenizer.padding_side = "left"
        
        # load peft config
        if self.peft_config_dir == None:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode = False,
                r=8,
                lora_alpha=16, 
                lora_dropout=0.05
            )
        else:
            self.peft_config =  LoraConfig.from_pretrained(self.peft_config_dir)

        # load model with peft
        if self.peft_config_dir == None:
            self.backbone = get_peft_model(self.base_model, self.peft_config)
        else:
            self.backbone = PeftModel.from_pretrained(model = self.base_model, 
                                                                                    model_id = self.peft_config_dir,
                                                                                    is_trainable = True)

        # show number of trainning parameters
        self.backbone.print_trainable_parameters()
        
    def forward(self, input):
        # infernece
        output = self.backbone(**input)
        return output
    
    def save_pretrained(self,path):
        # save lora config
        self.backbone.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def chat_with_context(self,context_path,max_new_tokens,device):      
        """ get data """  
        self.to(device)
        with open(context_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        messages = data["messages"]

        """ get context tensor"""
        input = self.tokenizer.apply_chat_template(messages, return_tensors="pt",padding=True,return_dict=True).to(device)
        
        """ inference """
        self.eval()
        with torch.no_grad():
            generated_ids = self.backbone.generate(**input, 
                                                                        max_new_tokens = max_new_tokens, 
                                                                        do_sample = False)
            
        """ analysis results """
        generate_language = self.tokenizer.batch_decode(generated_ids)
        print("generate language:\n",generate_language[0])
        return generate_language[0]

    def chat_with_prompt(self,prompt,max_new_tokens,device):
        """ get prompt tensor"""
        input = self.tokenizer(prompt, return_tensors="pt").to(device)
        self.to(device)

        """ inference """
        self.eval()
        with torch.no_grad():
            generated_ids = self.backbone.generate(**input, 
                                        max_new_tokens = max_new_tokens, 
                                        do_sample = False)
        
        """ analysis results """
        generate_language = self.tokenizer.batch_decode(generated_ids)
        print("generate language:\n",generate_language[0])
        return generate_language[0]

if __name__ == "__main__":
    model = Mistral()
    model = model.to("cuda")
    input = torch.zeros((16, 28), dtype=torch.long).to("cuda")
    model(input)