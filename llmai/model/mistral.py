import os
import sys
sys.path.append(os.getcwd())
import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType,PeftModel
from torch import nn

class Mistral(nn.Module):
    def __init__(self,base_model_name_or_path,pre_perf_path = None,trainning = True):
        super(Mistral, self).__init__()
        self.model_name_or_path = base_model_name_or_path
        self.pre_perf_path = pre_perf_path

        # load base model
        self.backbone =  AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16
        )
        
        # load peft config
        if pre_perf_path == None:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode = False,
                r=8,
                lora_alpha=16, 
                lora_dropout=0.05
            )
            self.backbone = get_peft_model(self.backbone, self.peft_config)
        else:
            self.backbone = PeftModel.from_pretrained(model = self.backbone, 
                                                    model_id = "/data2/xiaohui/work/LLM/saved_model/finetune",
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

if __name__ == "__main__":
    model = Mistral()
    model = model.to("cuda")
    input = torch.zeros((16, 28), dtype=torch.long).to("cuda")
    model(input)