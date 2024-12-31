import json
import math
import os
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType,PeftModel
from torch import nn
from torch.utils.data import DataLoader
from torch import optim

from llmai.utils import *

class Mistral(nn.Module):
    def __init__(self, base_model_name_or_path, peft_config_dir = None):
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
    
    def compute_loss(self, input):
        output = {}
        if "class_weights" not in input:
            input["class_weights"] = None
        if "mask" not in input:
            criterion = nn.CrossEntropyLoss(weight = input["class_weights"])
            output["total_loss"] = criterion(input["predict"],input["label"])
        else:
            criterion = nn.CrossEntropyLoss(weight = input["class_weights"], reduction = 'none')
            loss = criterion(input["predict"],input["label"])
            masked_loss = loss * input["mask"]
            output["total_loss"] = masked_loss.sum() / input["mask"].sum()
        return output
    
    def forward(self, input):
        # infernece
        batch_data = {
            "input_ids":input["input_ids"].reshape(-1,input["input_ids"].shape[2]).to(self.device),
            "attention_mask":input["attention_mask"].reshape(-1,input["attention_mask"].shape[2]).to(self.device)
        }
        outputs = self.backbone(**batch_data)
        logits = outputs.logits
        output = {
            "predict": logits[..., :-1, :],
            "label":input["input_ids"].reshape(-1,input["input_ids"].shape[2]).to(self.device)[:, 1:],
            "mask":input["attention_mask"].reshape(-1,input["attention_mask"].shape[2]).to(self.device)[:,1:]
        }
        return output
    
    def save_pretrained(self, save_model_dir):
        # save lora config
        self.backbone.save_pretrained(save_model_dir)
        self.tokenizer.save_pretrained(save_model_dir)

    def load_pretrained(self, save_model_dir):
        # load 
        pass
    
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


    def train_one_epoch(
        self, epoch,train_dataloader, optimizer, clip_max_norm, log_path = None
    ):
        self.train()
        pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
        total_loss = AverageMeter()
        average_hit_rate = AverageMeter()
        """ get data """
        for batch_id, inputs in enumerate(pbar):
            """ grad zeroing """
            optimizer.zero_grad()

            """ forward """
            used_memory = 0 if self.device == "cpu" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
            output = self.forward(inputs)

            """ calculate loss """
            out_criterion = self.compute_loss(output)
            out_criterion["total_loss"].backward()
            after_used_memory = torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
            total_loss.update(out_criterion["total_loss"].item())
            average_hit_rate.update(math.exp(-total_loss.avg))

            """ grad clip """
            if clip_max_norm > 0:
                clip_gradient(optimizer,clip_max_norm)

            """ modify parameters """
            optimizer.step()
            after_used_memory = 0 if self.device == "cpu" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
            postfix_str = "total_loss: {:.4f},average_hit_rate:{:.4f},use_memory: {:.1f}G".format(
                total_loss.avg, 
                average_hit_rate.avg,
                after_used_memory - used_memory
            )
            pbar.set_postfix_str(postfix_str)
            pbar.update()
        with open(log_path, "a") as file:
            file.write(postfix_str+"\n")
        return total_loss.avg


    def test_epoch(self, epoch, test_dataloader,trainning_log_path = None):
        total_loss = AverageMeter()
        average_hit_rate = AverageMeter()
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            for batch_id, inputs in enumerate(test_dataloader):
                """ forward """
                output = self.forward(inputs)

                """ calculate loss """
                out_criterion = self.compute_loss(output)
                total_loss.update(out_criterion["total_loss"])

            average_hit_rate.update(math.exp(-total_loss.avg))
            str = "Test Epoch: {:d}, total_loss: {:.4f},average_hit_rate:{:.4f}".format(
                epoch,
                total_loss.avg, 
                average_hit_rate.avg,
            )
        print(str)
        with open(trainning_log_path, "a") as file:
            file.write(str+"\n")
        return total_loss.avg

    def trainning(
        self,
        train_dataloader:DataLoader = None,
        test_dataloader:DataLoader = None,
        optimizer_name:str = "Adam",
        weight_decay:float = 1e-4,
        clip_max_norm:float = 0.5,
        factor:float = 0.3,
        patience:int = 15,
        lr:float = 1e-4,
        total_epoch:int = 1000,
        save_checkpoint_step:str = 10,
        save_model_dir:str = "models",
        first_train = True
    ):
        ## 1 trainning log path 
        check_point_path = save_model_dir  + "/checkpoint.pth"
        log_path = save_model_dir + "/train.log"

        ## 2 get net pretrain parameters if need 
        """
            If there is  training history record, load pretrain parameters
        """
        if  first_train == False and os.path.isdir(save_model_dir) and os.path.exists(check_point_path) and os.path.exists(log_path):
            self.load_pretrained(save_model_dir)  
            first_train = False

        else:
            if not os.path.isdir(save_model_dir):
                os.makedirs(save_model_dir)
            with open(log_path, "w") as file:
                pass
            first_train = True

        ##  3 get optimizer
        if optimizer_name == "Adam":
            optimizer = optim.Adam(params = self.parameters(), lr = lr, weight_decay = weight_decay)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(params = self.parameters(), lr = lr, weight_decay = weight_decay)
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(params = self.parameters(), lr = lr, weight_decay = weight_decay)
        else:
            optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer, 
            mode = "min", 
            factor = factor, 
            patience = patience
        )

        ## init trainng log
        if first_train:
            best_loss = float("inf")
            last_epoch = 0
        else:
            checkpoint = torch.load(check_point_path, map_location=self.device)
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            best_loss = checkpoint["loss"]
            last_epoch = checkpoint["epoch"] + 1

        try:
            for epoch in range(last_epoch,total_epoch):
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                train_loss = self.train_one_epoch(epoch,train_dataloader, optimizer,clip_max_norm,log_path)
                test_loss = self.test_epoch(epoch,test_dataloader,log_path)
                loss = train_loss + test_loss
                lr_scheduler.step(loss)
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
                check_point_path = save_model_dir  + "/checkpoint.pth"
                torch.save(                
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "optimizer": None,
                        "lr_scheduler": None
                    },
                    check_point_path
                )

                if epoch % save_checkpoint_step == 0:
                    os.makedirs(save_model_dir + "/" + "chaeckpoint-"+str(epoch))
                    torch.save(
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict()
                        },
                        save_model_dir + "/" + "chaeckpoint-"+str(epoch)+"/checkpoint.pth"
                    )
                if is_best:
                    self.save_pretrained(save_model_dir)

        # interrupt trianning
        except KeyboardInterrupt:
                torch.save(                
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict()
                    },
                    check_point_path
                )
