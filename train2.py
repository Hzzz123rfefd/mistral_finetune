import argparse
import os
import torch
from transformers import TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType,PeftModel

def main(args):
    """get train device"""
    device = args.device if torch.cuda.is_available() else "cpu"

    """ get model """
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit = True,                                             # The model parameters are saved in memory as 4 bits
    #     bnb_4bit_use_double_quant = True,                    # Double quantification
    #     bnb_4bit_quant_type = "nf4",                              # Normal Float 4
    #     bnb_4bit_compute_dtype = torch.float16            # Number of model parameter bits during inference
    # )
    model =  AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype = torch.float16,
        # load_in_4bit = True,
    )
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode = False,
        r = 8,
        lora_alpha = 16, 
        lora_dropout = 0.05
    )
        
    # if os.path.isdir(args.lora_config_dir):
    #     model = Mistral(base_model_name_or_path = args.base_model_name_or_path,
    #                                 peft_config_dir = args.lora_config_dir)
    # else:
    #     os.makedirs(args.lora_config_dir)
    #     model = Mistral(base_model_name_or_path = args.base_model_name_or_path)
    # model.to(device)

    """ get data """
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    """ get args """
    trainning_args = TrainingArguments(
        output_dir=args.lora_config_dir, 
        num_train_epochs = args.total_epoch,                     
        per_device_train_batch_size = args.batch_size,          
        gradient_accumulation_steps = 2,          # Number of steps before reverse/update
        gradient_checkpointing = True,              # Use gradient checkpoints to save memory
        optim="adamw_torch_fused",               
        logging_steps = 1,                                   # Record every 1 steps
        save_strategy="epoch",                          # Save checkpoints for each epoch
        learning_rate = args.lr,                     
        bf16=True,                                             # Using BFLOAT16 Precision
        tf32=True,                                              # Using TF32 precision
        max_grad_norm = 0.3,                          
        warmup_ratio = 0.03,                      
        lr_scheduler_type="constant",          
        push_to_hub=False,                       
        report_to="tensorboard",                
    )


    trainer = SFTTrainer(
        model = model,
        args = trainning_args,
        train_dataset = dataset,
        peft_config = peft_config,
        max_seq_length = args.max_seq_length,
        tokenizer = model.tokenizer,
        packing = True,
        dataset_kwargs={
            "add_special_tokens": False,  
            "append_concat_token": False, 
        }
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",
                        type=str,
                        default = "mistralai/Mistral-7B-Instruct-v0.3")
    
    parser.add_argument("--lora_config_dir",
                        type=str,
                        default = "./saved_model/test2")
    
    parser.add_argument("--data_path",
                        type=str,
                        default = "data/train.jsonl")
    
    parser.add_argument("--lr",
                        type=float,
                        default = 2e-4)
    
    parser.add_argument("--batch_size",
                        type=int,
                        default=6)
    
    parser.add_argument("--total_epoch",
                        type = int,
                        default = 3)
    
    parser.add_argument("--max_seq_length",
                        type = int,
                        default = 3072)
    
    parser.add_argument("--device",
                        type=str,
                        default = "cuda")
    
    args = parser.parse_args()
    main(args)

