import argparse
import csv
import json
import re
import shutil
import sys
import os
sys.path.append(os.getcwd())

def process_csv(input_file, output_file):
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row['query']
            answers = row['answers']
            result = {
                "messages": [
                    {
                        "content":answers,
                        "role": "user"
                    },
                    {
                        "content":query,
                        "role": "assistant"
                    },
                ]
            }
            with open(output_file, 'a', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')  



def main(args):
    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir,exist_ok=True)
    process_csv(args.data_dir + "ms_marco_train.csv", args.out_dir + "train.jsonl")
    process_csv(args.data_dir + "ms_marco_test.csv", args.out_dir + "test.jsonl")
    process_csv(args.data_dir + "ms_marco_validation.csv", args.out_dir + "valid.jsonl")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type = str,default="datasets/ms_marco/")
    parser.add_argument("--out_dir",type = str,default = "marco_train/")
    args = parser.parse_args()
    main(args)