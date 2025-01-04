import argparse
from datasets import load_dataset
import pandas as pd


def main():
    dataset = load_dataset("microsoft/ms_marco", "v1.1")
    for split in dataset:
        print(f"Processing {split} split...")
        df = pd.DataFrame(dataset[split])
        csv_filename = "datasets/ms_marco/" + f"ms_marco_{split}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Saved {split} split to {csv_filename}")
    
if __name__ == "__main__":
    main()

