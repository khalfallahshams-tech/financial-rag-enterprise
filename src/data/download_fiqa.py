import os
import pandas as pd
from datasets import load_dataset

def main():
    # Ensure data directory exists
    os.makedirs("data/raw", exist_ok=True)

    print("Downloading FiQA Full Corpus (The Raw Financial Documents)...")
    corpus_dataset = load_dataset("explodinggradients/fiqa", "corpus", split="corpus")
    df_corpus = corpus_dataset.to_pandas()
    
    corpus_output_path = "data/raw/fiqa_corpus.csv"
    df_corpus.to_csv(corpus_output_path, index=False)
    print(f"Success! Full Corpus saved to {corpus_output_path} with {len(df_corpus)} documents.")

    print("\nDownloading FiQA Evaluation Dataset (Questions & Ground Truth Answers)...")
    # Load the evaluation split used by Ragas
    eval_dataset = load_dataset("explodinggradients/fiqa", "ragas_eval", split="baseline")
    df_eval = eval_dataset.to_pandas()
    
    eval_output_path = "data/raw/fiqa_eval.csv"
    df_eval.to_csv(eval_output_path, index=False)
    print(f"Success! Evaluation Dataset saved to {eval_output_path} with {len(df_eval)} pairs.")

if __name__ == "__main__":
    main()
