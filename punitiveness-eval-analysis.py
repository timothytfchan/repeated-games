import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json

def extract_punitiveness_from_eval_results(eval_results_files):
    data = []
    for file in eval_results_files:
        with open(file) as f:
            eval_results = json.load(f)
            data.append({
                "experiment_name": eval_results["experiment_name"],
                "focal_agent": eval_results["focal_agent"],
                "punitiveness": eval_results["punitiveness"],
                "lower_ci": eval_results["lower_ci"],
                "upper_ci": eval_results["upper_ci"]
            })
    
    return pd.DataFrame(data)

def plot_punitiveness_bar_chart(eval_results_files):
    df = extract_punitiveness_from_eval_results(eval_results_files)
    df = df.sort_values("experiment_name")

    plt.figure(figsize=(10, 6))
    sns.barplot(x="experiment_name", y="punitiveness", data=df, yerr=df[["lower_ci", "upper_ci"]].T.values, capsize=0.2)
    plt.xlabel("Experiment Name")
    plt.ylabel("Punitiveness")
    plt.title(f"Punitiveness of {df['focal_agent'].iloc[0]}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval_results_files", nargs="+", help="List of JSON files with the results of the evaluation.")
    args = parser.parse_args()

    plot_punitiveness_bar_chart(args.eval_results_files)

if __name__ == "__main__":
    main()