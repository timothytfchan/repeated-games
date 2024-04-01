import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
#from textwrap import wrap

def extract_punitiveness_from_eval_results(eval_results_files):
    data = []
    for file in eval_results_files:
        with open(file) as f:
            eval_results = json.load(f)
            # Ignore if punitiveness is null but include if it is 0
            if eval_results["punitiveness"] is not None:
                data.append({
                    "focal_agent_model": eval_results["focal_agent_model"],
                    "experiment_name": eval_results["experiment_name"],
                    "focal_agent": eval_results["focal_agent"],
                    "punitiveness": eval_results["punitiveness"],
                    "lower_ci": eval_results["lower_ci"],
                    "upper_ci": eval_results["upper_ci"]
                })
    # save the data in a pandas dataframe and save to csv
    df = pd.DataFrame(data)
    df.to_csv("punitiveness_debug.csv", index=False)
    return df

def plot_punitiveness_bar_chart(eval_results_files, save_path="punitiveness_bar_chart.png"):
    df = extract_punitiveness_from_eval_results(eval_results_files)    
    # Calculate error margins as previously described
    df = df.reset_index(drop=True)
    df['error_lower'] = df['punitiveness'] - df['lower_ci']
    df['error_upper'] = df['upper_ci'] - df['punitiveness']
    yerr = df[['error_lower', 'error_upper']].T.values
    yerr = [df['punitiveness'] - df['lower_ci'], df['upper_ci'] - df['punitiveness']]

    df['experiment_name'] = df['focal_agent_model'] + ' - ' + df['experiment_name']
    df = df.sort_values("experiment_name")

    plt.figure(figsize=(20, 16))  # Increase the figure width plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="experiment_name", y="punitiveness", data=df, yerr=yerr, capsize=0.2) #
    plt.xlabel("Experiment Name")
    plt.ylabel("Punitiveness")
    plt.title(f"Punitiveness of {df['focal_agent'].iloc[0]}")
    
    # labels = ['\n'.join(wrap(label, 20)) for label in df['experiment_name']]  # Wrap labels to 20 characters per line
    # ax.set_xticklabels(labels)
    
    plt.xticks(rotation=90, ha="right")  # Rotate the labels by 90 degrees and align them to the right
    plt.ylim(-50, 100)  # Set the y-axis limits
    plt.tick_params(axis='x', which='major', pad=15)  # Increase the padding between the labels and the axis
    #plt.subplots_adjust(bottom=0.05)  # Adjust the bottom margin
    plt.tight_layout()  # Adjust the layout to make room for the labels
    # Save the plot
    plt.savefig(save_path)

def extract_exploitability_from_eval_results(eval_results_files):
    data = []
    for file in eval_results_files:
        with open(file) as f:
            eval_results = json.load(f)
            if eval_results["exploitability"] is not None:
                data.append({
                    "focal_agent_model": eval_results["focal_agent_model"],
                    "experiment_name": eval_results["experiment_name"],
                    "focal_agent": eval_results["focal_agent"],
                    "exploitability": eval_results["exploitability"],
                    "lower_ci": eval_results["lower_ci"],
                    "upper_ci": eval_results["upper_ci"]
                })
    df = pd.DataFrame(data)
    df.to_csv("exploitability_debug.csv", index=False)
    return df

def plot_exploitability_bar_chart(eval_results_files, save_path="exploitability_bar_chart.png"):
    df = extract_exploitability_from_eval_results(eval_results_files)
    df = df.reset_index(drop=True)
    df['error_lower'] = df['exploitability'] - df['lower_ci']
    df['error_upper'] = df['upper_ci'] - df['exploitability']
    yerr = [df['exploitability'] - df['lower_ci'], df['upper_ci'] - df['exploitability']]
    df['experiment_name'] = df['focal_agent_model'] + ' - ' + df['experiment_name']
    df = df.sort_values("experiment_name")
    plt.figure(figsize=(20, 16))
    ax = sns.barplot(x="experiment_name", y="exploitability", data=df, yerr=yerr, capsize=0.2)
    plt.xlabel("Experiment Name")
    plt.ylabel("Exploitability")
    plt.title(f"Exploitability of {df['focal_agent'].iloc[0]}")
    plt.xticks(rotation=90, ha="right")
    plt.ylim(-50, 100)
    plt.tick_params(axis='x', which='major', pad=15)
    plt.tight_layout()
    plt.savefig(save_path)

def plot_punitiveness_vs_exploitability(punitiveness_eval_results_files, exploitability_eval_results_file, save_path="punitiveness_vs_exploitability.png"):
    punitiveness_df = extract_punitiveness_from_eval_results(punitiveness_eval_results_files)
    exploitability_df = extract_exploitability_from_eval_results(exploitability_eval_results_file)

    # Merge the two dataframes based on 'focal_agent_model' and 'experiment_name'
    merged_df = pd.merge(punitiveness_df, exploitability_df, on=['focal_agent_model', 'experiment_name', 'focal_agent'], suffixes=('_punitiveness', '_exploitability'))

    # Filter out datapoints that have missing values for either punitiveness or exploitability
    merged_df = merged_df.dropna(subset=['punitiveness', 'exploitability'])

    # Add a new column to the dataframe that combines the focal agent model and experiment name
    merged_df['experiment_name'] = merged_df['focal_agent_model'] + ' - ' + merged_df['experiment_name']
    
    # Create a scatter plot
    plt.figure(figsize=(12, 10))
    ax = sns.scatterplot(x='punitiveness', y='exploitability', data=merged_df)

    # Add labels and title
    plt.xlabel("Punitiveness")
    plt.ylabel("Exploitability")
    plt.title(f"Punitiveness vs Exploitability of {merged_df['focal_agent'].iloc[0]}")

    # Set the limits for x and y axes
    plt.xlim(-50, 100)
    plt.ylim(-50, 100)

    # Add labels for each dot using arrows
    for i, row in merged_df.iterrows():
        x = row['punitiveness']
        y = row['exploitability']
        label = row['experiment_name']

        # Determine the quadrant of the plot
        if x >= 0 and y >= 0:  # Upper right quadrant
            xytext = (10, 10)
            ha = 'left'
            va = 'bottom'
        elif x < 0 and y >= 0:  # Upper left quadrant
            xytext = (-10, 10)
            ha = 'right'
            va = 'bottom'
        elif x < 0 and y < 0:  # Lower left quadrant
            xytext = (-10, -10)
            ha = 'right'
            va = 'top'
        else:  # Lower right quadrant
            xytext = (10, -10)
            ha = 'left'
            va = 'top'

        # Adjust the arrow length based on the distance from the origin
        distance = np.sqrt(x**2 + y**2)
        shrink = 5 + distance * 0.1

        # Set the arrow properties
        arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0', shrinkA=0, shrinkB=shrink)

        ax.annotate(label, xy=(x, y), xytext=xytext, textcoords='offset points',
                    arrowprops=arrowprops, fontsize=8, ha=ha, va=va)

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)

def plot_punitiveness_bar_chart(eval_results_files, save_path="punitiveness_bar_chart.png"):
    df = extract_punitiveness_from_eval_results(eval_results_files)    
    df = df.reset_index(drop=True)
    df['error_lower'] = df['punitiveness'] - df['lower_ci']
    df['error_upper'] = df['upper_ci'] - df['punitiveness']
    yerr = [df['punitiveness'] - df['lower_ci'], df['upper_ci'] - df['punitiveness']]
    df['experiment_name'] = df['focal_agent_model'] + ' - ' + df['experiment_name']
    df = df.sort_values("experiment_name")
    plt.figure(figsize=(20, 16))
    ax = sns.barplot(x="experiment_name", y="punitiveness", data=df, yerr=yerr, capsize=0.2)
    plt.xlabel("Experiment Name")
    plt.ylabel("Punitiveness")
    plt.title(f"Punitiveness of {df['focal_agent'].iloc[0]}")
    plt.xticks(rotation=90, ha="right")
    plt.yscale('symlog')  # Set the y-axis to symlog scale
    plt.tick_params(axis='x', which='major', pad=15)
    plt.tight_layout()
    plt.savefig(save_path)

def plot_exploitability_bar_chart(eval_results_files, save_path="exploitability_bar_chart.png"):
    df = extract_exploitability_from_eval_results(eval_results_files)
    df = df.reset_index(drop=True)
    df['error_lower'] = df['exploitability'] - df['lower_ci']
    df['error_upper'] = df['upper_ci'] - df['exploitability']
    yerr = [df['exploitability'] - df['lower_ci'], df['upper_ci'] - df['exploitability']]
    df['experiment_name'] = df['focal_agent_model'] + ' - ' + df['experiment_name']
    df = df.sort_values("experiment_name")
    plt.figure(figsize=(20, 16))
    ax = sns.barplot(x="experiment_name", y="exploitability", data=df, yerr=yerr, capsize=0.2)
    plt.xlabel("Experiment Name")
    plt.ylabel("Exploitability")
    plt.title(f"Exploitability of {df['focal_agent'].iloc[0]}")
    plt.xticks(rotation=90, ha="right")
    plt.yscale('symlog')  # Set the y-axis to symlog scale
    plt.tick_params(axis='x', which='major', pad=15)
    plt.tight_layout()
    plt.savefig(save_path)

def plot_punitiveness_vs_exploitability(punitiveness_eval_results_files, exploitability_eval_results_file, save_path="punitiveness_vs_exploitability.png"):
    punitiveness_df = extract_punitiveness_from_eval_results(punitiveness_eval_results_files)
    exploitability_df = extract_exploitability_from_eval_results(exploitability_eval_results_file)
    merged_df = pd.merge(punitiveness_df, exploitability_df, on=['focal_agent_model', 'experiment_name', 'focal_agent'], suffixes=('_punitiveness', '_exploitability'))
    merged_df = merged_df.dropna(subset=['punitiveness', 'exploitability'])
    merged_df['experiment_name'] = merged_df['focal_agent_model'] + ' - ' + merged_df['experiment_name']
    
    # Save merged_df to debug
    merged_df.to_csv("punitiveness_vs_exploitability_debug.csv", index=False)
    
    plt.figure(figsize=(12, 10))
    ax = sns.scatterplot(x='punitiveness', y='exploitability', data=merged_df)
    plt.xlabel("Punitiveness")
    plt.ylabel("Exploitability")
    plt.title(f"Punitiveness vs Exploitability of {merged_df['focal_agent'].iloc[0]}")
    plt.xscale('symlog')  # Set the x-axis to symlog scale
    plt.yscale('symlog')  # Set the y-axis to symlog scale
    for i, row in merged_df.iterrows():
        x = row['punitiveness']
        y = row['exploitability']
        label = row['experiment_name']
        if x >= 0 and y >= 0:
            xytext = (10, 10)
            ha = 'left'
            va = 'bottom'
        elif x < 0 and y >= 0:
            xytext = (-10, 10)
            ha = 'right'
            va = 'bottom'
        elif x < 0 and y < 0:
            xytext = (-10, -10)
            ha = 'right'
            va = 'top'
        else:
            xytext = (10, -10)
            ha = 'left'
            va = 'top'
        distance = np.sqrt(x**2 + y**2)
        shrink = 5 + distance * 0.1
        arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0', shrinkA=0, shrinkB=shrink)
        ax.annotate(label, xy=(x, y), xytext=xytext, textcoords='offset points',
                    arrowprops=arrowprops, fontsize=8, ha=ha, va=va)
    plt.tight_layout()
    plt.savefig(save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval_results_files", nargs="+", help="List of JSON files with the results of the evaluation.")
    args = parser.parse_args()
    eval_results_files = args.eval_results_files
    plot_punitiveness_bar_chart(eval_results_files)

if __name__ == "__main__":
    main()