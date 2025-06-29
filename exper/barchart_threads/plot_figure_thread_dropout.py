import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import ast

def plot_threads_bar_chart(input_csv, benchmark, platform):
    df = pd.read_csv(input_csv)

    df = df[df['method'] == "triton_tuned"].drop(columns=['method'])
    shapes_to_keep_first_three = {
        # "matmul": ["(32, 32, 32, 10)", "(64, 64, 64, 10)", "(128, 128, 128, 10)"],
        "dropout": ["(1024, 10)"],
        # "layernorm": ["(64, 64, 10)", "(256, 256, 10)", "(1024, 1024, 10)"],
        # "correlation": ["(1, 1, 32, 32, 10)", "(4, 4, 32, 32, 10)", "(4, 4, 64, 64, 10)"],
        # "softmax": ["(32, 64, 10)", "(128, 64, 10)", "(128, 256, 10)"],
        # "resize": ["(32, 32, 1, 100)", "(32, 128, 1, 100)", "(64, 128, 1, 100)"],
        # "rope": ["(64, 4, 4, 256, 100)", "(256, 1, 4, 256, 100)", "(256, 4, 4, 64, 100)"]
    }
    df = df[df.apply(lambda row: row["shape"] in shapes_to_keep_first_three.get(row["benchmark"], []), axis=1)]

    df["shape"] = df["shape"].apply(lambda x: str(tuple(eval(x)[:-1])))

    df = df[df["benchmark"] == benchmark]

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(4, 4))
    ax = sns.barplot(data=df, x="thread", y="speedup", palette="Set2")
    ax.yaxis.grid(False)
    # Set y-axis limits to 8-12
    ax.set_ylim(10, 12)
    
    # Hide y-axis tick values
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Change y-axis label to "efficiency"
    ax.set_ylabel("Efficiency", fontsize=18)
    
    # Change x-axis label and tick labels
    ax.set_xlabel("Thread Settings", fontsize=18)
    # ax.set_xticklabels(["t1", "t2", "t4", "$t8"])
    
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    # Remove legend since we're now showing threads on x-axis
    # ax.get_legend().remove()

    output_file = f"./bar_chart_additional_dropout.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Chart saved as {output_file}")

    plt.close()


if __name__ == "__main__":

    for platform in ["X86"]:
        for benchmark in ["dropout"]:
            plot_threads_bar_chart(f'./performance_report_overall_{platform}.csv', benchmark, platform)
