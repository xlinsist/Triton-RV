import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import ast

def plot_threads_bar_chart(input_csv, benchmark):
    df = pd.read_csv(input_csv)

    df = df[df["benchmark"] == benchmark]

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 8))
    ax = sns.barplot(data=df, x="shape", y="speedup", hue="thread", palette="Set2")

    ax.set_xlabel(f"Shape of {benchmark}", fontsize=14)
    ax.set_ylabel("Speedup", fontsize=14)
    # ax.set_title(f"{benchmark} performance for different shapes and threads", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    plt.legend(title='Thread', title_fontsize='14', fontsize='14')

    output_file = f"./bar_chart_{benchmark}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Chart saved as {output_file}")

    plt.close()


def filter_triton_tuned(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    df = df[df['method'] == "triton_tuned"].drop(columns=['method'])

    shapes_to_keep = {
        "matmul": ["(32, 32, 32, 10)", "(64, 64, 64, 10)", "(128, 128, 128, 10)", "(256, 256, 256, 10)", "(512, 512, 512, 10)"],
        "dropout": ["(1024, 10)", "(16384, 10)", "(262144, 10)", "(4194304, 10)"],
        "layernorm": ["(64, 64, 10)", "(256, 256, 10)", "(1024, 1024, 10)", "(4096, 4096, 10)"],
        "correlation": ["(1, 1, 64, 64, 10)", "(16, 16, 64, 64, 10)", "(32, 32, 64, 64, 10)", "(64, 64, 64, 64, 10)"],
        "softmax": ["(32, 1024, 10)", "(128, 1024, 10)", "(512, 1024, 10)", "(512, 4096, 10)", "(512, 16384, 10)"]
    }
    df = df[df.apply(lambda row: row["shape"] in shapes_to_keep.get(row["benchmark"], []), axis=1)]

    df["shape"] = df["shape"].apply(lambda x: str(tuple(eval(x)[:-1])))
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":

    filter_triton_tuned('./performance_report_overall.csv', 'performance_report_selected.csv')

    for benchmark in ["matmul", "layernorm", "correlation", "dropout", "softmax"]:
        plot_threads_bar_chart(f'performance_report_selected.csv', benchmark)
