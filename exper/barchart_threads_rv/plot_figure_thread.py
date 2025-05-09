import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import ast

def plot_threads_bar_chart(input_csv, benchmark, platform):
    df = pd.read_csv(input_csv)

    df = df[df["benchmark"] == benchmark]

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 7))
    ax = sns.barplot(data=df, x="shape", y="speedup", hue="thread", palette="Set2")

    ax.set_xlabel(f"{benchmark}", fontsize=40)
    ax.tick_params(axis='x', labelsize=36)
    ax.tick_params(axis='y', labelsize=36)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.xticks(range(3), ["s1", "s2", "s3"], rotation=0, ha="right", fontsize=36)
    if benchmark == "matmul":
        plt.legend(title='Thread', title_fontsize='36', fontsize='34')
        ax.set_ylabel("Speedup", fontsize=40)
    else:
        ax.get_legend().remove()  # 隐藏图例
        ax.set_ylabel("")  # 移除y轴标签

    output_file = f"./bar_chart_{benchmark}_{platform}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Chart saved as {output_file}")

    plt.close()


def filter_triton_tuned(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    df = df[df['method'] == "triton_tuned"].drop(columns=['method'])

    # shapes_to_keep_first_five = {
    #     "matmul": ["(32, 32, 32, 10)", "(64, 64, 64, 10)", "(128, 128, 128, 10)"],
    #     "dropout": ["(128, 10)", "(256, 10)", "(512, 10)", "(1024, 10)", "(2048, 10)"],
    #     "layernorm": ["(64, 64, 10)", "(128, 128, 10)", "(256, 256, 10)", "(512, 512, 10)", "(1024, 1024, 10)"],
    #     "conv2d": ["(1, 1, 16, 16, 10)", "(1, 1, 32, 32, 10)", "(4, 4, 32, 32, 10)", "(4, 4, 64, 64, 10)", "(8, 8, 64, 64, 10)"],
    #     "softmax": ["(32, 64, 10)", "(32, 256, 10)", "(128, 64, 10)", "(128, 256, 10)", "(32, 1024, 10)", "(128, 1024, 10)", "(512, 1024, 10)"],
    #     "resize": ["(32, 32, 1, 100)", "(32, 64, 1, 100)", "(32, 128, 1, 100)", "(64, 64, 1, 100)", "(64, 128, 1, 100)"],
    #     "rope": ["(64, 1, 4, 64, 100)", "(64, 1, 4, 256, 100)", "(64, 4, 4, 256, 100)", "(256, 1, 4, 256, 100)", "(256, 4, 4, 256, 100)"],
    # }
    shapes_to_keep_first_three = {
        "matmul": ["(32, 32, 32, 10)", "(64, 64, 64, 10)", "(128, 128, 128, 10)"],
        "dropout": ["(512, 10)", "(1024, 10)", "(2048, 10)"],
        "layernorm": ["(64, 64, 10)", "(256, 256, 10)", "(1024, 1024, 10)"],
        "conv2d": ["(1, 1, 32, 32, 10)", "(4, 4, 32, 32, 10)", "(4, 4, 64, 64, 10)"],
        "softmax": ["(32, 64, 10)", "(128, 64, 10)", "(128, 256, 10)"],
        # "resize": ["(32, 32, 1, 100)", "(32, 128, 1, 100)", "(64, 128, 1, 100)"],
        "rope": ["(64, 4, 4, 256, 100)", "(256, 1, 4, 256, 100)", "(256, 4, 4, 64, 100)"]
    }
    # shapes_to_keep = {
    #     "matmul": ["(32, 32, 32, 10)", "(64, 64, 64, 10)", "(128, 128, 128, 10)", "(256, 256, 256, 10)", "(512, 512, 512, 10)"],
    #     "dropout": ["(1024, 10)", "(16384, 10)", "(262144, 10)", "(4194304, 10)"],
    #     "layernorm": ["(64, 64, 10)", "(256, 256, 10)", "(1024, 1024, 10)", "(4096, 4096, 10)"],
    #     "conv2d": ["(1, 1, 64, 64, 10)", "(16, 16, 64, 64, 10)", "(32, 32, 64, 64, 10)", "(64, 64, 64, 64, 10)"],
    #     "softmax": ["(32, 1024, 10)", "(128, 1024, 10)", "(512, 1024, 10)", "(512, 4096, 10)", "(512, 16384, 10)"]
    # }
    df = df[df.apply(lambda row: row["shape"] in shapes_to_keep_first_three.get(row["benchmark"], []), axis=1)]

    df["shape"] = df["shape"].apply(lambda x: str(tuple(eval(x)[:-1])))
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":

    for platform in ["X86", "RISC-V"]:
        filter_triton_tuned(f'./performance_report_overall_{platform}.csv', f'performance_report_selected_{platform}.csv')
        for benchmark in ["matmul", "layernorm", "conv2d", "dropout", "softmax", "rope"]:
            plot_threads_bar_chart(f'performance_report_selected_{platform}.csv', benchmark, platform)
