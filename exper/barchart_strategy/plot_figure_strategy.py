import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import ast

def plot_all_benchmarks_bar_chart(input_csv):
    df = pd.read_csv(input_csv)
    
    # 筛选我们感兴趣的benchmarks
    df = df[df["benchmark"].isin(["matmul", "layernorm", "correlation", "dropout", "softmax"])]
    
    # 创建一个组合列，包含benchmark和shape信息
    df['benchmark_shape'] = df['benchmark'] + df['shape'].astype(str)
    
    sns.set_theme(style="whitegrid")
    
    # 创建更大的图形以适应更多数据
    plt.figure(figsize=(16, 10))
    ax = sns.barplot(
        data=df, 
        x="benchmark_shape", 
        y="speedup", 
        hue="strategy", 
        palette="Set2"
    )
    
    ax.set_xlabel("Benchmark and Shape", fontsize=14)
    ax.set_ylabel("Speedup", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=12)
    
    # 调整图例位置和大小，将其放在画布内靠右上
    plt.legend(
        title='Strategy', 
        title_fontsize='14', 
        fontsize='12',
        loc='upper right',
        bbox_to_anchor=(0.95, 0.95)
    )
    
    # 调整布局防止标签被截断
    plt.tight_layout()
    
    output_file = "./bar_chart_all_benchmarks.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Chart saved as {output_file}")
    
    plt.close()


def plot_threads_bar_chart(input_csv, benchmark):
    df = pd.read_csv(input_csv)

    df = df[df["benchmark"] == benchmark]

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 8))
    ax = sns.barplot(data=df, x="shape", y="speedup", hue="strategy", palette="Set2")

    ax.set_xlabel(f"Shape of {benchmark}", fontsize=14)
    ax.set_ylabel("Speedup", fontsize=14)
    # ax.set_title(f"{benchmark} performance for different shapes and threads", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    plt.legend(title='Thread', title_fontsize='14', fontsize='14')

    output_file = f"./bar_chart_strategy_{benchmark}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Chart saved as {output_file}")

    plt.close()


def get_best_thread(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    triton_df = df[df['method'] == "triton"]
    triton_tuned_df = df[df['method'] == "triton_tuned"]

    shapes_to_keep_first_three = {
        "matmul": ["(32, 32, 32, 10)", "(64, 64, 64, 10)", "(128, 128, 128, 10)"],
        "dropout": ["(1024, 10)", "(4096, 10)", "(16384, 10)"],
        "layernorm": ["(64, 64, 10)", "(256, 256, 10)", "(1024, 1024, 10)"],
        "correlation": ["(16, 16, 64, 64, 10)", "(32, 32, 64, 64, 10)", "(64, 64, 64, 64, 10)"],
        "softmax": ["(32, 1024, 10)", "(128, 1024, 10)", "(512, 1024, 10)"]
    }
    # shapes_to_keep_full = {
    #         "matmul": ["(32, 32, 32, 10)", "(64, 64, 64, 10)", "(128, 128, 128, 10)", "(256, 256, 256, 10)", "(512, 512, 512, 10)"],
    #         "dropout": ["(1024, 10)", "(4096, 10)", "(16384, 10)", "(65536, 10)", "(262144, 10)", "(1048576, 10)", "(4194304, 10)"],
    #         "layernorm": ["(64, 64, 10)", "(256, 256, 10)", "(256, 1024, 10)", "(1024, 1024, 10)", "(4096, 4096, 10)"],
    #         "correlation": ["(16, 16, 64, 64, 10)", "(32, 32, 64, 64, 10)", "(64, 64, 64, 64, 10)", "(64, 64, 128, 128, 10)", "(64, 64, 256, 256, 10)", "(64, 64, 512, 512, 10)"],
    #         "softmax": ["(32, 1024, 10)", "(128, 1024, 10)", "(512, 1024, 10)", "(512, 4096, 10)", "(512, 16384, 10)"]
    #     }
    # shapes_to_keep_old = {
    #     "matmul": ["(32, 32, 32, 10)", "(64, 64, 64, 10)", "(128, 128, 128, 10)", "(256, 256, 256, 10)", "(512, 512, 512, 10)"],
    #     "dropout": ["(1024, 10)", "(16384, 10)", "(262144, 10)", "(4194304, 10)"],
    #     "layernorm": ["(64, 64, 10)", "(256, 256, 10)", "(1024, 1024, 10)", "(4096, 4096, 10)"],
    #     "correlation": ["(1, 1, 64, 64, 10)", "(16, 16, 64, 64, 10)", "(32, 32, 64, 64, 10)", "(64, 64, 64, 64, 10)"],
    #     "softmax": ["(32, 1024, 10)", "(128, 1024, 10)", "(512, 1024, 10)", "(512, 4096, 10)", "(512, 16384, 10)"]
    # }
    triton_df = triton_df[triton_df.apply(lambda row: row["shape"] in shapes_to_keep_first_three.get(row["benchmark"], []), axis=1)]
    triton_df["shape"] = triton_df["shape"].apply(lambda x: str(tuple(eval(x)[:-1])))

    unique_shapes = triton_df['shape'].unique()
    unique_benchmarks = triton_df['benchmark'].unique()
    
    best_thread_rows = []
    for benchmark in unique_benchmarks:
        benchmark_df = triton_df[triton_df['benchmark'] == benchmark]
        unique_shapes = benchmark_df['shape'].unique()

        for shape in unique_shapes:
            sub_df = benchmark_df[benchmark_df['shape'] == shape]

            speedups = {thread: sub_df.loc[sub_df['thread'] == thread, 'speedup'].max() for thread in [1, 8, 32]}
            our_speedup = max(speedups.values())
            our_best_thread = [thread for thread, speedup in speedups.items() if speedup == our_speedup][0]
            our_best_row = sub_df[sub_df['thread'] == our_best_thread].sort_values(by='speedup', ascending=False).iloc[0]
            our_best_param = our_best_row['parameter']

            # Get T1_best_param and T1_best_speedup
            t1_best_row = sub_df[sub_df['thread'] == 1].sort_values(by='speedup', ascending=False).iloc[0]
            t1_best_param = t1_best_row['parameter']
            t1_best_speedup = t1_best_row['speedup']

            t32_best_row = sub_df[sub_df['thread'] == 32].sort_values(by='speedup', ascending=False).iloc[0]
            t32_best_param = t32_best_row['parameter']
            t32_best_speedup = t32_best_row['speedup']

            # Measure t32_best_param under T1, T8, and T32 threads to find the optimal result
            p2_speedup = 0
            for thread in [1, 8, 32]:
                thread_row = sub_df[(sub_df['thread'] == thread) & (sub_df['parameter'] == t32_best_param)]
                if not thread_row.empty:
                    p2_speedup = max(p2_speedup, thread_row['speedup'].max())
            for strategy, speedup in [("triton", t32_best_speedup), ("OUR", our_speedup)]:
                best_thread_rows.append({
                    "benchmark": benchmark,
                    "shape": shape,
                    "strategy": strategy,
                    "speedup": speedup
                })
            # best_thread_rows.append({
            #     "benchmark": benchmark,
            #     "shape": shape,
            #     # "T1_best_param": t1_best_param,
            #     "T32_best_param": t32_best_param,
            #     # "T1_best_speedup": t1_best_speedup,
            #     "P1_speedup": t32_best_speedup,
            #     "P2_speedup": p2_speedup,
            #     "our_best_thread": our_best_thread,
            #     "our_best_param": our_best_param,
            #     "our_best_speedup": our_speedup
            # })

    triton_df = pd.DataFrame(best_thread_rows)
    triton_df.to_csv(output_csv, index=False)


if __name__ == "__main__":

    get_best_thread('./performance_report_overall.csv', 'performance_report_strategy.csv')
    # for benchmark in ["matmul", "layernorm", "correlation", "dropout", "softmax"]:
    #     plot_threads_bar_chart(f'performance_report_strategy.csv', benchmark)

    plot_all_benchmarks_bar_chart('performance_report_strategy.csv')