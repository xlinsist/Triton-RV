import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import ast

def plot_all_benchmarks_bar_chart(input_csv, platform):
    df = pd.read_csv(input_csv)
    
    # 筛选我们感兴趣的benchmarks
    df = df[df["benchmark"].isin(["matmul", "layernorm", "correlation", "dropout", "resize", "rope"])]
    
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
    
    output_file = f"./bar_chart_all_benchmarks_{platform}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Chart saved as {output_file}")
    
    plt.close()


def plot_threads_bar_chart(input_csv, benchmark, platform):
    df = pd.read_csv(input_csv)

    df = df[df["benchmark"] == benchmark]

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 2))  # 修改这里，调整宽高比 5:1
    ax = sns.barplot(data=df, x="shape", y="speedup", hue="strategy", palette="Set2")

    ax.set_xlabel(f"Shape of {benchmark}", fontsize=14)
    ax.set_ylabel("Speedup", fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    # ax.set_title(f"{benchmark} performance for different shapes and threads", fontsize=12)
    plt.xticks(rotation=30, ha="right")

    plt.legend(title='Thread', title_fontsize='14', fontsize='12')

    output_file = f"./bar_chart_strategy_{benchmark}_{platform}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Chart saved as {output_file}")

    plt.close()


def get_best_thread(input_csv, output_synergy_csv, output_params_csv):
    df = pd.read_csv(input_csv)
    triton_df = df[df['method'] == "triton"]
    triton_tuned_df = df[df['method'] == "triton_tuned"]

    shapes_to_keep_first_five = {
        "matmul": ["(32, 32, 32, 10)", "(64, 64, 64, 10)", "(128, 128, 128, 10)"],
        "dropout": ["(128, 10)", "(256, 10)", "(512, 10)", "(1024, 10)", "(2048, 10)"],
        "layernorm": ["(64, 64, 10)", "(128, 128, 10)", "(256, 256, 10)", "(512, 512, 10)", "(1024, 1024, 10)"],
        "correlation": ["(1, 1, 16, 16, 10)", "(1, 1, 32, 32, 10)", "(4, 4, 32, 32, 10)", "(4, 4, 64, 64, 10)", "(8, 8, 64, 64, 10)"],
        "softmax": ["(32, 64, 10)", "(32, 256, 10)", "(128, 64, 10)", "(128, 256, 10)", "(32, 1024, 10)", "(128, 1024, 10)", "(512, 1024, 10)"],
        "resize": ["(32, 32, 1, 100)", "(32, 64, 1, 100)", "(32, 128, 1, 100)", "(64, 64, 1, 100)", "(64, 128, 1, 100)"],
        "rope": ["(64, 1, 4, 64, 100)", "(64, 1, 4, 256, 100)", "(64, 4, 4, 256, 100)", "(256, 1, 4, 256, 100)", "(256, 4, 4, 256, 100)"],
    }
    shapes_to_keep_first_three = {
        "matmul": ["(32, 32, 32, 10)", "(64, 64, 64, 10)", "(128, 128, 128, 10)"],
        "dropout": ["(512, 10)", "(1024, 10)", "(2048, 10)"],
        "layernorm": ["(64, 64, 10)", "(256, 256, 10)", "(1024, 1024, 10)"],
        "correlation": ["(1, 1, 32, 32, 10)", "(4, 4, 32, 32, 10)", "(4, 4, 64, 64, 10)"],
        "softmax": [],
        # "softmax": ["(32, 64, 10)", "(128, 64, 10)", "(128, 256, 10)"],
        "resize": ["(32, 32, 1, 100)", "(32, 128, 1, 100)", "(64, 128, 1, 100)"],
        "rope": ["(64, 4, 4, 256, 100)", "(256, 1, 4, 256, 100)", "(256, 4, 4, 64, 100)"]
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
    best_params_rows = []
    for benchmark in unique_benchmarks:
        benchmark_df = triton_df[triton_df['benchmark'] == benchmark]
        unique_shapes = benchmark_df['shape'].unique()

        for shape in unique_shapes:
            sub_df = benchmark_df[benchmark_df['shape'] == shape]

            speedups = {thread: sub_df.loc[sub_df['thread'] == thread, 'speedup'].max() for thread in [1, 4, 8]}

            our_speedup = max(speedups.values())
            our_best_thread = [thread for thread, speedup in speedups.items() if speedup == our_speedup][0]
            our_best_row = sub_df[sub_df['thread'] == our_best_thread].sort_values(by='speedup', ascending=False).iloc[0]
            our_best_param = our_best_row['parameter']

            # Get T1_best_param and T1_best_speedup
            t1_best_row = sub_df[sub_df['thread'] == 1].sort_values(by='speedup', ascending=False).iloc[0]
            t1_best_param = t1_best_row['parameter']
            t1_best_speedup = t1_best_row['speedup']

            t2_best_row = sub_df[sub_df['thread'] == 1].sort_values(by='speedup', ascending=False).iloc[0]
            t2_best_param = t2_best_row['parameter']
            t2_best_speedup = t2_best_row['speedup']

            t4_best_row = sub_df[sub_df['thread'] == 4].sort_values(by='speedup', ascending=False).iloc[0]
            t4_best_param = t4_best_row['parameter']
            t4_best_speedup = t4_best_row['speedup']

            t8_best_row = sub_df[sub_df['thread'] == 8].sort_values(by='speedup', ascending=False).iloc[0]
            t8_best_param = t8_best_row['parameter']
            t8_best_speedup = t8_best_row['speedup']

            # Measure t1_best_param under T8
            p2_speedup = 0
            thread_row = sub_df[(sub_df['thread'] == 8) & (sub_df['parameter'] == t1_best_param)]
            if not thread_row.empty:
                p2_speedup = max(p2_speedup, thread_row['speedup'].max())
            for strategy, speedup in [("clang", 1), ("triton", p2_speedup), ("OUR", our_speedup)]:
            # for strategy, speedup in [("triton-T1", t1_best_speedup), ("triton-T4", t4_best_speedup), ("triton-T8", t8_best_speedup), ("OUR", our_speedup)]:
                best_thread_rows.append({
                    "benchmark": benchmark,
                    "shape": shape,
                    "strategy": strategy,
                    "speedup": speedup
                })
            best_params_rows.append({
                "benchmark": benchmark,
                "shape": shape,
                "T1_best_param": t1_best_param,
                "T2_best_param": t2_best_param,
                "T4_best_param": t4_best_param,
                "T8_best_param": t8_best_param,
                "T1_best_speedup": t1_best_speedup,
                "T2_best_speedup": t2_best_speedup,
                "T4_best_speedup": t4_best_speedup,
                "T8_best_speedup": t8_best_speedup,
                # "P1_speedup": t4_best_speedup,
                # "P2_speedup": p2_speedup,
                "our_best_thread": our_best_thread,
                "our_best_param": our_best_param,
                "our_best_speedup": our_speedup
            })

    triton_thread_df = pd.DataFrame(best_thread_rows)
    triton_thread_df.to_csv(output_synergy_csv, index=False)
    triton_params_df = pd.DataFrame(best_params_rows)
    triton_params_df.to_csv(output_params_csv, index=False)


if __name__ == "__main__":

    get_best_thread('./performance_report_overall_rv.csv', 'performance_report_strategy_rv.csv', 'performance_report_params_rv.csv')
    plot_all_benchmarks_bar_chart('performance_report_strategy_rv.csv', "rv")

    get_best_thread('./performance_report_overall_x86.csv', 'performance_report_strategy_x86.csv', 'performance_report_params_x86.csv')
    plot_all_benchmarks_bar_chart('performance_report_strategy_x86.csv', "x86")
