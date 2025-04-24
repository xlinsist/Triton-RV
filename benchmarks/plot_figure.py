import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import ast

def plot_parameters_heat_map(input_csv, shape, thread):
    df = pd.read_csv(input_csv)
    df = df[(df['benchmark'] == 'matmul') & 
            (df['method'] == 'triton') & 
            (df['shape'] == shape) & 
            (df['thread'] == thread)]

    df['parameter'] = df['parameter'].apply(ast.literal_eval)

    df['BLOCK_M'] = df['parameter'].apply(lambda x: x[0])
    df['BLOCK_N'] = df['parameter'].apply(lambda x: x[1])
    df['TILE_K'] = df['parameter'].apply(lambda x: x[2])

    x_param = 'BLOCK_M'
    y_param = 'BLOCK_N'
    
    pivot = df.pivot_table(values='speedup', index=y_param, columns=x_param)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap='YlGnBu')
    plt.title(f"Parameter Sensitivity Heatmap of matmul shape {shape}, thread {thread}")
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    output_file = f"./heat_map_matmul_{ast.literal_eval(shape)[0]}_T{thread}_{x_param}_{y_param}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Chart saved as {output_file}")
    plt.close()


def plot_parameters_facet_heatmaps(input_csv, shape, thread):

    df = pd.read_csv(input_csv)
    df = df[(df['benchmark'] == 'matmul') & 
            (df['method'] == 'triton') & 
            (df['shape'] == shape) & 
            (df['thread'] == thread)]

    df['parameter'] = df['parameter'].apply(ast.literal_eval)
    df['BLOCK_M'] = df['parameter'].apply(lambda x: x[0])
    df['BLOCK_N'] = df['parameter'].apply(lambda x: x[1])
    df['BLOCK_K'] = df['parameter'].apply(lambda x: x[2])

    # Filter parameters where BLOCK_M, BLOCK_N, and TILE_K are all less than 64
    df = df[(df['BLOCK_M'] < 64) & (df['BLOCK_N'] < 64) & (df['BLOCK_K'] < 64)]
    # print(df)

    g = sns.FacetGrid(df, col="BLOCK_K", col_wrap=4, height=4)
    g.map_dataframe(lambda data, color: sns.heatmap(
        data.pivot(index="BLOCK_N", columns="BLOCK_M", values="speedup"),
        cmap="viridis_r", cbar=True, annot=True, fmt=".1f"
    ))
    g.set_titles("BLOCK_K = {col_name}")
    # plt.title(f"Parameter Sensitivity Heatmap of matmul shape {shape}, thread {thread}")
    plt.tight_layout()
    output_file = f"./facet_heatmaps_matmul_{ast.literal_eval(shape)[0]}_T{thread}.png"
    plt.savefig(output_file, dpi=300)
    print(f"Chart saved as {output_file}")
    plt.close()


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


def calculate_parameter_matrix(input_csv):
    BLOCK_SIZE_H = [1, 2, 4]
    BLOCK_SIZE_W = [4, 8, 16, 32]
    parameter_matrix = [(h, w) for h in BLOCK_SIZE_H for w in BLOCK_SIZE_W]


    BLOCK_SIZE_M = [8, 16, 32]
    BLOCK_SIZE_N = [8, 16, 32]
    BLOCK_SIZE_K = [16, 32, 64]
    parameter_matrix = [(m*n, k) for m in BLOCK_SIZE_M for n in BLOCK_SIZE_N for k in BLOCK_SIZE_K]

    num_matrix = np.zeros((len(BLOCK_SIZE_H), len(BLOCK_SIZE_W)))

    df = pd.read_csv(input_csv)
    df = df[(df['benchmark'] == 'correlation') & (df['method'] == "triton_tuned") & (df['thread'] == 8)]

    for _, row in df.iterrows():
        # shape = eval(row['shape'])
        parameter = eval(row['parameter'])  # Assuming parameter is stored as a string representation of a tuple
        # print(shape, parameter)
        if parameter in parameter_matrix:
            h_index = BLOCK_SIZE_H.index(parameter[0])
            w_index = BLOCK_SIZE_W.index(parameter[1])
            num_matrix[h_index, w_index] += 1

    print("number of Parameter Matrix:")
    print(num_matrix)


def get_best_thread(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    triton_df = df[df['method'] == "triton"]
    triton_tuned_df = df[df['method'] == "triton_tuned"]
    unique_shapes = triton_tuned_df['shape'].unique()
    
    best_thread_rows = []

    for shape in unique_shapes:
        sub_df = triton_tuned_df[triton_tuned_df['shape'] == shape]
        times = {thread: sub_df.loc[sub_df['thread'] == thread, 'time(s)'].min() for thread in [1, 8, 32]}
        min_time = min(times.values())

        best_thread = [thread for thread, time in times.items() if time == min_time][0]

        param_thread_1 = sub_df.loc[sub_df['thread'] == 1, 'parameter'].values[0]
        triton_T1_sub_df = triton_df[(triton_df['shape'] == shape) & (triton_df['thread'] == best_thread) & (triton_df['parameter'] == param_thread_1)]
        time_with_param_T1 = triton_T1_sub_df['time(s)'].values[0]
        speedup_with_param_T1 = triton_T1_sub_df['speedup'].values[0]

        best_thread_row = sub_df[sub_df['thread'] == best_thread].copy()
        best_thread_row['param_T1'] = param_thread_1
        best_thread_row['speedup_with_param_T1'] = speedup_with_param_T1
        best_thread_row['extra_time_cost_ratio_with_param_T1'] = str(round((time_with_param_T1 - min_time) / min_time * 100, 2)) + "%"
        best_thread_rows.append(best_thread_row)

    best_thread_df = pd.concat(best_thread_rows)
    best_thread_df.rename(columns={'parameter': 'param_best', 'thread': 'thread_best'}, inplace=True)
    best_thread_df.drop(columns=['time(s)', 'method', 'speedup', 'speedup_with_param_T1'], inplace=True)

    shapes_to_keep = {
            "matmul": ["(32, 32, 32, 10)", "(64, 64, 64, 10)", "(128, 128, 128, 10)", "(256, 256, 256, 10)", "(512, 512, 512, 10)"],
            "dropout": ["(1024, 10)", "(4096, 10)", "(16384, 10)", "(65536, 10)", "(262144, 10)", "(1048576, 10)", "(4194304, 10)"],
            "layernorm": ["(64, 64, 10)", "(256, 256, 10)", "(256, 1024, 10)", "(1024, 1024, 10)", "(4096, 4096, 10)"],
            "correlation": ["(16, 16, 64, 64, 10)", "(32, 32, 64, 64, 10)", "(64, 64, 64, 64, 10)", "(64, 64, 128, 128, 10)", "(64, 64, 256, 256, 10)", "(64, 64, 512, 512, 10)"],
            "softmax": ["(32, 1024, 10)", "(128, 1024, 10)", "(512, 1024, 10)", "(512, 4096, 10)", "(512, 16384, 10)"]
        }
    # shapes_to_keep = {
    #     "matmul": ["(32, 32, 32, 10)", "(64, 64, 64, 10)", "(128, 128, 128, 10)", "(256, 256, 256, 10)", "(512, 512, 512, 10)"],
    #     "dropout": ["(1024, 10)", "(16384, 10)", "(262144, 10)", "(4194304, 10)"],
    #     "layernorm": ["(64, 64, 10)", "(256, 256, 10)", "(1024, 1024, 10)", "(4096, 4096, 10)"],
    #     "correlation": ["(1, 1, 64, 64, 10)", "(16, 16, 64, 64, 10)", "(32, 32, 64, 64, 10)", "(64, 64, 64, 64, 10)"],
    #     "softmax": ["(32, 1024, 10)", "(128, 1024, 10)", "(512, 1024, 10)", "(512, 4096, 10)", "(512, 16384, 10)"]
    # }
    best_thread_df = best_thread_df[best_thread_df.apply(lambda row: row["shape"] in shapes_to_keep.get(row["benchmark"], []), axis=1)]

    best_thread_df["shape"] = best_thread_df["shape"].apply(lambda x: str(tuple(eval(x)[:-1])))

    best_thread_df.to_csv(output_csv, index=False)


if __name__ == "__main__":


    # plot_parameters_heat_map('./performance_report_overall.csv', "(128, 128, 128, 10)", 32)
    # plot_parameters_facet_heatmaps('./performance_report_overall.csv', "(128, 128, 128, 10)", 32)
    plot_parameters_facet_heatmaps('./performance_report_overall.csv', "(64, 64, 64, 10)", 1)

    # filter_triton_tuned('./performance_report_overall.csv', 'performance_report_selected.csv')
    # get_best_thread('./performance_report_overall.csv', 'performance_report_best_thread.csv')

    # for benchmark in ["matmul", "layernorm", "correlation", "dropout", "softmax"]:
    #     plot_threads_bar_chart(f'performance_report_selected.csv', benchmark)

    # calculate_parameter_matrix('./performance_report_overall.csv')
