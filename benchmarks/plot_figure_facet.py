import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import ast

def plot_parameters_heat_map(input_csv, benchmark, thread):

    df = pd.read_csv(input_csv)
    shapes = df[(df['benchmark'] == benchmark) & 
                (df['method'] == 'triton') & 
                (df['thread'] == thread)]['shape'].unique()
    for shape in shapes:
        df = df[(df['benchmark'] == benchmark) & 
                (df['method'] == 'triton') & 
                (df['shape'] == shape) & 
                (df['thread'] == thread)]
        df['parameter'] = df['parameter'].apply(ast.literal_eval)

        input_size = ast.literal_eval(shape)
        df['BLOCK_M'] = df['parameter'].apply(lambda x: x[0])
        df['BLOCK_N'] = df['parameter'].apply(lambda x: x[1])

        # df = df[(df['BLOCK_M'] < input_size[0]) & (df['BLOCK_N'] < input_size[1])]
        # print(input_size[0])
        df = df[(df['BLOCK_M'] < input_size[0])]
        print(df)
        break

        x_param = 'BLOCK_M'
        y_param = 'BLOCK_N'
        
        pivot = df.pivot_table(values='speedup', index=y_param, columns=x_param)
        if pivot.empty:
            print(f"Skipping heatmap for {benchmark} shape {shape}, thread {thread} as no data is available.")
            continue
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap='YlGnBu')
        plt.title(f"Parameter Sensitivity Heatmap of {benchmark} shape {shape}, thread {thread}")
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        output_file = f"./heat_map_{benchmark}_{input_size}_T{thread}_{x_param}_{y_param}.png"
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
    input_size = ast.literal_eval(shape)[0]
    df['BLOCK_M'] = df['parameter'].apply(lambda x: x[0])
    df['BLOCK_N'] = df['parameter'].apply(lambda x: x[1])
    df['BLOCK_K'] = df['parameter'].apply(lambda x: x[2])

    # Filter parameters where BLOCK_M, BLOCK_N, and BLOCK_K are all less than 64
    df = df[(df['BLOCK_M'] < input_size) & (df['BLOCK_N'] < input_size) & (df['BLOCK_K'] < input_size)]
    # print(df)

    g = sns.FacetGrid(df, col="BLOCK_K", col_wrap=4, height=4)
    g.map_dataframe(lambda data, color: sns.heatmap(
        data.pivot(index="BLOCK_N", columns="BLOCK_M", values="speedup"),
        cmap="viridis_r", cbar=True, annot=True, fmt=".1f"
    ))
    g.set_titles("BLOCK_K = {col_name}")
    # plt.title(f"Parameter Sensitivity Heatmap of matmul shape {shape}, thread {thread}")
    plt.tight_layout()
    output_file = f"./facet_heatmaps_matmul_{input_size}_T{thread}.png"
    plt.savefig(output_file, dpi=300)
    print(f"Chart saved as {output_file}")
    plt.close()


if __name__ == "__main__":


    for benchmark in ["dropout"]:
        for thread in [1]:
            plot_parameters_heat_map('./performance_report_overall.csv', benchmark, thread)
    
    # for thread in [1, 8, 32]:
    #     for shape in [64, 128, 256, 512]:
    #         plot_parameters_facet_heatmaps('./performance_report_overall.csv', f"({shape}, {shape}, {shape}, 10)", thread)
