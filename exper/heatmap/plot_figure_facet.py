import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import ast


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
    df['speedup'] = df['speedup'].round(2)
    g = sns.FacetGrid(df, col="BLOCK_N", col_wrap=4, height=4)
    g.map_dataframe(lambda data, color: sns.heatmap(
        data.pivot(index="BLOCK_K", columns="BLOCK_M", values="speedup"),
        cmap="viridis_r", cbar=True, annot=True, fmt="0.2f",
        cbar_kws={"format": "%.2f"}, annot_kws={"fontsize": 8},
    ))
    g.set_titles("BLOCK_N = {col_name}")
    # plt.title(f"Parameter Sensitivity Heatmap of matmul shape {shape}, thread {thread}")
    plt.tight_layout()
    output_file = f"./facet_heatmaps_matmul_{input_size}_T{thread}.png"
    plt.savefig(output_file, dpi=300)
    print(f"Chart saved as {output_file}")
    plt.close()


if __name__ == "__main__":


    # for thread in [1, 8, 32]:
    #     for shape in [64, 128, 256, 512]:
    for thread in [1, 32]:
        for shape in [64]:
            plot_parameters_facet_heatmaps('./performance_report_overall.csv', f"({shape}, {shape}, {shape}, 10)", thread)
