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

    input_size = ast.literal_eval(shape)
    df['parameter'] = df['parameter'].apply(ast.literal_eval)
    df['BLOCK_M'] = df['parameter'].apply(lambda x: x[0])
    df['BLOCK_N'] = df['parameter'].apply(lambda x: x[1])
    df['BLOCK_K'] = df['parameter'].apply(lambda x: x[2])
    df = df[(df['BLOCK_M'] < input_size[0])]
    df = df[(df['BLOCK_N'] < input_size[1])]
    df = df[(df['BLOCK_K'] < input_size[2])]

    x_param = 'BLOCK_M'
    y_param = 'BLOCK_K'
    
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


if __name__ == "__main__":

    for thread in [32]:
        for shape in [64, 128, 256, 512]:
            plot_parameters_heat_map('./performance_report_overall.csv', f"({shape}, {shape}, {shape}, 10)", thread)

