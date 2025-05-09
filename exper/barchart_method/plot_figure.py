import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# def plot_bar_chart_with_tuning_time(input_csv, benchmark):
#     df = pd.read_csv(input_csv)
#     df = df[df['Benchmark'] == benchmark]
    
#     shapes_to_keep = {
#         "matmul": ["(256, 256, 256)"],
#     }
#     df = df[df.apply(lambda row: row["Shape"] in shapes_to_keep.get(row["Benchmark"], []), axis=1)]
    
#     # 准备数据 - 将数据从宽格式转为长格式
#     df_melted = df.melt(id_vars=['Method'], 
#                         value_vars=['Speedup', 'TuningTime(s)'],
#                         var_name='Metric', 
#                         value_name='Value')
    
#     sns.set_theme(style="whitegrid")
#     plt.figure(figsize=(12, 8))
    
#     # 创建分组柱状图
#     ax = sns.barplot(data=df_melted, x='Method', y='Value', hue='Metric', 
#                      palette={'Speedup': '#66c2a5', 'TuningTime(s)': '#fc8d62'})
    
#     # 设置标签和标题
#     ax.set_xlabel('Method', fontsize=16)
#     ax.set_ylabel('Value', fontsize=16)
#     ax.set_title(f'{benchmark} - Speedup vs Tuning Time', fontsize=16)
    
#     # 调整图例
#     plt.legend(title='Metrics', title_fontsize='16', fontsize='16')
    
#     # 调整x轴标签旋转
#     plt.xticks(rotation=45, ha="right")
    
#     # 为Speedup添加数值标签
#     for p in ax.patches:
#         if p.get_height() > 0:
#             ax.annotate(f"{p.get_height():.1f}", 
#                         (p.get_x() + p.get_width() / 2., p.get_height()),
#                         ha='center', va='center', 
#                         xytext=(0, 10), 
#                         textcoords='offset points',
#                         fontsize=16)
    
#     output_file = f"./bar_chart_with_tuning_time_{benchmark}.png"
#     plt.savefig(output_file, dpi=300, bbox_inches="tight")
#     print(f"Bar chart saved as {output_file}")
#     plt.close()


# def plot_bar_chart(input_csv, benchmark):
#     df = pd.read_csv(input_csv)

#     # df = df[df['TuningTime(s)'] != 0.0]
#     df = df[df['Benchmark'] == benchmark]
#     shapes_to_keep = {
#         "matmul": ["(256, 256, 256)"],
#     }
#     df = df[df.apply(lambda row: row["Shape"] in shapes_to_keep.get(row["Benchmark"], []), axis=1)]

#     sns.set_theme(style="whitegrid")

#     plt.figure(figsize=(6, 8))
#     ax = sns.barplot(data=df, x="Benchmark", y="Speedup", hue="Method", palette="Set2")

#     ax.set_xlabel(f"Benchmark", fontsize=14)
#     ax.set_ylabel("Speedup", fontsize=14)
#     plt.xticks(rotation=45, ha="right")

#     plt.legend(title='Tuning Methods', title_fontsize='14', fontsize='14')

#     output_file = f"./bar_chart_{benchmark}.png"
#     plt.savefig(output_file, dpi=300, bbox_inches="tight")
#     print(f"Bar chart saved as {output_file}")

#     plt.close()


# def plot_dual_axis_bar_chart(input_csv, benchmark):
#     df = pd.read_csv(input_csv)
#     df = df[df['Benchmark'] == benchmark]
    
#     shapes_to_keep = {
#         "matmul": ["(256, 256, 256)"],
#         "softmax": ["(512, 512)"],
#         "conv2d": ["input: (8, 3, 224, 224), out: (64, 3, 7, 7)"],
#         "transpose" : ["(4096, 3072)"]
#     }
#     df = df[df.apply(lambda row: row["Shape"] in shapes_to_keep.get(row["Benchmark"], []), axis=1)]
    
#     method_order = ["OUR", "triton", "ansor", "autotvm", "tvm", "hidet", "torch"]
#     df['Method'] = pd.Categorical(df['Method'], categories=method_order, ordered=True)
#     df = df.sort_values('Method')  # 按指定顺序排序
#     sns.set_theme(style="whitegrid")
#     fig, ax1 = plt.subplots(figsize=(6, 4))
    
#     # 设置x轴位置
#     x = range(len(df))
#     plt.xticks(x, df['Method'], rotation=30, ha="right", fontsize=16)  # 增加 fontsize 参数
#     # 第一个y轴（左侧）- Speedup
#     # color = '#66c2a5'
#     # color = '#95A3C3'
#     color = '#72B6A1'
#     bar_width = 0.7  # 缩小柱状图宽度
#     bars = ax1.bar([i for i in x], df['Speedup'], 
#                width=bar_width, color=color, alpha=0.6, label='Speedup')
#     ax1.tick_params(axis='y', labelcolor=color)
    
#     # 添加Speedup数值标签
#     for bar in bars:
#         height = bar.get_height()
#         ax1.text(bar.get_x() + bar.get_width()/2., height,
#                  f'{height:.1f}',
#                  ha='center', va='bottom',
#                  color=color, fontsize=16)
    
#     tuning_time_data = df[df['TuningTime(s)'] != 0.0]
#     x_filtered = [x[i] for i in range(len(df)) if df['TuningTime(s)'].iloc[i] != 0.0]

#     # if benchmark == "matmul":
#     ax1.set_ylabel('Speedup', fontsize=16, color=color)

#     # 第二个y轴（右侧）- TuningTime
#     ax2 = ax1.twinx()
#     color = '#E99675'
#     line = ax2.plot(x_filtered, tuning_time_data['TuningTime(s)'], 
#                 color=color, 
#                 marker='o', 
#                 linestyle='none',  # 不显示折线
#                 markersize=10,     # 调大点的大小
#                 markeredgewidth=1, # 点边缘宽度
#                 markeredgecolor='k',  # 点边缘颜色（黑色）
#                 label='Tuning Time')
#     ax2.tick_params(axis='y', labelcolor=color)

#     # 获取当前y轴范围和刻度
#     current_ticks = ax2.get_yticks()
#     current_min, current_max = ax2.get_ylim()

#     # 设置新的y轴范围（最大值为原来的两倍）
#     new_max = current_max * 1.7
#     ax2.set_ylim(bottom=current_min, top=new_max)
#     ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

#     # 保持刻度间距不变
#     if len(current_ticks) > 1:
#         tick_interval = current_ticks[1] - current_ticks[0]
#         ax2.set_yticks(np.arange(current_min, max(0, new_max + tick_interval), tick_interval))

#     # 添加TuningTime数值标签
#     for i, (xi, txt) in enumerate(zip(x_filtered, tuning_time_data['TuningTime(s)'])):
#         ax2.annotate(f"{txt:.1f}", (xi, txt),
#                     textcoords="offset points", xytext=(0,0),
#                     ha='center', va='bottom', color=color, fontsize=16)

#     # 设置x轴标签
#     plt.xticks(x, df['Method'], rotation=30, ha="right")
    
#     # 合并图例
#     lines = [bars, line[0]]
#     labels = [l.get_label() for l in lines]


#     # ax1.set_xlabel('Method', fontsize=14)
#     # if benchmark == "matmul":
#     ax2.set_ylabel('Tuning Time (s)', fontsize=16, color=color)
#     ax1.legend(lines, labels, loc='upper right', fontsize=16)
    
#     # plt.title(f'{benchmark} - Speedup vs Tuning Time', fontsize=16, pad=20)
#     ax1.grid(False); ax2.grid(False)
    
#     # 调整布局防止标签重叠
#     # plt.tight_layout()
#     output_file = f"./dual_axis_chart_{benchmark}.png"
#     plt.savefig(output_file, dpi=300, bbox_inches="tight")
#     print(f"Dual axis chart saved as {output_file}")
#     plt.close()


def plot_dual_axis_bar_chart(input_csv, benchmark):
    df = pd.read_csv(input_csv)
    df = df[df['Benchmark'] == benchmark]
    
    shapes_to_keep = {
        "matmul": ["(256, 256, 256)"],
        "softmax": ["(512, 512)"],
        "conv2d": ["input: (8, 3, 224, 224), out: (64, 3, 7, 7)"],
        "transpose" : ["(4096, 3072)"]
    }
    df = df[df.apply(lambda row: row["Shape"] in shapes_to_keep.get(row["Benchmark"], []), axis=1)]
    
    method_order = ["OUR", "triton", "ansor", "autotvm", "tvm", "hidet", "torch"]
    df['Method'] = pd.Categorical(df['Method'], categories=method_order, ordered=True)
    df = df.sort_values('Method')  # 按指定顺序排序
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    # 设置x轴位置
    x = range(len(df))
    plt.xticks(x, df['Method'], rotation=30, ha="right", fontsize=18)  # 调整 fontsize 到18
    # 第一个y轴（左侧）- Speedup
    color = '#72B6A1'
    bar_width = 0.7  # 缩小柱状图宽度
    bars = ax1.bar([i for i in x], df['Speedup'], 
               width=bar_width, color=color, alpha=0.6, label='Speedup')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=18)
    
    # 添加Speedup数值标签
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom',
                 color=color, fontsize=18)
    
    tuning_time_data = df[df['TuningTime(s)'] != 0.0]
    x_filtered = [x[i] for i in range(len(df)) if df['TuningTime(s)'].iloc[i] != 0.0]

    ax1.set_ylabel('Speedup', fontsize=18, color=color)

    # 第二个y轴（右侧）- TuningTime
    ax2 = ax1.twinx()
    color = '#E99675'
    line = ax2.plot(x_filtered, tuning_time_data['TuningTime(s)'], 
                color=color, 
                marker='o', 
                linestyle='none',  # 不显示折线
                markersize=10,     # 调大点的大小
                markeredgewidth=1, # 点边缘宽度
                markeredgecolor='k',  # 点边缘颜色（黑色）
                label='Tuning Time')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=18)

    # 确保y轴最小值为0，避免出现负数
    current_min, current_max = ax2.get_ylim()
    new_max = max(current_max * 1.7, current_max + 1)  # 确保有足够的空间
    ax2.set_ylim(bottom=0, top=new_max)  # 设置最小值为0
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

    # 添加TuningTime数值标签
    for i, (xi, txt) in enumerate(zip(x_filtered, tuning_time_data['TuningTime(s)'])):
        ax2.annotate(f"{txt:.1f}", (xi, txt),
                    textcoords="offset points", xytext=(0,0),
                    ha='center', va='bottom', color=color, fontsize=18)

    # 设置x轴标签
    plt.xticks(x, df['Method'], rotation=30, ha="right", fontsize=18)
    
    # 合并图例
    lines = [bars, line[0]]
    labels = [l.get_label() for l in lines]

    ax2.set_ylabel('Tuning Time (s)', fontsize=18, color=color)
    ax1.legend(lines, labels, loc='upper right', fontsize=18)
    
    ax1.grid(False); ax2.grid(False)
    
    output_file = f"./dual_axis_chart_{benchmark}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Dual axis chart saved as {output_file}")
    plt.close()


if __name__ == "__main__":

    for benchmark in ["matmul", "softmax"]:
        # plot_bar_chart_with_tuning_time(f'./performance_report.csv', benchmark)
        plot_dual_axis_bar_chart(f'./performance_report.csv', benchmark)

    for benchmark in ["conv2d", "transpose"]:
        plot_dual_axis_bar_chart(f'./{benchmark}_performance_report.csv', benchmark)
