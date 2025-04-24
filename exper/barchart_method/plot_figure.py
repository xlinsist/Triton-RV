import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bar_chart_with_tuning_time(input_csv, benchmark):
    df = pd.read_csv(input_csv)
    df = df[df['Benchmark'] == benchmark]
    
    shapes_to_keep = {
        "matmul": ["(256, 256, 256)"],
    }
    df = df[df.apply(lambda row: row["Shape"] in shapes_to_keep.get(row["Benchmark"], []), axis=1)]
    
    # 准备数据 - 将数据从宽格式转为长格式
    df_melted = df.melt(id_vars=['Method'], 
                        value_vars=['Speedup', 'TuningTime(s)'],
                        var_name='Metric', 
                        value_name='Value')
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))
    
    # 创建分组柱状图
    ax = sns.barplot(data=df_melted, x='Method', y='Value', hue='Metric', 
                     palette={'Speedup': '#66c2a5', 'TuningTime(s)': '#fc8d62'})
    
    # 设置标签和标题
    ax.set_xlabel('Method', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.set_title(f'{benchmark} - Speedup vs Tuning Time', fontsize=16)
    
    # 调整图例
    plt.legend(title='Metrics', title_fontsize='14', fontsize='12')
    
    # 调整x轴标签旋转
    plt.xticks(rotation=45, ha="right")
    
    # 为Speedup添加数值标签
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.1f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        fontsize=10)
    
    output_file = f"./bar_chart_with_tuning_time_{benchmark}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Bar chart saved as {output_file}")
    plt.close()


def plot_dual_axis_bar_chart(input_csv, benchmark):
    df = pd.read_csv(input_csv)
    df = df[df['Benchmark'] == benchmark]
    
    shapes_to_keep = {
        "matmul": ["(256, 256, 256)"],
        "softmax": ["(512, 512)"],
    }
    df = df[df.apply(lambda row: row["Shape"] in shapes_to_keep.get(row["Benchmark"], []), axis=1)]
    
    # 排序数据以便更好的可视化
    df = df.sort_values('Speedup', ascending=False)
    
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(6, 8))
    
    # 设置x轴位置
    x = range(len(df))
    
    # 第一个y轴（左侧）- Speedup
    color = '#66c2a5'
    ax1.set_xlabel('Method', fontsize=14)
    ax1.set_ylabel('Speedup', fontsize=14, color=color)
    bar_width = 0.8  # 缩小柱状图宽度
    bars = ax1.bar([i - bar_width/2 for i in x], df['Speedup'], 
               width=bar_width, color=color, alpha=0.6, label='Speedup')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 添加Speedup数值标签
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom',
                 color=color, fontsize=10)
    
    tuning_time_data = df[df['TuningTime(s)'] != 0.0]
    x_filtered = [x[i] for i in range(len(df)) if df['TuningTime(s)'].iloc[i] != 0.0]

    # 第二个y轴（右侧）- TuningTime
    ax2 = ax1.twinx()
    color = '#fc8d62'
    ax2.set_ylabel('Tuning Time (s)', fontsize=14, color=color)
    line = ax2.plot(x_filtered, tuning_time_data['TuningTime(s)'], color=color, marker='o', 
                linestyle='--', linewidth=2, markersize=8, 
                label='Tuning Time')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 添加TuningTime数值标签
    for i, (xi, txt) in enumerate(zip(x_filtered, tuning_time_data['TuningTime(s)'])):
        ax2.annotate(f"{txt:.1f}", (xi, txt),
                    textcoords="offset points", xytext=(0,10),
                    ha='center', va='bottom', color=color, fontsize=10)
    
    # 设置x轴标签
    plt.xticks(x, df['Method'], rotation=45, ha="right")
    
    # 合并图例
    lines = [bars, line[0]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=12)
    
    plt.title(f'{benchmark} - Speedup vs Tuning Time', fontsize=16, pad=20)
    
    # 调整布局防止标签重叠
    plt.tight_layout()
    
    output_file = f"./dual_axis_chart_{benchmark}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Dual axis chart saved as {output_file}")
    plt.close()


def plot_bar_chart(input_csv, benchmark):
    df = pd.read_csv(input_csv)

    # df = df[df['TuningTime(s)'] != 0.0]
    df = df[df['Benchmark'] == benchmark]
    shapes_to_keep = {
        "matmul": ["(256, 256, 256)"],
    }
    df = df[df.apply(lambda row: row["Shape"] in shapes_to_keep.get(row["Benchmark"], []), axis=1)]

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 8))
    ax = sns.barplot(data=df, x="Benchmark", y="Speedup", hue="Method", palette="Set2")

    ax.set_xlabel(f"Benchmark", fontsize=14)
    ax.set_ylabel("Speedup", fontsize=14)
    plt.xticks(rotation=45, ha="right")

    plt.legend(title='Tuning Methods', title_fontsize='14', fontsize='14')

    output_file = f"./bar_chart_{benchmark}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Bar chart saved as {output_file}")

    plt.close()


if __name__ == "__main__":

    for benchmark in ["softmax"]:
        # plot_bar_chart_with_tuning_time(f'./performance_report.csv', benchmark)
        plot_dual_axis_bar_chart(f'./performance_report.csv', benchmark)
