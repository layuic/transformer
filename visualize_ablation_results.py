"""
可视化消融实验结果
生成对比图表和分析报告
"""
import os
import csv
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
# 配置中文字体（在常见中文字体中就地选择可用的一个）
from matplotlib import font_manager
_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
for _font in ["Microsoft YaHei", "SimHei", "Source Han Sans CN", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Sarasa UI SC"]:
    if _font in _available_fonts:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [_font]
        matplotlib.rcParams['axes.unicode_minus'] = False
        break
import pandas as pd

def load_results():
    """加载实验结果"""
    summary_file = "results/ablation/ablation_summary.csv"
    json_file = "results/ablation/ablation_results.json"
    
    results = []
    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
    
    return results

def plot_comparison(results):
    """绘制对比图表"""
    results_dir = "results/ablation"
    os.makedirs(results_dir, exist_ok=True)
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 提取数值列
    df['Final Val Loss'] = pd.to_numeric(df['Final Val Loss'], errors='coerce')
    df['Best Val Loss'] = pd.to_numeric(df['Best Val Loss'], errors='coerce')
    df['Heads'] = pd.to_numeric(df['Heads'], errors='coerce')
    df['d_ff'] = pd.to_numeric(df['d_ff'], errors='coerce')
    df['Dropout'] = pd.to_numeric(df['Dropout'], errors='coerce')
    df['Encoder Layers'] = pd.to_numeric(df['Encoder Layers'], errors='coerce')
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 所有实验的验证损失对比
    ax1 = axes[0, 0]
    experiments = df['Experiment'].tolist()
    val_losses = df['Best Val Loss'].tolist()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    ax1.bar(experiments, val_losses, color=colors[:len(experiments)])
    ax1.set_xlabel('实验配置', fontsize=12)
    ax1.set_ylabel('最佳验证损失', fontsize=12)
    ax1.set_title('消融实验：最佳验证损失对比', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. 不同注意力头数的影响
    ax2 = axes[0, 1]
    heads_experiments = df[df['Experiment'].isin(['baseline', 'heads_1', 'heads_8'])]
    ax2.bar(heads_experiments['Experiment'], heads_experiments['Best Val Loss'], 
            color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_xlabel('注意力头数配置', fontsize=12)
    ax2.set_ylabel('最佳验证损失', fontsize=12)
    ax2.set_title('注意力头数影响', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 不同Dropout率的影响
    ax3 = axes[1, 0]
    dropout_experiments = df[df['Experiment'].isin(['baseline', 'dropout_0.0', 'dropout_0.3'])]
    ax3.bar(dropout_experiments['Experiment'], dropout_experiments['Best Val Loss'],
            color=['#1f77b4', '#d62728', '#9467bd'])
    ax3.set_xlabel('Dropout率配置', fontsize=12)
    ax3.set_ylabel('最佳验证损失', fontsize=12)
    ax3.set_title('Dropout率影响', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 不同层数的影响
    ax4 = axes[1, 1]
    layers_experiments = df[df['Experiment'].isin(['baseline', 'layers_2', 'layers_4'])]
    ax4.bar(layers_experiments['Experiment'], layers_experiments['Best Val Loss'],
            color=['#1f77b4', '#8c564b', '#e377c2'])
    ax4.set_xlabel('层数配置', fontsize=12)
    ax4.set_ylabel('最佳验证损失', fontsize=12)
    ax4.set_title('Encoder/Decoder层数影响', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(results_dir, "ablation_comparison.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"对比图表已保存到: {output_path}")
    
    # 绘制前馈网络维度影响
    fig, ax = plt.subplots(figsize=(10, 6))
    dff_experiments = df[df['Experiment'].isin(['baseline', 'dff_512', 'dff_2048'])]
    ax.bar(dff_experiments['Experiment'], dff_experiments['Best Val Loss'],
           color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_xlabel('前馈网络维度配置', fontsize=12)
    ax.set_ylabel('最佳验证损失', fontsize=12)
    ax.set_title('前馈网络维度 (d_ff) 影响', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path2 = os.path.join(results_dir, "dff_comparison.png")
    plt.savefig(output_path2, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"前馈网络维度对比图已保存到: {output_path2}")

def generate_analysis_report(results):
    """生成分析报告"""
    results_dir = "results/ablation"
    report_file = os.path.join(results_dir, "ablation_analysis.txt")
    
    df = pd.DataFrame(results)
    df['Best Val Loss'] = pd.to_numeric(df['Best Val Loss'], errors='coerce')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("消融实验结果分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 总体结果
        f.write("1. 实验结果汇总\n")
        f.write("-" * 60 + "\n")
        f.write(f"总实验数: {len(results)}\n")
        f.write(f"最佳配置: {df.loc[df['Best Val Loss'].idxmin(), 'Experiment']}\n")
        f.write(f"最佳验证损失: {df['Best Val Loss'].min():.6f}\n\n")
        
        # 注意力头数分析
        f.write("2. 注意力头数影响分析\n")
        f.write("-" * 60 + "\n")
        heads_experiments = df[df['Experiment'].isin(['baseline', 'heads_1', 'heads_8'])]
        for _, row in heads_experiments.iterrows():
            heads_val = float(row['Heads']) if pd.notna(row['Heads']) else 0
            f.write(f"{row['Experiment']:20s} - 头数: {heads_val:2.0f}, "
                   f"验证损失: {float(row['Best Val Loss']):.6f}\n")
        best_heads = heads_experiments.loc[heads_experiments['Best Val Loss'].idxmin(), 'Experiment']
        f.write(f"\n结论: {best_heads} 表现最佳\n\n")
        
        # Dropout分析
        f.write("3. Dropout率影响分析\n")
        f.write("-" * 60 + "\n")
        dropout_experiments = df[df['Experiment'].isin(['baseline', 'dropout_0.0', 'dropout_0.3'])]
        for _, row in dropout_experiments.iterrows():
            dropout_val = float(row['Dropout']) if pd.notna(row['Dropout']) else 0
            f.write(f"{row['Experiment']:20s} - Dropout: {dropout_val:3.1f}, "
                   f"验证损失: {float(row['Best Val Loss']):.6f}\n")
        best_dropout = dropout_experiments.loc[dropout_experiments['Best Val Loss'].idxmin(), 'Experiment']
        f.write(f"\n结论: {best_dropout} 表现最佳\n\n")
        
        # 层数分析
        f.write("4. 层数影响分析\n")
        f.write("-" * 60 + "\n")
        layers_experiments = df[df['Experiment'].isin(['baseline', 'layers_2', 'layers_4'])]
        for _, row in layers_experiments.iterrows():
            layers_val = float(row['Encoder Layers']) if pd.notna(row['Encoder Layers']) else 0
            f.write(f"{row['Experiment']:20s} - 层数: {layers_val:2.0f}, "
                   f"验证损失: {float(row['Best Val Loss']):.6f}\n")
        best_layers = layers_experiments.loc[layers_experiments['Best Val Loss'].idxmin(), 'Experiment']
        f.write(f"\n结论: {best_layers} 表现最佳\n\n")
        
        # 前馈网络维度分析
        f.write("5. 前馈网络维度影响分析\n")
        f.write("-" * 60 + "\n")
        dff_experiments = df[df['Experiment'].isin(['baseline', 'dff_512', 'dff_2048'])]
        for _, row in dff_experiments.iterrows():
            dff_val = float(row['d_ff']) if pd.notna(row['d_ff']) else 0
            f.write(f"{row['Experiment']:20s} - d_ff: {dff_val:5.0f}, "
                   f"验证损失: {float(row['Best Val Loss']):.6f}\n")
        best_dff = dff_experiments.loc[dff_experiments['Best Val Loss'].idxmin(), 'Experiment']
        f.write(f"\n结论: {best_dff} 表现最佳\n\n")
        
        # 总结
        f.write("=" * 60 + "\n")
        f.write("总结\n")
        f.write("=" * 60 + "\n")
        f.write(f"所有实验均已完成，共 {len(results)} 个配置。\n")
        f.write(f"最佳配置为: {df.loc[df['Best Val Loss'].idxmin(), 'Experiment']}\n")
        f.write(f"验证损失: {df['Best Val Loss'].min():.6f}\n")
    
    print(f"分析报告已保存到: {report_file}")

def main():
    print("=" * 60)
    print("生成消融实验结果可视化")
    print("=" * 60)
    
    # 加载结果
    results = load_results()
    
    if not results:
        print("错误: 未找到实验结果")
        return
    
    print(f"\n加载了 {len(results)} 个实验结果")
    
    # 生成可视化
    print("\n生成对比图表...")
    plot_comparison(results)
    
    # 生成分析报告
    print("\n生成分析报告...")
    generate_analysis_report(results)
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()

