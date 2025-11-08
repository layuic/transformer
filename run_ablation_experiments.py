"""
运行消融实验并收集结果
测试不同的超参数组合对模型性能的影响
"""
import os
import subprocess
import json
import csv
from pathlib import Path

def run_experiment(config, experiment_name, results_dir):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"运行实验: {experiment_name}")
    print(f"配置: {config}")
    print(f"{'='*60}\n")
    
    # 构建命令
    cmd = [
        "python", "train_transformer.py",
        "--data_dir", "data",
        "--data_source", "simple",
        "--level", "word",
        "--max_len", str(config.get("max_len", 128)),
        "--d_model", str(config["d_model"]),
        "--num_encoder_layers", str(config["num_encoder_layers"]),
        "--num_decoder_layers", str(config["num_decoder_layers"]),
        "--num_heads", str(config["num_heads"]),
        "--d_ff", str(config["d_ff"]),
        "--dropout", str(config["dropout"]),
        "--batch_size", str(config.get("batch_size", 32)),
        "--epochs", str(config.get("epochs", 10)),
        "--lr", str(config.get("lr", 1e-4)),
        "--seed", str(config.get("seed", 42)),
        "--save_dir", os.path.join("checkpoints", experiment_name)
    ]
    
    # 运行实验
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 读取结果
        checkpoint_path = os.path.join("checkpoints", experiment_name, "best_model.pt")
        if os.path.exists(checkpoint_path):
            import torch
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            train_losses = checkpoint.get("train_losses", [])
            val_losses = checkpoint.get("val_losses", [])
            
            result_data = {
                "experiment": experiment_name,
                "config": config,
                "final_train_loss": train_losses[-1] if train_losses else None,
                "final_val_loss": val_losses[-1] if val_losses else None,
                "best_val_loss": checkpoint.get("best_val_loss", None),
                "num_epochs": len(train_losses),
                "checkpoint_path": checkpoint_path
            }
            
            # 保存实验特定的结果
            exp_results_dir = os.path.join(results_dir, experiment_name)
            os.makedirs(exp_results_dir, exist_ok=True)
            
            # 复制训练曲线
            if os.path.exists("results/training_curves.png"):
                import shutil
                shutil.copy("results/training_curves.png", 
                          os.path.join(exp_results_dir, "training_curves.png"))
                shutil.copy("results/training_losses.csv", 
                          os.path.join(exp_results_dir, "training_losses.csv"))
            
            return result_data
        else:
            print(f"警告: 检查点文件不存在: {checkpoint_path}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"实验失败: {e}")
        print(f"错误输出: {e.stderr}")
        return None

def main():
    # 创建结果目录
    results_dir = "results/ablation"
    os.makedirs(results_dir, exist_ok=True)
    
    # 定义实验配置
    experiments = [
        # 基础配置
        {
            "name": "baseline",
            "config": {
                "d_model": 256,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "num_heads": 4,
                "d_ff": 1024,
                "dropout": 0.1,
                "epochs": 10,
                "batch_size": 64
            }
        },
        
        # 不同注意力头数
        {
            "name": "heads_1",
            "config": {
                "d_model": 256,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "num_heads": 1,
                "d_ff": 1024,
                "dropout": 0.1,
                "epochs": 10,
                "batch_size": 64
            }
        },
        {
            "name": "heads_8",
            "config": {
                "d_model": 256,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "num_heads": 8,
                "d_ff": 1024,
                "dropout": 0.1,
                "epochs": 10,
                "batch_size": 64
            }
        },
        
        # 不同前馈网络维度
        {
            "name": "dff_512",
            "config": {
                "d_model": 256,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "num_heads": 4,
                "d_ff": 512,
                "dropout": 0.1,
                "epochs": 10,
                "batch_size": 64
            }
        },
        {
            "name": "dff_2048",
            "config": {
                "d_model": 256,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "num_heads": 4,
                "d_ff": 2048,
                "dropout": 0.1,
                "epochs": 10,
                "batch_size": 64
            }
        },
        
        # 不同dropout率
        {
            "name": "dropout_0.0",
            "config": {
                "d_model": 256,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "num_heads": 4,
                "d_ff": 1024,
                "dropout": 0.0,
                "epochs": 10,
                "batch_size": 64
            }
        },
        {
            "name": "dropout_0.3",
            "config": {
                "d_model": 256,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "num_heads": 4,
                "d_ff": 1024,
                "dropout": 0.3,
                "epochs": 10,
                "batch_size": 64
            }
        },
        
        # 不同层数
        {
            "name": "layers_2",
            "config": {
                "d_model": 256,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "num_heads": 4,
                "d_ff": 1024,
                "dropout": 0.1,
                "epochs": 10,
                "batch_size": 64
            }
        },
        {
            "name": "layers_4",
            "config": {
                "d_model": 256,
                "num_encoder_layers": 4,
                "num_decoder_layers": 4,
                "num_heads": 4,
                "d_ff": 1024,
                "dropout": 0.1,
                "epochs": 10,
                "batch_size": 64
            }
        },
    ]
    
    # 运行所有实验
    all_results = []
    
    for exp in experiments:
        result = run_experiment(exp["config"], exp["name"], results_dir)
        if result:
            all_results.append(result)
    
    # 保存汇总结果
    summary_file = os.path.join(results_dir, "ablation_summary.csv")
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Experiment", "d_model", "Encoder Layers", "Decoder Layers", 
            "Heads", "d_ff", "Dropout", "Final Train Loss", "Final Val Loss", 
            "Best Val Loss", "Epochs"
        ])
        
        for r in all_results:
            config = r["config"]
            writer.writerow([
                r["experiment"],
                config["d_model"],
                config["num_encoder_layers"],
                config["num_decoder_layers"],
                config["num_heads"],
                config["d_ff"],
                config["dropout"],
                f'{r["final_train_loss"]:.6f}' if r["final_train_loss"] else "N/A",
                f'{r["final_val_loss"]:.6f}' if r["final_val_loss"] else "N/A",
                f'{r["best_val_loss"]:.6f}' if r["best_val_loss"] else "N/A",
                r["num_epochs"]
            ])
    
    # 保存JSON格式的详细结果
    json_file = os.path.join(results_dir, "ablation_results.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"所有实验完成!")
    print(f"结果汇总保存在: {summary_file}")
    print(f"详细结果保存在: {json_file}")
    print(f"{'='*60}\n")
    
    # 打印汇总表
    print("\n实验汇总:")
    print("-" * 100)
    print(f"{'实验名称':<20} {'最终训练损失':<15} {'最终验证损失':<15} {'最佳验证损失':<15}")
    print("-" * 100)
    for r in all_results:
        train_loss = f"{r['final_train_loss']:.6f}" if r['final_train_loss'] else "N/A"
        val_loss = f"{r['final_val_loss']:.6f}" if r['final_val_loss'] else "N/A"
        best_loss = f"{r['best_val_loss']:.6f}" if r['best_val_loss'] else "N/A"
        print(f"{r['experiment']:<20} {train_loss:<15} {val_loss:<15} {best_loss:<15}")
    print("-" * 100)

if __name__ == "__main__":
    main()

