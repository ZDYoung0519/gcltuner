import argparse
import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    CLIPVisionModel,
    CLIPTextModel,
    CLIPTokenizer
)
from mmengine.config import Config, DictAction
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Compute ressa init')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--output-dir', default='./init_model', help='output directory for saved model')
    parser.add_argument('--output-path-lora', default='./init_lora.pth', help='output directory for saved model')
    parser.add_argument('--energy-ratio', default=0.99, type=float, help='energy ratio')
    parser.add_argument('--max-lora-rank', default=256, type=float, help='energy ratio')
    args = parser.parse_args()
    return args

def svd_decompose_with_energy_ratio(matrix, energy_ratio=0.9, max_lora_rank=128):
    """
    使用SVD分解矩阵，根据能量比确定主成分数量
    
    Args:
        matrix: 形状为 (out_d, in_d) 的权重矩阵
        energy_ratio: 保留的能量比例 (0-1)
    
    Returns:
        U: 主成分的左奇异向量 (out_d, rank)
        S: 主成分的奇异值 (rank,)
        Vh: 主成分的右奇异向量 (rank, in_d)
        U_residual: 残差的左奇异向量 (out_d, residual_rank)
        S_residual: 残差的奇异值 (residual_rank,)
        Vh_residual: 残差的右奇异向量 (residual_rank, in_d)
        rank: 主成分的秩
        residual_rank: 残差的秩
    """
    # matrix: (out_d, in_d)
    out_d, in_d = matrix.shape
    
    # 计算SVD: matrix = U @ S @ Vh
    U, S, Vh = torch.linalg.svd(matrix)
    print(U.shape,  S.shape, Vh.shape)
    max_rank = len(S)
    
    # 计算能量（奇异值的平方）
    energy = S ** 2
    total_energy = energy.sum()
    
    # 计算累积能量比
    cumulative_energy = torch.cumsum(energy, dim=0)
    cumulative_ratio = cumulative_energy / total_energy
    
    # 根据能量比确定主成分的秩
    # 找到第一个累积能量比 >= energy_ratio 的位置
    mask = cumulative_ratio >= energy_ratio
    if mask.any():
        rank = mask.nonzero()[0].item() + 1
    else:
        rank = max_rank
    rank = min(rank, max_rank)  # 确保不超过最大秩
    
    # 确保剩余部分（残差）最多有 max_lora_rank 个
    # 因此主成分至少要有 max_rank - max_lora_rank 个
    min_main_rank = max(max_rank - max_lora_rank, 0)
    rank = max(rank, min_main_rank)
    
    # 再次确保 rank 不超过 max_rank
    rank = min(rank, max_rank)
    
    # 计算残差秩
    residual_rank = max_rank - rank
    residual_rank = min(residual_rank, max_lora_rank)

    rank = int(rank)
    residual_rank = int(residual_rank)
    print(rank, residual_rank)
    
    # 如果残差秩为0，说明没有残差
    if residual_rank == 0:
        # 主成分: 所有
        U_principal = U[:, :rank]
        S_principal = S[:rank]
        Vh_principal = Vh[:rank, :]
        
        # 残差: 空
        U_residual = torch.tensor([], device=U.device)
        S_residual = torch.tensor([], device=S.device)
        Vh_residual = torch.tensor([], device=Vh.device)
        
        print(f"  No residual components (rank={rank}, max_rank={max_rank})")
    else:
        # 主成分: 前rank个
        print(rank)
        U_principal = U[:, :rank]
        S_principal = S[:rank]
        Vh_principal = Vh[:rank, :]
        
        # 残差: 剩余的
        U_residual = U[:, rank:(rank+residual_rank)]
        S_residual = S[rank:(rank+residual_rank)]
        Vh_residual = Vh[rank:(rank+residual_rank), :]
    
    return (U_principal, S_principal, Vh_principal, 
            U_residual, S_residual, Vh_residual, rank, residual_rank)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    llm_name_or_path = cfg.model['llm']['pretrained_model_name_or_path']
    
    # 1. 加载预训练模型（原始权重）
    print(f"Loading model from {llm_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="auto"  # 可选，自动分配设备
    )
    
    # 2. 对模型参数进行初始化处理
    print(f"Applying initialization...")
    energy_ratio = args.energy_ratio
    max_lora_rank = args.max_lora_rank
    
    lora_params = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if "lm_head" in name:
                continue

            print(f"\nProcessing layer: {name}")
            print(f"  Weight shape: {module.weight.shape}")
            
            # 获取权重矩阵 (out_d, in_d)
            weight = module.weight.data.clone()
            out_d, in_d = weight.shape
            
            # SVD分解
            U_p, S_p, Vh_p, U_r, S_r, Vh_r, rank, res_rank = \
                svd_decompose_with_energy_ratio(weight, energy_ratio, max_lora_rank)
            print(U_p.shape, S_p.shape, Vh_p.shape, U_r.shape, S_r.shape, Vh_r.shape)
            
            print(f"  Principal rank: {rank}")
            print(f"  Residual rank: {res_rank}")

            if res_rank <= 0:
                continue
            
            # 1. 使用主成分重新初始化权重
            # W_principal = U_p @ diag(S_p) @ Vh_p

            W_principal = U_p @ torch.diag(S_p) @ Vh_p
            module.weight.data.copy_(W_principal.to(module.weight.dtype))

            # 2. 使用剩余部分初始化lora
            A = Vh_r.T        # d_in*r, r*r => d_in*r
            B = U_r.T         # r*r, r*d_out=>r*d_out
            lora_params[name] = {
                    'A': A.detach().cpu(),
                    'B': B.detach().cpu(),
                    'S': S_r,
                    'rank': res_rank
            }
            
            print(f"Module name: {name}")
            print(f"  Weight shape: {weight.shape}")
            print(f"  Principal rank: {rank}")
            print(f"  Residual rank: {res_rank}")
            print(f"  Total rank: {rank + res_rank}")

            W_reconstructed = W_principal + (A @ torch.diag(S_r)@ B).T
            error = torch.norm(weight - W_reconstructed) / torch.norm(weight)
            print(f"  Reconstruction error: {error:.6f}")
    
    # 4. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 5. 保存模型和tokenizer，以便AutoModelForCausalLM可以直接读取
    print(f"Saving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    
    # 6. 保存配置文件（可选，但推荐）
    model.config.save_pretrained(args.output_dir)

    torch.save(lora_params, args.output_path_lora)
    
    print(f"Model successfully saved to {args.output_dir}")
    
    # 7. 验证：重新加载保存的模型（可选）
    print("\nVerifying saved model...")
    loaded_model = AutoModelForCausalLM.from_pretrained(
        args.output_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded successfully!")

if __name__ == '__main__':
    main()