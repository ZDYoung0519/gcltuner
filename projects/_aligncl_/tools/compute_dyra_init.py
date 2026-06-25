# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os

import torch
from tqdm import tqdm
from mmengine.config import Config
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from peft.tuners.lora import LoraLayer
from xtuner.registry import BUILDER


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLM")
    parser.add_argument("config", help="ckpt file name or path.")
    parser.add_argument("--checkpoint", help="ckpt file name or path.")
    parser.add_argument("--cur-task", default=0, type=int, help="task identity")
    parser.add_argument("--output-file", default=None, type=str, help="output file path")
    args = parser.parse_args()
    return args


def process_single_layer_svd(args):
    """对单个层的更新矩阵进行 SVD 分解，生成 LoRA 初始化权重。

    对更新矩阵进行低秩 SVD 分解，取前 lora_r 个奇异值/向量构造 LoRA 的 A/B 矩阵，
    并根据 lora_scale 进行缩放处理，使加载后的 \delta W = scale * AB 与原始更新量一致。
    """
    name, up_data, lora_r, lora_scale = args
    delta_w = torch.from_numpy(up_data) if not isinstance(up_data, torch.Tensor) else up_data
    result = {"name": name, "rank": lora_r}
    if lora_r <= 0:
        return result

    try:
        # SVD 分解：取前 q 个奇异值，q 略大于 lora_r 以提高精度
        q = min(10 * lora_r, min(delta_w.shape))
        U, S, V = torch.svd_lowrank(delta_w.float(), q=q, niter=5)
        error_svd = (U @ torch.diag_embed(S) @ V.T - delta_w.float()).norm(p=2)

        sqrt_S = torch.sqrt(S[:lora_r].clamp_min(1e-8))
        lora_B = U[:, :lora_r] @ torch.diag(sqrt_S)
        lora_A = V[:, :lora_r] @ torch.diag(sqrt_S)

        # 缩放处理：因为 lora 加载后 delta = lora_scale * B @ A，
        # 需要对 A/B 分别除以 sqrt(scale)，使得 scale * (B/s) @ (A/s)^T = B @ A^T
        scale_div = math.sqrt(float(lora_scale))
        result.update({
            "lora_A": (lora_A / scale_div).cpu().numpy(),
            "lora_B": (lora_B / scale_div).cpu().numpy(),
            "success": True,
        })
        error = (lora_B @ lora_A.T - delta_w.float()).norm(p=2)
        print(f"layer {name}, error: {error}, error_svd:{error_svd}")

    except Exception as e:
        result["error"] = str(e)
        result["success"] = False

    return result


def parallel_lora_init(updates, ranks_map, lora_scales, num_workers=4):
    """并行对多个层的更新矩阵执行 SVD，返回所有层的 LoRA 初始化权重字典。

    updates: {layer_name: 更新矩阵 (delta_w)}
    ranks_map: {layer_name: lora_rank}
    lora_scales: {layer_name: lora_scale}
    """
    tasks = []
    for name, lora_r in ranks_map.items():
        if lora_r > 0 and name in updates:
            delta_w = updates[name]
            if delta_w.dtype == torch.bfloat16:
                delta_w = delta_w.to(torch.float32)
            up_np = delta_w.cpu().numpy() if isinstance(delta_w, torch.Tensor) else delta_w
            scale = float(lora_scales.get(name, 1.0))
            tasks.append((name, up_np, lora_r, scale))

    lora_init = {}
    ctx = torch.multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        with tqdm(total=len(tasks), desc="SVD Processing") as pbar:
            for result in pool.imap_unordered(process_single_layer_svd, tasks):
                name = result["name"]

                if result.get("success"):
                    lora_init[f"{name}.lora_A"] = torch.from_numpy(result["lora_A"])
                    lora_init[f"{name}.lora_B"] = torch.from_numpy(result["lora_B"])
                    pbar.set_postfix({"layer": name[:20], "rank": result["rank"]})
                else:
                    print(f"Failed {name}: {result.get('error', 'unknown')}")
                pbar.update(1)
    return lora_init


def main():
    args = parse_args()

    # 加载配置文件
    cfg = Config.fromfile(args.config)

    from xtuner.model. utils import guess_load_checkpoint
    state_dict = guess_load_checkpoint(args.checkpoint)

    # 从 checkpoint 中提取每个层的base name
    base_layer_names = []
    for k, v in state_dict.items():
        if '.lora_A' in k or '.lora_B' in k:
            base_name = k.split('.lora_')[0]
            base_layer_names.append(base_name)
    base_layer_names = sorted(set(base_layer_names))

    def find_lora_param(base_layer_name, state_dict):
        """Return (lora_A, lora_B) weight tensors for a given base layer name."""
        key_a = f"{base_layer_name}.lora_A.default.weight"
        key_b = f"{base_layer_name}.lora_B.default.weight"
        return state_dict[key_a], state_dict[key_b]

    lora_rank = cfg.model.llm_lora['r']
    lora_alpha = cfg.model.llm_lora['lora_alpha']

    lora_scale = lora_alpha / lora_rank
    # 从 probing checkpoint 中提取每层 lora 的 \delta W = scaling * AB 作为该层的"更新量"
    # 一次性把所有 lora 搬到 GPU，减少 .to() 调用开销
    lora_pairs = [
        (key, *find_lora_param(key, state_dict))
        for key in base_layer_names
    ]
    updates = {}
    lora_scales = {}
    for key, lora_a, lora_b in tqdm(lora_pairs, desc="Computing delta W"):
        delta_weight = (lora_b.to("cuda") @ lora_a.to("cuda")) * lora_scale
        updates[key] = delta_weight.detach().cpu()
        lora_scales[key] = lora_scale

    # 计算每层更新量的 L2 范数作为重要性分数
    scores = {key: up.norm(p=2) for key, up in updates.items()}
    scores_sum = sum(scores.values())

    scale = cfg.llm_lora.get("scale", 1)
    L = len(scores)
    # 按重要性分数比例分配 rank：分数越高的层获得越大的 rank
    ranks_map = {}
    for name, score in tqdm(scores.items(), desc="Computing ranks"):
        lora_r = math.ceil(score / scores_sum * lora_rank * L * scale)
        ranks_map[name] = lora_r
        print(name, lora_r)

    # 并行 SVD 分解，得到每层的 LoRA A/B 初始化
    num_workers = 2
    lora_init = parallel_lora_init(
        updates,
        ranks_map,
        lora_scales,
        num_workers=num_workers,
    )

    # 统计参数量对比
    total_params_base = 0
    total_params_dynamic = 0

    for k, v in state_dict.items():
        if '.lora_' in k:
            total_params_base += v.numel()
    total_params_base *= lora_rank / 128

    for key, param in lora_init.items():
        total_params_dynamic += param.numel()
    print(f"Base Lora Parameters: {total_params_base}")
    print(f"Dynamic Lora Parameters: {total_params_dynamic}")

    # 与之前任务的 lora 初始化合并：加载已有文件，追加当前任务结果
    # pre_path = os.path.join(work_dir_parent, f"task{args.cur_task - 1}", "dyra_init.pth")
    pre_path = args.output_file.replace(f"/task{args.cur_task}", f"/task{args.cur_task - 1}")
    if os.path.exists(pre_path):
        lora_init_to_save = torch.load(pre_path)
    else:
        lora_init_to_save = {}

    lora_init_to_save[args.cur_task] = lora_init

    # 保存到 --output-file 指定的路径
    save_path = args.output_file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(lora_init_to_save, save_path)
    print_log(f"Successfully saved at: {save_path}")


if __name__ == "__main__":
    main()
