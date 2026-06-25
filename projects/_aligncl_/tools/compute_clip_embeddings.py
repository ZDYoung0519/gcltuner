import argparse
import logging
import os
import os.path as osp

import torch
from mmengine.config import Config, DictAction
from mmengine.device import get_device
from mmengine.dist import all_gather, barrier, get_rank, get_world_size, is_main_process
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.model import BaseModel
from tqdm import tqdm

from xtuner.registry import BUILDER


class CLIPEmbeddingModel(BaseModel):
    def __init__(self, visual_encoder, text_encoder, text_tokenizer):
        super().__init__()
        self.visual_encoder = BUILDER.build(visual_encoder)
        self.text_encoder = BUILDER.build(text_encoder)
        self.text_tokenizer = BUILDER.build(text_tokenizer)
        self.ddp_dummy_param = torch.nn.Parameter(torch.zeros(1))

        for n, p in self.named_parameters():
            if n == "ddp_dummy_param":
                continue
            p.requires_grad_(False)

    def forward(self, data, data_samples=None, mode="tensor"):
        images = data["pixel_values"]
        texts = data["text"]
        dtype = images.dtype
        device = get_device()

        visual_outputs = self.visual_encoder(
            images.to(dtype).to(device),
            output_hidden_states=True,
        )

        clip_text_inputs = self.text_tokenizer(
            texts,
            padding="longest",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        clip_text_inputs.to(device)
        text_outputs = self.text_encoder(**clip_text_inputs)

        return {
            "vision_embeddings": visual_outputs.last_hidden_state[:, 0, :],
            "text_embeddings": text_outputs.last_hidden_state[:, 0, :],
        }


@torch.no_grad()
def collect_train_embeddings(loader, model):
    rank = get_rank()
    vision_embeddings = []
    text_embeddings = []
    pbar = tqdm(loader, desc=f"Rank {rank}: collect train embeddings", position=rank, leave=False)

    for data in pbar:
        out = model(**data, mode="tensor")
        vision_embeddings.append(out["vision_embeddings"].detach().cpu())
        text_embeddings.append(out["text_embeddings"].detach().cpu())

    vision_embeddings = torch.cat(vision_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    vision_embeddings = torch.cat(all_gather(vision_embeddings), dim=0)
    text_embeddings = torch.cat(all_gather(text_embeddings), dim=0)
    return vision_embeddings, text_embeddings


@torch.no_grad()
def collect_test_embeddings(dataset, model):
    rank = get_rank()
    world_size = get_world_size()
    device = get_device()

    n_samples = len(dataset)
    per_rank_samples = (n_samples + world_size - 1) // world_size
    per_rank_ids = range(
        per_rank_samples * rank,
        min(n_samples, per_rank_samples * (rank + 1)),
    )

    vision_embeddings = []
    text_embeddings = []
    pbar = tqdm(per_rank_ids, desc=f"Rank {rank}: collect test embeddings", position=rank, leave=False)
    for i in pbar:
        data = dataset[i]
        data["input_ids"] = data["input_ids"].to(device).unsqueeze(0)
        data["pixel_values"] = data["pixel_values"].to(device).unsqueeze(0)
        data["text"] = [data["text"]]
        out = model(data, None, "tensor")
        vision_embeddings.append(out["vision_embeddings"].detach().cpu())
        text_embeddings.append(out["text_embeddings"].detach().cpu())

    vision_embeddings = torch.cat(vision_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    vision_embeddings = torch.cat(all_gather(vision_embeddings), dim=0)
    text_embeddings = torch.cat(all_gather(text_embeddings), dim=0)
    return vision_embeddings, text_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute CLIP vision and text embeddings")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--output_dir", help="directory to save per-task CLIP embeddings")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, in key=value format",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def maybe_save_embeddings(output_dir, split, vision_embeddings, text_embeddings):
    vision_path = osp.join(output_dir, f"vision_embedding_{split}.pt")
    text_path = osp.join(output_dir, f"text_embeddings_{split}.pt")
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        torch.save(vision_embeddings, vision_path)
        torch.save(text_embeddings, text_path)
        print_log(f"Saved {split} vision embeddings to {vision_path}", logger="current")
        print_log(f"Saved {split} text embeddings to {text_path}", logger="current")
    barrier()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])

    cfg.visualizer = None
    cfg.model = dict(
        type="projects.aligncl.tools.compute_clip_embeddings.CLIPEmbeddingModel",
        visual_encoder=cfg.model["visual_encoder"],
        text_encoder=cfg.model["text_encoder"],
        text_tokenizer=cfg.model["text_tokenizer"],
    )

    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    
    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    model.eval()

    all_train_datasets_cfg = cfg.train_dataset
    train_dataloader_cfg = cfg.train_dataloader

    for task_id in range(len(cfg.train_dataset)):
        task_output_dir = osp.join(args.output_dir, f"task{task_id}")
        train_vision_path = osp.join(task_output_dir, "vision_embedding_train.pt")
        train_text_path = osp.join(task_output_dir, "text_embeddings_train.pt")
        test_vision_path = osp.join(task_output_dir, "vision_embedding_test.pt")
        test_text_path = osp.join(task_output_dir, "text_embeddings_test.pt")

        if not (osp.exists(train_vision_path) and osp.exists(train_text_path)):
            train_dataloader_cfg["dataset"] = all_train_datasets_cfg[task_id]
            diff_rank_seed = runner._randomness_cfg.get("diff_rank_seed", False)
            train_dataloader = runner.build_dataloader(
                train_dataloader_cfg,
                seed=runner.seed,
                diff_rank_seed=diff_rank_seed,
            )
            vision_embeddings, text_embeddings = collect_train_embeddings(train_dataloader, model)
            maybe_save_embeddings(task_output_dir, "train", vision_embeddings, text_embeddings)
        else:
            print_log(f"Skip task{task_id} train embeddings: files already exist", logger="current")

        if not (osp.exists(test_vision_path) and osp.exists(test_text_path)):
            testset = BUILDER.build(cfg.test_dataset[task_id])
            vision_embeddings, text_embeddings = collect_test_embeddings(testset, model)
            maybe_save_embeddings(task_output_dir, "test", vision_embeddings, text_embeddings)
        else:
            print_log(f"Skip task{task_id} test embeddings: files already exist", logger="current")


if __name__ == "__main__":
    main()


