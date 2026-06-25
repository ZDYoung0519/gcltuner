import argparse
import json
import logging
import os
import os.path as osp
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
from timm.utils.metrics import accuracy

import math
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.dist import all_gather, is_main_process, get_rank, get_world_size, broadcast_object_list
from mmengine.device import get_device
from xtuner.registry import BUILDER


from mmengine.model import BaseModel
from xtuner.model.llava import ProjectorConfig, guess_load_checkpoint
from sklearn.cluster import KMeans
from tqdm import tqdm
from projects.aligncl.utils.metric import MetricLogger, SmoothedValue


def safe_accuracy(logits, targets):
    max_topk = min(5, logits.shape[1])
    topk = (1, max_topk)
    return accuracy(logits, targets, topk=topk)


class ToyModel(BaseModel):
    def __init__(
        self, 
        visual_encoder,
        text_encoder,
        text_tokenizer,
        router_args={},
        cur_task=0
    ):
        super().__init__()
        self.visual_encoder = BUILDER.build(visual_encoder)
        self.text_encoder = BUILDER.build(text_encoder)
        self.text_tokenizer = BUILDER.build(text_tokenizer)
        self.cur_task = cur_task

        self._setup_router(**router_args)

        for n, p in self.named_parameters():
            if 'router.' in n:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
    
    def _setup_router(
            self,
            num_experts: int = 10,
            router_bias: bool = False, 
            router_temp: float = 1.0,
            router_loss_ceof = 1e-3,
            router_topk = 1,
            router_frozen = True,
            forward_cur_expert_only = True,
            trained_router_path=None,
        ):
        self.router = nn.Linear(
            in_features=self.visual_encoder.config.hidden_size + self.text_encoder.config.hidden_size,
            out_features=num_experts,
            bias=router_bias
        )
        self.router_criterion = nn.CrossEntropyLoss()
        self.router_temp = router_temp
        self.router_loss_ceof = router_loss_ceof
        self.router_topk = router_topk
        self.router_frozen = router_frozen or router_loss_ceof == 0
        self.forward_cur_expert_only = forward_cur_expert_only

        if self.router_frozen:
            for n, p in self.router.named_parameters():
                p.requires_grad_(False)
        if trained_router_path:
            self.router.load_state_dict(torch.load(trained_router_path))

    def forward(self, data, data_samples, mode='loss'):
        if "pixel_values" not in data:
            return data

        images = data['pixel_values']
        texts = data['text']
        dtype = images.dtype
        device = get_device()

        visual_outputs = self.visual_encoder(
            images.to(dtype).to(device), 
            output_hidden_states=True
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

        visual_features = visual_outputs.last_hidden_state[:, 0, :]
        text_features = text_outputs.last_hidden_state[:, 0, :]
        features = torch.cat([visual_features, text_features], dim=-1).detach()
        logits = self.router(features) / self.router_temp
        logits[:, (self.cur_task+1):] = float('-inf')

        targets = torch.LongTensor([self.cur_task] * logits.shape[0]).to(logits.device)
        loss = self.router_criterion(logits, targets)            # B, T

        expert_weights = nn.Softmax(dim=-1)(logits).mean(0).detach()
        return {
            "loss": loss, 
            "logits": logits, 
            "expert_weights":expert_weights, 
            "features": features
        }

    def state_dict(self):
        state_dict  = super().state_dict()
        to_return = OrderedDict()
        to_return.update({k: v for k, v in state_dict.items() if "router." in k})
        return to_return


@torch.no_grad()
def collect_features(loader, model):
    rank = get_rank()
    all_features = []
    pbar = tqdm(loader, desc=f"Rank {rank}: collect features", position=rank, leave=False)
    for data in pbar:
        out = model(**data, mode="")
        all_features.append(out['features'].detach().cpu())
    features = torch.cat(all_features, dim=0)
    print("Rank{}: Features shape: {}".format(rank, features.shape))
    all_features = torch.cat(all_gather(features), dim=0)
    print("All features shape: {} {}".format(rank, all_features.shape))
    return all_features


@torch.no_grad()
def collect_test_features(dataset, model):
    rank = get_rank()
    world_size = get_world_size()
    device = get_device()

    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size)
    per_rank_ids = range(
        per_rank_samples * rank, min(n_samples, per_rank_samples * (rank + 1))
    )
    
    pbar = tqdm(per_rank_ids, desc=f"Rank {rank}")
    all_features = []
    for i in pbar:
        data = dataset[i]
        data["input_ids"] = data["input_ids"].to(device).unsqueeze(0)
        data["pixel_values"] = data["pixel_values"].to(device).unsqueeze(0)
        data['text'] = [data['text']]
        out = model(data, None, "")['features'].detach().cpu()
        all_features.append(out)
    all_features = torch.cat(all_features, dim=0)

    print("Rank{}: Features shape: {}".format(rank, all_features.shape))
    all_features = torch.cat(all_gather(all_features), dim=0)
    print("All features shape: {} {}".format(rank, all_features.shape))
    return all_features


def cal_mean_cov_with_kmeans(features, n_clusters, mode='covariance'):
    device = features.device
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)

    # cal mean and cov for all clusters
    cluster_lables = kmeans.labels_
    cluster_means = []
    cluster_vars = []
    for i in range(n_clusters):
        cluster_data = features[cluster_lables == i].detach()
        cluster_mean = cluster_data.mean(0).to(device)
        if mode == "var":
            cluster_var = torch.tensor(np.var(cluster_data.numpy(), axis=0), dtype=torch.float64).to(device)
        elif mode == "cov":
            cluster_var = torch.cov(torch.tensor(cluster_data, dtype=torch.float64).T) + torch.eye(
                cluster_mean.shape[-1]) * 1e-4
            cluster_var = cluster_var.to(device)
        else:
            raise NotImplementedError
        cluster_means.append(cluster_mean)
        cluster_vars.append(cluster_var)
    return cluster_means, cluster_vars


class GaussianMemory(object):
    def __init__(self, num_clusters_per_task, cov_mode, l2_norm=False):
        self.num_clusters_per_task = num_clusters_per_task
        self.cov_mode = cov_mode
        self.l2_norm = l2_norm
        self.means = {}
        self.covs = {}

        self._cache_ready = False
        self._cached_means = None
        self._cached_vars = None
        self._cached_chols = None
        self._cached_labels = None

    def _invalidate_cache(self):
        self._cache_ready = False
        self._cached_means = None
        self._cached_vars = None
        self._cached_chols = None
        self._cached_labels = None

    def update(self, all_features, task_id):
        if self.l2_norm:
            all_features = F.normalize(all_features.detach(), p=2.0, dim=1)
        else:
            all_features = all_features.detach()

        if is_main_process():
            cluster_means, cluster_covs = cal_mean_cov_with_kmeans(
                all_features,
                n_clusters=self.num_clusters_per_task,
                mode=self.cov_mode
            )
        else:
            cluster_means = None
            cluster_covs = None

        objects = [cluster_means, cluster_covs] if is_main_process() else [None, None]
        broadcast_object_list(objects, src=0)
        cluster_means, cluster_covs = objects[0], objects[1]

        self.means[task_id] = cluster_means
        self.covs[task_id] = cluster_covs
        self._invalidate_cache()

    def load_state_dict(self, state_dict):
        assert 'means' in state_dict
        assert 'covs' in state_dict
        self.means = state_dict['means']
        self.covs = state_dict['covs']
        self._invalidate_cache()

    def _build_sampling_cache(self, device):
        means = []
        covs_or_vars = []
        labels = []

        for task_id in sorted(self.means.keys()):
            task_means = self.means[task_id]
            task_covs = self.covs[task_id]

            for mean, cov in zip(task_means, task_covs):
                means.append(mean.float())
                covs_or_vars.append(cov.float())
                labels.append(task_id)

        means = torch.stack(means, dim=0).to(device)  # [M, D]
        labels = torch.tensor(labels, dtype=torch.long, device=device)  # [M]

        if self.cov_mode == "var":
            vars_ = torch.stack(covs_or_vars, dim=0).to(device)  # [M, D]
            vars_ = vars_.clamp_min(1e-8) + 1e-4
            self._cached_vars = vars_
            self._cached_chols = None

        elif self.cov_mode == "cov":
            covs = torch.stack(covs_or_vars, dim=0).to(device)  # [M, D, D]

            eye = torch.eye(covs.shape[-1], device=device, dtype=covs.dtype)
            covs = covs + eye.unsqueeze(0) * 1e-4

            # 关键优化：Cholesky 只做一次，不要每次 sample 都重新构造 MultivariateNormal
            chols = torch.linalg.cholesky(covs)  # [M, D, D]
            self._cached_chols = chols
            self._cached_vars = None

        else:
            raise NotImplementedError

        self._cached_means = means
        self._cached_labels = labels
        self._cache_ready = True

    @torch.no_grad()
    def sample_data(self, num_per_clusters, device=None):
        if device is None:
            device = get_device()

        if not self._cache_ready:
            self._build_sampling_cache(device)

        means = self._cached_means
        labels = self._cached_labels

        # 如果 cache 在 CPU，但本次训练要 GPU，迁移一次
        if means.device != torch.device(device):
            self._cache_ready = False
            self._build_sampling_cache(device)
            means = self._cached_means
            labels = self._cached_labels

        num_clusters_total, feat_dim = means.shape

        if self.cov_mode == "var":
            std = torch.sqrt(self._cached_vars)  # [M, D]
            eps = torch.randn(
                num_clusters_total,
                num_per_clusters,
                feat_dim,
                device=means.device,
                dtype=means.dtype,
            )
            samples = means[:, None, :] + eps * std[:, None, :]

        elif self.cov_mode == "cov":
            chols = self._cached_chols  # [M, D, D]
            eps = torch.randn(
                num_clusters_total,
                num_per_clusters,
                feat_dim,
                device=means.device,
                dtype=means.dtype,
            )
            samples = means[:, None, :] + torch.matmul(
                eps,
                chols.transpose(-1, -2)
            )

        else:
            raise NotImplementedError

        samples = samples.reshape(-1, feat_dim).float()
        sample_labels = labels[:, None].expand(
            num_clusters_total,
            num_per_clusters
        ).reshape(-1)

        return samples, sample_labels
    
    def state_dict(self):
        return {'means': self.means, "covs": self.covs}
    
    def load_state_dict(self, state_dict):
        assert 'means' in state_dict
        assert 'covs' in state_dict
        self.means = state_dict['means']
        self.covs = state_dict['covs']

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--saved-feature-dir', required=True)
    parser.add_argument('--cur-task', type=int, default=0, required=True)
    parser.add_argument('--num_clusters_per_task', type=int, default=10, help='the dir to save logs and models')
    parser.add_argument('--cov_mode', type=str, default='cov', help='the dir to save logs and models')
    parser.add_argument("--l2-norm", action='store_true')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def load_clip_features(saved_feature_dir, task_id, split):
    task_feature_dir = osp.join(saved_feature_dir, f"task{task_id}")
    vision_path = osp.join(task_feature_dir, f"vision_embedding_{split}.pt")
    text_path = osp.join(task_feature_dir, f"text_embeddings_{split}.pt")
    if osp.exists(vision_path) and osp.exists(text_path):
        vision_features = torch.load(vision_path, map_location="cpu")
        text_features = torch.load(text_path, map_location="cpu")
        return torch.cat([vision_features, text_features], dim=-1)

    legacy_feature_path = osp.join(saved_feature_dir, "train_features.pth", f"task{task_id}.pt")
    if osp.exists(legacy_feature_path):
        print_log(f"Load legacy {split} features from: {legacy_feature_path}")
        return torch.load(legacy_feature_path, map_location="cpu")

    raise FileNotFoundError(
        f"Can not find precomputed {split} features for task{task_id}. "
        f"Expected {vision_path} and {text_path}."
    )


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # customize
    cfg.model.cur_task = args.cur_task
    # set work dir and exp name
    cfg.visualizer = None
    cfg.model = dict(
        type="projects.aligncl.tools.train_router.ToyModel",
        visual_encoder=cfg.model['visual_encoder'],
        text_encoder=cfg.model['text_encoder'],
        text_tokenizer=cfg.model['text_tokenizer'],
        router_args=cfg.model['router_args'],
        cur_task=args.cur_task
    )

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    for n, p in model.named_parameters():
        if 'router.' in n:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    # build optimizer and scheaduler
    runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
    # runner.param_schedulers = runner.build_param_scheduler(runner.param_schedulers)
    optimizer = runner.optim_wrapper.optimizer

    gaussian_memory = GaussianMemory(args.num_clusters_per_task, args.cov_mode, args.l2_norm)
    # load previous gaussian_statics
    if args.cur_task > 0:
        pre_ckpt_dir = cfg.work_dir.replace(f"/task{args.cur_task}", f"/task{args.cur_task-1}")
        pre_ckpt_path = osp.join(pre_ckpt_dir, 'gaussian_memeory_state_dict.pth')
        if os.path.exists(pre_ckpt_path):
            gaussian_memory.load_state_dict(
                torch.load(pre_ckpt_path)
            )
            print_log(f"Load Gaussian Memory from: {pre_ckpt_path}")
    
    # update current gaussian_statics
    task_features = load_clip_features(args.saved_feature_dir, args.cur_task, "train")
    gaussian_memory.update(task_features,task_id=args.cur_task)

    if is_main_process():
        os.makedirs(cfg.work_dir, exist_ok=True)
        torch.save(gaussian_memory.state_dict(), osp.join(cfg.work_dir, f'gaussian_memeory_state_dict.pth'))

    best_metrics_path = osp.join(cfg.work_dir, "best_metrics.json")

    def save_best_metrics(metrics):
        if not is_main_process():
            return
        with open(best_metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def train_epoch():
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', SmoothedValue(window_size=1, fmt='{value:.4f}'))

        device = get_device()
        batch_size = cfg.batch_size // gaussian_memory.num_clusters_per_task
        sampled_data, sampled_label = gaussian_memory.sample_data(batch_size)
        inputs = sampled_data
        targets = sampled_label
        sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]
        # crct_num = args.cur_task * args.num_clusters_per_task
        crct_num = args.num_clusters_per_task
        rank = get_rank()
        pbar = tqdm(
            range(crct_num),
            desc=f"Epoch {epoch + 1}/{cfg.max_epochs}",
            position=rank,
            leave=False,
            disable=not is_main_process(),
        )
        for _iter in pbar:
            inp = inputs[_iter * batch_size:(_iter + 1) * batch_size].to(device)
            tgt = targets[_iter * batch_size:(_iter + 1) * batch_size].to(device)
            logits = model.router(inp)
            logits = logits[:, :args.cur_task+1]
            loss = F.cross_entropy(logits, tgt)
            with torch.no_grad():
                acc1, acc5 = safe_accuracy(logits, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc1=f"{acc1.item():.2f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}",
            )

            # print(f'  Iter [{_iter+1}/{crct_num}], Loss: {loss.item():.4f}, Acc@1: {acc1.item():.2f}%, Lr: {optimizer.param_groups[0]["lr"]:.6f}')
        
        metric_logger.synchronize_between_processes()
        if is_main_process():
            print('Epoch: {}, Train Loss: {:.3f}\tAcc@1: {:.2f}\tAcc@5: {:.2f}'.format(
                epoch+1,
                metric_logger.meters['Loss'].global_avg,
                metric_logger.meters['Acc@1'].global_avg,
                metric_logger.meters['Acc@5'].global_avg)
            )
        torch.cuda.synchronize()
        return model
    
    @torch.no_grad()
    def validate_until_now(model, cur_task):
        val_acc1, val_acc5 = 0, 0
        router = model.router
        device = get_device()
        for task_id in range(cur_task+1):
            task_features = load_clip_features(args.saved_feature_dir, task_id, "test").to(device)
            logits = router(task_features)
            logits[:, (cur_task+1):] = float('-inf')

            targets = torch.LongTensor([task_id] * logits.shape[0]).to(device)
            acc1, acc5 = safe_accuracy(logits, targets)

            print(f"Valid on task {task_id}, acc1: {acc1}, acc5: {acc5}")
            val_acc1 += acc1
            val_acc5 += acc5
        val_acc1 /= cur_task+1
        val_acc5 /= cur_task+1
        return val_acc1, val_acc5

    # load previous ckpt
    pre_ckpt_dir = cfg.work_dir.replace(f"/task{args.cur_task}", f"/task{args.cur_task-1}")
    pre_ckpt_path = osp.join(pre_ckpt_dir, 'best.pth')
    if osp.exists(pre_ckpt_path):
        model.load_state_dict(torch.load(pre_ckpt_path, map_location='cpu'), strict=False)

    # retrain the task head
    best_acc = 0.
    epoch_pbar = tqdm(
        range(cfg.max_epochs),
        desc=f"Task {args.cur_task} epochs",
        disable=not is_main_process(),
    )
    for epoch in epoch_pbar:
        train_epoch()
        if is_main_process():
            val_acc1, _ = validate_until_now(model, args.cur_task)
            val_acc1_value = float(val_acc1.item() if hasattr(val_acc1, "item") else val_acc1)
            epoch_pbar.set_postfix(acc1=f"{val_acc1_value:.4f}", best=f"{best_acc:.4f}")
            print(f"Valid until task {args.cur_task}, acc1: {val_acc1_value}, best: {best_acc}")
            if val_acc1 > best_acc:
                best_acc = val_acc1_value
                state_dict = model.router.state_dict()

                torch.save(state_dict, osp.join(cfg.work_dir, f'best.pth'))
                save_best_metrics({
                    "best_acc": best_acc,
                    "best_epoch": epoch + 1,
                    "cur_task": args.cur_task,
                    "num_clusters_per_task": args.num_clusters_per_task,
                    "cov_mode": args.cov_mode,
                    "l2_norm": args.l2_norm,
                    "work_dir": cfg.work_dir,
                })


    if is_main_process():
        state_dict = model.router.state_dict()

        torch.save(state_dict, osp.join(cfg.work_dir, f'last.pth'))
        if not osp.exists(best_metrics_path):
            save_best_metrics({
                "best_acc": best_acc,
                "best_epoch": None,
                "cur_task": args.cur_task,
                "num_clusters_per_task": args.num_clusters_per_task,
                "cov_mode": args.cov_mode,
                "l2_norm": args.l2_norm,
                "work_dir": cfg.work_dir,
            })


if __name__ == '__main__':
    main()
