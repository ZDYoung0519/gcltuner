import argparse
import os
import os.path as osp
from types import FunctionType

import torch
from torch import nn
import numpy as np
import scipy

from mmengine import print_log
from mmengine.model import BaseModel
from mmengine.runner import Runner
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS

from xtuner.model.llava import ProjectorConfig, guess_load_checkpoint
from xtuner.registry import MAP_FUNC, BUILDER

from projects.aligncl.modules.nsprojector import NSProjectorModel
from projects.aligncl.modules.nslinear import NSLinear


class ToyModel(BaseModel):
    def __init__(
        self,
        visual_encoder,
        text_encoder,
        text_tokenizer,
        projector_depth=2,
        llm_hidden_size=4096,
        visual_select_layer=-2,
        router_args={},
        projector_args={},
        cur_task=0,
    ):
        super().__init__()
        self.visual_encoder = BUILDER.build(visual_encoder)
        self.text_encoder = BUILDER.build(text_encoder)
        self.text_tokenizer = BUILDER.build(text_tokenizer)
        self.cur_task = cur_task

        self.projector_depth = projector_depth
        self.llm_hidden_size = llm_hidden_size
        self.visual_select_layer = visual_select_layer

        self._setup_projector(projector_args)
        self._setup_router(**router_args)

    def _setup_projector(self, projector_args):
        projector_type = projector_args.get('type', 'mlp')
        self.projector_type = projector_type
        if projector_type == 'mlp':
            # projector is already built
            strategy = projector_args.get('strategy', None)
            if strategy == 'fsa':
                # FSA: first session adaption, only train the projector during the first task
                if self.cur_task > 0:
                    for n, p in self.projector.named_parameters():
                        p.requires_grad_(False)
                return
            elif strategy == 'nsp':
                from ..modules.nslinear import NSLinear
                from ..modules.nsprojector import NSProjectorModel, ProjectorConfig
                
                # Null space projection, we need to first covert it
                projector_config = ProjectorConfig(
                    visual_hidden_size=self.visual_encoder.config.hidden_size,
                    llm_hidden_size=self.llm_hidden_size,
                    depth=self.projector_depth,
                )
                projector_new = NSProjectorModel(projector_config).to(self.visual_encoder.dtype)
                # projector_new.load_state_dict(self.projector.state_dict(), strict=True)
                # del self.projector
                self.projector = projector_new

                # load the projection matrix for projector parameters
                if self.cur_task > 0:
                    projection_path = projector_args.get("projection_matrix", "")
                    if projection_path:
                        projection_matrix = torch.load(projection_path)
                        for n, m in self.named_modules():
                            if isinstance(m, NSLinear):
                                m.weight._projection = projection_matrix[n]
                                print_log(f"Module {n}.weight is equipped with projection: {projection_matrix[n].shape}")
                                m.bias.requires_grad_(False)

        elif projector_type == "taa":
            raise NotImplementedError
        else:
            raise NotImplementedError(f"Unknown projector type: {projector_type}")

    def _setup_router(
        self,
        num_experts: int = 10,
        router_bias: bool = False,
        router_temp: float = 1.0,
        router_loss_ceof=1e-3,
        router_topk=1,
        router_frozen=True,
        forward_cur_expert_only=True,
        trained_router_path=None,
    ):
        self.router = nn.Linear(
            in_features=(
                self.visual_encoder.config.hidden_size
                + self.text_encoder.config.hidden_size
            ),
            out_features=num_experts,
            bias=router_bias,
        )

        self.router_criterion = nn.CrossEntropyLoss()
        self.router_temp = router_temp
        self.router_loss_ceof = router_loss_ceof
        self.router_topk = router_topk
        self.router_frozen = router_frozen or router_loss_ceof == 0
        self.forward_cur_expert_only = forward_cur_expert_only

        if self.router_frozen:
            for _, p in self.router.named_parameters():
                p.requires_grad_(False)

        if trained_router_path:
            self.router.load_state_dict(
                torch.load(trained_router_path, map_location="cpu")
            )

    def forward(self, data, data_samples=None, mode="predict"):
        device = next(self.parameters()).device

        visual_outputs = self.visual_encoder(
            data["pixel_values"].to(
                device=device,
                dtype=self.visual_encoder.dtype,
            ),
            output_hidden_states=True,
        )

        clip_text_inputs = self.text_tokenizer(
            data["text"],
            padding="longest",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        clip_text_inputs = clip_text_inputs.to(device)

        text_outputs = self.text_encoder(**clip_text_inputs)

        pixel_values = self._forward_projector(visual_outputs, text_outputs)
        return pixel_values

    def _forward_projector(self, visual_outputs, text_outputs=None):
        if self.projector_type == "mlp":
            pixel_values = self.projector(
                visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
            )
            return pixel_values

        raise NotImplementedError

    def load_checkpoint(self, ckpt_path):
        if ckpt_path is None:
            return

        pretrained_state_dict = guess_load_checkpoint(ckpt_path)

        missing_keys, unexpected_keys = self.load_state_dict(
            pretrained_state_dict,
            strict=False,
        )

        model_keys = set(self.state_dict().keys())
        loaded_keys = [k for k in pretrained_state_dict if k in model_keys]

        print_log(f"Load pretrained weight from {ckpt_path}", "current")
        print_log(f"Missing keys: {len(missing_keys)}", "current")
        print_log(f"Unexpected keys: {len(unexpected_keys)}", "current")

        for k in loaded_keys:
            v = pretrained_state_dict[k]
            if hasattr(v, "shape"):
                print_log(f"Load key {k}: {v.shape}", "current")


class CovarianceCollector:
    """
    Collect input covariance of NSLinear / NS2Linear.

    For every hooked layer, accumulate:

        cov = X^T X / N

    This is enough for computing low-energy / approximate null-space directions.
    """

    def __init__(self, model: nn.Module, save_last_features: bool = True):
        self._handles = []
        self._cov: dict[str, torch.Tensor | None] = {}
        self._count: dict[str, int] = {}
        self.features: dict[str, torch.Tensor] = {}
        self.save_last_features = save_last_features

        self._register_hooks(model)

    def _init_state(self, name: str, tag: str) -> None:
        if tag == "NSLinear":
            self._cov[name] = None
            self._count[name] = 0

        elif tag == "NS2Linear":
            self._cov[f"{name}.x1"] = None
            self._cov[f"{name}.x2"] = None
            self._count[f"{name}.x1"] = 0
            self._count[f"{name}.x2"] = 0

    def _register_hooks(self, root: nn.Module) -> None:
        from projects.aligncl.modules.nslinear import NSLinear, NS2Linear

        for name, module in root.named_modules():
            if isinstance(module, NSLinear):
                self._init_state(name, "NSLinear")
                handle = module.register_forward_hook(
                    self._make_hook(name, tag="NSLinear")
                )
                self._handles.append(handle)

            elif isinstance(module, NS2Linear):
                self._init_state(name, "NS2Linear")
                handle = module.register_forward_hook(
                    self._make_hook(name, tag="NS2Linear")
                )
                self._handles.append(handle)

    def _accumulate_cov(self, key: str, x: torch.Tensor) -> None:
        x = x.detach().float()
        n = x.shape[0]

        gram = x.T @ x
        gram = gram.cpu()

        if self._cov[key] is None:
            self._cov[key] = gram
        else:
            self._cov[key] += gram

        self._count[key] += n

        if self.save_last_features:
            self.features[key] = x.cpu()

    def _make_hook(self, name: str, tag: str):
        def hook(module, args, output):
            if tag == "NSLinear":
                x = args[0].detach().reshape(-1, module.in_features)
                self._accumulate_cov(name, x)

            elif tag == "NS2Linear":
                x1 = args[0].detach().reshape(-1, module.in_features)
                x2 = args[1].detach().reshape(-1, module.in_features)

                self._accumulate_cov(f"{name}.x1", x1)
                self._accumulate_cov(f"{name}.x2", x2)

        return hook

    def finalized(self) -> dict[str, torch.Tensor]:
        covariances = {}

        for k, v in self._cov.items():
            count = self._count.get(k, 0)

            if v is None or count == 0:
                continue

            covariances[k] = v / count

        return covariances

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def save(self, path: str) -> None:
        data = {
            k: v / self._count[k]
            for k, v in self._cov.items()
            if v is not None and self._count[k] > 0
        }
        torch.save({"cov": data, "count": dict(self._count)}, path)

    def summary(self) -> str:
        lines = []
        modules: dict[str, dict[str, torch.Tensor | int]] = {}

        for key, cov in self._cov.items():
            if cov is None:
                continue

            if key.endswith(".x1") or key.endswith(".x2"):
                base, field = key.rsplit(".", 1)
            else:
                base, field = key, "input"

            if base not in modules:
                modules[base] = {}

            modules[base][field] = cov
            modules[base][f"{field}_count"] = self._count[key]

        for base, fields in modules.items():
            is_ns2 = "x1" in fields
            module_type = "NS2Linear" if is_ns2 else "NSLinear"

            lines.append(f"{base} ({module_type}):")

            if is_ns2:
                lines.append(
                    f"  x1: dim={fields['x1'].shape[0]}, "
                    f"count={fields['x1_count']}"
                )
                lines.append(
                    f"  x2: dim={fields['x2'].shape[0]}, "
                    f"count={fields['x2_count']}"
                )
            else:
                lines.append(
                    f"  dim={fields['input'].shape[0]}, "
                    f"count={fields['input_count']}"
                )

        return "\n".join(lines) if lines else "(no NSLinear/NS2Linear modules found)"


def compute_low_energy_basis(
    cov: torch.Tensor,
    energy_keep: float = 0.99,
    eps: float = 1e-12,
):
    """
    Compute approximate null-space / low-energy basis.

    Given covariance:

        C = X^T X / N

    SVD:

        C = U S V^T

    We keep the leading directions whose squared singular-value energy reaches
    `energy_keep`, and use the remaining right singular vectors as low-energy
    directions.

    Returns:
        basis: [null_dim, d_in]
        S: singular values
        signal_rank: number of leading directions kept
    """
    cov = cov.float()
    d_in = cov.shape[0]

    U, S, Vh = torch.linalg.svd(cov, full_matrices=True)

    total_energy = (S ** 2).sum()

    if total_energy <= eps:
        signal_rank = 0
        basis = Vh
        return basis, S, signal_rank

    energy_ratio = torch.cumsum(S ** 2, dim=0) / total_energy

    valid = (energy_ratio >= energy_keep).nonzero(as_tuple=False)

    if len(valid) == 0:
        signal_rank = d_in
    else:
        signal_rank = int(valid[0].item()) + 1

    signal_rank = min(max(signal_rank, 0), d_in)

    basis = Vh[signal_rank:, :]

    return basis, S, signal_rank


def compute_strict_null_space_basis(
    cov: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    """
    Compute strict numerical null-space basis.

    This only returns directions whose singular values are almost zero.
    In most deep-feature covariance matrices, strict null-space may be empty.
    """
    cov = cov.float()
    U, S, Vh = torch.linalg.svd(cov, full_matrices=True)

    tol = max(atol, rtol * S.max().item())
    zero_idx = S <= tol

    basis = Vh[zero_idx, :]

    return basis, S, tol


def merge_covariances(
    previous_covs: dict[str, torch.Tensor],
    previous_counts: dict[str, list[int]],
    current_covs: dict[str, torch.Tensor],
    current_counts: dict[str, int],
):
    """
    Merge previous averaged covariance and current averaged covariance.

    If:
        C_prev = sum_i X_i^T X_i / N_prev
        C_cur  = X_cur^T X_cur / N_cur

    Then:
        C_new = (N_prev * C_prev + N_cur * C_cur) / (N_prev + N_cur)
    """
    merged_covs = dict(previous_covs)
    merged_counts = {k: list(v) for k, v in previous_counts.items()}

    for layer_name, cur_cov in current_covs.items():
        cur_count = int(current_counts[layer_name])

        if layer_name not in merged_covs:
            merged_covs[layer_name] = cur_cov
            merged_counts[layer_name] = [cur_count]
            continue

        old_counts = merged_counts[layer_name]
        old_total = int(sum(old_counts))
        new_total = old_total + cur_count

        old_cov = merged_covs[layer_name]

        new_cov = (
            old_total / new_total * old_cov
            + cur_count / new_total * cur_cov
        )

        merged_covs[layer_name] = new_cov
        merged_counts[layer_name].append(cur_count)

    return merged_covs, merged_counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect NSLinear/NS2Linear input features and compute projections"
    )

    parser.add_argument("config", help="config file name or path.")
    parser.add_argument("--checkpoint", default=None, help="checkpoint file")
    parser.add_argument("--cur-task", default=0, type=int, help="task identity")
    parser.add_argument("--work-dir", help="directory to save collected features")

    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="max batches to collect; 0 means use entire dataloader",
    )

    parser.add_argument(
        "--energy-ratio",
        type=float,
        default=0.99,
        help="energy ratio kept as signal subspace; remaining directions are used "
        "as approximate null-space",
    )

    parser.add_argument(
        "--strict-null-space",
        action="store_true",
        help="use strict numerical null-space instead of low-energy subspace",
        default=True
    )

    parser.add_argument(
        "--strict-rtol",
        type=float,
        default=1e-5,
        help="relative tolerance for strict null-space",
    )

    parser.add_argument(
        "--strict-atol",
        type=float,
        default=1e-7,
        help="absolute tolerance for strict null-space",
    )

    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, format: key=value",
    )

    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )

    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)

    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def register_function(cfg_dict):
    if isinstance(cfg_dict, dict):
        for key, value in dict.items(cfg_dict):
            if isinstance(value, FunctionType):
                value_str = str(value)
                if value_str not in MAP_FUNC:
                    MAP_FUNC.register_module(module=value, name=value_str)
                cfg_dict[key] = value_str
            else:
                register_function(value)

    elif isinstance(cfg_dict, (list, tuple)):
        for value in cfg_dict:
            register_function(value)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    register_function(cfg._cfg_dict)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join(
            "./work_dirs",
            osp.splitext(osp.basename(args.config))[0],
        )

    os.makedirs(cfg.work_dir, exist_ok=True)

    cfg.train_dataset = cfg.train_dataset[args.cur_task]
    cfg.train_dataloader["dataset"] = cfg.train_dataset
    cfg.visualizer = None

    cfg.model = dict(
        type="projects.aligncl.tools.compute_null_space.ToyModel",
        visual_encoder=cfg.model["visual_encoder"],
        text_encoder=cfg.model["text_encoder"],
        text_tokenizer=cfg.model["text_tokenizer"],
        projector_args=cfg.model["projector_args"],
        router_args=cfg.model["router_args"],
        cur_task=args.cur_task,
    )

    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    model.eval()
    model.load_checkpoint(args.checkpoint)

    diff_rank_seed = runner._randomness_cfg.get("diff_rank_seed", False)
    seed = runner.seed

    dataloader = runner.build_dataloader(
        cfg.train_dataloader,
        seed,
        diff_rank_seed,
    )

    from tqdm import tqdm

    rank = int(os.environ.get("LOCAL_RANK", 0))

    total = len(dataloader)
    if args.max_batches and args.max_batches > 0:
        total = min(total, args.max_batches)

    collector = CovarianceCollector(model, save_last_features=True)

    pbar = tqdm(
        total=total,
        desc="Collecting input covariances for null-space projection",
        disable=rank != 0,
    )

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(dataloader):
            if args.max_batches and args.max_batches > 0:
                if batch_idx >= args.max_batches:
                    break

            model(**data_batch)
            pbar.update(1)

    pbar.close()
    collector.remove_hooks()

    if rank != 0:
        return

    current_covariances = collector.finalized()
    current_counts = dict(collector._count)

    print("\n" + collector.summary())

    features_path = osp.join(cfg.work_dir, "features_last_batch.pt")
    torch.save(collector.features, features_path)
    print(f"\nLast-batch features saved to {features_path}")

    covariances_path = osp.join(cfg.work_dir, "covariances.pt")

    if args.cur_task > 0:
        previous_covariances_path = covariances_path.replace(
            f"/task{args.cur_task}/",
            f"/task{args.cur_task - 1}/",
        )

        previous_data = torch.load(
            previous_covariances_path,
            map_location="cpu",
        )

        previous_covariances = previous_data["covariances"]
        previous_counts = previous_data["counts"]

        merged_covariances, merged_counts = merge_covariances(
            previous_covs=previous_covariances,
            previous_counts=previous_counts,
            current_covs=current_covariances,
            current_counts=current_counts,
        )

        covariances_data = {
            "covariances": merged_covariances,
            "counts": merged_counts,
        }

    else:
        merged_covariances = current_covariances
        merged_counts = {
            k: [int(v)]
            for k, v in current_counts.items()
            if k in current_covariances
        }

        covariances_data = {
            "covariances": merged_covariances,
            "counts": merged_counts,
        }

    torch.save(covariances_data, covariances_path)
    print(f"\nFeature covariances saved to {covariances_path}")

    null_space_basis: dict[str, torch.Tensor] = {}
    svd_info: dict[str, dict] = {}

    for layer_name, cov in merged_covariances.items():
        if args.strict_null_space:
            basis, S, tol = compute_strict_null_space_basis(
                cov,
                rtol=args.strict_rtol,
                atol=args.strict_atol,
            )

            svd_info[layer_name] = {
                "singular_values": S.cpu(),
                "tol": tol,
                "mode": "strict",
            }

            print(
                f"  [task {args.cur_task}] {layer_name}: "
                f"d_in={cov.shape[0]}, "
                f"strict_null_dim={basis.shape[0]}, "
                f"tol={tol:.4e}, "
                f"min_s={S.min().item():.4e}, "
                f"max_s={S.max().item():.4e}"
            )

        else:
            basis, S, signal_rank = compute_low_energy_basis(
                cov,
                energy_keep=args.energy_ratio,
            )

            svd_info[layer_name] = {
                "singular_values": S.cpu(),
                "signal_rank": signal_rank,
                "energy_keep": args.energy_ratio,
                "mode": "low_energy",
            }

            print(
                f"  [task {args.cur_task}] {layer_name}: "
                f"d_in={cov.shape[0]}, "
                f"signal_rank={signal_rank}, "
                f"low_energy_dim={basis.shape[0]}, "
                f"min_s={S.min().item():.4e}, "
                f"max_s={S.max().item():.4e}"
            )

        null_space_basis[layer_name] = basis.cpu()

    null_space_basis_path = osp.join(cfg.work_dir, "null_space_basis.pt")
    torch.save(null_space_basis, null_space_basis_path)
    print(f"\nNull-space basis saved to {null_space_basis_path}")

    svd_info_path = osp.join(cfg.work_dir, "svd_info.pt")
    torch.save(svd_info, svd_info_path)
    print(f"SVD info saved to {svd_info_path}")

    projections: dict[str, torch.Tensor] = {}

    for layer_name, basis in null_space_basis.items():
        V0 = basis.float()
        proj = V0.T @ V0
        projections[layer_name] = proj.cpu()

    proj_path = osp.join(cfg.work_dir, "projection_matrices.pt")
    torch.save(projections, proj_path)
    print(f"\nProjection matrices saved to {proj_path}")

    print("\nVerifying projections:")

    for layer_name, proj in projections.items():
        cov = merged_covariances[layer_name].float()
        proj = proj.float()

        cov_proj = cov @ proj

        cov_norm = cov.norm().clamp_min(1e-12)
        proj_norm = proj.norm().clamp_min(1e-12)

        cov_proj_abs = cov_proj.norm()
        cov_proj_rel = cov_proj_abs / cov_norm

        sym_err = (proj - proj.T).norm() / proj_norm
        idem_err = (proj @ proj - proj).norm() / proj_norm

        msg = (
            f"  [task {args.cur_task}] {layer_name}: "
            f"cov_proj_abs={cov_proj_abs.item():.4e}, "
            f"cov_proj_rel={cov_proj_rel.item():.4e}, "
            f"proj_sym_err={sym_err.item():.4e}, "
            f"proj_idem_err={idem_err.item():.4e}"
        )

        if layer_name in collector.features:
            feat = collector.features[layer_name].float()
            feat_proj = feat @ proj
            feat_proj_rel = feat_proj.norm() / feat.norm().clamp_min(1e-12)

            msg += f", last_batch_feat_proj_rel={feat_proj_rel.item():.4e}"

        print(msg)


if __name__ == "__main__":
    main()

