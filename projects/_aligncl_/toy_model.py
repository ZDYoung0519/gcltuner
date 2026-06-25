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

