from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from mmengine import print_log
from xtuner.model.utils import (
    LoadWoInit, 
    prepare_inputs_labels_for_multimodal,
    make_inputs_require_grad
)

from gcltuner.model.llava import LLaVAModel, XTunerLLaVAModel, get_peft_model_state_dict


class AlignclLLaVAModel(LLaVAModel):
    def __init__(
        self,
        llm,
        visual_encoder,
        freeze_llm=False,
        freeze_visual_encoder=False,
        visual_select_layer=-2,
        pretrained_pth=None,
        projector_depth=2,
        llm_lora=None,
        visual_encoder_lora=None,
        use_activation_checkpointing=True,
        max_position_embeddings=None,
        cur_task=0,
        text_encoder=None,
        text_tokenizer=None,
        projector_args={},
        router_args={},
        disable_llm_lora=False,
        **kwargs
    ):
        super().__init__(
            llm,
            visual_encoder, 
            freeze_llm, 
            freeze_visual_encoder, 
            visual_select_layer, 
            None,
            projector_depth,
            llm_lora,
            visual_encoder_lora,
            use_activation_checkpointing,
            max_position_embeddings,
            **kwargs
        )

        self.cur_task = cur_task
        with LoadWoInit():
            if text_encoder:
                self.text_encoder = self._build_from_cfg_or_module(text_encoder)
                for n, p in self.text_encoder.named_parameters():
                    p.requires_grad_(False)
            if text_tokenizer:
                self.text_tokenizer = self._build_from_cfg_or_module(text_tokenizer)

        if use_activation_checkpointing:
            if hasattr(self.text_encoder, "enable_input_require_grads"):
                self.text_encoder.enable_input_require_grads()
            else:
                self.text_encoder.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )
        
        if disable_llm_lora:
            self.llm._set_adapter_layers(enabled=False)

        self.projector_depth = projector_depth
        self._setup_projector(projector_args)
        self._setup_router(**router_args)

        # we should ensure the pre-trained is loaded finally
        self.load_checkpoint(pretrained_pth)

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
                from .modules.nslinear import NSLinear
                from .modules.nsprojector import NSProjectorModel, ProjectorConfig
                
                # Null space projection, we need to first covert it
                projector_config = ProjectorConfig(
                    visual_hidden_size=self.visual_encoder.config.hidden_size,
                    llm_hidden_size=self.llm.config.hidden_size,
                    depth=self.projector_depth,
                )
                projector_new = NSProjectorModel(projector_config).to(self.visual_encoder.dtype)
                projector_new.load_state_dict(self.projector.state_dict(), strict=True)
                del self.projector
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
        elif projector_type == 'taa':
            from .modules.taa_projector import TaskAdaptiveAggregator
            self.projector_taa = TaskAdaptiveAggregator(
                visual_encoder_hidden_size=self.visual_encoder.config.hidden_size,
                text_encoder_hidden_size=self.text_encoder.config.hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                cur_task=self.cur_task,
                num_experts=projector_args.get('num_experts', 6),
                visual_select_layers=projector_args.get('visual_select_layers', [3, 8, 24]),
                num_visual_layers=projector_args.get('num_visual_layers', 24),
                freeze_previous_experts=projector_args.get('freeze_previous_experts', True),
                freeze_shared_expert_after_first_task=projector_args.get('freeze_shared_expert_after_first_task', True)
            )
            self.projector_residual = projector_args.get('projector_residual', True)
            if self.projector_residual and self.cur_task > 0:
                for n, p in self.projector.named_parameters():
                    p.requires_grad_(False)
        else:
            raise NotImplementedError
    
    def _forward_projector(self, visual_outputs, text_outputs=None):
        if self.projector_type == 'mlp':
            pixel_values = self.projector(
                visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
            )
            return pixel_values
        elif self.projector_type == 'taa':
            out = self.projector_taa(visual_outputs, text_outputs)
            if self.projector_residual:
                pixel_values = self.projector(
                    visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
                )
                out = out + pixel_values
            return out

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
            for _, p in self.router.named_parameters():
                p.requires_grad_(False)
        if trained_router_path:
            self.router.load_state_dict(torch.load(trained_router_path))
    
    def _forward_router(self, visual_outputs, text_outputs):
        bs = visual_outputs.last_hidden_state.shape[0]
        device = visual_outputs.last_hidden_state.device
        dtype = visual_outputs.last_hidden_state.dtype

        if self.forward_cur_expert_only:
            expert_weights = torch.zeros([bs, self.cur_task+1]).to(device).to(dtype)
            expert_weights[:, self.cur_task] = 1
            return expert_weights, 0
        
        router_input =  torch.cat(
            [
                visual_outputs.last_hidden_state[:, 0, :], 
                text_outputs.last_hidden_state[:, 0, :]
            ],
            dim=-1
        ).detach()

        logits = self.router(router_input) / self.router_temp
        logits[:, (self.cur_task+1):] = float('-inf')       # mask logits

        # select top-k
        if self.router_topk > 0 and self.router_topk < logits.shape[-1]:
            _, topk_indices = torch.topk(logits, self.router_topk, dim=-1)
            mask = torch.zeros_like(logits)
            mask.scatter_(1, topk_indices, 1)
            logits = logits.masked_fill(mask == 0, float('-inf'))
        
        with torch.no_grad():
            softmax = nn.Softmax()(logits)
        
        targets = torch.LongTensor([self.cur_task] * logits.shape[0]).to(logits.device)

        # compute loss
        loss = self.router_criterion(logits, targets)
        loss = loss  * self.router_loss_ceof
        return softmax, loss

    def _prepare_data_for_llm(self, data, mode='loss'):
        device = data["pixel_values"].device
        dtype = self.visual_encoder.dtype
        bs = data["pixel_values"].shape[0]
    
        # 1. forward visual encoder
        visual_outputs = self.visual_encoder(
            data["pixel_values"].to(self.visual_encoder.dtype),
            output_hidden_states=True,
        )

        # 2. forward text encoder
        clip_text_inputs = self.text_tokenizer(
            data['text'],
            padding="longest",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        clip_text_inputs.to(device)
        text_outputs = self.text_encoder(**clip_text_inputs)

        # 3. get and set expert weights for projector and llm
        expert_weights, router_loss = self._forward_router(visual_outputs, text_outputs)
        expert_weights = expert_weights.mean(0)
        self.expert_weights = expert_weights.detach().to(torch.float).cpu().numpy().tolist()
        if hasattr(self.llm, "_set_expert_weights"):
            self.llm._set_expert_weights(expert_weights)
        if hasattr(self, "projector_taa") and hasattr(self.projector_taa, "_set_expert_weights"):
            self.projector_taa._set_expert_weights(expert_weights)

        # 4. get pixel values for llm
        pixel_values = self._forward_projector(visual_outputs, text_outputs)
        data['pixel_values'] = pixel_values

        self.router_loss = router_loss

        # 5. prepare inputs for llm
        if mode == 'predict' or mode == 'generate':
            data = dict(
                input_ids=data["input_ids"],
                pixel_values=data["pixel_values"],
            )
        else:
            data = {
                "input_ids": data.get("input_ids", None),
                "position_ids": data.get("position_ids", None),
                "attention_mask": data.get("attention_mask", None),
                "past_key_values": data.get("past_key_values", None),
                "labels": data.get("labels", None),
                "pixel_values": data.get("pixel_values", None)
            }
        data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)
        return data
        
    def forward(self, data, data_samples=None, mode="loss"):
        if self.is_first_iter:
            self.to(data["input_ids"].device)
            self.is_first_iter = False
        
        if "pixel_values" in data:
            data = self._prepare_data_for_llm(data, mode)

        if mode == "loss":
            return self.compute_loss(data, data_samples)
        elif mode == 'predict' or mode == 'generate':
            output = self.generate(data, data_samples)
            output['expert_weights'] = self.expert_weights
            return output
        elif mode == "tensor":
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def compute_loss(self, data, data_samples=None):
        outputs = self.llm(**data)
        loss_dict = {"loss": outputs.loss + self.router_loss}
        return loss_dict
    
    def state_dict(self, *args, **kwargs):
        state_dict = super(XTunerLLaVAModel, self).state_dict()
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(self.visual_encoder, state_dict=state_dict)
            )
        elif not self.freeze_visual_encoder:
            to_return.update(
                {k: v for k, v in state_dict.items() if "visual_encoder." in k}
            )
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update({k: v for k, v in state_dict.items() if "llm." in k})
        # Step 3. Projector
        to_return.update({k: v for k, v in state_dict.items() if "projector." in k})
        # Step 3. router
        to_return.update({k: v for k, v in state_dict.items() if "router." in k})
        return to_return

