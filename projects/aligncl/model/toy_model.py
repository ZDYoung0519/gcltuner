import torch
from torch import nn

from collections import OrderedDict

from mmengine.device import get_device
from mmengine.model import BaseModel
from mmengine import print_log

from xtuner.registry import BUILDER


class ToyModel(BaseModel):
    def __init__(
        self, 
        visual_encoder,
        text_encoder,
        text_tokenizer,
        cur_task=0,
        expert_num=8,
        expert_router_args={},
    ):
        super().__init__()
        self.visual_encoder = BUILDER.build(visual_encoder)
        self.text_encoder = BUILDER.build(text_encoder)
        self.text_tokenizer = BUILDER.build(text_tokenizer)
        self.cur_task = cur_task

        self.expert_num = expert_num
        self._setup_expert_router(**expert_router_args)

        for n, p in self.named_parameters():
            if 'expert_router.' in n:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
    
    def _setup_expert_router(
            self,
            expert_router_bias: bool = False, 
            expert_router_input_featuers: str = "both",
            expert_router_temp: float = 1,
            pretrained_expert_router_path: str = "",  
        ):
        self.router_input_featuers = expert_router_input_featuers
        self.expert_router_temp = expert_router_temp
        if expert_router_input_featuers == "vision":
            in_features = self.visual_encoder.config.hidden_size
        elif expert_router_input_featuers == "text":
            in_features = self.text_encoder.config.hidden_size,
        elif expert_router_input_featuers == "both":
            in_features = self.visual_encoder.config.hidden_size + self.text_encoder.config.hidden_size

        self.expert_router = nn.Linear(
            in_features=in_features,
            out_features=self.expert_num,
            bias=expert_router_bias
        )
        self.router_criterion = nn.CrossEntropyLoss()
        
        if pretrained_expert_router_path:
            self.expert_router.load_state_dict(torch.load(pretrained_expert_router_path, map_location='cpu'))
            print_log(f"Load pretrained_router from {pretrained_expert_router_path}")

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
