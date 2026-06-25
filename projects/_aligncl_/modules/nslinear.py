from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from mmengine import print_log
from transformers.activations import ACT2FN
from xtuner.model.llava import ProjectorConfig, prepare_inputs_labels_for_multimodal, guess_load_checkpoint, get_peft_model_state_dict
from xtuner.registry import BUILDER, MAP_FUNC



class NSLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)


class NS2Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
    
    def forward(self, x1, x2):
        """
        x1: (batch_size, n1, dim1)
        x2: (batch_size, n2, dim2)
        out: (batch_size, n1, n2)
        """
        x2 =  F.linear(x2, self.weight, self.bias)     # b, n2, d1
        out = torch.bmm(x1, x2.transpose(-2, -1))      # b, n1, n2
        return out


def build_null_space_vision_projector(config):
    modules = [
        NSLinear(
            config.visual_hidden_size, config.llm_hidden_size, bias=config.bias
        )
    ]
    for _ in range(1, config.depth):
        modules.append(ACT2FN[config.hidden_act])
        modules.append(
            NSLinear(
                config.llm_hidden_size, config.llm_hidden_size, bias=config.bias
            )
        )
    return nn.Sequential(*modules)
