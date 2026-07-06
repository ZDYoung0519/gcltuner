import torch
from torch import nn
from xtuner.model.modules import ProjectorModel, ProjectorConfig
from transformers.activations import ACT2FN


class SeparateProjectorModel(nn.Module):
    def __init__(self, config: ProjectorConfig, num_expert: int=8) -> None:
        super().__init__()
        self.mm_projectors = nn.ModuleList()
        for _ in range(num_expert):
            self.mm_projectors.append(ProjectorModel(config))
        
        self.visual_hidden_size = config.visual_hidden_size
        self.llm_hidden_size = config.llm_hidden_size
        self.expert_weights = torch.ones([num_expert]) / num_expert
    
    def forward(self, x):
        result = torch.zeros(*x.shape[:-1], self.llm_hidden_size,
            device=x.device,
            dtype=x.dtype
        )
        expert_weights = self.expert_weights
        active_experts = torch.nonzero(expert_weights).squeeze(-1)
        for i in active_experts.tolist():
            result += self.mm_projectors[i](x)
        return result

    def _init_cur_weights_with_previous(self, cur_task):
        if cur_task > 0:
            self.mm_projectors[cur_task].load_state_dict(
                self.mm_projectors[cur_task-1].state_dict()
            )
    
    def _set_expert_weights(self, weights):
        self.expert_weights = weights

class LightWeightSeparateProjectorModel(nn.Module):
    def __init__(self, config: ProjectorConfig, num_expert: int=8, use_base=False, hidden_scale=0.5) -> None:
        super().__init__()

        self.use_base = use_base
        self.hidden_scale = hidden_scale

        self.mm_projectors = nn.ModuleList()
        for _ in range(num_expert):
            self.mm_projectors.append(self._create_light_weight_projector(config, hidden_scale))
        if use_base:
            self.base_projector = ProjectorModel(config)
        self.visual_hidden_size = config.visual_hidden_size
        self.llm_hidden_size = config.llm_hidden_size
        self.expert_weights = torch.ones([num_expert]) / num_expert
    
    def _create_light_weight_projector(self, config, hidden_scale):
        hidden_size=int(config.llm_hidden_size*hidden_scale)
        modules = [
            nn.Linear(
                config.visual_hidden_size, hidden_size, bias=config.bias
            )
        ]
        for idx in range(1, config.depth):
            modules.append(ACT2FN[config.hidden_act])
            out_size = config.llm_hidden_size if idx == config.depth - 1 else hidden_size
            modules.append(
                nn.Linear(
                    hidden_size, out_size, bias=config.bias
                )
            )
        return nn.Sequential(*modules)

    def forward(self, x):
        result = torch.zeros(*x.shape[:-1], self.llm_hidden_size,
            device=x.device,
            dtype=x.dtype
        )
        expert_weights = self.expert_weights
        active_experts = torch.nonzero(expert_weights).squeeze(-1)
        for i in active_experts.tolist():
            result += self.mm_projectors[i](x) * expert_weights[i]
            
        if self.use_base:
            result += self.base_projector(x)
        return result

    def _init_cur_weights_with_previous(self, cur_task):
        if cur_task > 0:
            self.mm_projectors[cur_task].load_state_dict(
                self.mm_projectors[cur_task-1].state_dict()
            )
    
    def _set_expert_weights(self, weights):
        self.expert_weights = weights
    
    def gradient_checkpointing_disable(self):
        pass