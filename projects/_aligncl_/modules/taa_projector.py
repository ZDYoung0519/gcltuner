import torch
from torch import nn
from transformers.activations import ACT2FN


class LayerWiseProjector(nn.Module):
    def __init__(self, in_features, out_features, hidden_size, hidden_act='relu', bias=False):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_size, bias=bias),
            nn.ReLU() if hidden_act == 'relu' else ACT2FN[hidden_act],
            nn.Linear(hidden_size, out_features, bias=bias)
        )
    
    def forward(self, x):
        return self.model(x)

class LayerWiseProjectorWithExpertPool(nn.Module):
    def __init__(
            self, 
            in_features, 
            out_features, 
            expert_size, 
            expert_hidden_size, 
            use_shared_expert=False,
            shared_expert_hidden_size=None,
            hidden_act='relu', bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_shared_expert = use_shared_expert

        if use_shared_expert:
            self.shared_expert = nn.Sequential(
                nn.Linear(in_features, shared_expert_hidden_size, bias=bias),
                nn.ReLU() if hidden_act == 'relu' else ACT2FN[hidden_act],
                nn.Linear(shared_expert_hidden_size, out_features, bias=bias)
            )
        self.expert_pool = nn.ModuleList([])
        for _ in range(expert_size):
            self.expert_pool.append(
                nn.Sequential(
                    nn.Linear(in_features, expert_hidden_size, bias=bias),
                    nn.ReLU() if hidden_act == 'relu' else ACT2FN[hidden_act],
                    nn.Linear(expert_hidden_size, out_features, bias=bias)
                )
            )        
        self.expert_weights = None

    def _set_expert_weights(self, expert_weights):
        self.expert_weights = expert_weights

    def forward(self, x):
        expert_weights = self.expert_weights
        if expert_weights is None:
            if self.use_shared_expert:
                return self.shared_expert(x)
            else:
                raise ValueError("Expert weights are not set for LayerWiseProjectorWithExpertPool.")

        if isinstance(expert_weights, list):
            expert_weights = torch.tensor(expert_weights, device=x.device)    # (expert_size,)
        active_experts = torch.nonzero(expert_weights).squeeze(-1)
        out = torch.zeros([x.shape[0], x.shape[1], self.out_features], device=x.device, dtype=x.dtype)
        for i in active_experts.tolist():
            expert_out = self.expert_pool[i](x)
            out = out + expert_weights[i] * expert_out
        if self.use_shared_expert:
            out = out + self.shared_expert(x)
        return out

class AlignLinearWithExpertPool(nn.Module):
    def __init__(self, in_features, out_features, expert_size, expert_hidden_size, hidden_act='relu', bias=False):
        super().__init__()
        self.shared_linear = nn.Linear(in_features, out_features, bias=bias)

        self.expert_pool = nn.ModuleList([])
        for _ in range(expert_size):
            self.expert_pool.append(
                nn.Sequential(
                    nn.Linear(in_features, expert_hidden_size, bias=bias),
                    nn.ReLU() if hidden_act == 'relu' else ACT2FN[hidden_act],
                    nn.Linear(expert_hidden_size, out_features, bias=bias)
                )
            )
        self.expert_weights = None
    
    def _set_expert_weights(self, expert_weights):
        self.expert_weights = expert_weights
    
    def forward(self, x):
        expert_weights = self.expert_weights
        if isinstance(expert_weights, list):
            expert_weights = torch.tensor(expert_weights, device=x.device)    # (expert_size,)
        active_experts = torch.nonzero(expert_weights).squeeze(-1)
        out = torch.zeros([x.shape[0], self.shared_linear.out_features], device=x.device, dtype=x.dtype)
        for i in active_experts.tolist():
            expert_out = self.expert_pool[i](x)
            out = out + expert_weights[i] * expert_out
        out = out + self.shared_linear(x)
        return out
    

class TaskAdaptiveAggregator(nn.Module):
    def __init__(
            self, 
            visual_encoder_hidden_size, 
            text_encoder_hidden_size, 
            llm_hidden_size, 
            visual_select_layers,
            num_experts, 
            num_visual_layers,
            topk=3,
            cur_task=0,
            freeze_previous_experts=True,
            freeze_shared_expert_after_first_task=True
            ):
        super().__init__()
        self.visual_hidden_size = visual_encoder_hidden_size
        self.text_hidden_size = text_encoder_hidden_size

        if visual_select_layers == 'all':
            visual_select_layers = [str(i) for i in range(num_visual_layers)]
        self.visual_select_layers = visual_select_layers

        self.align_linear_with_pool = AlignLinearWithExpertPool(
            text_encoder_hidden_size,
            visual_encoder_hidden_size,
            expert_size=6,
            expert_hidden_size=text_encoder_hidden_size//2,
            hidden_act='relu',
            bias=False
        )
        if freeze_previous_experts:
            for task_id in range(cur_task):
                for p in self.align_linear_with_pool.expert_pool[task_id].parameters():
                    p.requires_grad_(False)
        if freeze_shared_expert_after_first_task and cur_task > 0:
            for p in self.align_linear_with_pool.shared_linear.parameters():
                p.requires_grad_(False)

        self.topk = topk
        self.layer_wise_vision_projectors = nn.ModuleList()
        for _ in range(len(visual_select_layers)):
            layer_wise_vision_projector = LayerWiseProjectorWithExpertPool(
                visual_encoder_hidden_size,
                llm_hidden_size,
                expert_size=6, 
                expert_hidden_size=visual_encoder_hidden_size//2,
                use_shared_expert=True,
                shared_expert_hidden_size=visual_encoder_hidden_size
            )
            self.layer_wise_vision_projectors.append(layer_wise_vision_projector)
        
            if freeze_previous_experts:
                for task_id in range(cur_task):
                    for p in layer_wise_vision_projector.expert_pool[task_id].parameters():
                        p.requires_grad_(False)
            if freeze_shared_expert_after_first_task and cur_task > 0:
                for p in layer_wise_vision_projector.shared_expert.parameters():
                    p.requires_grad_(False)
                
    def _set_expert_weights(self, expert_weights):
        for layer_wise_vision_projector in self.layer_wise_vision_projectors:
            layer_wise_vision_projector._set_expert_weights(expert_weights)
        self.align_linear_with_pool._set_expert_weights(expert_weights)
    
    def _collect_visual_features(self, visual_outputs, i):
        layer = self.visual_select_layers[i]
        if isinstance(layer, str) and '-' in layer:
            start, end = layer.split('-')
            start, end = int(start), int(end)
            feature = [visual_outputs.hidden_states[idx][:, 1:, :] for idx in range(start, end+1)]
            feature = torch.stack(feature, dim=0).mean(dim=0)   # avg pooling
        else:
            idx = int(layer)
            feature = visual_outputs.hidden_states[idx][:, 1:, :]    # B, L-1, d
        return feature
    
    def _collect_visual_cls_token(self, visual_outputs, i):
        layer = self.visual_select_layers[i]
        if isinstance(layer, str) and '-' in layer:
            start, end = layer.split('-')
            start, end = int(start), int(end)
            feature = [visual_outputs.hidden_states[idx][:, 0, :] for idx in range(start, end+1)]
            feature = torch.stack(feature, dim=0).mean(dim=0)   # avg pooling
        else:
            idx = int(layer)
            feature = visual_outputs.hidden_states[idx][:, 0, :]    # B, d
        return feature

    # def forward(self, visual_outputs, text_outputs):
    #     text_cls_token = text_outputs['last_hidden_state'] [:, 0, :]    # B, d_t
    #     vision_cls_tokens = [
    #         self._collect_visual_cls_token(visual_outputs, i) for i in range(len(self.visual_select_layers))
    #     ]   # B, M, d_v

    #     query = self.align_linear_with_pool(text_cls_token).unsqueeze(1)    # B, 1, d
    #     key = torch.stack(vision_cls_tokens, dim=1)               # B, M, d
    #     attention = torch.bmm(query, key.transpose(-2, -1))         # (B, 1, d) * (B, M, d) -> (B, 1, M)
    #     attention = nn.Softmax(dim=-1)(attention / self.visual_hidden_size**0.5)    # (B, 1, M)

    #     self.topk = self.topk
    #     _, topk_indices = torch.topk(attention, self.topk, dim=-1)    # (B, 1, topk)
    #     topk_indices = topk_indices.squeeze(1)    # (B, topk)

    #     vision_tokens= []
    #     for top_idx in topk_indices.cpu().tolist():
    #         layer_wise_vision_tokens = self._collect_visual_features(visual_outputs, self.visual_select_layers[top_idx])   # B, L-1, d
    #         layer_wise_vision_tokens = self.layer_wise_vision_projectors[top_idx](layer_wise_vision_tokens)   # B, L-1, d_llm
    #         vision_tokens.append(layer_wise_vision_tokens)
    #     value = torch.stack(vision_tokens, dim=1)   # B, M, L ,d_llm
    #     weight = attention.transpose(-2, -1).unsqueeze(-1)   # B, M, 1, 1
    #     weighted_value = (weight * value).sum(dim=1)    # B, L, d_llm
    #     return weighted_value

    def forward(self, visual_outputs, text_outputs):
        # Input: text_outputs['last_hidden_state']: (B, L_text, d_text)
        text_cls_token = text_outputs['last_hidden_state'][:, 0, :]  # (B, d_text)
        
        # Collect CLS tokens from all visual layers
        vision_cls_tokens = torch.stack([
            self._collect_visual_cls_token(visual_outputs, i)  # (B, d_v)
            for i in range(len(self.visual_select_layers))
        ], dim=1)  # (B, M, d_v), M = #selected layers
        
        # Align text to visual space
        query = self.align_linear_with_pool(text_cls_token).unsqueeze(1)  # (B, 1, d_v)
        
        # Compute attention scores
        attention = torch.bmm(query, vision_cls_tokens.transpose(-2, -1))  # (B, 1, M)
        attention = nn.Softmax(dim=-1)(attention / self.visual_hidden_size**0.5)  # (B, 1, M)
        
        # Get top-k layers
        topk_values, topk_indices = torch.topk(attention, self.topk, dim=-1)  # (B, 1, topk)
        topk_indices = topk_indices.squeeze(1)  # (B, topk)
        topk_values = topk_values.squeeze(1)    # (B, topk)
        
        # Deduplicate layer indices
        unique_layers = torch.unique(topk_indices)  # (U,), U <= min(topk*B, M)
        
        # Cache projected features for unique layers
        layer_cache = {}
        for layer_idx in unique_layers.tolist():
            layer_features = self._collect_visual_features(visual_outputs, layer_idx)  # (B, L_v, d_v)
            projected = self.layer_wise_vision_projectors[layer_idx](layer_features)  # (B, L_v, d_llm)
            layer_cache[layer_idx] = projected
        
        # Gather selected features for each sample
        batch_size = topk_indices.shape[0]  # B
        seq_len = layer_cache[list(layer_cache.keys())[0]].shape[1]  # L_v
        llm_dim = layer_cache[list(layer_cache.keys())[0]].shape[2]  # d_llm
        
        selected_features = torch.zeros(
            batch_size, self.topk, seq_len, llm_dim,
            device=topk_indices.device,
            dtype=layer_cache[list(layer_cache.keys())[0]].dtype
        )  # (B, topk, L_v, d_llm)
        
        # Per-sample per-layer feature collection
        for b in range(batch_size):
            for k, layer_idx in enumerate(topk_indices[b].tolist()):
                selected_features[b, k] = layer_cache[layer_idx][b]  # (L_v, d_llm)
        
        # Weighted sum
        weights = topk_values.unsqueeze(-1).unsqueeze(-1)  # (B, topk, 1, 1)
        weighted_value = (weights * selected_features).sum(dim=1)  # (B, L_v, d_llm)
        
        return weighted_value
