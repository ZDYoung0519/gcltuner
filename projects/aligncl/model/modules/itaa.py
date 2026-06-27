import torch
from torch import nn
from transformers.activations import ACT2FN



class ITAAProjector(nn.Module):
    def __init__(self, visual_hidden_size, llm_hidden_size):
        super().__init__()

    def gradient_checkpointing_disable(self):
        pass


class LayerWiseProjectors(nn.Module):
    def __init__(self,
            visual_hidden_size, 
            llm_hidden_size, 
            expert_num, 
            expert_hidden_size,
            use_shared_expert=False,
            shared_expert_hidden_size=None,    
            hidden_act='relu', 
            bias=False
        ):
        super().__init__()

        self.projector_pool = nn.ModuleList([])
        for _ in range(expert_num):
            self.projector_pool.append(
                nn.Sequential(
                    nn.Linear(visual_hidden_size, expert_hidden_size, bias=bias),
                    nn.ReLU() if hidden_act == 'relu' else ACT2FN[hidden_act],
                    nn.Linear(expert_hidden_size, llm_hidden_size, bias=bias)
                )
            )
        
        if use_shared_expert:
            self.shared_expert = nn.Sequential(
                nn.Linear(visual_hidden_size, shared_expert_hidden_size, bias=bias),
                nn.ReLU() if hidden_act == 'relu' else ACT2FN[hidden_act],
                nn.Linear(shared_expert_hidden_size, llm_hidden_size, bias=bias)
            )


class AlignLinears(nn.Module):
    def __init__(self, in_features, out_features, expert_size, bias=False):
        super().__init__()
        self.shared_linear = nn.Linear(in_features, out_features, bias=bias)

        self.weights_pool = nn.ModuleList([])
        for _ in range(expert_size):
            self.weights_pool.append(
                nn.Linear(in_features, out_features, bias=bias),
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
            expert_out = self.weights_pool[i](x)
            out = out + expert_weights[i] * expert_out
        out = out + self.shared_linear(x)
        return out

class ITAAMoEProjectors(nn.Module):
    def __init__(
            self, 
            visual_select_layers,
            num_visual_layers,
            visual_hidden_size, 
            text_hidden_size,
            llm_hidden_size, 
            expert_num, 
            expert_hidden_size,
            use_shared_expert=False,
            shared_expert_hidden_size=None,    
            hidden_act='relu', 
            bias=False,
            cur_task=0,
            layer_topk=3
        ):
        super().__init__()
        self.expert_weights = None
        self.layer_topk=layer_topk
        
        if visual_select_layers == 'all':
            visual_select_layers = [str(i) for i in range(num_visual_layers)]
        self.visual_select_layers = visual_select_layers

        # layerwise projectors
        self.all_layer_projectors = nn.ModuleList([])
        for _ in range(len(visual_select_layers)):
            layer_wise_projectors = LayerWiseProjectors(
                visual_hidden_size,
                llm_hidden_size,
                expert_size=expert_num, 
                expert_hidden_size=expert_hidden_size,
                use_shared_expert=use_shared_expert,
                shared_expert_hidden_size=shared_expert_hidden_size,
                hidden_act=hidden_act,
                bias=bias
            )
            self.all_layer_projectors.append(layer_wise_projectors)
        
            # freeze previous projectros
            for task_id in range(cur_task):
                for p in layer_wise_projectors.projector_pool[task_id].parameters():
                    p.requires_grad_(False)

            # freeze shared projectors if cur_task > 0:
            if cur_task > 0:
                for p in layer_wise_projectors.shared_expert.parameters():
                    p.requires_grad_(False)
        

        # alignment weighs
        self.align_linears = AlignLinears(text_hidden_size, visual_hidden_size, expert_num)
    
    def _set_expert_weights(self, expert_weights):
        self.expert_weights = expert_weights

    def _collect_visual_features(self, visual_outputs, i):
        layer = self.visual_select_layers[i]
        if isinstance(layer, str) and '-' in layer:
            start, end = layer.split('-')
            start, end = int(start), int(end)
            feature = [visual_outputs.hidden_states[idx] for idx in range(start, end+1)]
            feature = torch.stack(feature, dim=0).mean(dim=0)   # avg pooling
        else:
            idx = int(layer)
            feature = visual_outputs.hidden_states[idx]
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
    
    def _collect_visual_features(self, visual_outputs, i):
        layer = self.visual_select_layers[i]
        if isinstance(layer, str) and '-' in layer:
            start, end = layer.split('-')
            start, end = int(start), int(end)
            feature = [visual_outputs.hidden_states[idx] for idx in range(start, end+1)]
            feature = torch.stack(feature, dim=0).mean(dim=0)   # avg pooling
        else:
            idx = int(layer)
            feature = visual_outputs.hidden_states[idx]    # B, L-1, d
        return feature

    def forward(self, visual_outputs, text_outputs):
        text_cls_token = text_outputs['last_hidden_state'][:, 0, :]  # (B, d_text)

        # (B, M, d_v), M = #selected layers
        vision_cls_tokens = torch.stack([
            self._collect_visual_cls_token(visual_outputs, i)  # (B, d_v)
            for i in range(len(self.visual_select_layers))
        ], dim=1)  


        # Align text to visual space as query
        query = self.align_linears(text_cls_token).unsqueeze(1)  # (B, 1, d_v)

        attention = torch.bmm(query, vision_cls_tokens.transpose(-2, -1))  # (B, 1, M)
        attention = nn.Softmax(dim=-1)(attention / query.shape[-1]**0.5)  # (B, 1, M)

        # Get top-k layers
        topk_values, topk_indices = torch.topk(attention, self.layer_topk, dim=-1)  # (B, 1, topk)
        topk_indices = topk_indices.squeeze(1)  # (B, topk)
        topk_values = topk_values.squeeze(1)    # (B, topk)

        # Deduplicate layer indices
        unique_layers = torch.unique(topk_indices)  # (U,), U <= min(topk*B, M)

        # Cache projected features for unique layers
        layer_cache = {}
        for layer_idx in unique_layers.tolist():
            layer_features = self._collect_visual_features(visual_outputs, layer_idx)  # (B, L, d_v)
            projected = self.all_layer_projectors[layer_idx](layer_features)  # (B, L, d_llm)
            layer_cache[layer_idx] = projected
        
        # Gather selected features for each sample
        batch_size = topk_indices.shape[0]  # B
        seq_len = layer_cache[list(layer_cache.keys())[0]].shape[1]  # L_v
        llm_dim = layer_cache[list(layer_cache.keys())[0]].shape[2]  # d_llm

        selected_features = torch.zeros(
            batch_size, self.layer_topk, seq_len, llm_dim,
            device=topk_indices.device,
            dtype=layer_cache[list(layer_cache.keys())[0]].dtype
        )  # (B, topk, L, d_llm)
        
        # Per-sample per-layer feature collection
        for b in range(batch_size):
            for k, layer_idx in enumerate(topk_indices[b].tolist()):
                selected_features[b, k] = layer_cache[layer_idx][b]  # (L_v, d_llm)
        
        # Weighted sum
        weights = topk_values.unsqueeze(-1).unsqueeze(-1)  # (B, topk, 1, 1)
        weighted_value = (weights * selected_features).sum(dim=1)  # (B, topk, 1, 1)* (B, topk, L, d_llm) ->(B, L_v, d_llm)

        return attention, weighted_value
        