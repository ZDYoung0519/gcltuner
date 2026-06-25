import torch
from torch import nn

from gcltuner.model.llava import LLaVAModel
from xtuner.model.utils import (
    LoadWoInit, 
    prepare_inputs_labels_for_multimodal,
    make_inputs_require_grad
)
from mmengine import print_log
from mmengine.dist import get_rank



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
        expert_num=6,
        text_encoder=None,
        text_tokenizer=None,
        expert_router_args={},
        init_cur_lora_method="pre",
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
        
        self.expert_num = expert_num
        self._setup_expert_router(**expert_router_args)

        self.projector_depth = projector_depth

        # freeze router 
        for n, p in self.named_parameters():
            if 'expert_router.' in n:
                p.requires_grad_(False)
        self.log_trainable_parameters()
        
        # ensure the ckpt is loaded when the model is ready.
        self.load_checkpoint(pretrained_pth)

        if init_cur_lora_method:
            self.llm._init_cur_lora(cur_task, init_cur_lora_method)    
    
    def log_trainable_parameters(self):
        rank = get_rank()
        if rank == 0:
            total = 0
            count = 0
            for n, p in self.named_parameters():
                if p.requires_grad:
                    print(n, p.shape)
                    count += p.numel()
                total += p.numel()
            print(f"Total parameters: {total / 1024 / 1024} M, trainable: {count/1024/1024} M")

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

        if pretrained_expert_router_path:
            self.expert_router.load_state_dict(torch.load(pretrained_expert_router_path, map_location='cpu'))
            print_log(f"Load pretrained_router from {pretrained_expert_router_path}")
    
    def _forward_expert_router(self, visual_outputs, text_outputs):
        router_input =  torch.cat(
            [
                visual_outputs.last_hidden_state[:, 0, :], 
                text_outputs.last_hidden_state[:, 0, :]
            ],
            dim=-1
        ).detach()
        return self.expert_router(router_input)
    
    def _forward_llava_projector(self, visual_outputs):
        pixel_values = self.projector(
            visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
        )
        return pixel_values

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
        # print("training", self.training)
        if self.training:
            expert_weights = torch.zeros([self.cur_task+1])
            expert_weights[-1] = 1            
        else:
            expert_router_logits = self._forward_expert_router(visual_outputs, text_outputs) / self.expert_router_temp
            expert_router_logits[:, (self.cur_task+1):] = -float('inf') # mask unseen task logits
            expert_weights = nn.Softmax(dim=-1)(expert_router_logits).mean(dim=0)
        self.llm._set_expert_weights(expert_weights)
        self.expert_weights = expert_weights.detach().cpu().tolist()

        # 4. get pixel values for llm
        pixel_values = self._forward_llava_projector(visual_outputs)
        data['pixel_values'] = pixel_values

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
    
    