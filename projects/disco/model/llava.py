import torch
from torch import nn
import torch.nn.functional as F

from gcltuner.model.llava import LLaVAModel
from xtuner.model.utils import (
    LoadWoInit, 
    prepare_inputs_labels_for_multimodal,
    make_inputs_require_grad
)
from mmengine import print_log
from mmengine.dist import get_rank



class DiscoLLaVAModel(LLaVAModel):
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
        

        self.global_text_feature = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.text_encoder.config.hidden_size, dtype=torch.bfloat16)) for _ in range(expert_num)]
            )
        self.global_image_feature = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.visual_encoder.config.hidden_size, dtype=torch.bfloat16)) for _ in range(expert_num)]
            )
        self.image_boundary = nn.ParameterList(
            [nn.Parameter(torch.ones(1, dtype=torch.bfloat16)) for _ in range(expert_num)]
            )
        self.text_boundary = nn.ParameterList(
            [nn.Parameter(torch.ones(1, dtype=torch.bfloat16)) for _ in range(expert_num)]
            )
        self.set_boundary_for_save()

        self.log_trainable_parameters()

        # ensure the ckpt is loaded when the model is ready.
        self.load_checkpoint(pretrained_pth)

    def set_boundary_for_save(self):
        for name, param in self.image_boundary.named_parameters():
            param.requires_grad = True
        
        for name, param in self.text_boundary.named_parameters():
            param.requires_grad = True

        for name, param in self.global_image_feature.named_parameters():
            param.requires_grad = True
        
        for name, param in self.global_text_feature.named_parameters():
            param.requires_grad = True
    
    def log_trainable_parameters(self):
        rank = get_rank()
        if rank == 0:
            total = 0
            count = 0
            for n, p in self.named_parameters():
                if p.requires_grad:
                    # print(n, p.shape)
                    count += p.numel()
                total += p.numel()
            print(f"Total parameters: {total / 1024 / 1024} M, trainable: {count/1024/1024} M")
    
    def _set_lora_id(self, lora_id):
        from peft.tuners.discomoelora import CoINMOELoraLinear
        for n, m in self.llm.named_modules():
            if isinstance(m, CoINMOELoraLinear):
                m.lora_id = lora_id
    
    def _set_mask_singal(self, mask_singal):
        from peft.tuners.discomoelora import CoINMOELoraLinear
        for n, m in self.llm.named_modules():
            if isinstance(m, CoINMOELoraLinear):
                m.mask_singal = mask_singal

    def _prepare_data_for_llm(self, data, mode='loss'):
        with torch.no_grad():
            # visual encoding
            images = data['pixel_values']
            visual_outputs = self.visual_encoder(images.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
            image_features = visual_outputs.hidden_states[self.visual_select_layer][1:]
            image_guide_features = visual_outputs.image_embeds.to(images.dtype)

            # text encoding
            clip_text_inputs = self.text_tokenizer(
                data['text'],
                padding="longest",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            text_outputs = self.text_encoder(clip_text_inputs)
            text_guide_features = text_outputs.pooler_output.to(images.dtype)

            # compute cos similarity for current image/text features
            text_guide_features_mean = text_guide_features.mean(dim=0, keepdim=True)
            image_guide_features_mean = image_guide_features.mean(dim=0, keepdim=True)

            global_text_feature = torch.stack([param for param in self.global_text_feature])
            global_image_feature = torch.stack([param for param in self.global_image_feature]) 
            cos_sim_text = F.cosine_similarity(global_text_feature, text_guide_features_mean, dim=1)
            cos_sim_img = F.cosine_similarity(global_image_feature, image_guide_features_mean, dim=1)
            # cos_sim = 0.5 * cos_sim_text + 0.5 * cos_sim_img
            cos_sim = cos_sim_text[:self.expert_num]        # HERE: only the text features are kept?
            cos_sim_softmax = F.softmax(cos_sim / 0.05)

            # update global features
            # Question: image/text boundary = 1, and global image/text features are not divided by the sample size
            current_image_features = image_guide_features  # [batch_size, feature_dim]
            current_text_features = text_guide_features  # [batch_size, feature_dim]
            task_id = self.cur_task

            image_sum = self.global_image_feature[task_id] * self.image_boundary[task_id] + current_image_features.sum(dim=0)
            text_sum = self.global_text_feature[task_id] * self.text_boundary[task_id] + current_text_features.sum(dim=0)

            # modified version:
            self.image_boundary[task_id].data += current_image_features.shape[0]
            self.text_boundary[task_id].data += current_text_features.shape[0]

            self.global_image_feature[task_id] = image_sum / self.image_boundary[task_id]
            self.global_text_feature[task_id] = text_sum / self.text_boundary[task_id]

        # forward projector
        image_features = self.projector(image_features)

        if self.training:
            self._set_lora_id(self.cur_task)
        
        else:
            cos_sim_list = cos_sim_softmax.tolist()
            self._set_mask_singal(cos_sim_list)
        
        data['pixel_values'] = image_features

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
            
    