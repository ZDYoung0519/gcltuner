import logging
from collections import OrderedDict
from transformers import GenerationConfig
from mmengine import print_log
from mmengine.dist import get_rank

from xtuner.model.llava import LLaVAModel as XTunerLLaVAModel, guess_load_checkpoint
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria


def get_peft_model_state_dict(model, state_dict=None, adapter_name="default"):
    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()

    bias = config.bias
    
    if bias == "none":
        to_return = {k: state_dict[k] for k in state_dict if "lora" in k}
    elif bias == "all":
        to_return = {
            k: state_dict[k] for k in state_dict if "lora" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        for k in state_dict:
            if "lora_" in k:
                to_return[k] = state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError
    to_return = {
        k: v
        for k, v in to_return.items()
        if (("lora_" in k and adapter_name in k) or ("bias" in k))
    }

    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(
                f"{module_name}.modules_to_save.{adapter_name}" in key
                for module_name in model.modules_to_save
            ):
                to_return[key] = value
    return to_return


class LLaVAModel(XTunerLLaVAModel):
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
            max_position_embeddings
        )
        self.load_checkpoint(pretrained_pth)
    
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

    def load_checkpoint(self, ckpt_path):
        if ckpt_path is not None:
            pretrained_state_dict = guess_load_checkpoint(ckpt_path)
            missing_keys, unexpected_keys = self.load_state_dict(pretrained_state_dict, strict=False)
            model_keys = set(self.state_dict().keys())
            loaded_keys = [k for k in pretrained_state_dict if k in model_keys]
            print_log(f"Load pretrained weight from {ckpt_path}", "current", level=logging.INFO)
            for k in loaded_keys:
                print_log(f"  Load key: {k}: {pretrained_state_dict[k].shape}", "current", level=logging.INFO)

    def _prepare_data_for_llm(self, data, mode='loss'):
        # prepare pixel values
        visual_outputs = self.visual_encoder(
            data["pixel_values"].to(self.visual_encoder.dtype),
            output_hidden_states=True,
        )
        pixel_values = self.projector(
            visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
        )
        data["pixel_values"] = pixel_values

        # prepare inputs for llm
        if mode == 'predict' or mode == 'generate':
            data = dict(
                input_ids=data["input_ids"],
                pixel_values=data["pixel_values"],
            )
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
            return self.generate(data, data_samples)
        elif mode == "tensor":
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError
        
    def preparing_for_generation(self, tokenizer, metainfo: dict = None):
        self.tokenizer = tokenizer
        default_generation_kwargs = dict(
            num_beams=1,
            max_new_tokens=256,
            temperature=0,
            do_sample=False,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
        default_generation_kwargs.update(metainfo.get('generation_kwargs', {}))
        self.gen_config = GenerationConfig(**default_generation_kwargs)

        stop_criteria = get_stop_criteria(tokenizer=tokenizer, stop_words="")
        self.stop_criteria = stop_criteria

    def generate(self, data, data_samples=None):
        generate_output = self.llm.generate(
            **data,
            generation_config=self.gen_config,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            streamer=None,
        )

        prediction = self.tokenizer.decode(
            generate_output[0], skip_special_tokens=True).strip()

        return dict(prediction=prediction)

    def state_dict(self, *args, **kwargs):
        """
        XTunerLLaVAModel only support `LoRA` finetune.
        We have to modify this to support differnt peft types
        """
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
        return to_return

