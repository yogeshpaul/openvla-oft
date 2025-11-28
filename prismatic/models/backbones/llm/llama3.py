"""
llama3.py

Class definition for all LLMs derived from LlamaForCausalLM.
"""

from typing import Optional, Sequence, Type

import torch
from torch import nn as nn
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import (
    LLaMa2ChatPromptBuilder,
    PromptBuilder,
    PurePromptBuilder,
)

from llama2 import LLaMa2LLMBackbone

# Registry =>> Support LLaMa-3 Models (from HF Transformers)
# fmt: off
LLAMA3_MODELS = {
    # === Meta LLaMa-3 Chat Models ===
    "llama3.2-3b": {
        "llm_family": "llama3.2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-3.2-3B-Instruct"
    },

    "llama3.1-8b": {
        "llm_family": "llama3.1", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-3.1-8B-Instruct"
    },
}
# fmt: on


class LLaMa3LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **LLAMA3_MODELS[llm_backbone_id],
        )

        # [Special Case] LLaMa-2 PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.startswith("Llama-3") and self.identifier.endswith("B"):
            return PurePromptBuilder

        elif self.identifier.startswith("Llama-3") and self.identifier.endswith("-Instruct"):
            return LLaMa2ChatPromptBuilder

        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return LlamaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        return (self.llm.model.embed_tokens, self.llm.model.layers[-1], self.llm.lm_head)
