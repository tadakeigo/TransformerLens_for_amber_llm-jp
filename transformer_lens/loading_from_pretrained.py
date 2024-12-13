"""Loading Pretrained Models Utilities.

This module contains functions for loading pretrained models from the Hugging Face Hub.
"""

import dataclasses
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    BertForPreTraining,
    T5ForConditionalGeneration,
)

import transformer_lens.utils as utils
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions import (
    convert_bert_weights,
    convert_bloom_weights,
    convert_coder_weights,
    convert_gemma_weights,
    convert_gpt2_weights,
    convert_gptj_weights,
    convert_llama_weights,
    convert_mingpt_weights,
    convert_mistral_weights,
    convert_mixtral_weights,
    convert_neel_solu_old_weights,
    convert_neo_weights,
    convert_neox_weights,
    convert_olmo2_weights,
    convert_olmo_weights,
    convert_opt_weights,
    convert_phi3_weights,
    convert_phi_weights,
    convert_qwen2_weights,
    convert_qwen_weights,
    convert_t5_weights,
)

OFFICIAL_MODEL_NAMES = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    "facebook/opt-125m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    "facebook/opt-66b",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-j-6B",
    "EleutherAI/gpt-neox-20b",
    "stanford-crfm/alias-gpt2-small-x21",
    "stanford-crfm/battlestar-gpt2-small-x49",
    "stanford-crfm/caprica-gpt2-small-x81",
    "stanford-crfm/darkmatter-gpt2-small-x343",
    "stanford-crfm/expanse-gpt2-small-x777",
    "stanford-crfm/arwen-gpt2-medium-x21",
    "stanford-crfm/beren-gpt2-medium-x49",
    "stanford-crfm/celebrimbor-gpt2-medium-x81",
    "stanford-crfm/durin-gpt2-medium-x343",
    "stanford-crfm/eowyn-gpt2-medium-x777",
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b-deduped",
    "EleutherAI/pythia-70m-v0",
    "EleutherAI/pythia-160m-v0",
    "EleutherAI/pythia-410m-v0",
    "EleutherAI/pythia-1b-v0",
    "EleutherAI/pythia-1.4b-v0",
    "EleutherAI/pythia-2.8b-v0",
    "EleutherAI/pythia-6.9b-v0",
    "EleutherAI/pythia-12b-v0",
    "EleutherAI/pythia-70m-deduped-v0",
    "EleutherAI/pythia-160m-deduped-v0",
    "EleutherAI/pythia-410m-deduped-v0",
    "EleutherAI/pythia-1b-deduped-v0",
    "EleutherAI/pythia-1.4b-deduped-v0",
    "EleutherAI/pythia-2.8b-deduped-v0",
    "EleutherAI/pythia-6.9b-deduped-v0",
    "EleutherAI/pythia-12b-deduped-v0",
    "EleutherAI/pythia-160m-seed1",
    "EleutherAI/pythia-160m-seed2",
    "EleutherAI/pythia-160m-seed3",
    "NeelNanda/SoLU_1L_v9_old",
    "NeelNanda/SoLU_2L_v10_old",
    "NeelNanda/SoLU_4L_v11_old",
    "NeelNanda/SoLU_6L_v13_old",
    "NeelNanda/SoLU_8L_v21_old",
    "NeelNanda/SoLU_10L_v22_old",
    "NeelNanda/SoLU_12L_v23_old",
    "NeelNanda/SoLU_1L512W_C4_Code",
    "NeelNanda/SoLU_2L512W_C4_Code",
    "NeelNanda/SoLU_3L512W_C4_Code",
    "NeelNanda/SoLU_4L512W_C4_Code",
    "NeelNanda/SoLU_6L768W_C4_Code",
    "NeelNanda/SoLU_8L1024W_C4_Code",
    "NeelNanda/SoLU_10L1280W_C4_Code",
    "NeelNanda/SoLU_12L1536W_C4_Code",
    "NeelNanda/GELU_1L512W_C4_Code",
    "NeelNanda/GELU_2L512W_C4_Code",
    "NeelNanda/GELU_3L512W_C4_Code",
    "NeelNanda/GELU_4L512W_C4_Code",
    "NeelNanda/Attn_Only_1L512W_C4_Code",
    "NeelNanda/Attn_Only_2L512W_C4_Code",
    "NeelNanda/Attn_Only_3L512W_C4_Code",
    "NeelNanda/Attn_Only_4L512W_C4_Code",
    "NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr",
    "NeelNanda/SoLU_1L512W_Wiki_Finetune",
    "NeelNanda/SoLU_4L512W_Wiki_Finetune",
    "ArthurConmy/redwood_attn_2l",
    "llama-7b-hf",
    "llama-13b-hf",
    "llama-30b-hf",
    "llama-65b-hf",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "CodeLlama-7b-hf",
    "CodeLlama-7b-Python-hf",
    "CodeLlama-7b-Instruct-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "Baidicoot/Othello-GPT-Transformer-Lens",
    "bert-base-cased",
    "roneneldan/TinyStories-1M",
    "roneneldan/TinyStories-3M",
    "roneneldan/TinyStories-8M",
    "roneneldan/TinyStories-28M",
    "roneneldan/TinyStories-33M",
    "roneneldan/TinyStories-Instruct-1M",
    "roneneldan/TinyStories-Instruct-3M",
    "roneneldan/TinyStories-Instruct-8M",
    "roneneldan/TinyStories-Instruct-28M",
    "roneneldan/TinyStories-Instruct-33M",
    "roneneldan/TinyStories-1Layer-21M",
    "roneneldan/TinyStories-2Layers-33M",
    "roneneldan/TinyStories-Instuct-1Layer-21M",
    "roneneldan/TinyStories-Instruct-2Layers-33M",
    "stabilityai/stablelm-base-alpha-3b",
    "stabilityai/stablelm-base-alpha-7b",
    "stabilityai/stablelm-tuned-alpha-3b",
    "stabilityai/stablelm-tuned-alpha-7b",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    "bigcode/santacoder",
    "Qwen/Qwen-1_8B",
    "Qwen/Qwen-7B",
    "Qwen/Qwen-14B",
    "Qwen/Qwen-1_8B-Chat",
    "Qwen/Qwen-7B-Chat",
    "Qwen/Qwen-14B-Chat",
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-0.5B-Chat",
    "Qwen/Qwen1.5-1.8B",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen1.5-4B",
    "Qwen/Qwen1.5-4B-Chat",
    "Qwen/Qwen1.5-7B",
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen1.5-14B",
    "Qwen/Qwen1.5-14B-Chat",
    "Qwen/Qwen2-0.5B",
    "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2-1.5B",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2-7B-Instruct",
    "microsoft/phi-1",
    "microsoft/phi-1_5",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "google/gemma-2b",
    "google/gemma-7b",
    "google/gemma-2b-it",
    "google/gemma-7b-it",
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b",
    "google/gemma-2-27b-it",
    "01-ai/Yi-6B",
    "01-ai/Yi-34B",
    "01-ai/Yi-6B-Chat",
    "01-ai/Yi-34B-Chat",
    "google-t5/t5-small",
    "google-t5/t5-base",
    "google-t5/t5-large",
    "ai-forever/mGPT",
    "LLM360/Amber",
    "yu-takagi/jv3-7",
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-7B-0724-hf",
    "allenai/OLMo-7B-0424-hf",
    "allenai/OLMo-7B",
]
"""Official model names for models on HuggingFace."""

# Model Aliases:
MODEL_ALIASES = {
    "NeelNanda/SoLU_1L_v9_old": ["solu-1l-pile", "solu-1l-old"],
    "NeelNanda/SoLU_2L_v10_old": ["solu-2l-pile", "solu-2l-old"],
    "NeelNanda/SoLU_4L_v11_old": ["solu-4l-pile", "solu-4l-old"],
    "NeelNanda/SoLU_6L_v13_old": ["solu-6l-pile", "solu-6l-old"],
    "NeelNanda/SoLU_8L_v21_old": ["solu-8l-pile", "solu-8l-old"],
    "NeelNanda/SoLU_10L_v22_old": ["solu-10l-pile", "solu-10l-old"],
    "NeelNanda/SoLU_12L_v23_old": ["solu-12l-pile", "solu-12l-old"],
    "NeelNanda/SoLU_1L512W_C4_Code": ["solu-1l", "solu-1l-new", "solu-1l-c4-code"],
    "NeelNanda/SoLU_2L512W_C4_Code": ["solu-2l", "solu-2l-new", "solu-2l-c4-code"],
    "NeelNanda/SoLU_3L512W_C4_Code": ["solu-3l", "solu-3l-new", "solu-3l-c4-code"],
    "NeelNanda/SoLU_4L512W_C4_Code": ["solu-4l", "solu-4l-new", "solu-4l-c4-code"],
    "NeelNanda/GELU_1L512W_C4_Code": ["gelu-1l", "gelu-1l-new", "gelu-1l-c4-code"],
    "NeelNanda/GELU_2L512W_C4_Code": ["gelu-2l", "gelu-2l-new", "gelu-2l-c4-code"],
    "NeelNanda/GELU_3L512W_C4_Code": ["gelu-3l", "gelu-3l-new", "gelu-3l-c4-code"],
    "NeelNanda/GELU_4L512W_C4_Code": ["gelu-4l", "gelu-4l-new", "gelu-4l-c4-code"],
    "NeelNanda/Attn_Only_1L512W_C4_Code": [
        "attn-only-1l",
        "attn-only-1l-new",
        "attn-only-1l-c4-code",
    ],
    "NeelNanda/Attn_Only_2L512W_C4_Code": [
        "attn-only-2l",
        "attn-only-2l-new",
        "attn-only-2l-c4-code",
    ],
    "NeelNanda/Attn_Only_3L512W_C4_Code": [
        "attn-only-3l",
        "attn-only-3l-new",
        "attn-only-3l-c4-code",
    ],
    "NeelNanda/Attn_Only_4L512W_C4_Code": [
        "attn-only-4l",
        "attn-only-4l-new",
        "attn-only-4l-c4-code",
    ],
    "NeelNanda/SoLU_6L768W_C4_Code": ["solu-6l", "solu-6l-new", "solu-6l-c4-code"],
    "NeelNanda/SoLU_8L1024W_C4_Code": ["solu-8l", "solu-8l-new", "solu-8l-c4-code"],
    "NeelNanda/SoLU_10L1280W_C4_Code": ["solu-10l", "solu-10l-new", "solu-10l-c4-code"],
    "NeelNanda/SoLU_12L1536W_C4_Code": ["solu-12l", "solu-12l-new", "solu-12l-c4-code"],
    "NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr": [
        "attn-only-2l-demo",
        "attn-only-2l-shortformer-6b-big-lr",
        "attn-only-2l-induction-demo",
        "attn-only-demo",
    ],
    "NeelNanda/SoLU_1L512W_Wiki_Finetune": [
        "solu-1l-wiki",
        "solu-1l-wiki-finetune",
        "solu-1l-finetune",
    ],
    "NeelNanda/SoLU_4L512W_Wiki_Finetune": [
        "solu-4l-wiki",
        "solu-4l-wiki-finetune",
        "solu-4l-finetune",
    ],
    "EleutherAI/pythia-14m": [
        "pythia-14m",
    ],
    "EleutherAI/pythia-31m": [
        "pythia-31m",
    ],
    "EleutherAI/pythia-70m": [
        "pythia-70m",
        "pythia",
        "EleutherAI/pythia-19m",
        "pythia-19m",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-160m": [
        "pythia-160m",
        "EleutherAI/pythia-125m",
        "pythia-125m",  # EleutherAI renamed this model"
    ],
    "EleutherAI/pythia-410m": [
        "pythia-410m",
        "EleutherAI/pythia-350m",
        "pythia-350m",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-1b": [
        "pythia-1b",
        "EleutherAI/pythia-800m",
        "pythia-800m",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-1.4b": [
        "pythia-1.4b",
        "EleutherAI/pythia-1.3b",
        "pythia-1.3b",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-2.8b": [
        "pythia-2.8b",
        "EleutherAI/pythia-2.7b",
        "pythia-2.7b",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-6.9b": [
        "pythia-6.9b",
        "EleutherAI/pythia-6.7b",
        "pythia-6.7b",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-12b": [
        "pythia-12b",
        "EleutherAI/pythia-13b",
        "pythia-13b",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-70m-deduped": [
        "pythia-70m-deduped",
        "EleutherAI/pythia-19m-deduped",  # EleutherAI renamed this model
        "pythia-19m-deduped",
    ],
    "EleutherAI/pythia-160m-deduped": [
        "pythia-160m-deduped",
        "EleutherAI/pythia-125m-deduped",  # EleutherAI renamed this model
        "pythia-125m-deduped",
    ],
    "EleutherAI/pythia-410m-deduped": [
        "pythia-410m-deduped",
        "EleutherAI/pythia-350m-deduped",  # EleutherAI renamed this model
        "pythia-350m-deduped",
    ],
    "EleutherAI/pythia-1b-deduped": [
        "pythia-1b-deduped",
        "EleutherAI/pythia-800m-deduped",  # EleutherAI renamed this model
        "pythia-800m-deduped",
    ],
    "EleutherAI/pythia-1.4b-deduped": [
        "pythia-1.4b-deduped",
        "EleutherAI/pythia-1.3b-deduped",  # EleutherAI renamed this model
        "pythia-1.3b-deduped",
    ],
    "EleutherAI/pythia-2.8b-deduped": [
        "pythia-2.8b-deduped",
        "EleutherAI/pythia-2.7b-deduped",  # EleutherAI renamed this model
        "pythia-2.7b-deduped",
    ],
    "EleutherAI/pythia-6.9b-deduped": [
        "pythia-6.9b-deduped",
        "EleutherAI/pythia-6.7b-deduped",  # EleutherAI renamed this model
        "pythia-6.7b-deduped",
    ],
    "EleutherAI/pythia-12b-deduped": [
        "pythia-12b-deduped",
        "EleutherAI/pythia-13b-deduped",  # EleutherAI renamed this model
        "pythia-13b-deduped",
    ],
    "EleutherAI/pythia-70m-v0": [
        "pythia-70m-v0",
        "pythia-v0",
        "EleutherAI/pythia-19m-v0",
        "pythia-19m-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-160m-v0": [
        "pythia-160m-v0",
        "EleutherAI/pythia-125m-v0",
        "pythia-125m-v0",  # EleutherAI renamed this model"
    ],
    "EleutherAI/pythia-410m-v0": [
        "pythia-410m-v0",
        "EleutherAI/pythia-350m-v0",
        "pythia-350m-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-1b-v0": [
        "pythia-1b-v0",
        "EleutherAI/pythia-800m-v0",
        "pythia-800m-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-1.4b-v0": [
        "pythia-1.4b-v0",
        "EleutherAI/pythia-1.3b-v0",
        "pythia-1.3b-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-2.8b-v0": [
        "pythia-2.8b-v0",
        "EleutherAI/pythia-2.7b-v0",
        "pythia-2.7b-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-6.9b-v0": [
        "pythia-6.9b-v0",
        "EleutherAI/pythia-6.7b-v0",
        "pythia-6.7b-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-12b-v0": [
        "pythia-12b-v0",
        "EleutherAI/pythia-13b-v0",
        "pythia-13b-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-70m-deduped-v0": [
        "pythia-70m-deduped-v0",
        "EleutherAI/pythia-19m-deduped-v0",  # EleutherAI renamed this model
        "pythia-19m-deduped-v0",
    ],
    "EleutherAI/pythia-160m-deduped-v0": [
        "pythia-160m-deduped-v0",
        "EleutherAI/pythia-125m-deduped-v0",  # EleutherAI renamed this model
        "pythia-125m-deduped-v0",
    ],
    "EleutherAI/pythia-410m-deduped-v0": [
        "pythia-410m-deduped-v0",
        "EleutherAI/pythia-350m-deduped-v0",  # EleutherAI renamed this model
        "pythia-350m-deduped-v0",
    ],
    "EleutherAI/pythia-1b-deduped-v0": [
        "pythia-1b-deduped-v0",
        "EleutherAI/pythia-800m-deduped-v0",  # EleutherAI renamed this model
        "pythia-800m-deduped-v0",
    ],
    "EleutherAI/pythia-1.4b-deduped-v0": [
        "pythia-1.4b-deduped-v0",
        "EleutherAI/pythia-1.3b-deduped-v0",  # EleutherAI renamed this model
        "pythia-1.3b-deduped-v0",
    ],
    "EleutherAI/pythia-2.8b-deduped-v0": [
        "pythia-2.8b-deduped-v0",
        "EleutherAI/pythia-2.7b-deduped-v0",  # EleutherAI renamed this model
        "pythia-2.7b-deduped-v0",
    ],
    "EleutherAI/pythia-6.9b-deduped-v0": [
        "pythia-6.9b-deduped-v0",
        "EleutherAI/pythia-6.7b-deduped-v0",  # EleutherAI renamed this model
        "pythia-6.7b-deduped-v0",
    ],
    "EleutherAI/pythia-12b-deduped-v0": [
        "pythia-12b-deduped-v0",
        "EleutherAI/pythia-13b-deduped-v0",  # EleutherAI renamed this model
        "pythia-13b-deduped-v0",
    ],
    "EleutherAI/pythia-160m-seed1": [
        "pythia-160m-seed1",
        "EleutherAI/pythia-125m-seed1",
        "pythia-125m-seed1",  # EleutherAI renamed this model"
    ],
    "EleutherAI/pythia-160m-seed2": [
        "pythia-160m-seed2",
        "EleutherAI/pythia-125m-seed2",
        "pythia-125m-seed2",  # EleutherAI renamed this model"
    ],
    "EleutherAI/pythia-160m-seed3": [
        "pythia-160m-seed3",
        "EleutherAI/pythia-125m-seed3",
        "pythia-125m-seed3",  # EleutherAI renamed this model"
    ],
    "gpt2": ["gpt2-small"],
    "distilgpt2": ["distillgpt2", "distill-gpt2", "distil-gpt2", "gpt2-xs"],
    "facebook/opt-125m": ["opt-125m", "opt-small", "opt"],
    "facebook/opt-1.3b": ["opt-1.3b", "opt-medium"],
    "facebook/opt-2.7b": ["opt-2.7b", "opt-large"],
    "facebook/opt-6.7b": ["opt-6.7b", "opt-xl"],
    "facebook/opt-13b": ["opt-13b", "opt-xxl"],
    "facebook/opt-30b": ["opt-30b", "opt-xxxl"],
    "facebook/opt-66b": ["opt-66b", "opt-xxxxl"],
    "EleutherAI/gpt-neo-125M": ["gpt-neo-125M", "gpt-neo-small", "neo-small", "neo"],
    "EleutherAI/gpt-neo-1.3B": ["gpt-neo-1.3B", "gpt-neo-medium", "neo-medium"],
    "EleutherAI/gpt-neo-2.7B": ["gpt-neo-2.7B", "gpt-neo-large", "neo-large"],
    "EleutherAI/gpt-j-6B": ["gpt-j-6B", "gpt-j", "gptj"],
    "EleutherAI/gpt-neox-20b": ["gpt-neox-20b", "gpt-neox", "neox"],
    "stanford-crfm/alias-gpt2-small-x21": [
        "stanford-gpt2-small-a",
        "alias-gpt2-small-x21",
        "gpt2-mistral-small-a",
        "gpt2-stanford-small-a",
    ],
    "stanford-crfm/battlestar-gpt2-small-x49": [
        "stanford-gpt2-small-b",
        "battlestar-gpt2-small-x49",
        "gpt2-mistral-small-b",
        "gpt2-mistral-small-b",
    ],
    "stanford-crfm/caprica-gpt2-small-x81": [
        "stanford-gpt2-small-c",
        "caprica-gpt2-small-x81",
        "gpt2-mistral-small-c",
        "gpt2-stanford-small-c",
    ],
    "stanford-crfm/darkmatter-gpt2-small-x343": [
        "stanford-gpt2-small-d",
        "darkmatter-gpt2-small-x343",
        "gpt2-mistral-small-d",
        "gpt2-mistral-small-d",
    ],
    "stanford-crfm/expanse-gpt2-small-x777": [
        "stanford-gpt2-small-e",
        "expanse-gpt2-small-x777",
        "gpt2-mistral-small-e",
        "gpt2-mistral-small-e",
    ],
    "stanford-crfm/arwen-gpt2-medium-x21": [
        "stanford-gpt2-medium-a",
        "arwen-gpt2-medium-x21",
        "gpt2-medium-small-a",
        "gpt2-stanford-medium-a",
    ],
    "stanford-crfm/beren-gpt2-medium-x49": [
        "stanford-gpt2-medium-b",
        "beren-gpt2-medium-x49",
        "gpt2-medium-small-b",
        "gpt2-stanford-medium-b",
    ],
    "stanford-crfm/celebrimbor-gpt2-medium-x81": [
        "stanford-gpt2-medium-c",
        "celebrimbor-gpt2-medium-x81",
        "gpt2-medium-small-c",
        "gpt2-medium-small-c",
    ],
    "stanford-crfm/durin-gpt2-medium-x343": [
        "stanford-gpt2-medium-d",
        "durin-gpt2-medium-x343",
        "gpt2-medium-small-d",
        "gpt2-stanford-medium-d",
    ],
    "stanford-crfm/eowyn-gpt2-medium-x777": [
        "stanford-gpt2-medium-e",
        "eowyn-gpt2-medium-x777",
        "gpt2-medium-small-e",
        "gpt2-stanford-medium-e",
    ],
    "ArthurConmy/redwood_attn_2l": ["redwood_attn_2l"],
    "llama-7b-hf": ["llama-7b"],
    "llama-13b-hf": ["llama-13b"],
    "llama-30b-hf": ["llama-30b"],
    "llama-65b-hf": ["llama-65b"],
    "meta-llama/Llama-2-7b-hf": ["Llama-2-7b", "meta-llama/Llama-2-7b-hf"],
    "meta-llama/Llama-2-7b-chat-hf": [
        "Llama-2-7b-chat",
        "meta-llama/Llama-2-7b-chat-hf",
    ],
    "meta-llama/Llama-2-13b-hf": ["Llama-2-13b", "meta-llama/Llama-2-13b-hf"],
    "meta-llama/Llama-2-13b-chat-hf": [
        "Llama-2-13b-chat",
        "meta-llama/Llama-2-13b-chat-hf",
    ],
    "meta-llama/Llama-2-70b-chat-hf": ["Llama-2-70b-chat", "meta-llama-2-70b-chat-hf"],
    "CodeLlama-7b-hf": ["CodeLlamallama-2-7b", "codellama/CodeLlama-7b-hf"],
    "CodeLlama-7b-Python-hf": [
        "CodeLlama-7b-python",
        "codellama/CodeLlama-7b-Python-hf",
    ],
    "CodeLlama-7b-Instruct-hf": [
        "CodeLlama-7b-instruct",
        "codellama/CodeLlama-7b-Instruct-hf",
    ],
    "Baidicoot/Othello-GPT-Transformer-Lens": ["othello-gpt"],
    "roneneldan/TinyStories-1M": ["tiny-stories-1M"],
    "roneneldan/TinyStories-3M": ["tiny-stories-3M"],
    "roneneldan/TinyStories-8M": ["tiny-stories-8M"],
    "roneneldan/TinyStories-28M": ["tiny-stories-28M"],
    "roneneldan/TinyStories-33M": ["tiny-stories-33M"],
    "roneneldan/TinyStories-Instruct-1M": ["tiny-stories-instruct-1M"],
    "roneneldan/TinyStories-Instruct-3M": ["tiny-stories-instruct-3M"],
    "roneneldan/TinyStories-Instruct-8M": ["tiny-stories-instruct-8M"],
    "roneneldan/TinyStories-Instruct-28M": ["tiny-stories-instruct-28M"],
    "roneneldan/TinyStories-Instruct-33M": ["tiny-stories-instruct-33M"],
    "roneneldan/TinyStories-1Layer-21M": ["tiny-stories-1L-21M"],
    "roneneldan/TinyStories-2Layers-33M": ["tiny-stories-2L-33M"],
    "roneneldan/TinyStories-Instuct-1Layer-21M": ["tiny-stories-instruct-1L-21M"],
    "roneneldan/TinyStories-Instruct-2Layers-33M": ["tiny-stories-instruct-2L-33M"],
    "stabilityai/stablelm-base-alpha-3b": [
        "stablelm-base-alpha-3b",
        "stablelm-base-3b",
    ],
    "stabilityai/stablelm-base-alpha-7b": [
        "stablelm-base-alpha-7b",
        "stablelm-base-7b",
    ],
    "stabilityai/stablelm-tuned-alpha-3b": [
        "stablelm-tuned-alpha-3b",
        "stablelm-tuned-3b",
    ],
    "stabilityai/stablelm-tuned-alpha-7b": [
        "stablelm-tuned-alpha-7b",
        "stablelm-tuned-7b",
    ],
    "mistralai/Mistral-7B-v0.1": ["mistral-7b"],
    "mistralai/Mistral-7B-Instruct-v0.1": ["mistral-7b-instruct"],
    "mistralai/Mixtral-8x7B-v0.1": ["mixtral", "mixtral-8x7b"],
    "mistralai/Mixtral-8x7B-Instruct-v0.1": [
        "mixtral-instruct",
        "mixtral-8x7b-instruct",
    ],
    "bigscience/bloom-560m": ["bloom-560m"],
    "bigscience/bloom-1b1": ["bloom-1b1"],
    "bigscience/bloom-1b7": ["bloom-1b7"],
    "bigscience/bloom-3b": ["bloom-3b"],
    "bigscience/bloom-7b1": ["bloom-7b1"],
    "bigcode/santacoder": ["santacoder"],
    "Qwen/Qwen-1_8B": ["qwen-1.8b"],
    "Qwen/Qwen-7B": ["qwen-7b"],
    "Qwen/Qwen-14B": ["qwen-14b"],
    "Qwen/Qwen-1_8B-Chat": ["qwen-1.8b-chat"],
    "Qwen/Qwen-7B-Chat": ["qwen-7b-chat"],
    "Qwen/Qwen-14B-Chat": ["qwen-14b-chat"],
    "Qwen/Qwen1.5-0.5B": ["qwen1.5-0.5b"],
    "Qwen/Qwen1.5-0.5B-Chat": ["qwen1.5-0.5b-chat"],
    "Qwen/Qwen1.5-1.8B": ["qwen1.5-1.8b"],
    "Qwen/Qwen1.5-1.8B-Chat": ["qwen1.5-1.8b-chat"],
    "Qwen/Qwen1.5-4B": ["qwen1.5-4b"],
    "Qwen/Qwen1.5-4B-Chat": ["qwen1.5-4b-chat"],
    "Qwen/Qwen1.5-7B": ["qwen1.5-7b"],
    "Qwen/Qwen1.5-7B-Chat": ["qwen1.5-7b-chat"],
    "Qwen/Qwen1.5-14B": ["qwen1.5-14b"],
    "Qwen/Qwen1.5-14B-Chat": ["qwen1.5-14b-chat"],
    "microsoft/phi-1": ["phi-1"],
    "microsoft/phi-1_5": ["phi-1_5"],
    "microsoft/phi-2": ["phi-2"],
    "microsoft/Phi-3-mini-4k-instruct": ["phi-3"],
    "google/gemma-2b": ["gemma-2b"],
    "google/gemma-7b": ["gemma-7b"],
    "google/gemma-2b-it": ["gemma-2b-it"],
    "google/gemma-7b-it": ["gemma-7b-it"],
    "google/gemma-2-2b": ["gemma-2-2b"],
    "google/gemma-2-9b": ["gemma-2-9b"],
    "google/gemma-2-27b": ["gemma-2-27b"],
    "google/gemma-2-2b-it": ["gemma-2-2b-it"],
    "google/gemma-2-9b-it": ["gemma-2-9b-it"],
    "google/gemma-2-27b-it": ["gemma-2-27b-it"],
    "01-ai/Yi-6B": ["yi-6b", "Yi-6B"],
    "01-ai/Yi-34B": ["yi-34b", "Yi-34B"],
    "01-ai/Yi-6B-Chat": ["yi-6b-chat", "Yi-6B-Chat"],
    "01-ai/Yi-34B-Chat": ["yi-34b-chat", "Yi-34B-Chat"],
    "google-t5/t5-small": ["t5-small"],
    "google-t5/t5-base": ["t5-base"],
    "google-t5/t5-large": ["t5-large"],
    "ai-forever/mGPT": ["mGPT"],
}
"""Model aliases for models on HuggingFace."""

NON_HF_HOSTED_MODEL_NAMES = [
    "llama-7b-hf",
    "llama-13b-hf",
    "llama-30b-hf",
    "llama-65b-hf",
]
"""Official model names for models not hosted on HuggingFace."""

# Sets a default model alias, by convention the first one in the model alias table, else the official name if it has no aliases
DEFAULT_MODEL_ALIASES = [
    MODEL_ALIASES[name][0] if name in MODEL_ALIASES else name for name in OFFICIAL_MODEL_NAMES
]

NEED_REMOTE_CODE_MODELS = (
    "bigcode/santacoder",
    "Qwen/Qwen-",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
)


def make_model_alias_map():
    """
    Converts OFFICIAL_MODEL_NAMES (the list of actual model names on
    HuggingFace) and MODEL_ALIASES (a dictionary mapping official model names to
    aliases) into a dictionary mapping all aliases to the official model name.
    """
    model_alias_map = {}
    for official_model_name in OFFICIAL_MODEL_NAMES:
        aliases = MODEL_ALIASES.get(official_model_name, [])
        for alias in aliases:
            model_alias_map[alias.lower()] = official_model_name
        model_alias_map[official_model_name.lower()] = official_model_name
    return model_alias_map


def get_official_model_name(model_name: str):
    """
    Returns the official model name for a given model name (or alias).
    """
    model_alias_map = make_model_alias_map()
    official_model_name = model_alias_map.get(model_name.lower(), None)
    if official_model_name is None:
        raise ValueError(
            f"{model_name} not found. Valid official model names (excl aliases): {OFFICIAL_MODEL_NAMES}"
        )
    return official_model_name


def convert_hf_model_config(model_name: str, **kwargs):
    """
    Returns the model config for a HuggingFace model, converted to a dictionary
    in the HookedTransformerConfig format.

    Takes the official_model_name as an input.
    """
    # In case the user passed in an alias
    if (Path(model_name) / "config.json").exists():
        logging.info("Loading model config from local directory")
        official_model_name = model_name
    else:
        official_model_name = get_official_model_name(model_name)

    # Load HuggingFace model config
    if "llama" in official_model_name.lower():
        architecture = "LlamaForCausalLM"
    elif "gemma-2" in official_model_name.lower():
        architecture = "Gemma2ForCausalLM"
    elif "gemma" in official_model_name.lower():
        architecture = "GemmaForCausalLM"
    else:
        huggingface_token = os.environ.get("HF_TOKEN", None)
        hf_config = AutoConfig.from_pretrained(
            official_model_name,
            token=huggingface_token,
            **kwargs,
        )
        architecture = hf_config.architectures[0]

    if official_model_name.startswith(
        ("llama-7b", "meta-llama/Llama-2-7b")
    ):  # same architecture for LLaMA and Llama-2
        cfg_dict = {
            "d_model": 4096,
            "d_head": 4096 // 32,
            "n_heads": 32,
            "d_mlp": 11008,
            "n_layers": 32,
            "n_ctx": 2048 if official_model_name.startswith("llama-7b") else 4096,
            "eps": 1e-6 if official_model_name.startswith("llama-7b") else 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 4096 // 32,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif official_model_name.startswith("CodeLlama-7b"):  # same architecture CodeLlama and Llama-2
        cfg_dict = {
            "d_model": 4096,
            "d_head": 4096 // 32,
            "n_heads": 32,
            "d_mlp": 11008,
            "n_layers": 32,
            "n_ctx": 4096,
            "eps": 1e-5,
            "d_vocab": 32016,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 4096 // 32,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 1000000,
        }
        if "python" in official_model_name.lower():
            # The vocab size of python version of CodeLlama-7b is 32000
            cfg_dict["d_vocab"] = 32000
    elif official_model_name.startswith(
        ("llama-13b", "meta-llama/Llama-2-13b")
    ):  # same architecture for LLaMA and Llama-2
        cfg_dict = {
            "d_model": 5120,
            "d_head": 5120 // 40,
            "n_heads": 40,
            "d_mlp": 13824,
            "n_layers": 40,
            "n_ctx": 2048 if official_model_name.startswith("llama-13b") else 4096,
            "eps": 1e-6 if official_model_name.startswith("llama-13b") else 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 5120 // 40,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "llama-30b" in official_model_name:
        cfg_dict = {
            "d_model": 6656,
            "d_head": 6656 // 52,
            "n_heads": 52,
            "d_mlp": 17920,
            "n_layers": 60,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 6656 // 52,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "llama-65b" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 8192 // 64,
            "n_heads": 64,
            "d_mlp": 22016,
            "n_layers": 80,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 8192 // 64,
            "rotary_adjacent_pairs": False,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "Llama-2-70b" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 4096,
            "eps": 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "Meta-Llama-3-8B" in official_model_name:
        cfg_dict = {
            "d_model": 4096,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 14336,
            "n_layers": 32,
            "n_ctx": 8192,
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "Meta-Llama-3-70B" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 8192,
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "GPTNeoForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_heads,
            "n_heads": hf_config.num_heads,
            "d_mlp": hf_config.hidden_size * 4,
            "n_layers": hf_config.num_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "attn_types": hf_config.attention_layers,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": False,
            "use_local_attn": True,
            "window_size": hf_config.window_size,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
        }
    elif architecture == "GPT2LMHeadModel":
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.n_embd * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_ctx,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
            "normalization_type": "LN",
        }
    elif architecture == "OPTForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.ffn_dim,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": 1e-5,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
        }
    elif architecture == "GPTJForCausalLM":
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": 4 * hf_config.n_embd,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_positions,
            "eps": 1e-5,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "rotary_dim": hf_config.rotary_dim,
            "rotary_adjacent_pairs": True,
            "normalization_type": "LN",
        }
    elif architecture == "GPTNeoXForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "normalization_type": "LN",
        }
        rotary_pct = hf_config.rotary_pct
        cfg_dict["rotary_dim"] = round(rotary_pct * cfg_dict["d_head"])
    elif architecture == "BertForMaskedLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": "gelu",
            "attention_dir": "bidirectional",
        }
    elif architecture == "MistralForCausalLM":
        cfg_dict = {
            "d_model": 4096,
            "d_head": 4096 // 32,
            "n_heads": 32,
            "d_mlp": 14336,
            "n_layers": 32,
            "n_ctx": 2048,  # Capped due to memory issues
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "window_size": 4096,
            "attn_types": ["local"] * 32,
            "eps": 1e-05,
            "n_key_value_heads": 8,
            "gated_mlp": True,
            "use_local_attn": True,
            "rotary_dim": 4096 // 32,
        }
    elif architecture == "MixtralForCausalLM":
        cfg_dict = {
            "dtype": torch.bfloat16,
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,  # Capped due to memory issues
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_base": hf_config.rope_theta,
            "window_size": hf_config.sliding_window,  # This is None, as no sliding window was used
            "attn_types": ["global"] * 32,
            "eps": hf_config.rms_norm_eps,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "gated_mlp": True,
            "use_local_attn": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "num_experts": hf_config.num_local_experts,
            "experts_per_token": hf_config.num_experts_per_tok,
        }
    elif architecture == "BloomForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.hidden_size * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": 2048,  # Capped due to HF Tokenizer Constraints
            "d_vocab": hf_config.vocab_size,
            "act_fn": "gelu_fast",
            "eps": hf_config.layer_norm_epsilon,
            "normalization_type": "LN",
            "post_embedding_ln": True,
            "positional_embedding_type": "alibi",
        }
    elif architecture == "GPT2LMHeadCustomModel":
        # santacoder
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.n_embd * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_positions,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "trust_remote_code": "santacoder"
            in official_model_name,  # Only santacoder needs trust_remote_code
            "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
            "normalization_type": "LN",
        }
    elif architecture == "LlamaForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "n_key_value_heads": (
                hf_config.num_key_value_heads
                if hf_config.num_key_value_heads != hf_config.num_attention_heads
                else None
            ),
            # This is done because the current implementation of GQA will use Grouped-Query Attention if
            # n_key_value_heads is not None, but hf_config.num_key_value_heads is sometimes specified as
            # the same as hf_config.num_attention_heads, in which case GQA should not be used.
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "QWenLMHeadModel":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size // 2,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,  # Capped bc the actual ctx length is 30k and the attn mask would be too big
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": "silu",
            "use_attn_scale": hf_config.scale_attn_weights,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": hf_config.kv_channels,
            "rotary_adjacent_pairs": False,
            "tokenizer_prepends_bos": True,
            "trust_remote_code": True,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "Qwen2ForCausalLM":
        # Note that Qwen1.5 models have architecture type Qwen2ForCausalLM.
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,  # Capped bc the actual ctx length is 30k and the attn mask would be too big
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_base": hf_config.rope_theta,
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "tokenizer_prepends_bos": True,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "PhiForCausalLM":
        # Architecture for microsoft/phi models
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "LN",
            "positional_embedding_type": "rotary",
            "trust_remote_code": True,
            "rotary_base": hf_config.rope_theta,
            "use_attn_scale": True,
            "parallel_attn_mlp": True,
        }
        partial_rotary_factor = hf_config.partial_rotary_factor
        cfg_dict["rotary_dim"] = round(partial_rotary_factor * cfg_dict["d_head"])
    elif architecture == "Phi3ForCausalLM":
        # Architecture for microsoft/phi3 models
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "trust_remote_code": True,
            "rotary_base": hf_config.rope_theta,
            "use_attn_scale": True,
            "gated_mlp": True,
            "parallel_attn_mlp": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
        }

    elif official_model_name.startswith("google/gemma-2b"):
        # Architecture for Gemma 2b and Gemma 2b Instruct models
        cfg_dict = {
            "d_model": 2048,
            "d_head": 256,
            "n_heads": 8,
            "d_mlp": 16384,
            "n_layers": 18,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_new",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "rotary_dim": 256,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 1,
            "gated_mlp": True,
            "final_rms": True,
        }
    elif official_model_name.startswith("google/gemma-7b"):
        # Architecture for Gemma 7b and Gemma 7b Instruct models
        cfg_dict = {
            "d_model": 3072,
            "d_head": 256,
            "n_heads": 16,
            "d_mlp": 24576,
            "n_layers": 28,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_new",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "rotary_dim": 256,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 16,
            "gated_mlp": True,
            "final_rms": True,
        }
    elif official_model_name.startswith("google/gemma-2-2b"):
        # Architecture for Gemma-2 2b and Gemma-2 2b Instruct models
        cfg_dict = {
            "d_model": 2304,
            "d_head": 256,
            "n_heads": 8,
            "d_mlp": 9216,
            "n_layers": 26,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 4,
            "window_size": 4096,
            "use_local_attn": True,
            "attn_types": ["global", "local"] * 21,  # Alternate global and local attn
            "attn_scores_soft_cap": 50.0,
            "output_logits_soft_cap": 30.0,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
        }
    elif official_model_name.startswith("google/gemma-2-9b"):
        # Architecture for Gemma-2 9b and Gemma-2 9b Instruct models
        cfg_dict = {
            "d_model": 3584,
            "d_head": 256,
            "n_heads": 16,
            "d_mlp": 14336,
            "n_layers": 42,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 8,
            "window_size": 4096,
            "use_local_attn": True,
            "attn_types": ["global", "local"] * 21,  # Alternate global and local attn
            "attn_scores_soft_cap": 50.0,
            "output_logits_soft_cap": 30.0,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
        }
    elif official_model_name.startswith("google/gemma-2-27b"):
        # Architecture for Gemma-2 27b and Gemma-2 27b Instruct models
        cfg_dict = {
            "d_model": 4608,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 36864,
            "n_layers": 46,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "attn_scale": 12.0,
            "n_key_value_heads": 16,
            "window_size": 4096,
            "use_local_attn": True,
            "attn_types": ["global", "local"] * 23,  # Alternate global and local attn
            "attn_scores_soft_cap": 50.0,
            "output_logits_soft_cap": 30.0,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
        }
    elif architecture == "T5ForConditionalGeneration":
        cfg_dict = {
            "d_model": hf_config.d_model,
            "d_head": hf_config.d_kv,
            "n_heads": hf_config.num_heads,
            "d_mlp": hf_config.d_ff,
            "d_vocab": hf_config.vocab_size,
            "n_layers": hf_config.num_layers,
            "n_ctx": hf_config.max_length,
            "eps": hf_config.layer_norm_epsilon,
            "act_fn": hf_config.feed_forward_proj,
            "positional_embedding_type": "relative_positional_bias",
            "relative_attention_max_distance": hf_config.relative_attention_max_distance,
            "relative_attention_num_buckets": hf_config.relative_attention_num_buckets,
            "decoder_start_token_id": hf_config.decoder_start_token_id,
            "attention_dir": "bidirectional",
            "use_attn_scale": False,
            "tie_word_embeddings": hf_config.tie_word_embeddings,
        }
    elif architecture == "Olmo2ForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "n_key_value_heads": (
                hf_config.num_key_value_heads
                if hf_config.num_key_value_heads != hf_config.num_attention_heads
                else None
            ),
            # This is done because the current implementation of GQA will use Grouped-Query Attention if
            # n_key_value_heads is not None, but hf_config.num_key_value_heads is sometimes specified as
            # the same as hf_config.num_attention_heads, in which case GQA should not be used.
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "final_rms": True,
            "gated_mlp": True,
            "use_normalization_before_and_after": True,
        }
    elif architecture == "OlmoForCausalLM" or architecture == "OLMoForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": 1e-05,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "n_key_value_heads": (
                hf_config.num_key_value_heads
                if hf_config.num_key_value_heads != hf_config.num_attention_heads
                else None
            ),
            # This is done because the current implementation of GQA will use Grouped-Query Attention if
            # n_key_value_heads is not None, but hf_config.num_key_value_heads is sometimes specified as
            # the same as hf_config.num_attention_heads, in which case GQA should not be used.
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "final_rms": True,
            "gated_mlp": True,
        }
    else:
        raise NotImplementedError(f"{architecture} is not currently supported.")
    # All of these models use LayerNorm
    cfg_dict["original_architecture"] = architecture
    # The name such that AutoTokenizer.from_pretrained works
    cfg_dict["tokenizer_name"] = official_model_name
    if kwargs.get("trust_remote_code", False):
        cfg_dict["trust_remote_code"] = True
    return cfg_dict


def convert_neel_model_config(official_model_name: str, **kwargs):
    """
    Loads the config for a model trained by me (NeelNanda), converted to a dictionary
    in the HookedTransformerConfig format.

    AutoConfig is not supported, because these models are in the HookedTransformer format, so we directly download and load the json.
    """
    official_model_name = get_official_model_name(official_model_name)
    cfg_json: dict = utils.download_file_from_hf(official_model_name, "config.json", **kwargs)
    cfg_arch = cfg_json.get(
        "architecture", "neel" if "_old" not in official_model_name else "neel-solu-old"
    )
    cfg_dict = {
        "d_model": cfg_json["d_model"],
        "n_layers": cfg_json["n_layers"],
        "d_mlp": cfg_json["d_mlp"],
        "d_head": cfg_json["d_head"],
        "n_heads": cfg_json["n_heads"],
        "n_ctx": cfg_json["n_ctx"],
        "d_vocab": cfg_json["d_vocab"],
        "tokenizer_name": cfg_json.get("tokenizer_name", None),
        "act_fn": cfg_json["act_fn"],
        "attn_only": cfg_json["attn_only"],
        "final_rms": cfg_json.get("final_rms", False),
        "original_architecture": cfg_arch,
    }
    if "normalization" in cfg_json:
        cfg_dict["normalization_type"] = cfg_json["normalization"]
    else:
        cfg_dict["normalization_type"] = cfg_json["normalization_type"]
    if "shortformer_pos" in cfg_json:
        cfg_dict["positional_embedding_type"] = (
            "shortformer" if cfg_json["shortformer_pos"] else "standard"
        )
    else:
        cfg_dict["positional_embedding_type"] = "standard"
    return cfg_dict


def get_pretrained_model_config(
    model_name: str,
    hf_cfg: Optional[dict] = None,
    checkpoint_index: Optional[int] = None,
    checkpoint_value: Optional[int] = None,
    fold_ln: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    n_devices: int = 1,
    default_prepend_bos: bool = True,
    dtype: torch.dtype = torch.float32,
    **kwargs,
):
    """Returns the pretrained model config as an HookedTransformerConfig object.

    There are two types of pretrained models: HuggingFace models (where
    AutoModel and AutoConfig work), and models trained by me (NeelNanda) which
    aren't as integrated with HuggingFace infrastructure.

    Args:
        model_name: The name of the model. This can be either the official
            HuggingFace model name, or the name of a model trained by me
            (NeelNanda).
        hf_cfg (dict, optional): Config of a loaded pretrained HF model,
            converted to a dictionary.
        checkpoint_index (int, optional): If loading from a
            checkpoint, the index of the checkpoint to load. Defaults to None.
        checkpoint_value (int, optional): If loading from a checkpoint, the
        value of
            the checkpoint to load, ie the step or token number (each model has
            checkpoints labelled with exactly one of these). Defaults to None.
        fold_ln (bool, optional): Whether to fold the layer norm into the
            subsequent linear layers (see HookedTransformer.fold_layer_norm for
            details). Defaults to False.
        device (str, optional): The device to load the model onto. By
            default will load to CUDA if available, else CPU.
        n_devices (int, optional): The number of devices to split the model across. Defaults to 1.
        default_prepend_bos (bool, optional): Default behavior of whether to prepend the BOS token when the
            methods of HookedTransformer process input text to tokenize (only when input is a string).
            Defaults to True - even for models not explicitly trained with this, heads often use the
            first position as a resting position and accordingly lose information from the first token,
            so this empirically seems to give better results. To change the default behavior to False, pass in
            default_prepend_bos=False. Note that you can also locally override the default behavior by passing
            in prepend_bos=True/False when you call a method that processes the input string.
        dtype (torch.dtype, optional): The dtype to load the TransformerLens model in.
        kwargs: Other optional arguments passed to HuggingFace's from_pretrained.
            Also given to other HuggingFace functions when compatible.

    """
    if Path(model_name).exists():
        # If the model_name is a path, it's a local model
        cfg_dict = convert_hf_model_config(model_name, **kwargs)
        official_model_name = model_name
    else:
        official_model_name = get_official_model_name(model_name)
    if (
        official_model_name.startswith("NeelNanda")
        or official_model_name.startswith("ArthurConmy")
        or official_model_name.startswith("Baidicoot")
    ):
        cfg_dict = convert_neel_model_config(official_model_name, **kwargs)
    else:
        if official_model_name.startswith(NEED_REMOTE_CODE_MODELS) and not kwargs.get(
            "trust_remote_code", False
        ):
            logging.warning(
                f"Loading model {official_model_name} requires setting trust_remote_code=True"
            )
            kwargs["trust_remote_code"] = True
        cfg_dict = convert_hf_model_config(official_model_name, **kwargs)
    # Processing common to both model types
    # Remove any prefix, saying the organization who made a model.
    cfg_dict["model_name"] = official_model_name.split("/")[-1]
    # Don't need to initialize weights, we're loading from pretrained
    cfg_dict["init_weights"] = False

    if (
        "positional_embedding_type" in cfg_dict
        and cfg_dict["positional_embedding_type"] == "shortformer"
        and fold_ln
    ):
        logging.warning(
            "You tried to specify fold_ln=True for a shortformer model, but this can't be done! Setting fold_ln=False instead."
        )
        fold_ln = False

    if device is not None:
        cfg_dict["device"] = device

    cfg_dict["dtype"] = dtype

    if fold_ln:
        if cfg_dict["normalization_type"] in ["LN", "LNPre"]:
            cfg_dict["normalization_type"] = "LNPre"
        elif cfg_dict["normalization_type"] in ["RMS", "RMSPre"]:
            cfg_dict["normalization_type"] = "RMSPre"
        else:
            logging.warning("Cannot fold in layer norm, normalization_type is not LN.")

    if checkpoint_index is not None or checkpoint_value is not None:
        checkpoint_labels, checkpoint_label_type = get_checkpoint_labels(
            official_model_name,
            **kwargs,
        )
        cfg_dict["from_checkpoint"] = True
        cfg_dict["checkpoint_label_type"] = checkpoint_label_type
        if checkpoint_index is not None:
            cfg_dict["checkpoint_index"] = checkpoint_index
            cfg_dict["checkpoint_value"] = checkpoint_labels[checkpoint_index]
        elif checkpoint_value is not None:
            assert (
                checkpoint_value in checkpoint_labels
            ), f"Checkpoint value {checkpoint_value} is not in list of available checkpoints"
            cfg_dict["checkpoint_value"] = checkpoint_value
            cfg_dict["checkpoint_index"] = checkpoint_labels.index(checkpoint_value)
    else:
        cfg_dict["from_checkpoint"] = False

    cfg_dict["device"] = device
    cfg_dict["n_devices"] = n_devices
    cfg_dict["default_prepend_bos"] = default_prepend_bos
    if hf_cfg is not None:
        cfg_dict["load_in_4bit"] = hf_cfg.get("quantization_config", {}).get("load_in_4bit", False)

    cfg = HookedTransformerConfig.from_dict(cfg_dict)
    return cfg


def get_num_params_of_pretrained(model_name):
    """
    Returns the number of parameters of a pretrained model, used to filter to only run code for sufficiently small models.
    """
    cfg = get_pretrained_model_config(model_name)
    return cfg.n_params


# %% Load checkpointed model state dicts
# The steps for which there are checkpoints in the stanford crfm models
STANFORD_CRFM_CHECKPOINTS = (
    list(range(0, 100, 10))
    + list(range(100, 2000, 50))
    + list(range(2000, 20000, 100))
    + list(range(20000, 400000 + 1, 1000))
)

# Linearly spaced checkpoints for Pythia models, taken every 1000 steps.
# Batch size 2,097,152 tokens, so checkpoints every 2.1B tokens
PYTHIA_CHECKPOINTS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(
    range(1000, 143000 + 1, 1000)
)
# Pythia V1 has log-spaced early checkpoints (see line above), but V0 doesn't
PYTHIA_V0_CHECKPOINTS = list(range(1000, 143000 + 1, 1000))

# The steps for which there are checkpoints in the LLM360/Amber model
AMBER_CHECKPOINTS = [f"{i:03}" for i in range(0, 359)]

LLM_JP_CHECKPOINTS = [f"{i:07}" for i in range(0, 10000000)]


OLMO_CHECKPOINTS = ['main', 'step557000-tokens2464B', 'step556000-tokens2460B', 'step555000-tokens2455B', 'step554000-tokens2451B', 'step553000-tokens2446B', 'step552000-tokens2442B', 'step551000-tokens2437B', 'step550000-tokens2433B', 'step549000-tokens2429B', 'step548000-tokens2424B', 'step547000-tokens2420B', 'step546000-tokens2415B', 'step545000-tokens2411B', 'step544000-tokens2406B', 'step543000-tokens2402B', 'step542000-tokens2398B', 'step541000-tokens2393B', 'step540000-tokens2389B', 'step539000-tokens2384B', 'step538000-tokens2380B', 'step537000-tokens2376B', 'step536000-tokens2371B', 'step535000-tokens2367B', 'step534000-tokens2362B', 'step533000-tokens2358B', 'step532000-tokens2353B', 'step531000-tokens2349B', 'step530000-tokens2345B', 'step529000-tokens2340B', 'step528000-tokens2336B', 'step527000-tokens2331B', 'step526000-tokens2327B', 'step525000-tokens2322B', 'step524000-tokens2318B', 'step523000-tokens2314B', 'step522000-tokens2309B', 'step521000-tokens2305B', 'step520000-tokens2300B', 'step519000-tokens2296B', 'step518000-tokens2291B', 'step517000-tokens2287B', 'step516000-tokens2283B', 'step515000-tokens2278B', 'step514000-tokens2274B', 'step513000-tokens2269B', 'step512000-tokens2265B', 'step511000-tokens2261B', 'step510000-tokens2256B', 'step509000-tokens2252B', 'step508000-tokens2247B', 'step507000-tokens2243B', 'step506000-tokens2238B', 'step505000-tokens2234B', 'step504000-tokens2230B', 'step503000-tokens2225B', 'step502000-tokens2221B', 'step501000-tokens2216B', 'step500000-tokens2212B', 'step499000-tokens2207B', 'step498000-tokens2203B', 'step497000-tokens2199B', 'step496000-tokens2194B', 'step495000-tokens2190B', 'step494000-tokens2185B', 'step493000-tokens2181B', 'step492000-tokens2176B', 'step491000-tokens2172B', 'step490000-tokens2168B', 'step489000-tokens2163B', 'step488000-tokens2159B', 'step487000-tokens2154B', 'step486000-tokens2150B', 'step485000-tokens2145B', 'step484000-tokens2141B', 'step483000-tokens2137B', 'step482000-tokens2132B', 'step481000-tokens2128B', 'step480000-tokens2123B', 'step479000-tokens2119B', 'step478000-tokens2115B', 'step477000-tokens2110B', 'step476000-tokens2106B', 'step475000-tokens2101B', 'step474000-tokens2097B', 'step473000-tokens2092B', 'step472000-tokens2088B', 'step471000-tokens2084B', 'step470000-tokens2079B', 'step469000-tokens2075B', 'step468000-tokens2070B', 'step467000-tokens2066B', 'step466000-tokens2061B', 'step465000-tokens2057B', 'step464000-tokens2053B', 'step463000-tokens2048B', 'step462000-tokens2044B', 'step461000-tokens2039B', 'step460000-tokens2035B', 'step459000-tokens2030B', 'step458000-tokens2026B', 'step457000-tokens2022B', 'step456000-tokens2017B', 'step455000-tokens2013B', 'step454000-tokens2008B', 'step453000-tokens2004B', 'step452000-tokens2000B', 'step451000-tokens1995B', 'step450000-tokens1991B', 'step449000-tokens1986B', 'step448000-tokens1982B', 'step447000-tokens1977B', 'step446000-tokens1973B', 'step445000-tokens1969B', 'step444000-tokens1964B', 'step443000-tokens1960B', 'step442000-tokens1955B', 'step441000-tokens1951B', 'step440000-tokens1946B', 'step439000-tokens1942B', 'step438000-tokens1938B', 'step437000-tokens1933B', 'step436000-tokens1929B', 'step435000-tokens1924B', 'step434000-tokens1920B', 'step433000-tokens1915B', 'step432000-tokens1911B', 'step431000-tokens1907B', 'step430000-tokens1902B', 'step429000-tokens1898B', 'step428000-tokens1893B', 'step427000-tokens1889B', 'step426000-tokens1884B', 'step425000-tokens1880B', 'step424000-tokens1876B', 'step423000-tokens1871B', 'step422000-tokens1867B', 'step421000-tokens1862B', 'step420000-tokens1858B', 'step419000-tokens1854B', 'step418000-tokens1849B', 'step417000-tokens1845B', 'step416000-tokens1840B', 'step415000-tokens1836B', 'step414000-tokens1831B', 'step413000-tokens1827B', 'step412000-tokens1823B', 'step411000-tokens1818B', 'step410000-tokens1814B', 'step409000-tokens1809B', 'step408000-tokens1805B', 'step407000-tokens1800B', 'step406000-tokens1796B', 'step405000-tokens1792B', 'step404000-tokens1787B', 'step403000-tokens1783B', 'step402000-tokens1778B', 'step401000-tokens1774B', 'step400000-tokens1769B', 'step399000-tokens1765B', 'step398000-tokens1761B', 'step397000-tokens1756B', 'step396000-tokens1752B', 'step395000-tokens1747B', 'step394000-tokens1743B', 'step393000-tokens1739B', 'step392000-tokens1734B', 'step391000-tokens1730B', 'step390000-tokens1725B', 'step389000-tokens1721B', 'step388000-tokens1716B', 'step387000-tokens1712B', 'step386000-tokens1708B', 'step385000-tokens1703B',
                    'step384000-tokens1699B', 'step383000-tokens1694B', 'step382000-tokens1690B', 'step381000-tokens1685B', 'step380000-tokens1681B', 'step379000-tokens1677B', 'step378000-tokens1672B', 'step377000-tokens1668B', 'step376000-tokens1663B', 'step375000-tokens1659B', 'step374000-tokens1654B', 'step373000-tokens1650B', 'step372000-tokens1646B', 'step371000-tokens1641B', 'step370000-tokens1637B', 'step369000-tokens1632B', 'step368000-tokens1628B', 'step367000-tokens1623B', 'step366000-tokens1619B', 'step365000-tokens1615B', 'step364000-tokens1610B', 'step363000-tokens1606B', 'step362000-tokens1601B', 'step361000-tokens1597B', 'step360000-tokens1593B', 'step359000-tokens1588B', 'step358000-tokens1584B', 'step357000-tokens1579B', 'step356000-tokens1575B', 'step355000-tokens1570B', 'step354000-tokens1566B', 'step353000-tokens1562B', 'step352000-tokens1557B', 'step351000-tokens1553B', 'step350000-tokens1548B', 'step349000-tokens1544B', 'step348000-tokens1539B', 'step347000-tokens1535B', 'step346000-tokens1531B', 'step345000-tokens1526B', 'step344000-tokens1522B', 'step343000-tokens1517B', 'step342000-tokens1513B', 'step341000-tokens1508B', 'step340000-tokens1504B', 'step339000-tokens1500B', 'step338000-tokens1495B', 'step337000-tokens1491B', 'step336000-tokens1486B', 'step335000-tokens1482B', 'step334000-tokens1478B', 'step333000-tokens1473B', 'step332000-tokens1469B', 'step331000-tokens1464B', 'step330000-tokens1460B', 'step329000-tokens1455B', 'step328000-tokens1451B', 'step327000-tokens1447B', 'step326000-tokens1442B', 'step325000-tokens1438B', 'step324000-tokens1433B', 'step323000-tokens1429B', 'step322000-tokens1424B', 'step321000-tokens1420B', 'step320000-tokens1416B', 'step319000-tokens1411B', 'step318000-tokens1407B', 'step317000-tokens1402B', 'step316000-tokens1398B', 'step315000-tokens1393B', 'step314000-tokens1389B', 'step313000-tokens1385B', 'step312000-tokens1380B', 'step311000-tokens1376B', 'step310000-tokens1371B', 'step309000-tokens1367B', 'step308000-tokens1362B', 'step307000-tokens1358B', 'step306000-tokens1354B', 'step305000-tokens1349B', 'step304000-tokens1345B', 'step303000-tokens1340B', 'step302000-tokens1336B', 'step301000-tokens1332B', 'step300000-tokens1327B', 'step299000-tokens1323B', 'step298000-tokens1318B', 'step297000-tokens1314B', 'step296000-tokens1309B', 'step295000-tokens1305B', 'step294000-tokens1301B', 'step293000-tokens1296B', 'step292000-tokens1292B', 'step291000-tokens1287B', 'step290000-tokens1283B', 'step289000-tokens1278B', 'step288000-tokens1274B', 'step287000-tokens1270B', 'step286000-tokens1265B', 'step285000-tokens1261B', 'step284000-tokens1256B', 'step283000-tokens1252B', 'step282000-tokens1247B', 'step281000-tokens1243B', 'step280000-tokens1239B', 'step279000-tokens1234B', 'step278000-tokens1230B', 'step277000-tokens1225B', 'step276000-tokens1221B', 'step275000-tokens1217B', 'step274000-tokens1212B', 'step273000-tokens1208B', 'step272000-tokens1203B', 'step271000-tokens1199B', 'step270000-tokens1194B', 'step269000-tokens1190B', 'step268000-tokens1186B', 'step267000-tokens1181B', 'step266000-tokens1177B', 'step265000-tokens1172B', 'step264000-tokens1168B', 'step263000-tokens1163B', 'step262000-tokens1159B', 'step261000-tokens1155B', 'step260000-tokens1150B', 'step259000-tokens1146B', 'step258000-tokens1141B', 'step257000-tokens1137B', 'step256000-tokens1132B', 'step255000-tokens1128B', 'step254000-tokens1124B', 'step253000-tokens1119B', 'step252000-tokens1115B', 'step251000-tokens1110B', 'step250000-tokens1106B', 'step249000-tokens1101B', 'step248000-tokens1097B', 'step247000-tokens1093B', 'step246000-tokens1088B', 'step245000-tokens1084B', 'step244000-tokens1079B', 'step243000-tokens1075B', 'step242000-tokens1071B', 'step241000-tokens1066B', 'step240000-tokens1062B', 'step239000-tokens1057B', 'step238000-tokens1053B', 'step237000-tokens1048B', 'step236000-tokens1044B', 'step235000-tokens1040B', 'step234000-tokens1035B', 'step233000-tokens1031B', 'step232000-tokens1026B', 'step231000-tokens1022B', 'step230000-tokens1017B', 'step229000-tokens1013B', 'step228000-tokens1009B', 'step227000-tokens1004B', 'step226000-tokens1000B', 'step225000-tokens995B', 'step224000-tokens991B', 'step223000-tokens986B', 'step222000-tokens982B', 'step221000-tokens978B', 'step220000-tokens973B', 'step219000-tokens969B', 'step218000-tokens964B', 'step217000-tokens960B', 'step216000-tokens956B', 'step215000-tokens951B', 'step214000-tokens947B', 'step213000-tokens942B', 'step212000-tokens938B', 'step211000-tokens933B',
                    'step210000-tokens929B', 'step209000-tokens925B', 'step208000-tokens920B', 'step207000-tokens916B', 'step206000-tokens911B', 'step205000-tokens907B', 'step204000-tokens902B', 'step203000-tokens898B', 'step202000-tokens894B', 'step201000-tokens889B', 'step200000-tokens885B', 'step199000-tokens880B', 'step198000-tokens876B', 'step197000-tokens871B', 'step196000-tokens867B', 'step195000-tokens863B', 'step194000-tokens858B', 'step193000-tokens854B', 'step192000-tokens849B', 'step191000-tokens845B', 'step190000-tokens840B', 'step189000-tokens836B', 'step188000-tokens832B', 'step187000-tokens827B', 'step186000-tokens823B', 'step185000-tokens818B', 'step184000-tokens814B', 'step183000-tokens810B', 'step182000-tokens805B', 'step181000-tokens801B', 'step180000-tokens796B', 'step179000-tokens792B', 'step178000-tokens787B', 'step177000-tokens783B', 'step176000-tokens779B', 'step175000-tokens774B', 'step174000-tokens770B', 'step173000-tokens765B', 'step172000-tokens761B', 'step171000-tokens756B', 'step170000-tokens752B', 'step169000-tokens748B', 'step168000-tokens743B', 'step167000-tokens739B', 'step166000-tokens734B', 'step165000-tokens730B', 'step164000-tokens725B', 'step163000-tokens721B', 'step162000-tokens717B', 'step161000-tokens712B', 'step160000-tokens708B', 'step159000-tokens703B', 'step158000-tokens699B', 'step157000-tokens695B', 'step156000-tokens690B', 'step155000-tokens686B', 'step154000-tokens681B', 'step153000-tokens677B', 'step152000-tokens672B', 'step151000-tokens668B', 'step150000-tokens664B', 'step149000-tokens659B', 'step148000-tokens655B', 'step147000-tokens650B', 'step146000-tokens646B', 'step145000-tokens641B', 'step144000-tokens637B', 'step143000-tokens633B', 'step142000-tokens628B', 'step141000-tokens624B', 'step140000-tokens619B', 'step139000-tokens615B', 'step138000-tokens610B', 'step137000-tokens606B', 'step136000-tokens602B', 'step135000-tokens597B', 'step134000-tokens593B', 'step133000-tokens588B', 'step132000-tokens584B', 'step131000-tokens580B', 'step130000-tokens575B', 'step129000-tokens571B', 'step128000-tokens566B', 'step127000-tokens562B', 'step126000-tokens557B', 'step125000-tokens553B', 'step124000-tokens549B', 'step123000-tokens544B', 'step122000-tokens540B', 'step121000-tokens535B', 'step120000-tokens531B', 'step119000-tokens526B', 'step118000-tokens522B', 'step117000-tokens518B', 'step116000-tokens513B', 'step115000-tokens509B', 'step114000-tokens504B', 'step113000-tokens500B', 'step112000-tokens495B', 'step111000-tokens491B', 'step110000-tokens487B', 'step109000-tokens482B', 'step108000-tokens478B', 'step107000-tokens473B', 'step106000-tokens469B', 'step105000-tokens464B', 'step104000-tokens460B', 'step103000-tokens456B', 'step102000-tokens451B', 'step101000-tokens447B', 'step100000-tokens442B', 'step99000-tokens438B', 'step98000-tokens434B', 'step97000-tokens429B', 'step96000-tokens425B', 'step95000-tokens420B', 'step94000-tokens416B', 'step93000-tokens411B', 'step92000-tokens407B', 'step91000-tokens403B', 'step90000-tokens398B', 'step89000-tokens394B', 'step88000-tokens389B', 'step87000-tokens385B', 'step86000-tokens380B', 'step85000-tokens376B', 'step84000-tokens372B', 'step83000-tokens367B', 'step82000-tokens363B', 'step81000-tokens358B', 'step80000-tokens354B', 'step79000-tokens349B', 'step78000-tokens345B', 'step77000-tokens341B', 'step76000-tokens336B', 'step75000-tokens332B', 'step74000-tokens327B', 'step73000-tokens323B', 'step72000-tokens319B', 'step71000-tokens314B', 'step70000-tokens310B', 'step69000-tokens305B', 'step68000-tokens301B', 'step67000-tokens296B', 'step66000-tokens292B', 'step65000-tokens288B', 'step64000-tokens283B', 'step63000-tokens279B', 'step62000-tokens274B', 'step61000-tokens270B', 'step60000-tokens265B', 'step59000-tokens261B', 'step58000-tokens257B', 'step57000-tokens252B', 'step56000-tokens248B', 'step55000-tokens243B', 'step54000-tokens239B', 'step53000-tokens234B', 'step52000-tokens230B', 'step51000-tokens226B', 'step50000-tokens221B', 'step49000-tokens217B', 'step48000-tokens212B', 'step47000-tokens208B', 'step46000-tokens203B', 'step45000-tokens199B', 'step44000-tokens195B', 'step43000-tokens190B', 'step42000-tokens186B', 'step41000-tokens181B', 'step40000-tokens177B', 'step39000-tokens173B', 'step38000-tokens168B', 'step37000-tokens164B', 'step36000-tokens159B', 'step35000-tokens155B', 'step34000-tokens150B', 'step33000-tokens146B', 'step32000-tokens142B', 'step31000-tokens137B', 'step30000-tokens133B', 'step29000-tokens128B', 'step28000-tokens124B',
                    'step27000-tokens119B', 'step26000-tokens115B', 'step25000-tokens111B', 'step24000-tokens106B', 'step23000-tokens102B', 'step22000-tokens97B', 'step21000-tokens93B', 'step20000-tokens88B', 'step19000-tokens84B', 'step18000-tokens80B', 'step17000-tokens75B', 'step16000-tokens71B', 'step15000-tokens66B', 'step14000-tokens62B', 'step13000-tokens58B', 'step12000-tokens53B', 'step11000-tokens49B', 'step10000-tokens44B', 'step9000-tokens40B', 'step7000-tokens31B', 'step8000-tokens35B', 'step6000-tokens27B', 'step5000-tokens22B', 'step4000-tokens18B', 'step3000-tokens13B', 'step2000-tokens9B', 'step1000-tokens4B', 'step0-tokens0B']

OLMO_0424_CHECKPOINTS = ['main', 'nitro', 'step651581-tokens2731B', 'step650650-tokens2728B', 'step649650-tokens2723B', 'step648650-tokens2719B', 'step647650-tokens2715B', 'step646650-tokens2711B', 'step645650-tokens2707B', 'step644650-tokens2702B', 'step643650-tokens2698B', 'step642650-tokens2694B', 'step641650-tokens2690B', 'step640650-tokens2686B', 'step639650-tokens2681B', 'step639000-tokens2679B', 'step638000-tokens2675B', 'step637000-tokens2670B', 'step636000-tokens2666B', 'step635000-tokens2662B', 'step634000-tokens2658B', 'step633000-tokens2654B', 'step632000-tokens2649B', 'step631000-tokens2645B', 'step630000-tokens2641B', 'step629000-tokens2637B', 'step628000-tokens2633B', 'step627000-tokens2628B', 'step626000-tokens2624B', 'step625000-tokens2620B', 'step624000-tokens2616B', 'step623000-tokens2612B', 'step622000-tokens2607B', 'step621100-tokens2604B', 'step621000-tokens2603B', 'step620000-tokens2599B', 'step619000-tokens2595B', 'step618000-tokens2591B', 'step617000-tokens2587B', 'step616000-tokens2582B', 'step615000-tokens2578B', 'step614000-tokens2574B', 'step613000-tokens2570B', 'step612000-tokens2566B', 'step611000-tokens2561B', 'step610000-tokens2557B', 'step609000-tokens2553B', 'step608000-tokens2549B', 'step607000-tokens2545B', 'step606000-tokens2540B', 'step605000-tokens2536B', 'step604000-tokens2532B', 'step603000-tokens2528B', 'step602000-tokens2524B', 'step601000-tokens2519B', 'step600000-tokens2515B', 'step599000-tokens2511B', 'step598000-tokens2507B', 'step597000-tokens2503B', 'step596000-tokens2498B', 'step595000-tokens2494B', 'step594000-tokens2490B', 'step593000-tokens2486B', 'step592000-tokens2482B', 'step591000-tokens2477B', 'step590000-tokens2473B', 'step589000-tokens2469B', 'step588000-tokens2465B', 'step587000-tokens2461B', 'step586000-tokens2457B', 'step585000-tokens2452B', 'step584000-tokens2448B', 'step583000-tokens2444B', 'step582000-tokens2440B', 'step581000-tokens2436B', 'step580000-tokens2431B', 'step579000-tokens2427B', 'step578000-tokens2423B', 'step577000-tokens2419B', 'step576000-tokens2415B', 'step575000-tokens2410B', 'step574000-tokens2406B', 'step573000-tokens2402B', 'step572000-tokens2398B', 'step571000-tokens2394B', 'step570000-tokens2389B', 'step569000-tokens2385B', 'step568000-tokens2381B', 'step567000-tokens2377B', 'step566000-tokens2373B', 'step565000-tokens2368B', 'step564000-tokens2364B', 'step563000-tokens2360B', 'step562000-tokens2356B', 'step561000-tokens2352B', 'step560000-tokens2348B', 'step559000-tokens2343B', 'step558000-tokens2339B', 'step557000-tokens2335B', 'step556000-tokens2331B', 'step555000-tokens2327B', 'step554000-tokens2322B', 'step553000-tokens2318B', 'step552000-tokens2314B', 'step551000-tokens2310B', 'step550000-tokens2306B', 'step549000-tokens2301B', 'step548450-tokens2299B', 'step548000-tokens2297B', 'step547000-tokens2293B', 'step546000-tokens2289B', 'step545000-tokens2285B', 'step544000-tokens2280B', 'step543350-tokens2278B', 'step543000-tokens2276B', 'step542000-tokens2272B', 'step541000-tokens2268B', 'step540000-tokens2264B', 'step539000-tokens2259B', 'step538000-tokens2255B', 'step537000-tokens2251B', 'step536000-tokens2247B', 'step535000-tokens2243B', 'step534000-tokens2238B', 'step533000-tokens2234B', 'step532000-tokens2230B', 'step531000-tokens2226B', 'step530000-tokens2222B', 'step529000-tokens2218B', 'step528000-tokens2213B', 'step527000-tokens2209B', 'step526000-tokens2205B', 'step525000-tokens2201B', 'step524000-tokens2197B', 'step523000-tokens2192B', 'step522000-tokens2188B', 'step521000-tokens2184B', 'step520000-tokens2180B', 'step519000-tokens2176B', 'step518000-tokens2171B', 'step517000-tokens2167B', 'step516000-tokens2163B', 'step515000-tokens2159B', 'step514000-tokens2155B', 'step513000-tokens2150B', 'step512000-tokens2146B', 'step511350-tokens2144B', 'step511000-tokens2142B', 'step510000-tokens2138B', 'step509200-tokens2135B', 'step509000-tokens2134B', 'step508500-tokens2132B', 'step508000-tokens2129B', 'step507000-tokens2125B', 'step506000-tokens2121B', 'step505550-tokens2119B', 'step505000-tokens2117B', 'step504500-tokens2115B', 'step504000-tokens2113B', 'step503000-tokens2109B', 'step502000-tokens2104B', 'step501000-tokens2100B', 'step500000-tokens2096B', 'step499000-tokens2092B', 'step498000-tokens2088B', 'step497000-tokens2083B', 'step496000-tokens2079B', 'step495000-tokens2075B', 'step494000-tokens2071B', 'step493000-tokens2067B', 'step492000-tokens2062B', 'step491000-tokens2058B', 'step490000-tokens2054B', 'step489000-tokens2050B', 'step488000-tokens2046B',
                         'step487000-tokens2041B', 'step486000-tokens2037B', 'step485000-tokens2033B', 'step484000-tokens2029B', 'step483000-tokens2025B', 'step482000-tokens2020B', 'step481000-tokens2016B', 'step480000-tokens2012B', 'step479000-tokens2008B', 'step478000-tokens2004B', 'step470000-tokens1970B', 'step409000-tokens1714B', 'step408923-tokens1714B', 'step408000-tokens1710B', 'step407000-tokens1706B', 'step406000-tokens1702B', 'step405000-tokens1698B', 'step404000-tokens1693B', 'step403000-tokens1689B', 'step402000-tokens1685B', 'step401000-tokens1681B', 'step400000-tokens1677B', 'step200000-tokens838B', 'step201000-tokens842B', 'step202000-tokens846B', 'step203000-tokens851B', 'step204000-tokens855B', 'step205000-tokens859B', 'step206000-tokens863B', 'step207000-tokens867B', 'step208000-tokens872B', 'step209000-tokens876B', 'step210000-tokens880B', 'step211000-tokens884B', 'step212000-tokens888B', 'step213000-tokens893B', 'step214000-tokens897B', 'step215000-tokens901B', 'step216000-tokens905B', 'step217000-tokens909B', 'step218000-tokens914B', 'step219000-tokens918B', 'step220000-tokens922B', 'step221000-tokens926B', 'step222000-tokens930B', 'step223000-tokens935B', 'step224000-tokens939B', 'step225000-tokens943B', 'step226000-tokens947B', 'step227000-tokens951B', 'step228000-tokens955B', 'step229000-tokens960B', 'step230000-tokens964B', 'step231000-tokens968B', 'step232000-tokens972B', 'step233000-tokens976B', 'step234000-tokens981B', 'step235000-tokens985B', 'step236000-tokens989B', 'step237000-tokens993B', 'step238000-tokens997B', 'step239000-tokens1002B', 'step240000-tokens1006B', 'step241000-tokens1010B', 'step242000-tokens1014B', 'step243000-tokens1018B', 'step244000-tokens1023B', 'step245000-tokens1027B', 'step246000-tokens1031B', 'step247000-tokens1035B', 'step248000-tokens1039B', 'step249000-tokens1044B', 'step250000-tokens1048B', 'step251000-tokens1052B', 'step476000-tokens1995B', 'step475000-tokens1991B', 'step474000-tokens1987B', 'step473000-tokens1983B', 'step472000-tokens1979B', 'step471000-tokens1974B', 'step469000-tokens1966B', 'step468000-tokens1962B', 'step467000-tokens1958B', 'step466000-tokens1953B', 'step465000-tokens1949B', 'step464000-tokens1945B', 'step463000-tokens1941B', 'step462000-tokens1937B', 'step461000-tokens1932B', 'step460000-tokens1928B', 'step459000-tokens1924B', 'step458000-tokens1920B', 'step457000-tokens1916B', 'step456000-tokens1911B', 'step455000-tokens1907B', 'step454000-tokens1903B', 'step453000-tokens1899B', 'step452000-tokens1895B', 'step451000-tokens1890B', 'step450000-tokens1886B', 'step449000-tokens1882B', 'step448000-tokens1878B', 'step447000-tokens1874B', 'step446000-tokens1870B', 'step445000-tokens1865B', 'step444000-tokens1861B', 'step443000-tokens1857B', 'step442000-tokens1853B', 'step441000-tokens1849B', 'step440000-tokens1844B', 'step439000-tokens1840B', 'step438000-tokens1836B', 'step437000-tokens1832B', 'step436000-tokens1828B', 'step435000-tokens1823B', 'step434000-tokens1819B', 'step433000-tokens1815B', 'step432000-tokens1811B', 'step431000-tokens1807B', 'step430000-tokens1802B', 'step429000-tokens1798B', 'step428000-tokens1794B', 'step427000-tokens1790B', 'step426000-tokens1786B', 'step425000-tokens1781B', 'step424000-tokens1777B', 'step423000-tokens1773B', 'step422000-tokens1769B', 'step421000-tokens1765B', 'step420000-tokens1761B', 'step419000-tokens1756B', 'step418000-tokens1752B', 'step417000-tokens1748B', 'step416000-tokens1744B', 'step415000-tokens1740B', 'step414000-tokens1735B', 'step413000-tokens1731B', 'step412000-tokens1727B', 'step411000-tokens1723B', 'step410000-tokens1719B', 'step399000-tokens1672B', 'step398000-tokens1668B', 'step397000-tokens1664B', 'step396000-tokens1660B', 'step395000-tokens1656B', 'step394000-tokens1651B', 'step393000-tokens1647B', 'step392000-tokens1643B', 'step391000-tokens1639B', 'step390000-tokens1635B', 'step389000-tokens1631B', 'step388000-tokens1626B', 'step387000-tokens1622B', 'step386000-tokens1618B', 'step385000-tokens1614B', 'step384000-tokens1610B', 'step383000-tokens1605B', 'step382000-tokens1601B', 'step381000-tokens1597B', 'step380000-tokens1593B', 'step379000-tokens1589B', 'step378000-tokens1584B', 'step377000-tokens1580B', 'step376000-tokens1576B', 'step375000-tokens1572B', 'step374000-tokens1568B', 'step373000-tokens1563B', 'step372000-tokens1559B', 'step371000-tokens1555B', 'step370000-tokens1551B', 'step369000-tokens1547B', 'step368000-tokens1542B', 'step367000-tokens1538B', 'step366000-tokens1534B', 'step365000-tokens1530B',
                         'step364000-tokens1526B', 'step363000-tokens1522B', 'step362000-tokens1517B', 'step361000-tokens1513B', 'step360000-tokens1509B', 'step359000-tokens1505B', 'step358000-tokens1501B', 'step357000-tokens1496B', 'step356000-tokens1492B', 'step355000-tokens1488B', 'step354000-tokens1484B', 'step353000-tokens1480B', 'step352000-tokens1475B', 'step351000-tokens1471B', 'step350000-tokens1467B', 'step349000-tokens1463B', 'step348000-tokens1459B', 'step347000-tokens1454B', 'step346000-tokens1450B', 'step345000-tokens1446B', 'step344000-tokens1442B', 'step343000-tokens1438B', 'step342000-tokens1433B', 'step341000-tokens1429B', 'step340000-tokens1425B', 'step339000-tokens1421B', 'step338000-tokens1417B', 'step337000-tokens1412B', 'step336000-tokens1408B', 'step335000-tokens1404B', 'step334000-tokens1400B', 'step333000-tokens1396B', 'step332000-tokens1392B', 'step331000-tokens1387B', 'step330000-tokens1383B', 'step329000-tokens1379B', 'step328000-tokens1375B', 'step327000-tokens1371B', 'step326000-tokens1366B', 'step325000-tokens1362B', 'step324000-tokens1358B', 'step323000-tokens1354B', 'step322000-tokens1350B', 'step321000-tokens1345B', 'step320000-tokens1341B', 'step319000-tokens1337B', 'step318000-tokens1333B', 'step317000-tokens1329B', 'step316000-tokens1324B', 'step315000-tokens1320B', 'step314000-tokens1316B', 'step313000-tokens1312B', 'step312000-tokens1308B', 'step311000-tokens1303B', 'step310000-tokens1299B', 'step309000-tokens1295B', 'step308000-tokens1291B', 'step307000-tokens1287B', 'step306000-tokens1283B', 'step305000-tokens1278B', 'step304000-tokens1274B', 'step303000-tokens1270B', 'step302000-tokens1266B', 'step301000-tokens1262B', 'step300000-tokens1257B', 'step299000-tokens1253B', 'step298000-tokens1249B', 'step297000-tokens1245B', 'step296000-tokens1241B', 'step295000-tokens1236B', 'step294000-tokens1232B', 'step293000-tokens1228B', 'step292000-tokens1224B', 'step291000-tokens1220B', 'step290000-tokens1215B', 'step289000-tokens1211B', 'step288000-tokens1207B', 'step287000-tokens1203B', 'step286000-tokens1199B', 'step285000-tokens1194B', 'step284000-tokens1190B', 'step283000-tokens1186B', 'step282000-tokens1182B', 'step281000-tokens1178B', 'step280000-tokens1174B', 'step279000-tokens1169B', 'step278000-tokens1165B', 'step277000-tokens1161B', 'step276000-tokens1157B', 'step275000-tokens1153B', 'step274000-tokens1148B', 'step273000-tokens1144B', 'step272000-tokens1140B', 'step271000-tokens1136B', 'step270000-tokens1132B', 'step269000-tokens1127B', 'step268000-tokens1123B', 'step267000-tokens1119B', 'step266000-tokens1115B', 'step265000-tokens1111B', 'step264000-tokens1106B', 'step263000-tokens1102B', 'step262000-tokens1098B', 'step261000-tokens1094B', 'step260000-tokens1090B', 'step259000-tokens1085B', 'step258000-tokens1081B', 'step257000-tokens1077B', 'step256000-tokens1073B', 'step255000-tokens1069B', 'step254000-tokens1064B', 'step253000-tokens1060B', 'step252000-tokens1056B', 'step99500-tokens417B', 'step99000-tokens415B', 'step98500-tokens412B', 'step98000-tokens410B', 'step97500-tokens408B', 'step97000-tokens406B', 'step96500-tokens404B', 'step96000-tokens402B', 'step95500-tokens400B', 'step95000-tokens398B', 'step9500-tokens39B', 'step94500-tokens396B', 'step94000-tokens394B', 'step93500-tokens392B', 'step93000-tokens389B', 'step92500-tokens387B', 'step92000-tokens385B', 'step91500-tokens383B', 'step91000-tokens381B', 'step90500-tokens379B', 'step90000-tokens377B', 'step9000-tokens37B', 'step89500-tokens375B', 'step89000-tokens373B', 'step88500-tokens371B', 'step88000-tokens368B', 'step87500-tokens366B', 'step87000-tokens364B', 'step86500-tokens362B', 'step86000-tokens360B', 'step85500-tokens358B', 'step85000-tokens356B', 'step8500-tokens35B', 'step84500-tokens354B', 'step84000-tokens352B', 'step83500-tokens350B', 'step83000-tokens348B', 'step82500-tokens345B', 'step82000-tokens343B', 'step81500-tokens341B', 'step81000-tokens339B', 'step80500-tokens337B', 'step80000-tokens335B', 'step8000-tokens33B', 'step79500-tokens333B', 'step79000-tokens331B', 'step78500-tokens329B', 'step78000-tokens327B', 'step77500-tokens324B', 'step77000-tokens322B', 'step76500-tokens320B', 'step76000-tokens318B', 'step75500-tokens316B', 'step75000-tokens314B', 'step7500-tokens31B', 'step74500-tokens312B', 'step74000-tokens310B', 'step73500-tokens308B', 'step73100-tokens306B', 'step73000-tokens306B', 'step72500-tokens303B', 'step72000-tokens301B', 'step71500-tokens299B', 'step71000-tokens297B', 'step70500-tokens295B', 'step70000-tokens293B',
                         'step7000-tokens29B', 'step69500-tokens291B', 'step69000-tokens289B', 'step68500-tokens287B', 'step68000-tokens285B', 'step67500-tokens283B', 'step67000-tokens280B', 'step66500-tokens278B', 'step66000-tokens276B', 'step65500-tokens274B', 'step65000-tokens272B', 'step6500-tokens27B', 'step64500-tokens270B', 'step64000-tokens268B', 'step63500-tokens266B', 'step63000-tokens264B', 'step62500-tokens262B', 'step62000-tokens259B', 'step61500-tokens257B', 'step61000-tokens255B', 'step60500-tokens253B', 'step60000-tokens251B', 'step6000-tokens25B', 'step59500-tokens249B', 'step59000-tokens247B', 'step58500-tokens245B', 'step58000-tokens243B', 'step57500-tokens241B', 'step57000-tokens238B', 'step56500-tokens236B', 'step56000-tokens234B', 'step55500-tokens232B', 'step55000-tokens230B', 'step5500-tokens23B', 'step54500-tokens228B', 'step54000-tokens226B', 'step53500-tokens224B', 'step53000-tokens222B', 'step52500-tokens220B', 'step52300-tokens219B', 'step52000-tokens218B', 'step51500-tokens215B', 'step51450-tokens215B', 'step51000-tokens213B', 'step50500-tokens211B', 'step50000-tokens209B', 'step5000-tokens20B', 'step500-tokens2B', 'step49500-tokens207B', 'step49000-tokens205B', 'step48500-tokens203B', 'step48000-tokens201B', 'step47000-tokens197B', 'step46500-tokens194B', 'step46000-tokens192B', 'step45500-tokens190B', 'step45000-tokens188B', 'step4500-tokens18B', 'step44500-tokens186B', 'step44000-tokens184B', 'step43500-tokens182B', 'step43000-tokens180B', 'step42500-tokens178B', 'step42200-tokens176B', 'step42000-tokens176B', 'step41500-tokens174B', 'step41400-tokens173B', 'step41000-tokens171B', 'step40500-tokens169B', 'step40000-tokens167B', 'step4000-tokens16B', 'step39500-tokens165B', 'step39350-tokens164B', 'step39000-tokens163B', 'step38500-tokens161B', 'step38000-tokens159B', 'step37500-tokens157B', 'step37000-tokens155B', 'step36500-tokens153B', 'step36000-tokens150B', 'step35500-tokens148B', 'step35000-tokens146B', 'step3500-tokens14B', 'step34500-tokens144B', 'step34000-tokens142B', 'step33500-tokens140B', 'step33000-tokens138B', 'step32500-tokens136B', 'step32000-tokens134B', 'step31500-tokens132B', 'step31000-tokens129B', 'step30500-tokens127B', 'step30000-tokens125B', 'step3000-tokens12B', 'step29500-tokens123B', 'step29000-tokens121B', 'step28500-tokens119B', 'step28000-tokens117B', 'step27500-tokens115B', 'step27000-tokens113B', 'step26500-tokens111B', 'step26000-tokens109B', 'step25500-tokens106B', 'step47500-tokens199B', 'step25000-tokens104B', 'step2500-tokens10B', 'step24500-tokens102B', 'step24000-tokens100B', 'step23500-tokens98B', 'step23000-tokens96B', 'step22500-tokens94B', 'step22000-tokens92B', 'step21500-tokens90B', 'step21000-tokens88B', 'step20500-tokens85B', 'step20000-tokens83B', 'step142000-tokens595B', 'step14200-tokens59B', 'step141500-tokens593B', 'step141000-tokens591B', 'step140500-tokens589B', 'step140000-tokens587B', 'step14000-tokens58B', 'step139500-tokens584B', 'step139000-tokens582B', 'step138500-tokens580B', 'step138000-tokens578B', 'step137500-tokens576B', 'step137000-tokens574B', 'step136500-tokens572B', 'step136000-tokens570B', 'step135500-tokens568B', 'step128000-tokens536B', 'step127500-tokens534B', 'step127000-tokens532B', 'step126500-tokens530B', 'step126000-tokens528B', 'step125500-tokens526B', 'step125000-tokens524B', 'step12500-tokens52B', 'step124500-tokens522B', 'step124000-tokens519B', 'step123500-tokens517B', 'step123000-tokens515B', 'step122500-tokens513B', 'step122000-tokens511B', 'step121500-tokens509B', 'step121000-tokens507B', 'step120500-tokens505B', 'step120000-tokens503B', 'step12000-tokens50B', 'step119500-tokens501B', 'step119000-tokens498B', 'step118500-tokens496B', 'step118000-tokens494B', 'step117500-tokens492B', 'step117000-tokens490B', 'step116500-tokens488B', 'step116000-tokens486B', 'step115500-tokens484B', 'step11500-tokens48B', 'step114500-tokens480B', 'step114000-tokens477B', 'step113500-tokens475B', 'step113000-tokens473B', 'step112500-tokens471B', 'step112000-tokens469B', 'step111500-tokens467B', 'step111000-tokens465B', 'step110500-tokens463B', 'step110000-tokens461B', 'step11000-tokens46B', 'step109500-tokens459B', 'step109000-tokens457B', 'step108500-tokens454B', 'step108000-tokens452B', 'step107500-tokens450B', 'step107000-tokens448B', 'step106500-tokens446B', 'step106000-tokens444B', 'step105500-tokens442B', 'step105000-tokens440B', 'step10500-tokens44B', 'step104500-tokens438B', 'step104000-tokens436B', 'step103500-tokens433B', 'step103000-tokens431B',
                         'step102500-tokens429B', 'step102000-tokens427B', 'step101500-tokens425B', 'step101000-tokens423B', 'step100500-tokens421B', 'step100000-tokens419B', 'step10000-tokens41B', 'step1000-tokens4B', 'step2000-tokens8B', 'step199000-tokens834B', 'step198000-tokens830B', 'step197000-tokens825B', 'step196000-tokens821B', 'step195000-tokens817B', 'step19500-tokens81B', 'step194000-tokens813B', 'step193000-tokens809B', 'step192000-tokens805B', 'step191000-tokens800B', 'step190000-tokens796B', 'step19000-tokens79B', 'step189000-tokens792B', 'step188000-tokens788B', 'step187000-tokens784B', 'step186000-tokens779B', 'step185000-tokens775B', 'step18500-tokens77B', 'step184000-tokens771B', 'step183000-tokens767B', 'step182000-tokens763B', 'step181000-tokens758B', 'step180000-tokens754B', 'step18000-tokens75B', 'step179000-tokens750B', 'step178000-tokens746B', 'step177000-tokens742B', 'step176000-tokens737B', 'step175000-tokens733B', 'step17500-tokens73B', 'step174000-tokens729B', 'step173000-tokens725B', 'step172000-tokens721B', 'step171000-tokens716B', 'step170000-tokens712B', 'step17000-tokens71B', 'step169000-tokens708B', 'step168000-tokens704B', 'step167000-tokens700B', 'step166000-tokens696B', 'step165000-tokens691B', 'step16500-tokens69B', 'step164000-tokens687B', 'step163000-tokens683B', 'step162000-tokens679B', 'step161000-tokens675B', 'step160600-tokens673B', 'step160550-tokens673B', 'step160500-tokens672B', 'step160000-tokens670B', 'step16000-tokens67B', 'step159500-tokens668B', 'step159000-tokens666B', 'step158500-tokens664B', 'step158000-tokens662B', 'step157500-tokens660B', 'step157000-tokens658B', 'step156500-tokens656B', 'step156000-tokens654B', 'step155500-tokens651B', 'step155000-tokens649B', 'step15500-tokens64B', 'step154500-tokens647B', 'step154000-tokens645B', 'step153500-tokens643B', 'step153000-tokens641B', 'step152500-tokens639B', 'step152000-tokens637B', 'step151500-tokens635B', 'step151000-tokens633B', 'step150500-tokens631B', 'step150000-tokens628B', 'step15000-tokens62B', 'step1500-tokens6B', 'step149500-tokens626B', 'step149000-tokens624B', 'step148500-tokens622B', 'step148000-tokens620B', 'step147500-tokens618B', 'step147000-tokens616B', 'step146500-tokens614B', 'step146000-tokens612B', 'step145500-tokens610B', 'step145000-tokens607B', 'step14500-tokens60B', 'step144500-tokens605B', 'step144000-tokens603B', 'step143500-tokens601B', 'step143000-tokens599B', 'step142500-tokens597B', 'step135000-tokens566B', 'step13500-tokens56B', 'step134500-tokens563B', 'step134000-tokens561B', 'step133500-tokens559B', 'step133000-tokens557B', 'step132500-tokens555B', 'step132000-tokens553B', 'step131500-tokens551B', 'step131000-tokens549B', 'step130500-tokens547B', 'step130000-tokens545B', 'step13000-tokens54B', 'step129500-tokens542B', 'step129000-tokens540B', 'step128500-tokens538B', 'step0-tokens0B', 'step477000-tokens2000B', 'step115000-tokens482B']

OLMO_0724_CHECKPOINTS = ['step650650-tokens2729B', 'step69500-tokens291B', 'step69000-tokens289B', 'step50000-tokens209B', 'step190000-tokens796B', 'step98500-tokens413B', 'step98000-tokens411B', 'step97500-tokens408B', 'step97000-tokens406B', 'step96500-tokens404B', 'step96000-tokens402B', 'step95500-tokens400B', 'step95000-tokens398B', 'step9500-tokens39B', 'step94500-tokens396B', 'step94000-tokens394B', 'step93500-tokens392B', 'step93000-tokens390B', 'step92500-tokens387B', 'step92000-tokens385B', 'step91500-tokens383B', 'step91000-tokens381B', 'step90500-tokens379B', 'step90000-tokens377B', 'step9000-tokens37B', 'step89500-tokens375B', 'step89000-tokens373B', 'step88500-tokens371B', 'step88000-tokens369B', 'step87500-tokens367B', 'step87000-tokens364B', 'step86500-tokens362B', 'step86000-tokens360B', 'step85500-tokens358B', 'step85000-tokens356B', 'step8500-tokens35B', 'step84500-tokens354B', 'step84000-tokens352B', 'step83500-tokens350B', 'step83000-tokens348B', 'step82500-tokens346B', 'step82000-tokens343B', 'step81500-tokens341B', 'step81000-tokens339B', 'step80500-tokens337B', 'step80000-tokens335B', 'step8000-tokens33B', 'step79500-tokens333B', 'step79000-tokens331B', 'step78500-tokens329B', 'step78000-tokens327B', 'step77500-tokens325B', 'step77000-tokens322B', 'step76500-tokens320B', 'step76000-tokens318B', 'step75500-tokens316B', 'step75000-tokens314B', 'step7500-tokens31B', 'step74500-tokens312B', 'step74000-tokens310B', 'step73500-tokens308B', 'step73100-tokens306B', 'step73000-tokens306B', 'step72500-tokens304B', 'step72000-tokens301B', 'step71500-tokens299B', 'step71000-tokens297B', 'step70500-tokens295B', 'step70000-tokens293B', 'step7000-tokens29B', 'step68500-tokens287B', 'step68000-tokens285B', 'step67500-tokens283B', 'step67000-tokens281B', 'step66500-tokens278B', 'step66000-tokens276B', 'step65000-tokens272B', 'step6500-tokens27B', 'step649650-tokens2724B', 'step648650-tokens2720B', 'step647650-tokens2716B', 'step646650-tokens2712B', 'step645650-tokens2708B', 'step64500-tokens270B', 'step644650-tokens2703B', 'step643650-tokens2699B', 'step642650-tokens2695B', 'step641650-tokens2691B', 'step640650-tokens2687B', 'step639650-tokens2682B', 'step639000-tokens2680B', 'step638000-tokens2675B', 'step637000-tokens2671B', 'step636000-tokens2667B', 'step635000-tokens2663B', 'step63500-tokens266B', 'step634000-tokens2659B', 'step633000-tokens2654B', 'step632000-tokens2650B', 'step631000-tokens2646B', 'step630000-tokens2642B', 'step63000-tokens264B', 'step629000-tokens2638B', 'step628000-tokens2634B', 'step627000-tokens2629B', 'step626000-tokens2625B', 'step625000-tokens2621B', 'step62500-tokens262B', 'step624000-tokens2617B', 'step623000-tokens2613B', 'step622000-tokens2608B', 'step621100-tokens2605B', 'step621000-tokens2604B', 'step620000-tokens2600B', 'step62000-tokens260B', 'step619000-tokens2596B', 'step618000-tokens2592B', 'step617000-tokens2587B', 'step616000-tokens2583B', 'step615000-tokens2579B', 'step61500-tokens257B', 'step614000-tokens2575B', 'step613000-tokens2571B', 'step612000-tokens2566B', 'step611000-tokens2562B', 'step610000-tokens2558B', 'step61000-tokens255B', 'step609000-tokens2554B', 'step608000-tokens2550B', 'step607000-tokens2545B', 'step606000-tokens2541B', 'step605000-tokens2537B', 'step60500-tokens253B', 'step604000-tokens2533B', 'step603000-tokens2529B', 'step602000-tokens2524B', 'step601000-tokens2520B', 'step600000-tokens2516B', 'step60000-tokens251B', 'step6000-tokens25B', 'step599000-tokens2512B', 'step598000-tokens2508B', 'step597000-tokens2503B', 'step596000-tokens2499B', 'step595000-tokens2495B', 'step59500-tokens249B', 'step594000-tokens2491B', 'step593000-tokens2487B', 'step592000-tokens2483B', 'step591000-tokens2478B', 'step590000-tokens2474B', 'step59000-tokens247B', 'step589000-tokens2470B', 'step588000-tokens2466B', 'step587000-tokens2462B', 'step586000-tokens2457B', 'step585000-tokens2453B', 'step58500-tokens245B', 'step584000-tokens2449B', 'step583000-tokens2445B', 'step582000-tokens2441B', 'step581000-tokens2436B', 'step580000-tokens2432B', 'step58000-tokens243B', 'step579000-tokens2428B', 'step578000-tokens2424B', 'step577000-tokens2420B', 'step576000-tokens2415B', 'step575000-tokens2411B', 'step57500-tokens241B', 'step574000-tokens2407B', 'step573000-tokens2403B', 'step572000-tokens2399B', 'step571000-tokens2394B', 'step570000-tokens2390B', 'step57000-tokens239B', 'step569000-tokens2386B', 'step568000-tokens2382B', 'step567000-tokens2378B', 'step566000-tokens2373B', 'step565000-tokens2369B',
                         'step56500-tokens236B', 'step564000-tokens2365B', 'step563000-tokens2361B', 'step562000-tokens2357B', 'step561000-tokens2353B', 'step560000-tokens2348B', 'step56000-tokens234B', 'step559000-tokens2344B', 'step558000-tokens2340B', 'step557000-tokens2336B', 'step556000-tokens2332B', 'step555000-tokens2327B', 'step55500-tokens232B', 'step554000-tokens2323B', 'step553000-tokens2319B', 'step552000-tokens2315B', 'step551000-tokens2311B', 'step550000-tokens2306B', 'step55000-tokens230B', 'step5500-tokens23B', 'step549000-tokens2302B', 'step548450-tokens2300B', 'step548000-tokens2298B', 'step547000-tokens2294B', 'step546000-tokens2290B', 'step545000-tokens2285B', 'step54500-tokens228B', 'step544000-tokens2281B', 'step543350-tokens2278B', 'step543000-tokens2277B', 'step542000-tokens2273B', 'step541000-tokens2269B', 'step540000-tokens2264B', 'step54000-tokens226B', 'step539000-tokens2260B', 'step538000-tokens2256B', 'step537000-tokens2252B', 'step536000-tokens2248B', 'step535000-tokens2243B', 'step53500-tokens224B', 'step534000-tokens2239B', 'step533000-tokens2235B', 'step532000-tokens2231B', 'step531000-tokens2227B', 'step530000-tokens2222B', 'step53000-tokens222B', 'step529000-tokens2218B', 'step528000-tokens2214B', 'step527000-tokens2210B', 'step526000-tokens2206B', 'step525000-tokens2202B', 'step52500-tokens220B', 'step524000-tokens2197B', 'step523000-tokens2193B', 'step52300-tokens219B', 'step522000-tokens2189B', 'step521000-tokens2185B', 'step520000-tokens2181B', 'step52000-tokens218B', 'step519000-tokens2176B', 'step518000-tokens2172B', 'step517000-tokens2168B', 'step516000-tokens2164B', 'step515000-tokens2160B', 'step51500-tokens216B', 'step51450-tokens215B', 'step514000-tokens2155B', 'step513000-tokens2151B', 'step512000-tokens2147B', 'step511350-tokens2144B', 'step511000-tokens2143B', 'step510000-tokens2139B', 'step51000-tokens213B', 'step509200-tokens2135B', 'step509000-tokens2134B', 'step508500-tokens2132B', 'step508000-tokens2130B', 'step507000-tokens2126B', 'step506000-tokens2122B', 'step505550-tokens2120B', 'step505000-tokens2118B', 'step50500-tokens211B', 'step504500-tokens2116B', 'step504000-tokens2113B', 'step503000-tokens2109B', 'step502000-tokens2105B', 'step501000-tokens2101B', 'step500000-tokens2097B', 'step5000-tokens20B', 'step500-tokens2B', 'step499000-tokens2092B', 'step498000-tokens2088B', 'step497000-tokens2084B', 'step496000-tokens2080B', 'step495000-tokens2076B', 'step49500-tokens207B', 'step494000-tokens2071B', 'step493000-tokens2067B', 'step492000-tokens2063B', 'step491000-tokens2059B', 'step490000-tokens2055B', 'step49000-tokens205B', 'step489000-tokens2051B', 'step488000-tokens2046B', 'step487000-tokens2042B', 'step486000-tokens2038B', 'step485000-tokens2034B', 'step48500-tokens203B', 'step484000-tokens2030B', 'step483000-tokens2025B', 'step482000-tokens2021B', 'step481000-tokens2017B', 'step480000-tokens2013B', 'step48000-tokens201B', 'step479000-tokens2009B', 'step478000-tokens2004B', 'step477000-tokens2000B', 'step476000-tokens1996B', 'step475000-tokens1992B', 'step47500-tokens199B', 'step474000-tokens1988B', 'step473000-tokens1983B', 'step472000-tokens1979B', 'step471000-tokens1975B', 'step470000-tokens1971B', 'step47000-tokens197B', 'step469000-tokens1967B', 'step468000-tokens1962B', 'step467000-tokens1958B', 'step466000-tokens1954B', 'step465000-tokens1950B', 'step46500-tokens195B', 'step464000-tokens1946B', 'step463000-tokens1941B', 'step462000-tokens1937B', 'step461000-tokens1933B', 'step460000-tokens1929B', 'step46000-tokens192B', 'step459000-tokens1925B', 'step458000-tokens1920B', 'step457000-tokens1916B', 'step456000-tokens1912B', 'step455000-tokens1908B', 'step45500-tokens190B', 'step454000-tokens1904B', 'step453000-tokens1900B', 'step452000-tokens1895B', 'step451000-tokens1891B', 'step450000-tokens1887B', 'step45000-tokens188B', 'step4500-tokens18B', 'step449000-tokens1883B', 'step448000-tokens1879B', 'step447000-tokens1874B', 'step446000-tokens1870B', 'step445000-tokens1866B', 'step44500-tokens186B', 'step444000-tokens1862B', 'step443000-tokens1858B', 'step442000-tokens1853B', 'step441000-tokens1849B', 'step440000-tokens1845B', 'step44000-tokens184B', 'step439000-tokens1841B', 'step438000-tokens1837B', 'step437000-tokens1832B', 'step436000-tokens1828B', 'step435000-tokens1824B', 'step43500-tokens182B', 'step434000-tokens1820B', 'step433000-tokens1816B', 'step432000-tokens1811B', 'step431000-tokens1807B', 'step430000-tokens1803B', 'step43000-tokens180B', 'step429000-tokens1799B',
                         'step428000-tokens1795B', 'step427000-tokens1790B', 'step426000-tokens1786B', 'step425000-tokens1782B', 'step42500-tokens178B', 'step424000-tokens1778B', 'step423000-tokens1774B', 'step422000-tokens1769B', 'step42200-tokens176B', 'step421000-tokens1765B', 'step420000-tokens1761B', 'step42000-tokens176B', 'step419000-tokens1757B', 'step418000-tokens1753B', 'step417000-tokens1749B', 'step416000-tokens1744B', 'step415000-tokens1740B', 'step41500-tokens174B', 'step414000-tokens1736B', 'step41400-tokens173B', 'step413000-tokens1732B', 'step412000-tokens1728B', 'step411000-tokens1723B', 'step410000-tokens1719B', 'step41000-tokens171B', 'step409000-tokens1715B', 'step408923-tokens1715B', 'step408000-tokens1711B', 'step407000-tokens1707B', 'step406000-tokens1702B', 'step405000-tokens1698B', 'step40500-tokens169B', 'step404000-tokens1694B', 'step403000-tokens1690B', 'step402000-tokens1686B', 'step401000-tokens1681B', 'step400000-tokens1677B', 'step40000-tokens167B', 'step4000-tokens16B', 'step399000-tokens1673B', 'step398000-tokens1669B', 'step397000-tokens1665B', 'step396000-tokens1660B', 'step395000-tokens1656B', 'step39500-tokens165B', 'step394000-tokens1652B', 'step39350-tokens165B', 'step393000-tokens1648B', 'step392000-tokens1644B', 'step391000-tokens1639B', 'step390000-tokens1635B', 'step39000-tokens163B', 'step389000-tokens1631B', 'step388000-tokens1627B', 'step387000-tokens1623B', 'step386000-tokens1619B', 'step385000-tokens1614B', 'step38500-tokens161B', 'step384000-tokens1610B', 'step383000-tokens1606B', 'step382000-tokens1602B', 'step381000-tokens1598B', 'step380000-tokens1593B', 'step38000-tokens159B', 'step379000-tokens1589B', 'step378000-tokens1585B', 'step377000-tokens1581B', 'step376000-tokens1577B', 'step375000-tokens1572B', 'step37500-tokens157B', 'step374000-tokens1568B', 'step373000-tokens1564B', 'step372000-tokens1560B', 'step371000-tokens1556B', 'step370000-tokens1551B', 'step37000-tokens155B', 'step369000-tokens1547B', 'step368000-tokens1543B', 'step367000-tokens1539B', 'step366000-tokens1535B', 'step365000-tokens1530B', 'step36500-tokens153B', 'step364000-tokens1526B', 'step363000-tokens1522B', 'step362000-tokens1518B', 'step361000-tokens1514B', 'step360000-tokens1509B', 'step36000-tokens150B', 'step359000-tokens1505B', 'step358000-tokens1501B', 'step357000-tokens1497B', 'step356000-tokens1493B', 'step355000-tokens1488B', 'step35500-tokens148B', 'step354000-tokens1484B', 'step353000-tokens1480B', 'step352000-tokens1476B', 'step351000-tokens1472B', 'step350000-tokens1468B', 'step35000-tokens146B', 'step3500-tokens14B', 'step349000-tokens1463B', 'step348000-tokens1459B', 'step347000-tokens1455B', 'step346000-tokens1451B', 'step345000-tokens1447B', 'step34500-tokens144B', 'step344000-tokens1442B', 'step343000-tokens1438B', 'step342000-tokens1434B', 'step341000-tokens1430B', 'step340000-tokens1426B', 'step34000-tokens142B', 'step339000-tokens1421B', 'step338000-tokens1417B', 'step337000-tokens1413B', 'step336000-tokens1409B', 'step335000-tokens1405B', 'step33500-tokens140B', 'step334000-tokens1400B', 'step333000-tokens1396B', 'step332000-tokens1392B', 'step331000-tokens1388B', 'step330000-tokens1384B', 'step33000-tokens138B', 'step329000-tokens1379B', 'step328000-tokens1375B', 'step327000-tokens1371B', 'step326000-tokens1367B', 'step325000-tokens1363B', 'step32500-tokens136B', 'step324000-tokens1358B', 'step323000-tokens1354B', 'step322000-tokens1350B', 'step321000-tokens1346B', 'step320000-tokens1342B', 'step32000-tokens134B', 'step319000-tokens1337B', 'step318000-tokens1333B', 'step317000-tokens1329B', 'step316000-tokens1325B', 'step315000-tokens1321B', 'step31500-tokens132B', 'step314000-tokens1317B', 'step313000-tokens1312B', 'step312000-tokens1308B', 'step311000-tokens1304B', 'step310000-tokens1300B', 'step31000-tokens130B', 'step309000-tokens1296B', 'step308000-tokens1291B', 'step307000-tokens1287B', 'step306000-tokens1283B', 'step305000-tokens1279B', 'step30500-tokens127B', 'step304000-tokens1275B', 'step303000-tokens1270B', 'step302000-tokens1266B', 'step301000-tokens1262B', 'step300000-tokens1258B', 'step30000-tokens125B', 'step3000-tokens12B', 'step299000-tokens1254B', 'step298000-tokens1249B', 'step297000-tokens1245B', 'step296000-tokens1241B', 'step295000-tokens1237B', 'step29500-tokens123B', 'step294000-tokens1233B', 'step293000-tokens1228B', 'step292000-tokens1224B', 'step291000-tokens1220B', 'step290000-tokens1216B', 'step29000-tokens121B', 'step289000-tokens1212B', 'step288000-tokens1207B',
                         'step287000-tokens1203B', 'step286000-tokens1199B', 'step285000-tokens1195B', 'step28500-tokens119B', 'step284000-tokens1191B', 'step283000-tokens1186B', 'step282000-tokens1182B', 'step281000-tokens1178B', 'step280000-tokens1174B', 'step28000-tokens117B', 'step279000-tokens1170B', 'step278000-tokens1166B', 'step277000-tokens1161B', 'step276000-tokens1157B', 'step275000-tokens1153B', 'step27500-tokens115B', 'step274000-tokens1149B', 'step273000-tokens1145B', 'step272000-tokens1140B', 'step271000-tokens1136B', 'step270000-tokens1132B', 'step27000-tokens113B', 'step269000-tokens1128B', 'step268000-tokens1124B', 'step267000-tokens1119B', 'step266000-tokens1115B', 'step265000-tokens1111B', 'step26500-tokens111B', 'step264000-tokens1107B', 'step263000-tokens1103B', 'step262000-tokens1098B', 'step261000-tokens1094B', 'step260000-tokens1090B', 'step26000-tokens109B', 'step259000-tokens1086B', 'step258000-tokens1082B', 'step257000-tokens1077B', 'step256000-tokens1073B', 'step255000-tokens1069B', 'step25500-tokens106B', 'step254000-tokens1065B', 'step253000-tokens1061B', 'step252000-tokens1056B', 'step251000-tokens1052B', 'step250000-tokens1048B', 'step25000-tokens104B', 'step2500-tokens10B', 'step249000-tokens1044B', 'step248000-tokens1040B', 'step247000-tokens1035B', 'step246000-tokens1031B', 'step245000-tokens1027B', 'step24500-tokens102B', 'step244000-tokens1023B', 'step243000-tokens1019B', 'step242000-tokens1015B', 'step241000-tokens1010B', 'step240000-tokens1006B', 'step24000-tokens100B', 'step239000-tokens1002B', 'step238000-tokens998B', 'step237000-tokens994B', 'step236000-tokens989B', 'step235000-tokens985B', 'step23500-tokens98B', 'step234000-tokens981B', 'step233000-tokens977B', 'step232000-tokens973B', 'step231000-tokens968B', 'step230000-tokens964B', 'step23000-tokens96B', 'step229000-tokens960B', 'step228000-tokens956B', 'step227000-tokens952B', 'step226000-tokens947B', 'step225000-tokens943B', 'step22500-tokens94B', 'step224000-tokens939B', 'step223000-tokens935B', 'step222000-tokens931B', 'step221000-tokens926B', 'step220000-tokens922B', 'step22000-tokens92B', 'step219000-tokens918B', 'step218000-tokens914B', 'step217000-tokens910B', 'step216000-tokens905B', 'step215000-tokens901B', 'step21500-tokens90B', 'step214000-tokens897B', 'step213000-tokens893B', 'step212000-tokens889B', 'step211000-tokens884B', 'step210000-tokens880B', 'step21000-tokens88B', 'step209000-tokens876B', 'step208000-tokens872B', 'step207000-tokens868B', 'step206000-tokens864B', 'step205000-tokens859B', 'step20500-tokens85B', 'step204000-tokens855B', 'step203000-tokens851B', 'step202000-tokens847B', 'step201000-tokens843B', 'step200000-tokens838B', 'step20000-tokens83B', 'step2000-tokens8B', 'step199000-tokens834B', 'step19000-tokens79B', 'step189000-tokens792B', 'step188000-tokens788B', 'step187000-tokens784B', 'step186000-tokens780B', 'main', 'step185000-tokens775B', 'step18500-tokens77B', 'step184000-tokens771B', 'step183000-tokens767B', 'step182000-tokens763B', 'step181000-tokens759B', 'step180000-tokens754B', 'step18000-tokens75B', 'step179000-tokens750B', 'step178000-tokens746B', 'step177000-tokens742B', 'step176000-tokens738B', 'step175000-tokens734B', 'step17500-tokens73B', 'step174000-tokens729B', 'step173000-tokens725B', 'step172000-tokens721B', 'step171000-tokens717B', 'step170000-tokens713B', 'step17000-tokens71B', 'step169000-tokens708B', 'step168000-tokens704B', 'step167000-tokens700B', 'step166000-tokens696B', 'step165000-tokens692B', 'step16500-tokens69B', 'step164000-tokens687B', 'step163000-tokens683B', 'step162000-tokens679B', 'step161000-tokens675B', 'step160600-tokens673B', 'step160550-tokens673B', 'step160500-tokens673B', 'step160000-tokens671B', 'step16000-tokens67B', 'step159500-tokens668B', 'step159000-tokens666B', 'step158500-tokens664B', 'step158000-tokens662B', 'step157500-tokens660B', 'step157000-tokens658B', 'step156500-tokens656B', 'step156000-tokens654B', 'step155500-tokens652B', 'step155000-tokens650B', 'step15500-tokens65B', 'step154500-tokens648B', 'step154000-tokens645B', 'step153500-tokens643B', 'step153000-tokens641B', 'step152500-tokens639B', 'step152000-tokens637B', 'step151500-tokens635B', 'step151000-tokens633B', 'step150500-tokens631B', 'step150000-tokens629B', 'step15000-tokens62B', 'step1500-tokens6B', 'step149500-tokens627B', 'step149000-tokens624B', 'step148500-tokens622B', 'step148000-tokens620B', 'step147500-tokens618B', 'step147000-tokens616B', 'step146500-tokens614B', 'step146000-tokens612B',
                         'step145500-tokens610B', 'step145000-tokens608B', 'step14500-tokens60B', 'step144500-tokens606B', 'step144000-tokens603B', 'step143500-tokens601B', 'step143000-tokens599B', 'step142500-tokens597B', 'step142000-tokens595B', 'step14200-tokens59B', 'step141500-tokens593B', 'step141000-tokens591B', 'step140500-tokens589B', 'step140000-tokens587B', 'step14000-tokens58B', 'step139500-tokens585B', 'step139000-tokens583B', 'step138500-tokens580B', 'step138000-tokens578B', 'step137500-tokens576B', 'step137000-tokens574B', 'step136500-tokens572B', 'step136000-tokens570B', 'step135500-tokens568B', 'step135000-tokens566B', 'step13500-tokens56B', 'step134500-tokens564B', 'step134000-tokens562B', 'step133500-tokens559B', 'step133000-tokens557B', 'step132500-tokens555B', 'step132000-tokens553B', 'step131500-tokens551B', 'step131000-tokens549B', 'step130500-tokens547B', 'step130000-tokens545B', 'step13000-tokens54B', 'step129500-tokens543B', 'step129000-tokens541B', 'step128500-tokens538B', 'step128000-tokens536B', 'step127500-tokens534B', 'step127000-tokens532B', 'step126500-tokens530B', 'step126000-tokens528B', 'step125500-tokens526B', 'step125000-tokens524B', 'step12500-tokens52B', 'step124500-tokens522B', 'step124000-tokens520B', 'step123500-tokens517B', 'step123000-tokens515B', 'step122500-tokens513B', 'step122000-tokens511B', 'step121500-tokens509B', 'step121000-tokens507B', 'step120500-tokens505B', 'step120000-tokens503B', 'step12000-tokens50B', 'step119500-tokens501B', 'step119000-tokens499B', 'step118500-tokens497B', 'step118000-tokens494B', 'step117500-tokens492B', 'step117000-tokens490B', 'step116500-tokens488B', 'step116000-tokens486B', 'step115500-tokens484B', 'step11500-tokens48B', 'step114500-tokens480B', 'step114000-tokens478B', 'step113500-tokens476B', 'step113000-tokens473B', 'step112500-tokens471B', 'step112000-tokens469B', 'step111500-tokens467B', 'step111000-tokens465B', 'step110500-tokens463B', 'step110000-tokens461B', 'step11000-tokens46B', 'step109500-tokens459B', 'step109000-tokens457B', 'step108500-tokens455B', 'step108000-tokens452B', 'step107500-tokens450B', 'step107000-tokens448B', 'step106500-tokens446B', 'step106000-tokens444B', 'step105500-tokens442B', 'step105000-tokens440B', 'step10500-tokens44B', 'step104500-tokens438B', 'step104000-tokens436B', 'step103500-tokens434B', 'step103000-tokens432B', 'step102500-tokens429B', 'step102000-tokens427B', 'step101500-tokens425B', 'step101000-tokens423B', 'step100500-tokens421B', 'step100000-tokens419B', 'step10000-tokens41B', 'step1000-tokens4B', 'step0-tokens0B', 'step99500-tokens417B', 'step99000-tokens415B']

def get_checkpoint_labels(model_name: str, **kwargs):
    """Returns the checkpoint labels for a given model, and the label_type
    (step or token). Raises an error for models that are not checkpointed."""
    official_model_name = get_official_model_name(model_name)
    if official_model_name.startswith("stanford-crfm/"):
        return STANFORD_CRFM_CHECKPOINTS, "step"
    elif official_model_name.startswith("EleutherAI/pythia"):
        if "v0" in official_model_name:
            return PYTHIA_V0_CHECKPOINTS, "step"
        else:
            logging.warning(
                "Pythia models on HF were updated on 4/3/23! add '-v0' to model name to access the old models."
            )
            return PYTHIA_CHECKPOINTS, "step"
    elif official_model_name.startswith("NeelNanda/"):
        api = HfApi()
        files_list = api.list_repo_files(
            official_model_name,
            **utils.select_compatible_kwargs(kwargs, api.list_repo_files),
        )
        labels = []
        for file_name in files_list:
            match = re.match(r"checkpoints/.*_(\d*)\.pth", file_name)
            if match:
                labels.append(int(match.group(1)))
        if labels[-1] > 1e9:
            label_type = "token"
        else:
            label_type = "step"
        return labels, label_type
    elif official_model_name.startswith("LLM360/Amber"):
        return AMBER_CHECKPOINTS, "ckpt_"
    elif official_model_name.startswith("yu-takagi/jv3-7"):
        return LLM_JP_CHECKPOINTS, "step"
    #need fix
    elif official_model_name.startswith("allenai/OLMo-7B-0424-hf"):
        return OLMO_0424_CHECKPOINTS, "step"
    elif official_model_name.startswith("allenai/OLMo-7B-0724-hf"):
        return OLMO_0724_CHECKPOINTS, "step"
    elif official_model_name.startswith("allenai/OLMo-7B"):
        return OLMO_CHECKPOINTS, "step"
    else:
        raise ValueError(f"Model {official_model_name} is not checkpointed.")


# %% Loading state dicts
def get_pretrained_state_dict(
    official_model_name: str,
    cfg: HookedTransformerConfig,
    hf_model=None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Loads in the model weights for a pretrained model, and processes them to
    have the HookedTransformer parameter names and shapes. Supports checkpointed
    models (and expects the checkpoint info to be stored in the config object)

    hf_model: Optionally, a HuggingFace model object. If provided, we will use
        these weights rather than reloading the model.
    dtype: The dtype to load the HuggingFace model in.
    kwargs: Other optional arguments passed to HuggingFace's from_pretrained.
        Also given to other HuggingFace functions when compatible.
    """
    if "torch_dtype" in kwargs:
        dtype = kwargs["torch_dtype"]
        del kwargs["torch_dtype"]
    if Path(official_model_name).exists():
        official_model_name = str(Path(official_model_name).resolve())
        logging.info(f"Loading model from local path {official_model_name}")
    else:
        official_model_name = get_official_model_name(official_model_name)
    if official_model_name.startswith(NEED_REMOTE_CODE_MODELS) and not kwargs.get(
        "trust_remote_code", False
    ):
        logging.warning(
            f"Loading model {official_model_name} state dict requires setting trust_remote_code=True"
        )
        kwargs["trust_remote_code"] = True
    if (
        official_model_name.startswith("NeelNanda")
        or official_model_name.startswith("ArthurConmy")
        or official_model_name.startswith("Baidicoot")
    ):
        api = HfApi()
        repo_files = api.list_repo_files(
            official_model_name,
            **utils.select_compatible_kwargs(kwargs, api.list_repo_files),
        )
        if cfg.from_checkpoint:
            file_name = list(
                filter(lambda x: x.endswith(f"{cfg.checkpoint_value}.pth"), repo_files)
            )[0]
        else:
            file_name = list(filter(lambda x: x.endswith("final.pth"), repo_files))[0]
        state_dict = utils.download_file_from_hf(official_model_name, file_name, **kwargs)

        # Convert to dtype
        state_dict = {k: v.to(dtype) for k, v in state_dict.items()}

        if cfg.original_architecture == "neel-solu-old":
            state_dict = convert_neel_solu_old_weights(state_dict, cfg)
        elif cfg.original_architecture == "mingpt":
            state_dict = convert_mingpt_weights(state_dict, cfg)
        return state_dict
    else:
        if cfg.from_checkpoint:
            huggingface_token = os.environ.get("HF_TOKEN", None)
            if official_model_name.startswith("stanford-crfm"):
                hf_model = AutoModelForCausalLM.from_pretrained(
                    official_model_name,
                    revision=f"checkpoint-{cfg.checkpoint_value}",
                    torch_dtype=dtype,
                    token=huggingface_token,
                    **kwargs,
                )
            elif official_model_name.startswith("EleutherAI/pythia"):
                hf_model = AutoModelForCausalLM.from_pretrained(
                    official_model_name,
                    revision=f"step{cfg.checkpoint_value}",
                    torch_dtype=dtype,
                    token=huggingface_token,
                    **kwargs,
                )
            elif official_model_name.startswith("LLM360/Amber"):
                hf_model = AutoModelForCausalLM.from_pretrained(
                    official_model_name,
                    revision=f"ckpt_{cfg.checkpoint_value}",
                    torch_dtype=dtype,
                    token=huggingface_token,
                    **kwargs,
                )
            elif official_model_name.startswith("yu-takagi/jv3-7"):
                hf_model = AutoModelForCausalLM.from_pretrained(
                    official_model_name,
                    revision=f"step{cfg.checkpoint_value}",
                    torch_dtype=dtype,
                    token=huggingface_token,
                    **kwargs,
                )
            elif official_model_name.startswith("allenai/OLMo-7B-0424-hf"):
                hf_model = AutoModelForCausalLM.from_pretrained(
                    official_model_name,
                    revision=f"{cfg.checkpoint_value}",
                    torch_dtype=dtype,
                    token=huggingface_token,
                    **kwargs,
                )
            elif official_model_name.startswith("allenai/OLMo-7B-0724-hf"):
                hf_model = AutoModelForCausalLM.from_pretrained(
                    official_model_name,
                    revision=f"{cfg.checkpoint_value}",
                    torch_dtype=dtype,
                    token=huggingface_token,
                    **kwargs,
                )
            elif official_model_name.startswith("allenai/OLMo-7B"):
                hf_model = AutoModelForCausalLM.from_pretrained(
                    official_model_name,
                    revision=f"{cfg.checkpoint_value}",
                    torch_dtype=dtype,
                    token=huggingface_token,
                    **kwargs,
                )
            else:
                raise ValueError(f"Checkpoints for model {official_model_name} are not supported")
        elif hf_model is None:
            huggingface_token = os.environ.get("HF_TOKEN", None)
            if official_model_name in NON_HF_HOSTED_MODEL_NAMES:
                raise NotImplementedError("Model not hosted on HuggingFace, must pass in hf_model")
            elif "bert" in official_model_name:
                hf_model = BertForPreTraining.from_pretrained(
                    official_model_name,
                    torch_dtype=dtype,
                    token=huggingface_token,
                    **kwargs,
                )
            elif "t5" in official_model_name:
                hf_model = T5ForConditionalGeneration.from_pretrained(
                    official_model_name,
                    torch_dtype=dtype,
                    token=huggingface_token,
                    **kwargs,
                )
            else:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    official_model_name,
                    torch_dtype=dtype,
                    token=huggingface_token,
                    **kwargs,
                )

            # Load model weights, and fold in layer norm weights

        for param in hf_model.parameters():
            param.requires_grad = False

        if cfg.original_architecture == "GPT2LMHeadModel":
            state_dict = convert_gpt2_weights(hf_model, cfg)
        elif cfg.original_architecture == "GPTNeoForCausalLM":
            state_dict = convert_neo_weights(hf_model, cfg)
        elif cfg.original_architecture == "OPTForCausalLM":
            state_dict = convert_opt_weights(hf_model, cfg)
        elif cfg.original_architecture == "GPTJForCausalLM":
            state_dict = convert_gptj_weights(hf_model, cfg)
        elif cfg.original_architecture == "GPTNeoXForCausalLM":
            state_dict = convert_neox_weights(hf_model, cfg)
        elif cfg.original_architecture == "LlamaForCausalLM":
            state_dict = convert_llama_weights(hf_model, cfg)
        elif cfg.original_architecture == "BertForMaskedLM":
            state_dict = convert_bert_weights(hf_model, cfg)
        elif cfg.original_architecture == "T5ForConditionalGeneration":
            state_dict = convert_t5_weights(hf_model, cfg)
        elif cfg.original_architecture == "MistralForCausalLM":
            state_dict = convert_mistral_weights(hf_model, cfg)
        elif cfg.original_architecture == "MixtralForCausalLM":
            state_dict = convert_mixtral_weights(hf_model, cfg)
        elif cfg.original_architecture == "BloomForCausalLM":
            state_dict = convert_bloom_weights(hf_model, cfg)
        elif cfg.original_architecture == "GPT2LMHeadCustomModel":
            state_dict = convert_coder_weights(hf_model, cfg)
        elif cfg.original_architecture == "QWenLMHeadModel":
            state_dict = convert_qwen_weights(hf_model, cfg)
        elif cfg.original_architecture == "Qwen2ForCausalLM":
            state_dict = convert_qwen2_weights(hf_model, cfg)
        elif cfg.original_architecture == "PhiForCausalLM":
            state_dict = convert_phi_weights(hf_model, cfg)
        elif cfg.original_architecture == "Phi3ForCausalLM":
            state_dict = convert_phi3_weights(hf_model, cfg)
        elif cfg.original_architecture == "GemmaForCausalLM":
            state_dict = convert_gemma_weights(hf_model, cfg)
        elif cfg.original_architecture == "Gemma2ForCausalLM":
            state_dict = convert_gemma_weights(hf_model, cfg)
        elif cfg.original_architecture == "Olmo2ForCausalLM":
            state_dict = convert_olmo2_weights(hf_model, cfg)
        elif cfg.original_architecture == "OlmoForCausalLM" or cfg.original_architecture == "OLMoForCausalLM":
            state_dict = convert_olmo_weights(hf_model, cfg)
        else:
            raise ValueError(
                f"Loading weights from the architecture is not currently supported: {cfg.original_architecture}, generated from model name {cfg.model_name}. Feel free to open an issue on GitHub to request this feature."
            )

        return state_dict


def fill_missing_keys(model, state_dict):
    """Takes in a state dict from a pretrained model, and fills in any missing keys with the default initialization.

    This function is assumed to be run before weights are initialized.

    Args:
        state_dict (dict): State dict from a pretrained model

    Returns:
        dict: State dict with missing keys filled in
    """
    # Get the default state dict
    default_state_dict = model.state_dict()
    # Get the keys that are missing from the pretrained model
    missing_keys = set(default_state_dict.keys()) - set(state_dict.keys())
    # Fill in the missing keys with the default initialization
    for key in missing_keys:
        if "hf_model" in key:
            # Skip keys that are from the HuggingFace model, if loading from HF.
            continue
        if "W_" in key:
            logging.warning(
                "Missing key for a weight matrix in pretrained, filled in with an empty tensor: {}".format(
                    key
                )
            )
        state_dict[key] = default_state_dict[key]
    return state_dict


@dataclasses.dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


# Returns the configuration parameters of the model as a basic Config dataclass
def get_basic_config(model_name: str, **kwargs) -> Config:
    return Config(
        **{
            k: v
            for k, v in get_pretrained_model_config(model_name, **kwargs).to_dict().items()
            if k
            in [
                "d_model",
                "debug",
                "layer_norm_eps",
                "d_vocab",
                "init_range",
                "n_ctx",
                "d_head",
                "d_mlp",
                "n_heads",
                "n_layers",
            ]
        }
    )
