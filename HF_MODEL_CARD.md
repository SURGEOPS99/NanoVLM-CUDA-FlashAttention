---
language:
- en
base_model: HuggingFaceTB/SmolVLM-500M-Instruct
library_name: peft
pipeline_tag: image-text-to-text
datasets:
- mdwiratathya/SLAKE-vqa-english
tags:
- visual-question-answering
- medical-vqa
- qlora
- triton
- custom-cuda-attention
---

# NanoVLM-SLAKE: SmolVLM-500M Medical VQA QLoRA Adapter

This repository contains **PEFT/QLoRA adapter weights and the processor**, **not a complete base VLM**. The adapter is trained on the English SLAKE medical visual question answering dataset, starting from [`HuggingFaceTB/SmolVLM-500M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct).

The associated GitHub project contains the custom **Triton-JIT CUDA attention backend** named `nano_cuda_attn`, tests and the training notebook. For matching the original implementation, install that source code and register the attention backend **before** loading the base model. Inference with a different attention implementation may produce numerically different results.

## Reported evaluation (single-run, unverified independently)

- Dataset: SLAKE English VQA, reported test split of **1,061** examples
- Exact Match: **0.7455 (74.55%)**
- Token F1: **0.7864 (78.64%)**
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- Sequential inference: **1,237.1 ms/image**, ~**0.81 images/s** on that system
- Training peak PyTorch GPU allocation: **4.471 GiB**
- Training runtime: **40,692.4 s** (~11 h 18 min, 3 epochs)

These metrics are author-reported from a local run. They are not independent clinical diagnostics metrics and are not directly comparable to publications using different splits, prompts or normalization.

## Loading the adapter with the custom CUDA backend

Use the **local GitHub project directory**, which contains `nanoflash_cuda.py`:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from nanoflash_cuda import register_nanoflash

MODEL_REPO = "YOUR_HF_USERNAME/nanovlm-slake-500m-qlora"  # change this
BASE_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
ATTENTION_BACKEND = register_nanoflash()
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=DTYPE,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    quantization_config=quantization,
    device_map={"": 0},
    dtype=DTYPE,
    attn_implementation=ATTENTION_BACKEND,
)
model = PeftModel.from_pretrained(base, MODEL_REPO).eval()
processor = AutoProcessor.from_pretrained(MODEL_REPO)
```

To reproduce inference, format prompts and image inputs using the chat-template and processor methods in the project notebook; merely loading weights does not reproduce the evaluation pipeline.

## Intended use

Research, education, experimentation with resource-efficient VLM fine-tuning and custom GPU attention kernels.

**Not for medical diagnosis, triage, treatment planning, or clinical decision-making.** Performance on a VQA benchmark does not establish safety or robustness in clinical practice.

## Known limitations

- Finetuned on one dataset and evaluated once; clinical and cross-dataset generalization are not established.
- Limited sample size, answer-format sensitivity and unknown susceptibility to shortcut learning.
- Mixed precision and custom-kernel limitations; benchmark GPU was an NVIDIA laptop GPU.
- No verified speedup of custom attention over PyTorch SDPA.
- Requires the separately published base model and respects its licensing and usage terms.

## Training configuration (reported)

4-bit NF4 + QLoRA, LoRA rank 8 / alpha 16, 3 training epochs, micro-batch 2 and gradient accumulation 4. The notebook contains fuller configuration and evaluation details.

## Attribution

Base model: Hugging Face TB's SmolVLM-500M-Instruct. Dataset: SLAKE English VQA (`mdwiratathya/SLAKE-vqa-english`). Preserve upstream license and attribution when redistributing adapted artifacts.
