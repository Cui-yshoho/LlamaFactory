#!/usr/bin/env python3
"""Prepare a tiny random text-path preflight for the Qwen3.8 workflow."""

from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path

import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3_5TextConfig


SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<unk>",
]
WORDS = [
    "system",
    "user",
    "assistant",
    "check",
    "the",
    "training",
    "pipeline",
    "return",
    "ok",
    "one",
    "plus",
    "two",
    "three",
]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _build_tokenizer(model_dir: Path, shard_multiple: int) -> PreTrainedTokenizerFast:
    tokens = SPECIAL_TOKENS + WORDS
    tokens += [f"unused_{index}" for index in range((-len(tokens)) % shard_multiple)]
    vocab = {token: index for index, token in enumerate(tokens)}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<unk>",
        additional_special_tokens=SPECIAL_TOKENS[3:7],
    )
    tokenizer.save_pretrained(model_dir)
    return tokenizer


def prepare(output_dir: Path, num_processes: int) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")

    model_dir = output_dir / "model"
    data_dir = output_dir / "data"
    result_dir = output_dir / "output"
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _build_tokenizer(model_dir, num_processes)
    num_heads = math.lcm(4, num_processes)
    config = Qwen3_5TextConfig(
        vocab_size=len(tokenizer),
        hidden_size=32 * num_heads,
        intermediate_size=64 * num_heads,
        num_hidden_layers=2,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=32,
        linear_num_key_heads=num_heads,
        linear_num_value_heads=num_heads,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=256,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    config.architectures = ["Qwen3_5ForCausalLM"]
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(model_dir)
    del model

    records = [
        {"instruction": "check the training pipeline", "input": "", "output": "ok"},
        {"instruction": "return one plus two", "input": "", "output": "three"},
    ]
    _write_json(data_dir / "train.json", records)
    _write_json(
        data_dir / "dataset_info.json",
        {
            "qwen38_no_weight_smoke": {
                "file_name": "train.json",
                "formatting": "alpaca",
                "columns": {"prompt": "instruction", "query": "input", "response": "output"},
            }
        },
    )

    multi_device = num_processes > 1
    _write_yaml(
        output_dir / "train.yaml",
        {
            "model_name_or_path": str(model_dir),
            "train_from_scratch": False,
            "trust_remote_code": False,
            "use_v1_kernels": True,
            "flash_attn": "auto",
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "full",
            "use_hyper_parallel": multi_device,
            "dataset": "qwen38_no_weight_smoke",
            "dataset_dir": str(data_dir),
            "template": "qwen3",
            "cutoff_len": 128,
            "max_samples": 2,
            "preprocessing_num_workers": 1,
            "dataloader_num_workers": 0,
            "output_dir": str(result_dir),
            "logging_steps": 1,
            "save_strategy": "no",
            "plot_loss": False,
            "overwrite_output_dir": True,
            "report_to": "none",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1.0e-5,
            "max_steps": 1,
            "bf16": True,
            "ddp_timeout": 1800,
        },
    )
    if multi_device:
        _write_yaml(
            output_dir / "fsdp2.yaml",
            {
                "compute_environment": "LOCAL_MACHINE",
                "debug": False,
                "distributed_type": "FSDP",
                "downcast_bf16": "no",
                "fsdp_config": {
                    "fsdp_version": 2,
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "fsdp_transformer_layer_cls_to_wrap": "Qwen3_5DecoderLayer",
                    "fsdp_cpu_ram_efficient_loading": True,
                    "fsdp_offload_params": False,
                    "fsdp_reshard_after_forward": True,
                    "fsdp_state_dict_type": "FULL_STATE_DICT",
                },
                "machine_rank": 0,
                "main_training_function": "main",
                "mixed_precision": "bf16",
                "num_machines": 1,
                "num_processes": num_processes,
                "rdzv_backend": "static",
                "same_network": True,
                "use_cpu": False,
            },
        )
    _write_json(
        output_dir / "scope.json",
        {
            "label": "pipeline-only",
            "downloads_pretrained_weights": False,
            "covers": [
                "Qwen3.5 text GDN",
                "full attention",
                "locally generated checkpoint loading",
                "Hyper FSDP2" if multi_device else "LlamaFactory native trainer",
                "optimizer step",
            ],
            "does_not_cover": [
                "Qwen3.8 pretrained checkpoint loading",
                "Qwen3.8 multimodal processor and template",
                "vision tower",
                "numerical alignment",
                "full-model memory capacity",
                "artifact reload or deployment",
            ],
        },
    )

    print(f"Prepared no-weight smoke run in {output_dir}")
    print("Launch from the LlamaFactory repository with one command matching the detected backend:")
    if multi_device:
        launch = shlex.join(
            [
                "accelerate",
                "launch",
                "--config_file",
                str(output_dir / "fsdp2.yaml"),
                "src/train.py",
                str(output_dir / "train.yaml"),
            ]
        )
        print(f"CUDA_VISIBLE_DEVICES=<ids> {launch}")
        print(f"ASCEND_RT_VISIBLE_DEVICES=<ids> {launch}")
    else:
        launch = shlex.join(["llamafactory-cli", "train", str(output_dir / "train.yaml")])
        print(f"CUDA_VISIBLE_DEVICES=<id> {launch}")
        print(f"ASCEND_RT_VISIBLE_DEVICES=<id> {launch}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    args = parser.parse_args()
    if args.num_processes < 1:
        parser.error("--num-processes must be positive")
    prepare(args.output_dir.resolve(), args.num_processes)


if __name__ == "__main__":
    main()
