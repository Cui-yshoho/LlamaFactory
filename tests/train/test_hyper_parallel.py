# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hyper_parallel.core.optimizer import ChainedOptimizer

from llamafactory.train.hyper_parallel.loader import load_hyper_parallel_model
from llamafactory.train.hyper_parallel.trainer import (
    HyperParallelTrainer,
    _create_hyper_muon_optimizer,
    _shard_inputs_for_cp,
)
from llamafactory.train.hyper_parallel.workflow import _prepare_hp_args


def test_create_hyper_muon_optimizer_preserves_llamafactory_parameter_split():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 4)
            self.proj = torch.nn.Linear(4, 4)
            self.norm = torch.nn.LayerNorm(4)
            self.lm_head = torch.nn.Linear(4, 8, bias=False)

    model = _Model()
    training_args = SimpleNamespace(
        learning_rate=3.0e-4,
        weight_decay=0.1,
        adam_beta1=0.8,
        adam_beta2=0.95,
        adam_epsilon=1.0e-8,
    )
    expected_optimizer = ChainedOptimizer(
        model,
        {"adamw": torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)},
    )

    with patch(
        "llamafactory.train.hyper_parallel.trainer.get_hyper_optimizer",
        return_value=expected_optimizer,
    ) as get_optimizer:
        optimizer = _create_hyper_muon_optimizer(model, training_args)

    assert isinstance(optimizer, torch.optim.Optimizer)
    assert optimizer.optimizer is expected_optimizer
    call_kwargs = get_optimizer.call_args.kwargs
    assert call_kwargs["muon_params"][0]["params"] == [model.proj.weight]
    assert set(call_kwargs["adamw_params"][0]["params"]) == {
        model.embed_tokens.weight,
        model.proj.bias,
        model.norm.weight,
        model.norm.bias,
        model.lm_head.weight,
    }
    assert call_kwargs["muon_kwargs"] == {"muon_lr": 3.0e-4, "muon_weight_decay": 0.1}
    assert call_kwargs["adamw_kwargs"] == {
        "adamw_lr": 3.0e-4,
        "adamw_weight_decay": 0.1,
        "adamw_betas": (0.8, 0.95),
        "adamw_eps": 1.0e-8,
    }
    optimizer.param_groups[0]["lr"] = 1.0e-4
    assert expected_optimizer.chained_optimizers[0].param_groups[0]["lr"] == 1.0e-4


def test_cp_batch_adapter_keeps_global_mask_and_generates_local_positions():
    class _CPMesh:
        @staticmethod
        def size():
            return 2

        @staticmethod
        def get_local_rank():
            return 1

    inputs = {
        "input_ids": torch.arange(8).unsqueeze(0),
        "labels": torch.arange(8).unsqueeze(0),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    }

    sharded = _shard_inputs_for_cp(inputs, _CPMesh())

    torch.testing.assert_close(sharded["input_ids"], torch.arange(4, 8).unsqueeze(0))
    torch.testing.assert_close(sharded["position_ids"], torch.arange(4, 8).unsqueeze(0))
    torch.testing.assert_close(sharded["attention_mask"], inputs["attention_mask"])


def test_accelerate_patches_skip_torch_auto_wrap_for_prepared_hyper_model():
    class _FSDPPlugin:
        def __init__(self):
            self.calls = []

        def set_auto_wrap_policy(self, model):
            self.calls.append(model)

    plugin = _FSDPPlugin()
    original_clip = object()
    trainer = object.__new__(HyperParallelTrainer)
    trainer.accelerator = SimpleNamespace(
        state=SimpleNamespace(fsdp_plugin=plugin),
        clip_grad_norm_=original_clip,
    )
    trainer._orig_accelerator_clip_grad_norm = original_clip
    trainer._orig_fsdp2_prepare_model = None
    trainer._orig_fsdp2_set_auto_wrap_policy = None
    trainer._accelerator_patches_active = False
    prepared_models = []
    trainer._prepare_model_for_hyper_parallel = lambda model: prepared_models.append(model) or model

    trainer._activate_accelerator_patches()
    model = object()
    plugin.set_auto_wrap_policy(model)
    trainer._restore_accelerator_patches()

    assert prepared_models == [model]
    assert plugin.calls == []
    assert plugin.set_auto_wrap_policy == trainer._orig_fsdp2_set_auto_wrap_policy


def test_hyper_model_loader_uses_meta_then_parallelize_path():
    config = SimpleNamespace(model_type="test")
    model = torch.nn.Linear(2, 2, device="meta")
    model.config = config

    class _AutoModel:
        @staticmethod
        def from_config(input_config, trust_remote_code=False):
            assert input_config is config
            assert trust_remote_code is False
            return model

    model_args = SimpleNamespace(
        model_name_or_path="checkpoint",
        train_from_scratch=False,
        trust_remote_code=False,
        use_kt=False,
        use_v1_kernels=False,
        print_param_status=False,
    )
    finetuning_args = SimpleNamespace(stage="sft")
    hp_args = SimpleNamespace(activation_mode="none", activation_swap_inputs=True)
    distributed_setup = object()

    with (
        patch("llamafactory.train.hyper_parallel.loader._get_init_kwargs", return_value={}),
        patch("llamafactory.train.hyper_parallel.loader.load_config", return_value=config),
        patch("llamafactory.train.hyper_parallel.loader.patch_config"),
        patch("llamafactory.train.hyper_parallel.loader.apply_liger_kernel"),
        patch("llamafactory.train.hyper_parallel.loader._get_model_class", return_value=_AutoModel),
        patch("llamafactory.train.hyper_parallel.loader._get_no_init_weights", return_value=nullcontext),
        patch("llamafactory.train.hyper_parallel.loader.init_empty_weights", return_value=nullcontext()),
        patch("llamafactory.train.hyper_parallel.loader.patch_model"),
        patch("llamafactory.train.hyper_parallel.loader.register_autoclass"),
        patch("llamafactory.train.hyper_parallel.loader.init_adapter", return_value=model),
        patch("llamafactory.train.hyper_parallel.loader.parallelize_model", return_value=model) as parallelize,
        patch("llamafactory.train.hyper_parallel.loader._validate_conv3d_compatibility"),
    ):
        result = load_hyper_parallel_model(
            tokenizer=object(),
            model_args=model_args,
            finetuning_args=finetuning_args,
            distributed_setup=distributed_setup,
            hp_args=hp_args,
            is_trainable=True,
        )

    assert result is model
    parallelize.assert_called_once_with(
        model,
        distributed_setup,
        pretrained_path="checkpoint",
        activation_checkpoint=None,
        swap_inputs=False,
    )


def _make_hyper_workflow_args(finetuning_type="full"):
    finetuning_args = SimpleNamespace(
        hyper_parallel_args=None,
        hyper_parallel_cp_size=1,
        hyper_parallel_ep_size=1,
        hyper_parallel_efsdp_size=1,
        hyper_parallel_token_dispatcher="all_to_all",
        finetuning_type=finetuning_type,
        use_asft_loss=False,
    )
    model_args = SimpleNamespace(
        disable_gradient_checkpointing=False,
        quantization_bit=None,
        use_unsloth=False,
        mixture_of_depths=None,
    )
    return finetuning_args, model_args


def test_prepare_hp_args_accepts_supported_full_tuning():
    finetuning_args, model_args = _make_hyper_workflow_args()

    with patch("hyper_parallel.integration.llamafactory.utils.dist.get_world_size", return_value=1):
        hp_args = _prepare_hp_args(finetuning_args, model_args)

    assert hp_args.cp_size == 1
    assert hp_args.ep_size == 1


def test_prepare_hp_args_rejects_parameter_efficient_tuning():
    finetuning_args, model_args = _make_hyper_workflow_args(finetuning_type="lora")

    try:
        _prepare_hp_args(finetuning_args, model_args)
    except ValueError as exc:
        assert "requires full fine-tuning" in str(exc)
    else:
        raise AssertionError("Expected HyperParallel to reject LoRA tuning.")
