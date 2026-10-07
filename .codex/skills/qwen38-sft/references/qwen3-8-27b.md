# Qwen3.8-27B compatibility

This route is only for `Qwen/Qwen3.8-27B`, not Qwen3-8B or Qwen3.5-27B.

## Architecture gates

The released configuration declares:

```text
model_type: qwen3_5
architectures: [Qwen3_5ForConditionalGeneration]
language_model_only: false
```

LlamaFactory registers it as multimodal with template `qwen3_8`. Hyper resolves the `Qwen3_5ForConditionalGeneration` adapter, and FSDP2 wraps `Qwen3_5DecoderLayer,Qwen3_5VisionBlock`. The GPU and NPU kernel patches are selected by the active torch backend; do not change the model identity or template.

Before training, verify the downloaded config, all indexed safetensors shards, tokenizer and processor loading, and meta model construction.

## No-weight quick check

When the user explicitly forbids downloading pretrained weights, keep this separate from the normal smoke test:

1. With no local Qwen3.8 metadata, run only environment imports, accelerator allocation, dependency checks, YAML/dataclass validation, and distributed initialization. Exact tokenization and model execution remain pending.
2. For a pipeline-only execution check, run `python scripts/prepare_no_weight_smoke.py --output-dir <run-dir> --num-processes <devices>` from the skill directory. Here no-weight means no pretrained checkpoint download: the script creates a tiny local tokenizer and random two-layer Qwen3.5 text checkpoint containing both GDN and full attention, plus test data and launch configs. It uses the text-only `qwen3` template because the Qwen3.8 template requires the pretrained multimodal processor. It uses the native LlamaFactory trainer for one device and Hyper FSDP2 for multiple devices. Select the printed CUDA or Ascend command for the detected backend, require one finite optimizer step, then remove the generated output directory after recording the result.
3. Label the result `pipeline-only`. It does not validate the Qwen3.8 processor/template, pretrained checkpoint loading, numerical alignment, task quality, full-model memory capacity, artifact reload, or deployment. Do not promote it to a successful Qwen3.8 training smoke test.

This text-path smoke does not hard-code a CUDA or Ascend implementation: LlamaFactory selects the installed backend and `flash_attn: auto` selects an available attention path. A finite optimizer step on the target machine is still required before calling that hardware tested. On multiple devices the smoke also reaches the Hyper adapter used by Qwen3.8, but it does not cover the Qwen3.8 vision tower. Prefer an already complete shared checkpoint when available because using it in place provides stronger validation without another download.

## One device: QLoRA

Start from `examples/train_qlora/qwen3_8_lora_sft_bnb.yaml`. It uses 4-bit bitsandbytes quantization, LoRA, and `template: qwen3_8`.

Confirm that the backend-compatible bitsandbytes build imports and passes a small tensor test. Copy the YAML into the run directory and change only approved data, length, duration, accumulation, and output fields.

```bash
# CUDA
CUDA_VISIBLE_DEVICES=<id> \
  llamafactory-cli train llamafactory_runs/<run-id>/train.yaml

# Ascend
ASCEND_RT_VISIBLE_DEVICES=<id> \
  llamafactory-cli train llamafactory_runs/<run-id>/train.yaml
```

Reload the adapter with `model_name_or_path: Qwen/Qwen3.8-27B` and `adapter_name_or_path: <output_dir>`.

## Multiple devices: HyperParallel FSDP2 full SFT

Start from `examples/train_full/qwen3_8_full_sft_hyper_fsdp2.yaml` and `examples/accelerate/fsdp2_config_qwen35.yaml`. Retain:

```yaml
model_name_or_path: Qwen/Qwen3.8-27B
template: qwen3_8
stage: sft
finetuning_type: full
use_hyper_parallel: true
```

For this single-host route, set `num_processes` to the visible GPU or NPU count. Leave CP and EP at 1; this dense-model route validates Hyper FSDP2 only. Qwen3.8 uses Hyper's `qwen3_5` adapter, which contains CP support, but CP is outside this workflow until the exact model and selected backend pass a separate optimizer-step and accuracy validation. EP does not apply to this dense architecture. Hyper resolves `device_type: auto` from the active torch backend.

```bash
# CUDA
CUDA_VISIBLE_DEVICES=<id0,id1,...> accelerate launch \
  --config_file llamafactory_runs/<run-id>/fsdp2.yaml \
  src/train.py llamafactory_runs/<run-id>/train.yaml

# Ascend
ASCEND_RT_VISIBLE_DEVICES=<id0,id1,...> accelerate launch \
  --config_file llamafactory_runs/<run-id>/fsdp2.yaml \
  src/train.py llamafactory_runs/<run-id>/train.yaml
```

Confirm that the log enters the Hyper workflow, every rank updates, and the full-model output reloads without `adapter_name_or_path`.

## Acceptance

Both routes require held-out evaluation, fixed-prompt comparison, artifact reload, direct inference, and a loopback API request. Record backend, package versions, peak memory, and runtime. Do not compare QLoRA and full SFT as equivalent optimization methods, and do not report one backend as tested from results obtained on the other.
