# Qwen3.8 GPU/NPU server runbook

Resolve paths and device IDs from the active server. Use one backend per environment.

This is an execution runbook for the agent. Perform the checks and setup directly; do not hand these commands or implementation choices to a beginner unless they ask. A training request authorizes ordinary user-space inspection, run-file creation, smoke testing, and training. Obtain confirmation only for large downloads, gated/license-restricted data, privileged system changes, destructive cleanup, a material method/model change, or external service exposure.

## Detect the backend

```bash
nvidia-smi  # CUDA GPU
npu-smi info  # Ascend NPU
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Do not choose a backend merely because its command exists. Confirm the matching PyTorch runtime can allocate a small tensor on the selected device.

If CUDA and NPU runtimes are both active, do not launch from that mixed environment. Create or select a backend-specific environment in a user-writable location. `device_type` is a Hyper strategy field supplied through `hyper_parallel_args`, not a top-level LlamaFactory YAML key; do not use it to mask a contaminated environment.

## Install

Install the accelerator-compatible PyTorch stack first, then the current LlamaFactory checkout.

- CUDA: match the NVIDIA driver, CUDA-enabled PyTorch, and ordinary Triton. Do not install `torch-npu` or `triton-ascend`.
- Ascend: match CANN, PyTorch, and `torch-npu`; source the selected CANN `set_env.sh` in every process. Keep `triton-ascend`, not ordinary Triton. Do not modify host driver or firmware.

Then validate the common stack:

```bash
python -m pip install -e .
python -m pip check
llamafactory-cli version
python -c 'import torch, transformers, accelerate, peft; print(torch.__version__, transformers.__version__, accelerate.__version__, peft.__version__)'
```

Environment setup is complete only when the active Python and pip resolve to the selected isolated environment, the current checkout imports, `pip check` has no selected-stack conflict, the CLI starts, and a tensor operation succeeds on the chosen device. If the installed driver, CUDA, or CANN cannot support a compatible user-space stack, stop with the exact incompatibility and required administrator action; do not attempt a broad system upgrade.

For CUDA, inspect `torch.version.cuda` and allocate a CUDA tensor. For Ascend, import `torch_npu`, inspect its version, and allocate an NPU tensor. Single-device QLoRA also requires the backend-compatible bitsandbytes build. Multi-device training requires `hyper_parallel` and all imports under `src/llamafactory/train/hyper_parallel/`.

The editable LlamaFactory install does not install HyperParallel. For a multi-device route, reuse a compatible installed package or install the official wheel/source revision required by the current checkout, then record its version or commit and import the LlamaFactory Hyper workflow. Do not guess an unpinned package version when the checkout declares no compatible source; report that dependency as unresolved before training.

Record the exact accelerator model or SoC, not only "GPU" or "NPU", and select its compatible software stack. Do not transfer a server-tested label between hardware generations; each untested target must pass the model load, required operator backward, and optimizer-step smoke gates first.

Classify `pip check` failures. Selected-stack conflicts block the run; unrelated managed-platform conflicts should be recorded rather than repaired broadly. Never disable TLS verification to download packages or models.

## Model and data

For model weights, search in this order: a user-supplied path, shared model directories such as `/cache` when present, the Hugging Face cache, and the ModelScope cache. Validate `config.json`, the safetensors index, every referenced shard, tokenizer, and processor files. If the snapshot is complete, use that path directly without downloading or copying it. If it is absent or incomplete, confirm the large transfer, choose one cache location with enough space, and use an official resumable client, for example `hf download Qwen/Qwen3.8-27B --local-dir <cache-dir>` or `modelscope download --model Qwen/Qwen3.8-27B --local_dir <cache-dir>`. Pin and record the resolved revision when the service supports it.

For a local dataset path, inspect it in place without modification. For a dataset URL or Hub ID, use its official client when available, pin the resolved revision, support resumable caching, and record provenance, license/access status, file hashes or manifest, and row counts. Do not execute remote dataset scripts or upload local data. Keep raw downloads immutable; place conversions and deterministic train/evaluation splits under the run directory. Validate prepared files with `data/README.md` or `data/README_zh.md` and register them in `data/dataset_info.json`.

Create `llamafactory_runs/<run-id>/` for copied YAMLs, commands, logs, and reports. Keep model cache and `output_dir` separate. Record every difference from the exact Qwen3.8 example.

## Launch

The smoke test must cover an optimizer update, save, and reload. Use a fresh output directory for a new formal run. For recovery, inspect the failure and newest complete checkpoint, preserve the original run files, set `resume_from_checkpoint` to that explicit path, and disable `overwrite_output_dir`. Restart into a new output directory when optimizer, scheduler, RNG, or trainer state is incomplete or the original effective configuration cannot be reproduced.

```bash
# One CUDA GPU
CUDA_VISIBLE_DEVICES=<id> \
  llamafactory-cli train llamafactory_runs/<run-id>/train.yaml

# One Ascend NPU
ASCEND_RT_VISIBLE_DEVICES=<id> \
  llamafactory-cli train llamafactory_runs/<run-id>/train.yaml

# Multiple CUDA GPUs
CUDA_VISIBLE_DEVICES=<id0,id1,...> accelerate launch \
  --config_file llamafactory_runs/<run-id>/fsdp2.yaml \
  src/train.py llamafactory_runs/<run-id>/train.yaml

# Multiple Ascend NPUs
ASCEND_RT_VISIBLE_DEVICES=<id0,id1,...> accelerate launch \
  --config_file llamafactory_runs/<run-id>/fsdp2.yaml \
  src/train.py llamafactory_runs/<run-id>/train.yaml
```

For long jobs, use the server's scheduler when available; otherwise use a detached supervisor such as `tmux`, `systemd --user`, or `nohup`. Do not enable blind automatic restart. Record the job ID or PID, exact command, environment activation, and log path. After detaching, confirm the real rank processes remain alive and reach an optimizer step. Declare completion only after final metrics are written and no workers remain; a launcher process exiting is not completion. On failure, validate an explicit checkpoint before resuming. Monitor loss, step, and device memory. Prefer `eval_dataset` or `val_size`; the interactive CLI is not quantitative evaluation.

## Reload and local API

Use the same template and generation settings for base and fine-tuned comparisons. QLoRA loads the base plus adapter; full SFT loads its trained model directly. Re-evaluate device count and serving engine for inference: a distributed training topology does not imply the same deployment topology, and HyperParallel training is not automatically the API serving backend.

```bash
API_HOST=127.0.0.1 API_PORT=8000 API_MODEL_NAME=<name> \
  llamafactory-cli api llamafactory_runs/<run-id>/inference.yaml

curl -sS http://127.0.0.1:8000/v1/models
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<name>","messages":[{"role":"user","content":"<held-out-prompt>"}],"max_tokens":128}'
```

Inspect both JSON responses. If `API_KEY` is set, send a Bearer token without storing its value in YAML, logs, or documentation.
