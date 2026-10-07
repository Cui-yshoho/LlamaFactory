---
name: qwen38-sft
description: Run Qwen/Qwen3.8-27B SFT with LlamaFactory on CUDA GPU or Ascend NPU from environment and dataset acquisition through training, evaluation, and local API validation. Use one-device QLoRA or multi-device HyperParallel FSDP2. Do not use for other models.
---

# Qwen3.8 SFT

Drive the existing LlamaFactory CLI. Do not build a separate Agent runtime. Treat the current checkout, its official examples, and installed packages as the source of truth.

## Beginner contract

A beginner may provide a local dataset path, a dataset URL/Hub ID, or only the intended task and ask to train Qwen3.8-27B. Take ownership of backend and device discovery, isolated Python environment setup, framework installation, dataset acquisition and preparation, model/data validation, configuration, smoke testing, formal training, evaluation, artifact reload, and loopback API validation. Infer facts from the server instead of asking the user to identify CUDA/NPU, card models, memory, package versions, or data format.

Do not make the user operate the terminal or choose low-level settings. Give short progress updates and the recommended route, not the internal checklist or a command tutorial, unless commands are requested. Ask only when an action needs new authority or cannot be inferred safely: choosing training data when the task is unspecified, a large model/dataset download, a gated or license-restricted dataset, a privileged driver/CANN change, insufficient resources requiring a different method/model, destructive cleanup, or non-loopback API exposure.

If the user asks only for a plan, simulation, or explanation, do not probe the server, data path, environment, or devices. Read only the skill guidance needed to answer, state that execution evidence is pending, and keep the response concise. Do not retry blocked probes in simulation mode.

## Route

1. Detect exactly one backend: CUDA GPU or Ascend NPU. Never mix their packages or visibility variables in one environment. When both are usable, choose by verified free capacity, compatible existing environment, model availability, and route support; do not hard-code a vendor preference.
2. After checking memory and model capacity, use one visible accelerator for 4-bit bitsandbytes QLoRA and multiple visible accelerators for HyperParallel + Accelerate FSDP2 full SFT.
3. These are workflow defaults, not permission to override an explicitly requested method. If the requested method and available capacity disagree, stop and present the mismatch.
4. Read [server-runbook.md](references/server-runbook.md) only when executing on a server. Read [qwen3-8-27b.md](references/qwen3-8-27b.md) when deriving or validating the model route. A simulation normally needs neither reference.

Do not silently replace the model, method, backend, or hardware route when a gate fails. Ask one compact question only for an unresolved choice that materially changes the run.

## Rules

1. Start from the exact Qwen3.8 example. Copy it into the run directory; never edit the repository example in place. Record every changed field in the run report and summarize only material choices to the user.
2. Inspect `config.json`, the LlamaFactory registry, and the relevant loader or adapter instead of inferring architecture from the model name.
3. Check process ownership and memory before selecting devices. Pin the same devices for all later commands.
4. Do not invent configuration keys. Resolve every field against the current LlamaFactory dataclasses or the referenced Hyper strategy schema.
5. Keep immutable base weights, run records, and outputs separate:

   ```text
   llamafactory_runs/<run-id>/  # YAML, command, logs, report
   saves/<model>/<method>/...   # checkpoints or adapters
   <model cache/path>           # base weights
   ```

6. Accept local data, a dataset URL, or a Hub dataset ID. Keep downloaded raw data immutable and separate from converted data. Record source, revision, license/access status, counts, and integrity evidence; never execute remote dataset code without explicit trust. Register prepared data in `data/dataset_info.json`. If no validation set exists, create a deterministic split in the run directory so evaluation samples are excluded from training.
7. A checkpoint-based training smoke test covers load, forward, backward, optimizer update, save, and reload. The explicitly requested no-weight preflight is narrower and must be labeled as such. A successful launcher exit alone is insufficient.
8. Keep the same template through train, evaluation, inference, export, and serving. QLoRA reloads the base plus `adapter_name_or_path`; full SFT reloads its output directly.
9. Bind API services to `127.0.0.1` unless networking and authentication are explicitly approved.
10. Label results as code-confirmed, server-tested, or pending. Do not imply that a backend was tested without its hardware.
11. Derive task-level acceptance checks from the user's goal before training. Loss stability is necessary but not sufficient; evaluate held-out task behavior and safety boundaries.

## Workflow

### 1. Inspect and install

Record the repository SHA, Python/framework versions, accelerator stack, occupancy, model availability, and disk budget. Reuse a compatible isolated environment when possible; otherwise create one in a user-writable location without changing the host driver, firmware, CUDA, or CANN installation. Install only the dependencies for the detected backend and selected route, then rerun imports and `pip check`.

### 2. Acquire and validate model/data

Resolve a supplied dataset path, URL, or Hub ID. If none is supplied, ask one task-level question before selecting formal training data; repository demo data may be used only for smoke testing. Before downloading model weights, search the supplied path, shared model directories, and existing Hub caches. Reuse a complete snapshot in place; do not download or copy it again. Download only missing content into a resumable cache after confirmation. If the user explicitly requests a no-weight quick check, do not fetch a checkpoint and follow the limited preflight in the model reference. Preserve raw data snapshots, convert only into the run directory, and validate schema, duplicates, leakage, empty records, and token-length distribution against the repository data guide. Verify every model shard in `model.safetensors.index.json` exists, is nonzero, and opens with `safetensors.safe_open`.

### 3. Derive configuration

Copy the exact example to `llamafactory_runs/<run-id>/`. Record model, data, template, method, length, batch/accumulation, optimizer, precision, duration, output, backend, and distributed settings. If the user's request already authorizes training, proceed after the smoke test without asking them to approve inferred low-level settings.

For multiple devices, retain LlamaFactory's entrypoint and trainer, set `use_hyper_parallel: true`, and launch with Accelerate FSDP2. Do not invoke a standalone Hyper trainer. Leave CP and EP at 1 unless explicitly requested and verified for the exact model.

Treat a requested device-count change as a topology change. Recheck that exact number of free compatible devices, pin their IDs, set Accelerate `num_processes` to the count, and let FSDP2 use the resulting world size. Recalculate the effective global batch (`per_device_train_batch_size × gradient_accumulation_steps × world_size`): preserve the previous value when the user asks only to change device count, or report the intentional batch change when exact preservation is impossible. Regenerate the run configuration and repeat the optimizer-step smoke test before formal training.

### 4. Smoke and train

Run the smallest smoke test that completes an optimizer update and save/reload. Immediately before the formal launch, recheck the pinned devices and disk budget; do not take a device that became busy. Use a fresh output directory for a new formal run. Resume only from an explicit, complete checkpoint with the original effective configuration; never combine resume with output overwrite or configure a supervisor to restart blindly. Launch long jobs through an available detached supervisor, record its job/PID, command, environment activation, and log, then verify the real worker survives the launcher session and reaches a training step. Report actual step, loss, and accelerator memory; never interpret a detached launcher exit as training completion.

### 5. Evaluate and serve

Evaluate held-out data against the recorded task-level checks, reload the artifact, and run fixed prompts with fixed generation settings. Export only when requested. Choose serving devices and engine from the artifact format, current backend support, and inference memory; do not assume the training topology or HyperParallel is the serving topology. After direct inference succeeds, start the local LlamaFactory API and verify its process health plus `/v1/models` and `/v1/chat/completions` response bodies.

### 6. Report

Summarize code SHA, backend and hardware count, environment, model/data, effective config and diff, commands, results, artifacts, API result, fixes, and limitations. Remove credentials, internal addresses, usernames, private paths, and sensitive samples.
