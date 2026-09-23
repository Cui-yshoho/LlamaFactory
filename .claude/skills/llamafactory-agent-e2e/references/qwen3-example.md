# Qwen3 演示配置

本页仅演示如何套用当前仓库中的示例；不是对所有模型或设备的默认配置。示例模型 `Qwen/Qwen3-4B-Instruct-2507`、`qwen3_nothink` 和 `identity,alpaca_en_demo` 均来自对应 LlamaFactory 示例。正式实验应替换为获准使用的模型、训练集和未参与训练的评测集，并先核算显存。

## 单卡 CUDA QLoRA

从 `examples/train_qlora/qwen3_lora_sft_otfq.yaml` 复制 `train.yaml` 到独立运行目录。保留 `finetuning_type: lora`、`quantization_bit: 4`、`quantization_method: bnb`，修改模型、数据、`output_dir`、精度与步数。NPU 则参考 `examples/train_qlora/qwen3_lora_sft_bnb_npu.yaml` 和当前 NPU 安装说明，不能复用 CUDA 的 bitsandbytes 安装方式。

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train runs/qwen3-demo-qlora/train.yaml
```

此命令中的设备号、目录和 YAML 需要先按实际服务器创建与修改。先短步数烟测，再用不同输出目录正式训练。

## 多卡 HyperParallel FSDP2

完整参数演示可从 `examples/train_full/qwen3_full_sft.yaml` 复制，移除其中的 `deepspeed` 字段，加入 `use_hyper_parallel: true`，再修改模型、数据和独立 `output_dir`。这是 full 示例，不意味着 HyperParallel 只能或总应使用 full；如要改 LoRA，先检查当前代码支持并实测保存/加载。复制 `examples/accelerate/fsdp2_config.yaml`，确认 `distributed_type: FSDP`、`fsdp_config.fsdp_version: 2` 和实际进程数。不要照搬其他分支的 CP/EP/TP 字段。

```bash
accelerate launch --config_file runs/qwen3-demo-hp/fsdp2_config.yaml src/train.py runs/qwen3-demo-hp/train.yaml
```

确认 `src/llamafactory/train/tuner.py` 转到 HyperParallel workflow，且 trainer 使用 Accelerate FSDP2；仅 FSDP2 跑通不能证明 CP/EP/TP 已跑通。

## 测试与部署

从 `examples/inference/qwen3_lora_sft.yaml` 或 `examples/inference/qwen3_full_sft.yaml` 复制与实际产物匹配的推理 YAML；分别检查 `adapter_name_or_path` 或 `model_name_or_path`。训练前预留独立评测集，评测后再检查固定问题的输出。以下是单卡 LoRA 的 API 示例；多卡 full 产物改用自己的 inference YAML：

```bash
API_HOST=127.0.0.1 API_PORT=8000 API_MODEL_NAME=qwen3-demo llamafactory-cli api runs/qwen3-demo-qlora/inference.yaml
```

在另一终端：

```bash
curl -sS http://127.0.0.1:8000/v1/models
curl -sS http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen3-demo","messages":[{"role":"user","content":"请用一句话介绍你自己。"}],"max_tokens":64}'
```

如设置 `API_KEY`，请求须带 Bearer token；不要把真实 token 写进教程。
