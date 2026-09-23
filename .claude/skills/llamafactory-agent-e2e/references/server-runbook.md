# 服务器操作手册

在服务器的 LlamaFactory 仓库根目录执行。先以当前 checkout 的 `pyproject.toml`、`examples/`、`src/llamafactory/hparams/` 和 `src/llamafactory/train/hyper_parallel/` 为准核对命令与参数；不要把另一版本的示例直接用于本次环境。需要一组具体命令时，可参考 [Qwen3 演示](qwen3-example.md)，但其中的模型、数据和路径都不是通用默认值。

## 1. 确认环境与安装

记录操作系统、Python、驱动/CANN、设备型号、可用显存与设备占用；用 `nvidia-smi` 或 `npu-smi info` 检查实际硬件。确认模型许可、下载权限、磁盘空间和预计训练预算。为实验创建隔离环境，先按设备供应商的版本矩阵安装匹配的 PyTorch 组件；NPU 额外核对 CANN 和 torch-npu。不要为了解决依赖冲突盲目升级 torch。

```bash
python -m pip install -e .
python -m pip check
llamafactory-cli version
python -c 'import torch, transformers, accelerate, peft; print(torch.__version__, transformers.__version__, accelerate.__version__, peft.__version__)'
```

单卡 QLoRA：CUDA 安装与 torch/CUDA 匹配的 bitsandbytes；NPU 按当前 `README_zh.md` 的 NPU 安装说明及 `requirements/npu.txt` 核对依赖，不能照搬 CUDA wheel。安装后执行 `python -c 'import bitsandbytes'` 并做短步数训练验证。

多卡 HyperParallel：安装与当前 LlamaFactory 集成匹配的 HyperParallel 包或检出的源码，确认安装过程没有替换已选定的 torch/torch-npu，然后检查集成入口：

```bash
python -m pip install hyper-parallel
python -m pip check
python -c 'from hyper_parallel.integration.llamafactory import HyperParallelArguments; print(HyperParallelArguments())'
```

记录 LlamaFactory 的 commit SHA；HyperParallel 若从源码安装则记录 SHA，若从包安装则记录包版本和来源。完整 `pip freeze` 只存于私有实验记录，不直接贴进公开教程。

## 2. 选模型、数据与训练方法

先核对模型是否被当前 LlamaFactory 支持、对应 template、权重大小与硬件预算；从 `examples/` 选同模型家族、同训练方法的最近示例。自定义数据按 `data/README_zh.md` 校验并登记到 `data/dataset_info.json`。在训练前固定不重叠的留出集、评测指标和少量固定问题。

单设备默认从 LoRA 示例加入 `quantization_bit: 4` 和已验证的量化方法，或直接使用匹配模型的 QLoRA 示例。保留 `finetuning_type: lora`；训练 YAML、adapter 输出、评测输出各用独立目录。

多设备先确认当前代码对目标模型、`use_hyper_parallel` 和训练方法的支持。选 full 或 LoRA 时同时核算显存、训练目标及保存/加载路径；full 不是多卡的固定默认值。从对应方法的示例出发，移除互斥的其他分布式后端配置，加入 `use_hyper_parallel: true`。基于仓库的 `examples/accelerate/fsdp2_config.yaml` 配置 FSDP2，核对 `num_processes`、精度和设备可见性。CP、EP、TP 参数只在当前 LlamaFactory 与 HyperParallel 代码都存在对应配置和模型实现、且已做专项验证时加入；不开启额外并行维度时只验证 FSDP2 路径。

## 3. 训练与检查

把复制并修改过的 YAML 放在本次运行目录，设置唯一的 `output_dir`。先用少量 `max_steps` 做烟测，确认模型加载、前向/反向、保存和重载；正式训练另用新输出目录，避免混淆烟测模型。单卡和多卡启动方式分别为：

```bash
llamafactory-cli train "$TRAIN_YAML"
accelerate launch --config_file "$FSDP2_YAML" src/train.py "$TRAIN_YAML"
```

运行前把 `TRAIN_YAML` 和 `FSDP2_YAML` 设为本次实际文件路径；单卡不使用第二条命令。多卡必须确认 Accelerate 使用 FSDP version 2，且训练 YAML 确实走 HyperParallel workflow。记录日志、运行时长、峰值显存、训练产物；保存后重新读取产物，不能只凭进程退出码判断成功。

## 4. 评测与推理

训练配置中设置独立 `eval_dataset` 或从训练数据中预留 `val_size`，启用 `do_eval: true`；检查产生的评测指标与样本数。若训练时未评测，另建评测配置，指向正确的基础模型、adapter 或完整模型输出，`do_train: false` 并使用独立 `output_dir`。当前 CLI 的 `eval` 子命令不是正式评测入口；使用 `llamafactory-cli train` 运行评测/预测 YAML。

从 `examples/inference/` 选择与模型和产物类型匹配的配置，确认训练与推理 template 一致。LoRA 检查 `adapter_name_or_path`；完整模型检查 `model_name_or_path`。不要仅凭训练方法推断 HyperParallel 保存产物的类型，应检查实际文件并完成一次加载。用相同生成参数记录基础模型与微调模型对固定问题的输出。

## 5. API 部署

先用经过推理验证的配置，以 LlamaFactory HuggingFace backend 启动本机 API；不在仍被训练占用的设备上部署。使用 `/v1/models` 和 `/v1/chat/completions` 完成功能验收。需要吞吐时再评估 vLLM/SGLang 对当前模型、adapter 与量化格式的支持。服务先绑定 `127.0.0.1`；远程访问优先使用 SSH 端口转发，对外开放须另行确认鉴权和网络策略。

## 6. 整理教程与脱敏

私有运行记录保留代码 SHA/包版本、设备与依赖版本、实际配置、命令、日志、指标和失败处理。公开教程只提取必要信息；检查并去除 `pip freeze` 中的私有包地址/凭据，以及 YAML 和日志中的用户名、绝对路径、内网地址、token、私有模型与数据集名称、敏感提示词或样本。未获得用户许可不要公开原始数据、权重、完整日志或机器信息。给每个结论标明“代码确认”“服务器实测”或“待验证”。
