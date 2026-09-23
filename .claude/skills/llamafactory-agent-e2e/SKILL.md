---
name: llamafactory-agent-e2e
description: Guide an Agent through a reproducible LlamaFactory SFT workflow from environment setup to training, evaluation, and API deployment. Use for end-to-end tutorials or server runs; prefer QLoRA on one device and HyperParallel FSDP2 on multiple devices when supported by the checked-out code.
---

# LlamaFactory 端到端训练教程

以服务器上实际检出的 LlamaFactory 和 HyperParallel 版本为准，完成「环境 → 安装 → SFT → 测试 → 部署」，并把实测结果整理成教程。通用流程见 [服务器操作手册](references/server-runbook.md)；只有需要 Qwen3 演示时才读 [Qwen3 示例](references/qwen3-example.md)。未跑通的步骤不得写成已验证。

## 工作规则

1. 先确认硬件、可用设备与显存、驱动/CANN、Python、模型、数据、训练目标和部署目标。不要占用已有任务的设备；没有计算设备时只准备配置，不声称完成运行。
2. 单设备优先考虑 4-bit QLoRA；多设备优先考虑 HyperParallel FSDP2 后端。并行后端不决定微调方式：结合当前代码支持、显存和目标选择 full 或 LoRA，先做短步数验证。不能跑通时说明限制，不悄悄切换后端。
3. 从当前仓库的配置定义、解析器和模型适配代码确认可用参数。不要根据另一版本的文档添加 CP、EP 或 TP 字段；并行规模大于 1 时还要验证模型适配及进程数约束。
4. 记录框架版本与代码 SHA，按设备供应商的兼容矩阵安装 PyTorch / torch-npu，再安装 LlamaFactory 和 HyperParallel。安装后核对依赖，避免附加依赖覆盖已选定的 torch 版本。
5. 从与所选模型和方法匹配的官方示例复制配置，使用独立的训练、评测与部署产物路径。自定义数据按仓库的数据说明校验并登记；训练与留出测试数据隔离。
6. 按设备可用、包可导入、配置可解析、短步数训练、正式训练、独立评测、推理与 API 请求逐步验证。保存失败现场；不把训练 loss 或 HTTP 200 当作效果验证。
7. 服务先绑定本机地址。对外开放前确认鉴权和网络策略；使用 vLLM/SGLang 前核对模型与量化格式兼容性。
8. 原始配置、`pip freeze`、日志和输出只保存在受控环境。发布前删除凭据、内网地址、用户名/绝对路径、私有模型与数据标识、敏感提示词；仅发布必要的版本、脱敏配置、命令和指标，并标明“代码确认”“服务器实测”或“待验证”。

## 执行顺序

按 [服务器操作手册](references/server-runbook.md) 执行并记录结果。需要下载大模型、运行长时间训练或开放端口时先确认资源预算与目标。教程仅包含实际验证的组合；具体示例不得冒充所有模型的默认配置。
