# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Chinese span semantics for name-based lookup in the offline viewer.

Descriptions live in this catalog, rather than in every collected span. The
optional processor remains available for legacy callers only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DESCRIPTION_ATTRIBUTE = "relax.span.description"
ROLLOUT_IDS = (
    "G：全局提示词组编号，同组样本共享提示词。\nS：全局样本编号，不是组内编号。\n两者均从 0 开始，跨 Step 连续递增。"
)

DESCRIPTIONS = {
    "train.wait_rollout": "训练前等待 Rollout 数据及共卡服务就绪。",
    "train.wait_rollout_offload": "等待 Rollout 释放 GPU 显存。",
    "train.wait_peer": "等待共卡服务完成工作或释放 GPU 显存。",
    "train.service_finalize": "清理已消费的数据分区、上报进度并推进 Actor Step。",
    "train.advantages": "计算训练样本的优势与回报。",
    "weights.backup": "保存本地 Actor 权重及可选的参考权重副本。",
    "weights.prepare": "为权重传输准备推理消费端及通信组。",
    "weights.finalize": "释放传输资源并同步各训练 Rank。",
    "weights.onload_rollout": "权重传输后恢复 Rollout KV Cache 及已配置的奖励模型。",
    "service.allocate": "部署前准备组件服务并分配其 Placement Group。",
    "service.init.resources": "分配 Placement Group、等待资源并确定实际 GPU 顺序。",
    "controller.init.health": "检查控制面资源并创建健康管理系统。",
    "controller.init.checkpoint_service": "部署用于权重同步的分布式 Checkpoint 协调服务。",
    "rollout.init.checkpoint_prepare": "确定推理 Checkpoint 来源并应用配置的模型加载策略。",
    "job.shutdown": "结束作业观测前，尝试正常清理控制器、推理服务、缓存代理及 Ray 资源。",
    "controller.shutdown": "协调关闭各组件服务及其管理的运行时资源。",
    "job.init.runtime": "加载运行配置并连接 Ray 和 Ray Serve，随后创建训练服务。",
    "job.init.tracking": "初始化实验跟踪并确定本次运行使用的基线。",
    "controller.init.data_system": "初始化生产端与训练消费端共享的数据管理系统。",
    "service.deploy": "绑定并启动 Ray Serve Deployment，等待构造函数完成。",
    "service.init.data_client": "初始化服务到 TransferQueue 的连接。",
    "actor.init.allocate_workers": "分配训练组并创建远端 Worker Actor。",
    "actor.init.workers": "分发训练 Worker 初始化任务，等待所有 Rank 完成。",
    "worker.init.runtime": "准备训练 Worker 运行环境、参数处理及兼容补丁。",
    "worker.init.checkpoint_prepare": "分布式初始化读取元数据前，确定或准备 Checkpoint。",
    "worker.init.distributed": "初始化 Megatron 及其分布式通信组。",
    "worker.init.data_client": "将当前训练 Worker 连接到 TransferQueue。",
    "worker.init.tokenizer": "按 Rank 顺序加载模型配置和 Tokenizer，包含已有同步屏障。",
    "worker.init.model_optimizer": "构建训练模型、优化器和调度器，并恢复 Checkpoint 状态。",
    "worker.init.build_model": "构建模型分块、封装分布式训练模型，并检查可训练参数。",
    "worker.init.optimizer": "构建并检查优化器配置、优化器状态及参数调度器。",
    "worker.init.weight_management": "准备当前模式所需的权重快照、参考模型及权重更新客户端。",
    "rollout.init.manager_wait": "创建 RolloutManager Actor，并获取数据集和 Rollout 数量信息。",
    "rollout.init.manager": "初始化 RolloutManager、数据客户端、推理引擎组及生命周期协调。",
    "rollout.init.servers": "分配并启动 Rollout 服务组及其推理引擎。",
    "rollout.init.engine": "确定推理服务配置，启动或连接服务，并注册权重消费端。",
    "rollout.init.server_launch": "启动推理服务进程，并在其主节点等待服务健康就绪。",
    "rollout.init.health_wait": "轮询推理服务健康状态，包含启动等待；不表示 GPU 计算。",
    "rollout.init.page_cache": "加载推理模型前，将 Checkpoint 预热到操作系统页缓存。",
    "rollout.init.register_weights": "将就绪的推理引擎注册为 Actor 权重更新的消费端。",
    "rollout.init.register_router": "向请求路由器注册已就绪的推理引擎。",
    "rollout.weights.receive_apply": "接收并应用 Actor 权重，直到恢复生成。\n包含分桶间等待；版本号标识权重，不是批次。",
    "job.run": "观测 Relax 服务准备与执行过程，直到请求关闭作业。",
    "controller.loop": "协调各组件服务任务，并等待它们执行结束。",
    "service.init": "服务循环开始前，初始化当前角色的服务及后端 Worker。",
    "service.run": "运行当前角色跨 Step 的服务循环；异步角色的循环可以重叠。",
    "rollout.wait_start": "服务初始化完成后，等待 Controller 启动及首次权重同步。\n此区间可能与权重接收重叠。",
    "worker.init": "初始化一个后端 Worker，并记录其固定身份和通信组元数据。",
    "checkpoint.load": "将 Checkpoint 状态加载到训练 Worker，按需进行模型转换。",
    "checkpoint.save": "为当前 Worker 和 Step 保存已配置的训练 Checkpoint 状态。",
    "actor.offload": "为共享显存，迁移或释放训练 Worker 配置的 GPU 常驻状态。",
    "actor.onload": "训练 Worker 恢复 GPU 工作前，恢复已卸载的状态。",
    "weights.restore": "在训练 Worker 中恢复已保存的模型权重副本。",
    "weights.sync": "将训练权重同步到推理消费端，包含协调与传输。",
    "weights.initialize": "服务循环开始前，Actor 服务向各训练 Rank 分发首次异步权重同步。",
    "actor.configure_rollout": "将 RolloutManager 绑定到 Actor Worker，并在共卡或混合模式下完成首次权重同步。",
    "weights.connect": "发现权重消费端，并创建或复用权重传输通信组。",
    "weights.transfer_apply": "传输并应用一次完整权重更新；不为单个分桶创建 Span。",
    "weights.wait_ready": "等待权重消费端就绪、检查或恢复服务，并使各 Rank 达成一致。",
    "train.step": "在服务层协调各 Worker 完成一个训练 Step。",
    "train.worker": "单个 Rank 的训练 Step，包含数据准备、计算、协调与权重同步。",
    "train.prepare_data": "准备数据字段和迭代器；动态批处理可能等待并持续取出 Rollout 数据。",
    "train.compute": "执行当前 Worker 的训练迭代，包含取数据、前向、反向及优化器更新。",
    "train.postprocess": "汇总已消费样本、推进性能采集、记录 Rollout 数据，并按需导出调试数据。",
    "train.wait_eval": "检查或等待上一次评估，再同步各 Rank；未执行评估时也可能存在此 Span。",
    "train.finalize": "同步各 Rank、触发已配置的评估、在权重更新后放行 Rollout，并完成清理。",
    "train.log_metrics": "跨 Rank 汇总序列与 Token 数量、计算性能指标并刷新日志。",
    "train.log_probs": "运行模型推理，计算训练目标使用的对数概率。",
    "data.fetch": "从 TransferQueue 获取训练数据，并分发给消费数据的模型并行 Rank。",
    "data.get_batch": "读取并准备一个微批次，包含 Token、掩码及模型输入。",
    "megatron.iteration": "一次 Megatron 训练迭代，从梯度准备到优化器更新及损失上报。",
    "megatron.forward_backward": "执行 Megatron 前向与反向调度，并完成梯度收尾。",
    "megatron.forward_only": "仅执行 Megatron 前向调度，不执行反向或优化器更新。",
    "megatron.microbatch.forward": "在当前 Rank 上执行 Megatron 调度中的一次微批次前向调用。",
    "megatron.microbatch.backward": "在当前 Rank 上执行 Megatron 调度中的一次微批次反向调用。",
    "megatron.optimizer": "执行优化器更新，包含已配置的 CPU 卸载与同步。",
    "megatron.scheduler": "更新成功后，推进优化器的学习率及相关参数调度。",
    "megatron.loss_reduce": "在流水线末阶段汇总本次迭代的损失，用于训练指标。",
    "rollout.step": "在服务层协调一个 Rollout 批次，包含生成及下游数据交接。",
    "rollout.generate": "生成一个包含提示词组和样本的 Rollout 批次，包含奖励计算与筛选。",
    "rollout.data_fetch": "从 Rollout 数据源读取下一组提示词。",
    "rollout.group": "生成并评分同一提示词组中的样本。",
    "rollout.sample": "生成并评分一个样本，即所属提示词的一条回复。",
    "rollout.engine_request": "向推理引擎提交一次生成请求，并等待响应。",
    "rollout.reward": "计算一个已生成样本的奖励。",
    "rollout.group_reward": "计算需要整个提示词组样本共同参与的奖励。",
    "rollout.transfer": "将 Rollout 结果发布到下游数据管道。",
    "rollout.transfer_wait": "等待尚未完成的 Rollout 数据传输任务。",
    "rollout.restore_data": "服务循环开始前，恢复 Rollout 数据源位置。",
    "rollout.wait_existing_run": "等待已有的 Rollout 循环，不再启动第二个循环。",
    "rollout.wait_eval": "下一次 Rollout 前等待尚未完成的评估；next_step 标识即将开始的批次。",
    "rollout.evaluate": "通过推理管理器执行已配置的 Rollout 评估。",
    "rollout.wait_resume": "检查暂停标志并等待恢复，包含权重更新引起的暂停；next_step 标识即将开始的批次。",
    "rollout.check_completion": "检查 Rollout 数量上限，以及异步模式下最后的数据分区是否需要补齐。",
    "rollout.offload": "生成结束后，等待 Rollout 引擎释放已配置的 GPU 状态。",
    "rollout.report_progress": "向健康监控上报一个 Rollout 批次生成完成。",
    "rollout.wait_staleness": "根据下游消费进度和 max_staleness 限制下一次 Rollout；共卡模式还需等待共享 GPU 的服务。包含轮询与休眠，不表示生成。",
    "rollout.save_data": "检查保存策略，并在启用时保存 Rollout 数据源状态。",
    "rollout.finalize_step": "推进 Rollout Step，并检查最后补齐分区是否完成。",
}

NATIVE_DESCRIPTIONS = {
    "scheduler.abort": "请求计时阶段结束后，记录 Scheduler 请求取消。",
    "scheduler.events": "线程完成时记录待处理的 Scheduler 请求事件；这是零时长标记。",
    "sglang.init.scheduler": "初始化一个 SGLang Scheduler 进程，包含推理 Worker 和运行时状态。",
    "sglang.init.ipc": "创建 Scheduler 用于请求与结果交换的 IPC 通道。",
    "sglang.init.tokenizer": "加载当前 Scheduler 使用的 Tokenizer 和 Processor。",
    "sglang.init.model_worker": "创建张量并行推理 Worker 及其 Model Runner。",
    "sglang.init.memory_pools": "初始化推理请求、Token 及 KV Cache 内存池。",
    "sglang.init.attention": "初始化 Scheduler Worker 的 Attention 后端。",
    "sglang.init.cuda_graph": "初始化或捕获 Scheduler Worker 的 CUDA Graph。",
    "sglang.init.distributed": "初始化推理 Model Runner 的分布式运行环境及通信组。",
    "sglang.init.load_model": "加载推理模型参数，并完成 Model Runner 在加载时的准备工作。",
    "sglang.init.attention_backend": "构建当前 Model Runner 的 Attention 后端。",
    "sglang.init.decode_cuda_graph": "初始化或捕获当前 Model Runner 用于 Decode 的 CUDA Graph。",
    "sglang.init.prefill_cuda_graph": "初始化或捕获当前 Model Runner 用于 Prefill 的 CUDA Graph。",
    "prefill_forward": "SGLang 请求级 Prefill 阶段，从调度执行 Prefill 到生成首个 Token。",
    "decode_forward": "SGLang 请求级 Decode 阶段，覆盖多个生成 Token；不表示单个 Token 或单个 Kernel。",
    "prefill_waiting": "SGLang 请求在 Prefill 被调度前的等待。",
    "decode_waiting": "SGLang 请求在 Decode 开始或恢复前的等待。",
    "tokenize": "在 SGLang 中对当前请求的输入进行分词。",
    "detokenize": "在 SGLang 中将生成的 Token ID 转换为输出文本。",
    "request_process": "在 SGLang 请求处理层处理当前请求。",
    "api_server_dispatch": "将当前请求从 SGLang API 服务分发到请求处理器。",
    "dpc_dispatch": "通过 SGLang 数据并行控制器分发当前请求。",
    "decode_loop": "SGLang 针对当前请求的一次 Decode 循环迭代。",
    "chunked_prefill": "在 SGLang Prefill 阶段处理当前请求的一段提示词。",
    "mm_encode": "为当前 SGLang 请求编码多模态输入。",
    "prefill_prepare": "在分离式推理的 Prefill 端准备请求。",
    "prefill_bootstrap": "建立 Prefill 端到 Decode Worker 的连接。",
    "prefill_transfer_kv_cache": "将当前请求的 Prefill KV Cache 传输到 Decode Worker。",
    "decode_prepare": "在分离式推理的 Decode 端准备请求。",
    "decode_bootstrap": "建立 Decode 端与 Prefill Worker 的连接。",
    "decode_transferred": "在 Decode 端处理 KV Cache 传输完成。",
    "fake_output": "处理 SGLang 分离式 Decode 请求的占位输出。",
    "quick_finish": "通过提前完成路径结束 SGLang 请求。",
}

VIEW_DESCRIPTIONS = {
    "train.wait_rollout.merged": "训练前等待当前 Step 的 Rollout 数据及共卡服务就绪。\n此为合并后的轮询区间，包含各次轮询之间的间隔。",
    "sglang.forward": "合并当前 Rank、当前 Rollout Step 的 Prefill 与 Decode。\n显示最早开始至最晚结束的 Host 区间，包含其中的间隔。",
}


def describe_span(name: str, attributes: Mapping[str, Any] | None = None) -> str:
    """Return Chinese span semantics without changing training execution."""
    attributes = attributes or {}
    if name == "train.wait_rollout" and attributes.get("relax.simulation.synthetic") is True:
        text = VIEW_DESCRIPTIONS["train.wait_rollout.merged"]
    elif name == "sglang.forward" and attributes.get("relax.simulation.synthetic") is True:
        text = VIEW_DESCRIPTIONS["sglang.forward"]
    elif name == "worker.init" and attributes.get("relax.init.kind") == "metadata_snapshot":
        text = "首个采样请求时记录的 SGLang Worker 身份及通信组零时长快照；不表示模型初始化耗时。"
    elif name in DESCRIPTIONS:
        text = DESCRIPTIONS[name]
        if name in {
            "rollout.group",
            "rollout.sample",
            "rollout.engine_request",
            "rollout.reward",
            "rollout.group_reward",
        }:
            text += "\n\n" + ROLLOUT_IDS
    else:
        native_name = name.strip()
        text = NATIVE_DESCRIPTIONS.get(native_name)
        if native_name.startswith("Req "):
            text = "一个 SGLang 请求的生命周期；Req 表示该请求的 ID。"
        elif native_name.startswith("Scheduler"):
            text = "一个请求在 Scheduler 线程上的处理过程；并发请求可以重叠。\nTP/PP/DP 表示该线程的并行 Rank。"
        elif native_name.startswith("Tokenizer"):
            text = "一个请求在 Tokenizer 线程上的处理过程，包含输出等待。\ntokenize 子阶段表示分词耗时。"
        elif native_name.startswith("Detokenizer"):
            text = "一个请求在 SGLang Detokenizer 线程上的生命周期，包含输出处理与等待。"
        if text:
            if not native_name.startswith("sglang.init."):
                text += "\n\n" + ROLLOUT_IDS
        else:
            function = attributes.get("code.function.name")
            text = f"已埋点操作“{name}”" + (f"，位于函数 {function}。" if function else "。")
    return text


def description_processor() -> Any:
    """Create the alternative processor; importing this file loads no SDK."""
    from opentelemetry.sdk.trace import SpanProcessor

    class DescriptionProcessor(SpanProcessor):
        def on_start(self, span: Any, parent_context: Any = None) -> None:
            span.set_attribute(DESCRIPTION_ATTRIBUTE, describe_span(span.name, span.attributes))

    return DescriptionProcessor()


def description_table() -> dict[str, str]:
    """Return one shared table; extra keys identify existing semantic
    variants."""
    table = {name: describe_span(name) for name in (*DESCRIPTIONS, *NATIVE_DESCRIPTIONS)}
    for name, key, attrs in (
        ("worker.init", "worker.init#metadata_snapshot", {"relax.init.kind": "metadata_snapshot"}),
        ("train.wait_rollout", "train.wait_rollout#merged", {"relax.simulation.synthetic": True}),
        ("sglang.forward", "sglang.forward#merged", {"relax.simulation.synthetic": True}),
    ):
        table[key] = describe_span(name, attrs)
    for prefix in ("Req ", "Scheduler", "Tokenizer", "Detokenizer"):
        table[prefix + "*"] = describe_span(prefix + "catalog-example")
    return table
