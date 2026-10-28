# SGLang ETTR patch

2026-10-27 本地采集变更：Relax 与 Native provider 默认不再注册 description processor，因此新 Span 不重复保存 `relax.span.description`。网页按名称和已有语义属性查询中文说明表，英文表保留；历史 trace 与冻结 overlay 不改写。保留 RNG 隔离、active-span 跟踪与采集策略。当前补丁已更新并验证 round-trip，尚未重新进行 GPU 验证；下方逐 Span 英文说明记录为历史版本行为。

2026-10-26 本地修复（尚未部署或 GPU 验证）：明确的请求取消/HTTP499与丢弃未完成请求写入 `relax.span.outcome=aborted`、`relax.cancel.reason` 和 cancellation event；未知错误、503及计算异常继续 ERROR。结束请求时同时关闭本地 owner 的 Req 和 thread/slice；正常线程完成与跨进程 copy 不得提前关闭 owner Req。析构是遗漏显式结束路径的兜底，保留 `have unclosed span, auto closed` 原因。新增本进程 active-span 记录及正常退出前结束/flush。此版本必须同时同步 `relax/utils/tracing_shutdown.py`，完整目标、累计 patch 与 manifest 已更新，历史 frozen 配方和 trace 不变。

Relax SDK 同时区分 cancelled/aborted 与 ERROR；Controller 在阻塞清理前结束本进程观测，并向 service replicas 并发发送有总等待时限的遥测收尾 RPC。`relax.span.end_reason` 标明观测截止原因，不把被打断的操作当成成功完成，也不改训练调度。已结束 span 原状态保留。SIGKILL、节点掉电、不可达 replica、被强制终止的 engine/worker 无法保证导出；未补造历史父级。

2026-10-23 后续已完成35B colocate/10step GPU验证：新主trace5,675span，Scheduler请求包装为0，1,280条prefill/decode直接挂Req，10step×8TP×8sample及G/S完整。43源码和原生五目标双节点SHA、每节点76项CPU及native探针通过。主链无ERROR/缺父级/重复；完整原始说明与源码AST校验通过。见工作区artifacts/otel-flat-colocate/2026-10-23。下方待部署状态保留为此前准备记录，不更新已冻结manifest或历史trace。

2026-10-23 本地待部署：启用 ETTR 时移除每请求的 `Scheduler [TP …] [PP …]` 包装，直接导出 Req 下的 prefill/decode 等阶段；保留线程/rank 归属、嵌套调用、links、原始时间和每线程/trace一次的 worker.init 拓扑快照。Tokenizer 等线程和未启用 ETTR 的原生行为不变。未结束阶段的错误/events记在实际阶段；阶段外的未消费诊断用零时长 scheduler.abort/events 标记，正常请求不新增包装。完整目标、累计patch及manifest已同步；36项相关CPU测试、level1/2/3网页模型父链/G/S/rank解析与五文件patch/hash/语法校验通过，尚未GPU验证。以下保留历史实验边界。

2026-10-23 新扩展已完成两节点35B async/10step GPU验证：主trace5,148span，native init10阶段×8TP、receiver版本1–11×8TP均完整；主链无ERROR/缺父级/重复，五文件SHA和源码匹配通过。结果见工作区 artifacts/otel-lifecycle/2026-10-23。原manifest的本地验证文字保留冻结准备阶段边界，previous_gpu_verified_target_sha256仍指更早版本。

新增 `sglang-v0.5.17-lifecycle.patch` / `lifecycle-manifest.json`，与累计 trace patch 一起使用，共五个原生目标：trace、io_struct、scheduler、model_runner、scheduler_components/weight_updater。每个目标严格检查原安装 SHA，版本不同会拒绝应用。

- 12 个 `sglang.init.*` scope 覆盖 scheduler、IPC、tokenizer、model worker、分布式初始化、模型加载、内存池、attention 和 CUDA graph；记录真实 host 调用区间，不添加 CUDA 同步。
- `rollout.weights.receive_apply` 覆盖一个 rank 的一次权重更新，从首个带 context 的 bucket 调用开始，到 scheduler 处理 continue_generation 结束。并非从暂停生成开始，也不是纯 NCCL 时间。`relax.weights.apply_host_ns` 累计各接收方法的 host 时间；`bucket_count` 统计次数，不创建 bucket span。
- DCS distributed 和 colocate tensor 入口携带 Actor 父级、step 和 weight_version。routed 更新尚未发布版本时按 Actor transfer context 识别，不编造版本。LoRA adapter、其他 IPC checkpoint-engine 协议未扩展。
- 接收报错、返回 success=False、恢复失败、旧版本被替代、进程正常退出时仍未恢复，均结束为 ERROR。SIGKILL 无法保证导出未结束 span。结束 generation service loop 不会停止 engine 后续权重 span。
- resume 的 HTTP/IPC 调用原本就可能异步返回，receiver span 可以晚于 Actor 父 span 结束；保留真实时间，不改变业务等待或裁剪区间。

准备私有 overlay（不修改原安装，输出必须是全新目录）：

```bash
python3 patches/ettr/prepare-lifecycle-overlay.py \
  --sglang-root /sgl-workspace/sglang/python/sglang \
  --output /tmp/ettr/<new-run>-native
```

两节点使用匹配的 Relax 源码和完整 overlay，将 overlay 放在 PYTHONPATH 首位。启动环境增加 `RELAX_OTEL_SGLANG_LIFECYCLE=1`（默认 0，兼容旧服务），使用 OTLP/both exporter、`RELAX_OTEL_LEVEL>=2`。原生业务/init 是连续 L2；请求生成细节仍服从 Relax L3 窗口和 SGLANG_TRACE_LEVEL。不能只替换 trace.py 后启用此开关。

本地 58 项 CPU 测试包括真实补丁方法体的 distributed/tensor 调用、父级与源码、采样开关、上下文恢复、RNG、失败/中断；9 个模块剥离 trace 包装后业务 AST 与改前一致。五目标的原始 hash、补丁应用、语法和最终 hash 均通过。CPU 测试环境设置 `ETTR_NATIVE_LIFECYCLE_OVERLAY=/path/to/private-overlay` 后运行 `tests/utils/test_tracing_lifecycle.py`，可覆盖真实 native 方法体；未设置时这两项明确 skip。本轮 GPU 接收区间、版本/rank/父级及源码覆盖已验收；这是host观测，不是GPU kernel或纯NCCL计时。

累计原生 tracing 补丁适用于 SGLang 0.5.17；原始源码版本、目标文件和 SHA 见 manifest.json。sglang-trace.py 是完整目标文件，sglang-v0.5.17-ettr.patch 可从指定 before_sha256 正向生成该文件。应用前必须核对版本和原文件，不能覆盖其他版本。

保留 OTLP 随机数隔离、固定元数据集中、代码位置采集和系统熵 ID。2026-10-22 增加与 Relax SDK 共用的 description processor，在每个 span 上写英文 relax.span.description，包括 request/thread 区别、G/S 索引和 host 计时边界。原生名称保持 SGLang 名称，网页上的 G/S 后缀来自祖先属性。

本轮使用 /tmp/ettr/20261022T121017Z-native 私有 PYTHONPATH overlay；两节点原安装文件保持原 SHA。44 项 CPU 测试和原生传播/采样/ID/RNG/说明探针已在两节点通过，新的 35B async 10-step GPU 实验已完成：主trace4,876span、全部5,548条均有英文说明和源码AST匹配。历史 Trial2266400 与本次 Job345810/Trial2262600 的验证边界分别保留；完整结果见工作区 configs/otel-descriptions/2026-10-22 与 artifacts/otel-descriptions/2026-10-22。
