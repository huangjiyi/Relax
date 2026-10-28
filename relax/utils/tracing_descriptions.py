# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""English span semantics shared by Relax and the opt-in SGLang trace overlay."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DESCRIPTION_ATTRIBUTE = "relax.span.description"
ROLLOUT_IDS = (
    "G: global prompt-group index (shared prompt).\n"
    "S: global sample index, not within-group.\n"
    "Both start at 0 and continue across steps."
)

DESCRIPTIONS = {
    'train.wait_rollout': 'Wait for rollout data and colocated peers before training.',
    'train.wait_rollout_offload': 'Wait for Rollout to release GPU memory.',
    'train.wait_peer': 'Wait for colocated peers to finish or release GPU memory.',
    'train.service_finalize': 'Clear consumed partitions, report progress and advance the Actor step.',
    'train.advantages': 'Compute advantages and returns for the training samples.',
    'weights.backup': 'Save local Actor and optional reference weight copies.',
    'weights.prepare': 'Prepare inference consumers and communication groups for weight transfer.',
    'weights.finalize': 'Release transfer resources and synchronize training ranks.',
    'weights.onload_rollout': 'Restore Rollout KV cache and configured reward models after weight transfer.',
    'service.allocate': 'Prepare one component service and allocate its placement group before deployment.',
    'service.init.resources': 'Allocate the placement group, wait for resources and resolve the assigned GPU ordering.',
    'controller.init.health': 'Check control-plane resources and create the health-management system.',
    'controller.init.checkpoint_service': 'Deploy the distributed checkpoint coordinator used for weight synchronization.',
    'rollout.init.checkpoint_prepare': 'Resolve inference checkpoint sources and apply the configured model-loading policy.',
    'job.shutdown': 'Attempt graceful controller, inference-server, cache-agent and Ray cleanup before ending the job trace.',
    'controller.shutdown': 'Coordinate shutdown of the component services and their managed runtime resources.',
    'job.init.runtime': 'Load runtime configuration and connect to Ray and Ray Serve before creating training services.',
    'job.init.tracking': 'Initialize experiment tracking and resolve the baseline used by this run.',
    'controller.init.data_system': 'Initialize the shared data-management system used by producers and training consumers.',
    'service.deploy': 'Bind and start a Ray Serve deployment, waiting for its constructor to finish.',
    'service.init.data_client': 'Initialize the service connection to TransferQueue.',
    'actor.init.allocate_workers': 'Allocate the training group and create its remote worker actors.',
    'actor.init.workers': 'Dispatch training-worker initialization and wait for all ranks to finish.',
    'worker.init.runtime': 'Prepare the training worker runtime, argument handling and compatibility patches.',
    'worker.init.checkpoint_prepare': 'Resolve or materialize the checkpoint before distributed initialization reads its metadata.',
    'worker.init.distributed': 'Initialize Megatron and its distributed communication groups.',
    'worker.init.data_client': 'Connect this training worker to TransferQueue.',
    'worker.init.tokenizer': 'Load model configuration and tokenizer in serialized rank order, including the existing barriers.',
    'worker.init.model_optimizer': 'Construct the training model, optimizer and scheduler and restore checkpoint state.',
    'worker.init.build_model': 'Build model chunks and wrap them for distributed training, then validate trainable parameters.',
    'worker.init.optimizer': 'Build and validate optimizer configuration, optimizer state and its parameter scheduler.',
    'worker.init.weight_management': 'Prepare weight snapshots, reference models and the weight-update client required by this mode.',
    'rollout.init.manager_wait': 'Create the RolloutManager actor and resolve its dataset and rollout-count information.',
    'rollout.init.manager': 'Initialize the RolloutManager, data client, engine groups and lifecycle coordination.',
    'rollout.init.servers': 'Allocate and start rollout server groups and their inference engines.',
    'rollout.init.engine': 'Resolve inference-server settings, start or connect to the server and register its weight consumer.',
    'rollout.init.server_launch': 'Spawn the inference server and, on its head node, wait for it to become healthy.',
    'rollout.init.health_wait': 'Poll inference-server health, including startup waits; this is not GPU computation.',
    'rollout.init.page_cache': 'Warm the checkpoint in the operating-system page cache before loading the inference model.',
    'rollout.init.register_weights': 'Register the ready inference engine as a consumer of Actor weight updates.',
    'rollout.init.register_router': 'Register the ready inference engine with the request router.',
    'rollout.weights.receive_apply': 'Receive and apply Actor weights until generation resumes.\nIncludes inter-bucket waits; version identifies weights, not a batch.',
    "job.run": "Observed Relax service setup and execution, ending when job shutdown is requested.",
    "controller.loop": "Coordinate the component service tasks and wait for their execution to finish.",
    "service.init": "Initialize a role's service and its backend workers before the service loop starts.",
    "service.run": "Run a role's service loop across steps; asynchronous role loops can overlap.",
    "rollout.wait_start": (
        "Wait between service init and run for Controller startup and initial weight sync.\n"
        "Weight reception may overlap this interval."
    ),
    "worker.init": "Initialize one backend worker and record its fixed identity and communication-group metadata.",
    "checkpoint.load": "Load checkpoint state into a training worker, including model conversion when required.",
    "checkpoint.save": "Save the configured training checkpoint state for this worker and step.",
    "actor.offload": "Move or release the training worker's configured GPU-resident state for memory sharing.",
    "actor.onload": "Restore the training worker's offloaded state before it resumes GPU work.",
    "weights.restore": "Restore a saved model-weight copy in the training worker.",
    "weights.sync": "Synchronize training weights to inference consumers, including coordination and transfer.",
    "weights.initialize": (
        "The Actor service dispatches the initial async weight synchronization to its training ranks "
        "before service loops start."
    ),
    "actor.configure_rollout": (
        "Attach the rollout manager to Actor workers and perform the initial weight sync in colocate or hybrid mode."
    ),
    "weights.connect": "Discover weight consumers and create or reuse the communication groups for weight transfer.",
    "weights.transfer_apply": "Transfer and apply one complete weight update; individual bucket sends are not spans.",
    "weights.wait_ready": "Wait for weight consumers to be ready, check or recover services, and agree across ranks.",
    "train.step": "Service-level orchestration of one training step across its workers.",
    "train.worker": (
        "One rank's training step, including preparation, compute, coordination and weight synchronization."
    ),
    "train.prepare_data": "Prepare fields and data iterators; dynamic batching may wait for and drain rollout data.",
    "train.compute": (
        "Execute this worker's training iterations, including batch access, forward/backward and optimizer."
    ),
    "train.postprocess": (
        "Collect the consumed samples, advance profiling, log rollout data and optionally dump debug data."
    ),
    "train.wait_eval": (
        "Check or wait for previous evaluation, then synchronize ranks; a span can exist with no eval work."
    ),
    "train.finalize": (
        "Synchronize ranks, trigger configured evaluation, release rollout after weight update and clean up."
    ),
    "train.log_metrics": (
        "Aggregate sequence and token counts across ranks, compute performance metrics and flush logging."
    ),
    "train.log_probs": "Run model inference to compute the log probabilities used by the training objective.",
    "data.fetch": "Fetch training data from TransferQueue and distribute it to the consuming model-parallel ranks.",
    "data.get_batch": "Read and prepare one microbatch, including token, mask and model-input preparation.",
    "megatron.iteration": (
        "One Megatron training iteration, from gradient preparation through optimizer and loss reporting."
    ),
    "megatron.forward_backward": "Run the Megatron forward/backward schedule and its gradient finalization.",
    "megatron.forward_only": "Run a Megatron forward-only schedule without backward or optimizer updates.",
    "megatron.microbatch.forward": "Execute the Megatron schedule's forward call for one microbatch on this rank.",
    "megatron.microbatch.backward": "Execute the Megatron schedule's backward call for one microbatch on this rank.",
    "megatron.optimizer": "Execute the optimizer update, including any configured CPU offload and synchronization.",
    "megatron.scheduler": (
        "Advance the optimizer's learning-rate and related parameter schedule after a successful update."
    ),
    "megatron.loss_reduce": "Aggregate the iteration's loss values for training metrics on the last pipeline stage.",
    "rollout.step": "Service-level orchestration of one rollout batch, including generation and downstream handoff.",
    "rollout.generate": (
        "Generate a rollout batch of prompt groups and samples, including reward computation and filtering."
    ),
    "rollout.data_fetch": "Read the next prompt groups from the rollout data source.",
    "rollout.group": "Generate and score the samples belonging to one shared prompt group.",
    "rollout.sample": "Generate and score one sample (a completion for its group's prompt).",
    "rollout.engine_request": "Submit one generation request to the inference engine and await its response.",
    "rollout.reward": "Compute the reward for one generated sample.",
    "rollout.group_reward": "Compute rewards that require the samples of an entire prompt group.",
    "rollout.transfer": "Publish rollout results to the downstream data pipeline.",
    "rollout.transfer_wait": "Wait for pending rollout-data transfer tasks to finish.",
    "rollout.restore_data": "Restore the rollout data-source position before starting the service loop.",
    "rollout.wait_existing_run": "Wait for an already active rollout loop instead of starting a second one.",
    "rollout.wait_eval": (
        "Wait for an outstanding evaluation task before the next rollout; next_step names that batch."
    ),
    "rollout.evaluate": "Run the configured rollout evaluation through the inference manager.",
    "rollout.wait_resume": (
        "Check the pause flag and wait for resume, including pauses requested for weight updates; "
        "next_step names the upcoming batch."
    ),
    "rollout.check_completion": (
        "Check the rollout limit and, in async mode, whether the final data partition needs backfill."
    ),
    "rollout.offload": "Wait for the rollout engine to release its configured GPU state after generation.",
    "rollout.report_progress": "Report completion of a generated rollout batch to the health monitor.",
    "rollout.wait_staleness": (
        "Gate the next rollout on downstream consumption and max_staleness; colocate also waits for "
        "GPU-sharing peers. Includes polling and sleeps, not generation."
    ),
    "rollout.save_data": "Check the save policy and, when enabled, persist rollout data-source state.",
    "rollout.finalize_step": "Advance the rollout step and verify completion of any final backfill partition.",
}
NATIVE_DESCRIPTIONS = {
    "scheduler.abort": "Record a scheduler request abort after its timed phases have ended.",
    "scheduler.events": "Record pending scheduler request events at thread completion; zero-duration marker.",
    "sglang.init.scheduler": 'Initialize one SGLang scheduler process, including its inference workers and runtime state.',
    "sglang.init.ipc": 'Create the scheduler IPC channels used for request and result exchange.',
    "sglang.init.tokenizer": 'Load the tokenizer and processor used by this scheduler.',
    "sglang.init.model_worker": 'Create the tensor-parallel inference worker and its model runner.',
    "sglang.init.memory_pools": 'Initialize inference request, token and KV-cache memory pools.',
    "sglang.init.attention": "Initialize the scheduler workers' attention backends.",
    "sglang.init.cuda_graph": "Initialize or capture the scheduler workers' CUDA graphs.",
    "sglang.init.distributed": "Initialize the inference model runner's distributed runtime and groups.",
    "sglang.init.load_model": "Load inference model parameters and perform the model runner's load-time preparation.",
    "sglang.init.attention_backend": "Construct this model runner's attention backend.",
    "sglang.init.decode_cuda_graph": 'Initialize or capture CUDA graphs for decoding in this model runner.',
    "sglang.init.prefill_cuda_graph": 'Initialize or capture CUDA graphs for prefill in this model runner.',
    "prefill_forward": "SGLang's request-level prefill phase, from scheduled prefill to the first generated token.",
    "decode_forward": (
        "SGLang's request-level decoding phase across generated tokens; this is not one token or one kernel."
    ),
    "prefill_waiting": "SGLang's request wait before prefill is scheduled.",
    "decode_waiting": "SGLang's request wait before decoding resumes or starts.",
    "tokenize": "Tokenize this request's input in SGLang.",
    "detokenize": "Convert generated token IDs into output text in SGLang.",
    "request_process": "Handle this request in SGLang's request-processing layer.",
    "api_server_dispatch": "Dispatch this request from the SGLang API server to its request handler.",
    "dpc_dispatch": "Dispatch this request through SGLang's data-parallel controller.",
    "decode_loop": "One iteration of the SGLang decoding loop for this request.",
    "chunked_prefill": "Process a chunk of this request's prompt during SGLang prefill.",
    "mm_encode": "Encode multimodal inputs for this SGLang request.",
    "prefill_prepare": "Prepare a request on the prefill side of disaggregated inference.",
    "prefill_bootstrap": "Bootstrap the prefill side's connection to the decode worker.",
    "prefill_transfer_kv_cache": "Transfer this request's prefill KV cache to its decode worker.",
    "decode_prepare": "Prepare a request on the decode side of disaggregated inference.",
    "decode_bootstrap": "Bootstrap the decode side's connection to the prefill worker.",
    "decode_transferred": "Handle completion of the KV-cache transfer on the decode side.",
    "fake_output": "Handle SGLang's placeholder output for a disaggregated decode request.",
    "quick_finish": "Finish a SGLang request through the early-completion path.",
}


def describe_span(name: str, attributes: Mapping[str, Any] | None = None) -> str:
    """Describe span semantics without modifying IDs, names, sampling or training RNG."""
    attributes = attributes or {}
    if name == "worker.init" and attributes.get("relax.init.kind") == "metadata_snapshot":
        text = (
            "A zero-duration SGLang worker identity and communication-group snapshot at its first sampled request; "
            "this is not model initialization time."
        )
    elif name in DESCRIPTIONS:
        text = DESCRIPTIONS[name]
        if name in {"rollout.group", "rollout.sample", "rollout.engine_request", "rollout.reward", "rollout.group_reward"}:
            text += "\n\n" + ROLLOUT_IDS
    else:
        native_name = name.strip()
        text = NATIVE_DESCRIPTIONS.get(native_name)
        if native_name.startswith("Req "):
            text = "One SGLang request's lifetime. Req is its request ID."
        elif native_name.startswith("Scheduler"):
            text = (
                "One request on a scheduler thread; concurrent requests overlap.\n"
                "TP/PP/DP identify the thread's parallel ranks."
            )
        elif native_name.startswith("Tokenizer"):
            text = (
                "One request on a tokenizer thread, including output waits.\n"
                "The tokenize child measures tokenization."
            )
        elif native_name.startswith("Detokenizer"):
            text = "One request's lifetime on a SGLang detokenizer thread, including output handling and waiting."
        if text:
            if not native_name.startswith("sglang.init."):
                text += "\n\n" + ROLLOUT_IDS
        else:
            function = attributes.get("code.function.name")
            text = f"Instrumented operation '{name}'" + (f" in {function}." if function else ".")
    return text


def description_processor() -> Any:
    """Import the optional SDK only when a tracing provider is being configured."""
    from opentelemetry.sdk.trace import SpanProcessor

    class DescriptionProcessor(SpanProcessor):
        def on_start(self, span: Any, parent_context: Any = None) -> None:
            span.set_attribute(DESCRIPTION_ATTRIBUTE, describe_span(span.name, span.attributes))

    return DescriptionProcessor()
