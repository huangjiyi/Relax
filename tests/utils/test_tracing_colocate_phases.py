# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise real Actor waiting/housekeeping and colocate training on CPU doubles."""

import threading
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch
from opentelemetry import trace

from relax.utils import tracing
from relax.utils.timer import Timer
from tests.utils.test_tracing_async_phases import records as _records, source_class

records = _records


def test_service_wait_and_housekeeping_preserve_poll_and_barrier_order(records, monkeypatch):
    observed = []
    for enabled in ('0', '1'):
        monkeypatch.setenv('RELAX_OTEL_ENABLED', enabled)
        events, polls = [], iter([[], ['0'], ['1']])
        def event(name, value=None):
            events.append(name)
            return value
        cls = source_class('relax/components/actor.py', 'Actor', {'_background_run', '_wait_for_rollout_data'}, {
            'stage': tracing.stage, 'traced': tracing.traced, 'run': lambda x: x,
            'time': NS(sleep=lambda _: event('poll_sleep')), 'is_offline_mode': lambda _: False,
            'sft_partition_ids': lambda config, step: [str(step)],
        })
        actor = cls()
        actor.step = 0
        actor.config = NS(num_rollout=2, fully_async=False, colocate=True, debug_train_only=False,
                          offload_rollout=True, num_critic_only_steps=0, use_health_check=False)
        actor._stop_event, actor._lock = threading.Event(), threading.Lock()
        actor._logger = NS(info=lambda _: None, exception=lambda _: None)
        actor.data_system_client = NS(async_get_partition_list=lambda: event('poll', next(polls)),
                                      async_clear_partition=lambda **kw: event('clear'))
        actor._rollout_barrier = NS(wait_offloaded_sync=lambda: event('offload'))
        actor._peer_barrier = None
        actor.healthy = NS(update_heartbeat=NS(remote=lambda *a: event('heartbeat')),
                           report_error=NS(remote=lambda *a: event('error')))
        actor._execute_training = lambda: event('train', True)
        with tracing.stage('service.run', **{'relax.role': 'actor'}):
            actor._background_run()
        assert actor.step == 2
        observed.append(events)
    assert observed[0] == observed[1] == ['poll', 'poll_sleep', 'poll', 'offload', 'train', 'clear',
                                        'heartbeat', 'poll', 'offload', 'train', 'clear', 'heartbeat']
    spans = records()
    assert len([s for s in spans if s['name'] == 'train.wait_rollout']) == 3
    offload_spans = [s for s in spans if s['name'] == 'train.wait_rollout_offload']
    assert len(offload_spans) == 2
    by_id = {s['context']['span_id']: s for s in spans}
    for span in offload_spans:
        parent = by_id[span['parent_id']]
        assert parent['name'] == 'train.wait_rollout'
        assert parent['start_time'] <= span['start_time'] <= span['end_time'] <= parent['end_time']
        assert 'relax.span.description' not in span['attributes']
    assert {s['attributes']['relax.step'] for s in spans if s['name'] == 'train.service_finalize'} == {0, 1}


@pytest.mark.parametrize('fail', [False, True])
def test_colocate_prepare_backup_metrics_and_finalize_preserve_calls(records, monkeypatch, fail):
    sequences = []
    for enabled in ('0', '1'):
        monkeypatch.setenv('RELAX_OTEL_ENABLED', enabled)
        events = []
        def event(name, result=None):
            events.append((name, getattr(trace.get_current_span(), 'name', None)))
            return result
        def gather(out, value, **kw):
            event('gather'); out[:] = [value]
        def advantages(*args):
            event('advantages')
            if fail:
                raise RuntimeError('advantage failure')
        scope = {
            'stage': tracing.stage, 'torch': torch, 'Timer': Timer,
            'is_preference_mode': lambda _: False, 'algorithm_needs_critic': lambda _: False,
            'is_offline_mode': lambda _: False, 'get_data_iterator': lambda *a, **kw: event('iterator', ([NS()], [1])),
            'inverse_timer': lambda _: nullcontext(), 'timer': lambda _: nullcontext(),
            'compute_advantages_and_returns': advantages,
            'capture_hooks': NS(begin_rollout_for=lambda *a: event('capture_begin'),
                                capture_rollout_advantage=lambda **k: event('capture'),
                                end_rollout_for=lambda: event('capture_end')),
            'log_rollout_data': lambda *a: event('rollout_log'), 'train': lambda *a: event('compute'),
            'train_dump_utils': NS(save_debug_train_data=lambda *a, **k: event('dump')),
            'mpu': NS(get_data_parallel_world_size=lambda **k: 1, get_data_parallel_group=lambda **k: 'dp'),
            'dist': NS(all_gather_object=gather, barrier=lambda **k: event('barrier')),
            'log_perf_data': lambda *a, **k: event('performance'),
            'tracking_utils': NS(flush_metrics=lambda *a: event('flush')),
            'compute_rollout_step': lambda args, step: step,
            'get_gloo_group': lambda: 'gloo', 'evaluation_step_for_rollout': lambda args, step: step+1,
        }
        cls = source_class('relax/backends/megatron/actor.py', 'MegatronTrainRayActor', {'train_actor'}, scope)
        actor = cls()
        actor.args = NS(use_dynamic_batch_size=False, use_rollout_routing_replay=False, use_routing_replay=False,
                        compute_advantages_and_returns=True, use_critic=False, use_rollout_logprobs=True,
                        get_mismatch_metrics=False, keep_old_actor=False, ref_update_interval=None,
                        num_rollout=10, save=None, offload_train=True)
        actor.model = actor.optimizer = actor.opt_param_scheduler = actor.tokenizer = actor.flops_counter = None
        actor._active_model_tag, actor._per_step_rollout = 'actor', True
        actor.rollout_data_postprocess = None
        actor._is_standard_dpo = lambda: False
        actor._switch_model = lambda *a: event('switch')
        actor.weights_backuper = NS(backup_tags=[], backup=lambda *a: event('backup'))
        actor.prof = NS(step=lambda **k: event('profiler'))
        actor.rollout_manager = NS()
        actor.sleep = lambda: event('offload')
        actor.update_weights = lambda: event('sync')
        actor._run_step_evaluation = lambda *a: event('eval')
        actor._wait_for_previous_eval = lambda: event('wait_eval')
        with tracing.stage('train.worker', **{'relax.step': 9, 'relax.rank': 0, 'relax.role': 'actor'}):
            if fail:
                with pytest.raises(RuntimeError, match='advantage failure'):
                    actor.train_actor(9, {'total_lengths': [2], 'loss_masks': [[1, 1]]})
            else:
                actor.train_actor(9, {'total_lengths': [2], 'loss_masks': [[1, 1]]})
        sequences.append([name for name, _ in events])
    assert sequences[0] == sequences[1]
    spans = records()
    if fail:
        assert events[-1][0] == 'capture_end'
        assert next(s for s in spans if s['name'] == 'train.advantages')['status']['status_code'] == 'ERROR'
    else:
        expected = {'iterator': 'train.prepare_data', 'advantages': 'train.advantages',
                    'dump': 'train.postprocess', 'backup': 'weights.backup',
                    'performance': 'train.log_metrics', 'flush': 'train.log_metrics', 'wait_eval': 'train.finalize'}
        assert all(phase == expected[name] for name, phase in events if name in expected)
        assert all(s['attributes']['relax.step'] == 9 for s in spans)
        assert all(s['status']['status_code'] != 'ERROR' for s in spans)
