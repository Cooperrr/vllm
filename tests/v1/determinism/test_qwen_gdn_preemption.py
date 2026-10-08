# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact Qwen GDN recovery after cache-pressure and shared-cache preemption."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.utils import create_new_process_for_each_test
from tests.v1.determinism.test_qwen_gdn_prefix_caching import (
    _engine_settings,
    _prompts,
    _snapshot,
)
from vllm import SamplingParams, TokensPrompt
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.attention.backends.gdn_attn import QwenGDNAttentionBackend
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

pytestmark = [
    pytest.mark.skipif(
        not QwenGDNAttentionBackend.supports_batch_invariance(),
        reason="Requires the batch-invariant Qwen GDN CUDA backend",
    ),
    pytest.mark.timeout(600),
]


class ObservedScheduler(Scheduler):
    """Observe real scheduler decisions without changing allocation or ordering."""

    def _record(self, event, request, **extra):
        path = self.vllm_config.additional_config["preemption_trace_path"]
        row = {
            "event": event,
            "request_id": request.request_id,
            "prompt_tokens": request.num_prompt_tokens,
            "output_tokens": request.num_output_tokens,
            "computed_tokens": request.num_computed_tokens,
            "preemptions": request.num_preemptions,
            **extra,
        }
        with open(path, "a") as stream:
            stream.write(json.dumps(row) + "\n")

    def _preempt_request(self, request, timestamp):
        self._record("preempt", request)
        return super()._preempt_request(request, timestamp)

    def _update_after_schedule(self, output):
        for request_id, count in output.num_scheduled_tokens.items():
            request = self.requests[request_id]
            if request.num_preemptions:
                self._record("resume", request, scheduled_tokens=count)
        return super()._update_after_schedule(output)


class ObservedAsyncScheduler(ObservedScheduler, AsyncScheduler):
    pass


class SharedCachePreemptionScheduler(ObservedScheduler):
    """Trigger one normal preemption without evicting the donor's cached state."""

    def schedule(self):
        allocate = self.kv_cache_manager.allocate_slots

        def allocate_with_failure(request, *args, **kwargs):
            params = request.sampling_params
            threshold = (
                (params.extra_args or {}).get("test_preempt_after") if params else None
            )
            if (
                threshold is not None
                and request.num_output_tokens >= threshold
                and request.num_preemptions == 0
            ):
                # Observe the longer donor state before forcing an allocation
                # failure. Only this failure is injected; the normal scheduler
                # performs preemption, lookup, replay, and resumption.
                _, unrestricted_hit = (
                    self.kv_cache_manager.coordinator.find_longest_cache_hit(
                        request.block_hashes, request.num_tokens - 1
                    )
                )
                self._record(
                    "allocation_failure", request, unrestricted_hit=unrestricted_hit
                )
                return None
            return allocate(request, *args, **kwargs)

        with patch.object(
            self.kv_cache_manager, "allocate_slots", side_effect=allocate_with_failure
        ):
            return super().schedule()


class AsyncSharedCachePreemptionScheduler(
    SharedCachePreemptionScheduler, AsyncScheduler
):
    pass


@pytest.mark.parametrize(
    "eager,async_scheduling",
    [(True, False), (False, True)],
    ids=["eager-sync", "compiled-async"],
)
@create_new_process_for_each_test(method="spawn")
def test_qwen_gdn_preemption_shared_prompt_cache(
    vllm_runner, tmp_path: Path, eager: bool, async_scheduling: bool
):
    """A longer donor prompt cannot substitute prefill state for decoded output."""
    model, settings = _engine_settings(eager, async_scheduling)
    trace = tmp_path / "shared-cache-scheduler.jsonl"
    trace.write_text("")
    scheduler_name = (
        "AsyncSharedCachePreemptionScheduler"
        if async_scheduling
        else "SharedCachePreemptionScheduler"
    )
    settings["scheduler_cls"] = f"{__name__}.{scheduler_name}"
    settings.setdefault("additional_config", {})["preemption_trace_path"] = str(trace)
    sampling = SamplingParams(
        temperature=0, seed=3407, max_tokens=32, ignore_eos=True, logprobs=5
    )
    report: dict = {"settings": settings, "runs": [], "complete": False}

    def events():
        return [json.loads(line) for line in trace.read_text().splitlines()]

    def checkpoint():
        (tmp_path / "shared-cache-results.json").write_text(
            json.dumps(report, indent=2)
        )

    with vllm_runner(
        model, enable_prefix_caching=True, mamba_cache_mode="align", **settings
    ) as runner:
        llm = runner.llm
        [layout] = llm.collective_rpc("prefix_cache_layout")
        report["layout"] = layout
        block_size = layout["attention_block_size"]
        prompt = _prompts(llm.get_tokenizer(), block_size, 2048)["block_minus1"]
        report["prompt"] = prompt

        def run(label, tokens, params):
            start = len(events())
            [output] = llm.generate(
                [TokensPrompt(prompt_token_ids=tokens)], params, use_tqdm=False
            )
            row = {
                "label": label,
                "output": _snapshot(output),
                "events": events()[start:],
            }
            report["runs"].append(row)
            checkpoint()
            assert len(row["output"]["token_ids"]) == params.max_tokens
            return row

        assert llm.reset_prefix_cache()
        reference = run("uninterrupted", prompt, sampling)
        assert not reference["events"]
        # Output from A is valid prefill input for another request, but the
        # resulting state is not valid for replaying A's recurrent decoding.
        donor_tokens = prompt + reference["output"]["token_ids"][:1]
        one_token = sampling.clone()
        one_token.max_tokens = 1
        donor = run("donor", donor_tokens, one_token)
        assert donor["output"]["num_cached_tokens"] == 0
        probe = run(
            "donor-hit-check", prompt + reference["output"]["token_ids"][:2], one_token
        )
        assert probe["output"]["num_cached_tokens"] == block_size
        warm_control = run("donor-warmed-uninterrupted", prompt, sampling)
        replay_sampling = sampling.clone()
        replay_sampling.extra_args = {"test_preempt_after": 16}
        resumed = run("donor-warmed-preempted", prompt, replay_sampling)

    for control in (donor, probe, warm_control):
        assert not control["events"]
    failures = [e for e in resumed["events"] if e["event"] == "allocation_failure"]
    preemptions = [e for e in resumed["events"] if e["event"] == "preempt"]
    resumes = [e for e in resumed["events"] if e["event"] == "resume"]
    assert len(failures) == len(preemptions) == 1 and resumes
    assert failures[0]["unrestricted_hit"] == block_size
    # Save the numerical outcome even when an old lookup crosses the boundary.
    report["exact"] = all(
        row["output"][field] == reference["output"][field]
        for row in (warm_control, resumed)
        for field in ("token_ids", "logprobs")
    )
    checkpoint()
    assert report["exact"], f"See {tmp_path / 'shared-cache-results.json'}"
    assert resumes[0]["computed_tokens"] == 0
    assert resumes[0]["scheduled_tokens"] == len(prompt)
    assert resumes[0]["output_tokens"] >= 16
    assert all(e["scheduled_tokens"] == 1 for e in resumes[1:])
    assert resumed["output"]["num_cached_tokens"] == 0
    report["complete"] = True
    checkpoint()


@pytest.mark.parametrize(
    "eager,async_scheduling,cache_enabled,late,sampling_mode",
    [
        pytest.param(True, False, True, False, "greedy", id="eager-sync"),
        pytest.param(False, True, True, False, "greedy", id="compiled-async"),
        pytest.param(False, True, False, False, "greedy", id="cache-off"),
        pytest.param(False, True, True, True, "greedy", id="late"),
        pytest.param(False, True, True, False, "seeded", id="seeded"),
        pytest.param(False, True, True, False, "structured", id="structured"),
    ],
)
@create_new_process_for_each_test(method="spawn")
def test_qwen_gdn_preemption_exactness(
    vllm_runner,
    tmp_path: Path,
    eager: bool,
    async_scheduling: bool,
    cache_enabled: bool,
    late: bool,
    sampling_mode: str,
):
    model, settings = _engine_settings(eager, async_scheduling)
    # The pressure budget is sized for this model's four hybrid cache groups.
    assert model == "Qwen/Qwen3.5-0.8B", "Pressure test requires the 0.8B model"
    trace = tmp_path / "scheduler.jsonl"
    trace.write_text("")
    scheduler_name = (
        "ObservedAsyncScheduler" if async_scheduling else "ObservedScheduler"
    )
    settings.update(
        max_num_seqs=2,
        max_model_len=1536 if late else 1024,
        num_gpu_blocks_override=17 if late else 9,
        scheduler_cls=f"{__name__}.{scheduler_name}",
    )
    settings.setdefault("additional_config", {})["preemption_trace_path"] = str(trace)
    generation_tokens = 624 if late else 48
    sampling = SamplingParams(
        temperature=0.8 if sampling_mode == "seeded" else 0.0,
        seed=3407,
        max_tokens=generation_tokens,
        ignore_eos=True,
        logprobs=5,
        structured_outputs=(
            StructuredOutputsParams(regex=r"( report| bank| rules| records)+")
            if sampling_mode == "structured"
            else None
        ),
    )
    report = {"settings": settings, "sampling_mode": sampling_mode, "runs": []}

    def events():
        return [json.loads(line) for line in trace.read_text().splitlines()]

    with vllm_runner(
        model,
        enable_prefix_caching=cache_enabled,
        mamba_cache_mode="align" if cache_enabled else "none",
        **settings,
    ) as runner:
        llm = runner.llm
        [layout] = llm.collective_rpc("prefix_cache_layout")
        report["layout"] = layout
        block_size = layout["attention_block_size"]
        assert block_size == 576, "Recheck the pressure budget if cache layout changes"
        prompts = _prompts(llm.get_tokenizer(), block_size, 2048)
        prompts = [
            prompts[name][: block_size - 16] for name in ("block_minus1", "block_exact")
        ]

        def run(indices):
            if cache_enabled:
                assert llm.reset_prefix_cache()
            start = len(events())
            inputs = [TokensPrompt(prompt_token_ids=prompts[i]) for i in indices]
            if late and len(indices) == 2:
                # Both requests must cross the later block boundary together.
                # Otherwise staggered arrivals can release old running-state
                # blocks before the peer needs them and avoid memory pressure.
                # Queue both while paused, then let normal allocation decide.
                core = llm.llm_engine.engine_core
                core.call_utility("pause_scheduler", "keep", False)
                try:
                    llm.enqueue(inputs, sampling, use_tqdm=False)
                finally:
                    core.call_utility("resume_scheduler")
                outputs = llm.wait_for_completion(use_tqdm=False)
            else:
                outputs = llm.generate(inputs, sampling, use_tqdm=False)
            row = {
                "indices": indices,
                "outputs": [_snapshot(output) for output in outputs],
                "events": events()[start:],
            }
            report["runs"].append(row)
            (tmp_path / "results.json").write_text(json.dumps(report, indent=2))
            return row

        controls = [run([i]) for i in range(2)]
        pressure = run([0, 1])

    assert all(not control["events"] for control in controls)
    preemptions = [e for e in pressure["events"] if e["event"] == "preempt"]
    resumes = [e for e in pressure["events"] if e["event"] == "resume"]
    assert len(preemptions) == 1 and resumes
    first = resumes[0]
    assert first["computed_tokens"] == 0
    assert first["output_tokens"] >= (576 if late else 8)
    assert first["output_tokens"] < generation_tokens
    assert first["scheduled_tokens"] == first["prompt_tokens"]
    assert all(e["preemptions"] == 1 for e in resumes)
    assert all(e["scheduled_tokens"] == 1 for e in resumes[1:])
    replay_end = first["prompt_tokens"] + first["output_tokens"]
    replay = [e for e in resumes if e["computed_tokens"] < replay_end]
    assert sum(e["scheduled_tokens"] for e in replay) == replay_end
    for index, control in enumerate(controls):
        expected = control["outputs"][0]
        actual = pressure["outputs"][index]
        assert len(actual["token_ids"]) == generation_tokens
        assert actual["num_cached_tokens"] == expected["num_cached_tokens"] == 0
        assert actual["token_ids"] == expected["token_ids"]
        assert actual["logprobs"] == expected["logprobs"]
