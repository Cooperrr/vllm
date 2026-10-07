# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact cache-hit comparisons for dense Qwen3.5, sized for one RTX 3090.

See qwen_gdn_prefix_caching.md for setup, commands and validation limits.
"""

import json
import math
import os
import random
from pathlib import Path

import pytest
import torch

import vllm
import vllm.envs as envs
from tests.utils import create_new_process_for_each_test
from vllm import SamplingParams, TokensPrompt
from vllm.outputs import RequestOutput
from vllm.v1.attention.backends.gdn_attn import QwenGDNAttentionBackend

pytestmark = [
    pytest.mark.skipif(
        not QwenGDNAttentionBackend.supports_batch_invariance(),
        reason="Requires the patched Qwen GDN CUDA backend (SM86/89/90/120)",
    ),
    pytest.mark.timeout(1800),
]


def _snapshot(output: RequestOutput) -> dict:
    """Keep exact generated-token probabilities, without requesting prompt logits."""
    assert output.finished
    assert len(output.outputs) == 1
    completion = output.outputs[0]
    assert completion.logprobs is not None
    assert len(completion.logprobs) == len(completion.token_ids)
    steps = []
    for token_id, probabilities in zip(completion.token_ids, completion.logprobs):
        assert token_id in probabilities
        assert all(math.isfinite(value.logprob) for value in probabilities.values())
        steps.append(
            {
                str(token): (value.logprob.hex(), value.rank)
                for token, value in sorted(probabilities.items())
            }
        )
    return {
        "token_ids": list(completion.token_ids),
        "logprobs": steps,
        "num_cached_tokens": output.num_cached_tokens,
    }


def _prompts(tokenizer, block_size: int, token_budget: int) -> dict[str, list[int]]:
    lengths = {
        "gdn_minus1": 63,
        "gdn_exact": 64,
        "gdn_plus1": 65,
        "block_minus1": block_size - 1,
        "block_exact": block_size,
        "block_plus1": block_size + 1,
        "two_blocks_minus1": 2 * block_size - 1,
        "two_blocks_exact": 2 * block_size,
        "two_blocks_plus1": 2 * block_size + 1,
        "chunk_minus1": token_budget + 63,
        "chunk_exact": token_budget + 64,
        "chunk_plus1": token_budget + 65,
    }
    words = "report bank rules records review policy market risk customer annual".split()
    prompts = {}
    for index, (name, length) in enumerate(lengths.items()):
        rng = random.Random(3407 + index)
        text = f"Document {index}: " + " ".join(rng.choices(words, k=length))
        tokens = tokenizer.encode(text, add_special_tokens=False)
        assert len(tokens) >= length
        prompts[name] = tokens[:length]
    return prompts


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "compiled"])
@create_new_process_for_each_test(method="spawn")
def test_qwen_gdn_prefix_cache_exactness(
    vllm_runner,
    tmp_path: Path,
    enforce_eager: bool,
    async_scheduling: bool,
):
    """Compare uncached, cold, repeated and shared-prefix requests exactly."""
    assert envs.VLLM_BATCH_INVARIANT
    model = os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3.5-0.8B")
    token_budget = int(os.getenv("VLLM_GDN_TEST_TOKEN_BUDGET", "2048"))
    assert 128 <= token_budget <= 3072, "Use a token budget in [128, 3072]"
    model_len = 8192
    generation_tokens = 32
    settings = dict(
        dtype="bfloat16",
        language_model_only=True,
        trust_remote_code=False,
        tensor_parallel_size=1,
        max_model_len=model_len,
        max_num_seqs=4,
        max_num_batched_tokens=token_budget,
        enable_chunked_prefill=True,
        enforce_eager=enforce_eager,
        async_scheduling=async_scheduling,
        attention_config={"backend": "FLASH_ATTN"},
        additional_config={"gdn_prefill_backend": "triton"},
        gpu_memory_utilization=float(
            os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.65")
        ),
        seed=3407,
    )
    sampling = SamplingParams(
        temperature=0.0,
        seed=3407,
        max_tokens=generation_tokens,
        ignore_eos=True,
        logprobs=5,
    )
    donor_sampling = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    failures: list[str] = []
    report = {
        "model": model,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "vllm": vllm.__version__,
        "vllm_source": vllm.__file__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "settings": settings,
        "engines": [],
        "runs": [],
        "failures": failures,
        "complete": False,
    }
    report_path = tmp_path / "prefix-cache-results.json"

    def checkpoint():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    def check(condition: bool, message: str):
        if not condition:
            failures.append(message)

    def generate(llm, names, prompts, label):
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=prompts[name]) for name in names],
            sampling,
            use_tqdm=False,
        )
        assert len(outputs) == len(names)
        snapshots = {}
        for name, output in zip(names, outputs):
            assert output.prompt_token_ids == prompts[name]
            snapshot = _snapshot(output)
            assert len(snapshot["token_ids"]) == generation_tokens
            snapshots[name] = snapshot
            report["runs"].append({"scenario": label, "prompt": name, **snapshot})
            print(
                f"{label}/{name}: prompt={len(prompts[name])}, "
                f"cached={snapshot['num_cached_tokens']}",
                flush=True,
            )
        checkpoint()
        return snapshots

    def compare(actual, expected, label):
        for name, snapshot in actual.items():
            reference = expected[name]
            for field in ("token_ids", "logprobs"):
                if snapshot[field] != reference[field]:
                    first = next(
                        i
                        for i, (left, right) in enumerate(
                            zip(snapshot[field], reference[field])
                        )
                        if left != right
                    )
                    failures.append(f"{label}/{name}: {field} differ at step {first}")
        checkpoint()

    def check_hits(snapshots, names_with_hits, label):
        for name, snapshot in snapshots.items():
            hits = snapshot["num_cached_tokens"]
            check(isinstance(hits, int), f"{label}/{name}: missing cache-hit counter")
            if isinstance(hits, int):
                if name in names_with_hits:
                    check(hits > 0, f"{label}/{name}: expected a real prefix-cache hit")
                    check(
                        hits <= (len(prompts[name]) - 1) // block_size * block_size,
                        f"{label}/{name}: invalid hit length {hits}",
                    )
                    check(
                        hits % block_size == 0,
                        f"{label}/{name}: unaligned hit length {hits}",
                    )
                else:
                    check(hits == 0, f"{label}/{name}: unexpected cache hit {hits}")
        checkpoint()

    def describe(llm):
        config = llm.llm_engine.vllm_config
        assert config.model_config.hf_text_config.model_type == "qwen3_5_text"
        cache = config.cache_config
        report["engines"].append(
            {
                "prefix_caching": cache.enable_prefix_caching,
                "block_size": cache.block_size,
                "mamba_cache_mode": cache.mamba_cache_mode,
                "mamba_ssm_cache_dtype": cache.mamba_ssm_cache_dtype,
                "compilation_mode": str(config.compilation_config.mode),
                "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
                "async_scheduling": config.scheduler_config.async_scheduling,
            }
        )
        checkpoint()
        return cache

    checkpoint()
    print(f"Prefix-cache report: {report_path}", flush=True)
    # Separate engines ensure that 'uncached' really disables prefix caching,
    # rather than just clearing the cache while retaining cache-mode scheduling.
    with vllm_runner(model, enable_prefix_caching=False, **settings) as runner:
        llm = runner.llm
        cache = describe(llm)
        block_size = cache.block_size
        assert block_size is not None and block_size >= 64
        assert block_size <= token_budget, (
            f"Cache block {block_size} exceeds budget {token_budget}; "
            "increase VLLM_GDN_TEST_TOKEN_BUDGET"
        )
        prompts = _prompts(llm.get_tokenizer(), block_size, token_budget)
        assert max(map(len, prompts.values())) + generation_tokens < model_len
        report["prompts"] = prompts
        reference = {}
        for name in prompts:
            result = generate(llm, [name], prompts, "uncached-solo")
            check_hits(result, set(), "uncached-solo")
            reference.update(result)
        mixed_names = ["chunk_plus1", "block_plus1", "gdn_plus1", "two_blocks_plus1"]
        for order, names in enumerate((mixed_names, mixed_names[::-1])):
            label = f"uncached-mixed-{order}"
            result = generate(llm, names, prompts, label)
            check_hits(result, set(), label)
            compare(result, reference, label)
        del llm

    with vllm_runner(
        model, enable_prefix_caching=True, mamba_cache_mode="align", **settings
    ) as runner:
        llm = runner.llm
        cache = describe(llm)
        assert cache.block_size == block_size
        assert cache.mamba_cache_mode == "align"
        for name in prompts:
            assert llm.reset_prefix_cache()
            cold = generate(llm, [name], prompts, "cold-solo")
            check_hits(cold, set(), "cold-solo")
            compare(cold, reference, "cold-solo")
            warm = generate(llm, [name], prompts, "warm-solo")
            expected_hits = {name} if len(prompts[name]) > block_size else set()
            check_hits(warm, expected_hits, "warm-solo")
            compare(warm, reference, "warm-solo")
            if expected_hits:
                assert llm.reset_prefix_cache()
                # Ending the donor exactly at a block boundary makes that state
                # available even in 'align' mode, which does not save every block.
                llm.generate(
                    [TokensPrompt(prompt_token_ids=prompts[name][:block_size])],
                    donor_sampling,
                    use_tqdm=False,
                )
                partial = generate(llm, [name], prompts, "partial-solo")
                check_hits(partial, {name}, "partial-solo")
                check(
                    partial[name]["num_cached_tokens"] == block_size,
                    f"partial-solo/{name}: expected exactly one cached block",
                )
                compare(partial, reference, "partial-solo")

        for order, names in enumerate((mixed_names, mixed_names[::-1])):
            assert llm.reset_prefix_cache()
            label = f"cold-mixed-{order}"
            cold = generate(llm, names, prompts, label)
            check_hits(cold, set(), label)
            compare(cold, reference, label)
            assert llm.reset_prefix_cache()
            # Warm just one request: the batch contains a real hit alongside
            # fresh short and long requests, exercising mixed decode/prefill.
            needle = "chunk_plus1"
            llm.generate(
                [TokensPrompt(prompt_token_ids=prompts[needle][:block_size])],
                donor_sampling,
                use_tqdm=False,
            )
            label = f"partial-mixed-{order}"
            mixed = generate(llm, names, prompts, label)
            check_hits(mixed, {needle}, label)
            compare(mixed, reference, label)

        assert llm.reset_prefix_cache()
        reset = generate(llm, ["chunk_plus1"], prompts, "after-reset")
        check_hits(reset, set(), "after-reset")
        compare(reset, reference, "after-reset")
        del llm

    report["complete"] = True
    checkpoint()
    assert not failures, f"See {report_path}\n" + "\n".join(failures)
