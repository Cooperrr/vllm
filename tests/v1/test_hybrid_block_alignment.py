# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.config import CacheConfig
from vllm.model_executor.models import ModelRegistry
from vllm.platforms.interface import Platform
from vllm.v1.attention.backend import MultipleOf


@pytest.mark.parametrize("batch_invariant", [False, True])
@pytest.mark.parametrize("cache_mode", ["none", "align"])
@pytest.mark.parametrize("model_type", ["qwen3_5_text", "other_hybrid"])
@pytest.mark.parametrize("state_tokens,backend_alignment", [(524, 16), (530, 32)])
def test_hybrid_page_alignment_preserves_gdn_chunks(
    monkeypatch,
    batch_invariant,
    cache_mode,
    model_type,
    state_tokens,
    backend_alignment,
):
    # Reproduce the 528- and 544-token pages seen in serving and local tests.
    # Both need 576 tokens to preserve the GDN 64-token chunk grid.
    state_bytes = state_tokens * 1024
    model_cls = SimpleNamespace(
        get_mamba_state_shape_from_config=lambda _: ((state_bytes // 4,),),
        get_mamba_state_dtype_from_config=lambda _: (torch.float32,),
    )
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    monkeypatch.setattr(
        ModelRegistry, "resolve_model_cls", lambda *args, **kwargs: (model_cls, None)
    )
    monkeypatch.setattr(
        "vllm.config.vllm.set_current_vllm_config", lambda _: nullcontext()
    )
    cache = CacheConfig(block_size=16, mamba_cache_mode=cache_mode)
    config = SimpleNamespace(
        cache_config=cache,
        parallel_config=None,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            use_mla=False,
            architecture="TestHybrid",
            hf_text_config=SimpleNamespace(model_type=model_type),
            get_num_kv_heads=lambda _: 2,
            get_head_size=lambda: 128,
        ),
    )
    backend = SimpleNamespace(
        get_supported_kernel_block_sizes=lambda: [MultipleOf(backend_alignment)]
    )

    Platform._align_hybrid_block_size(config, backend)

    expected = (
        576
        if batch_invariant and model_type == "qwen3_5_text"
        else (528 if state_tokens == 524 else 544)
    )
    assert cache.block_size == expected
    assert cache.mamba_page_size_padded == expected * 1024
    assert cache.mamba_page_size_padded >= state_bytes
    if cache_mode == "align":
        assert cache.mamba_block_size == expected
