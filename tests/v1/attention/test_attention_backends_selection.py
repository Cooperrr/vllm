# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for mamba attention backend selectors."""

from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.short_conv import ShortConv
from vllm.model_executor.models.minimax_text_01 import MiniMaxText01LinearAttention
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    QwenGDNAttentionBackend,
)
from vllm.v1.attention.backends.linear_attn import LinearAttentionBackend
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionBackend
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionBackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionBackend
from vllm.v1.attention.selector import (
    _cached_get_mamba_attn_backend,
    get_mamba_attn_backend,
)


@pytest.mark.parametrize("batch_invariant", [False, True])
@pytest.mark.parametrize(
    "model_type", ["qwen3_5_text", "qwen3_5_moe_text", "qwen3_next", "other_gdn"]
)
def test_gdn_batch_invariant_model_gate(monkeypatch, batch_invariant, model_type):
    """Use the real model-to-backend selection and startup support check."""
    from vllm import platforms

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability=lambda capability: capability == 120,
        ),
    )
    layer = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type)
        )
    )
    mamba_type = QwenGatedDeltaNetAttention.mamba_type.__get__(layer)
    _cached_get_mamba_attn_backend.cache_clear()
    try:
        if batch_invariant and model_type != "qwen3_5_text":
            with pytest.raises(
                RuntimeError,
                match=r"supported only for dense Qwen3\.5.*qwen3_5_text",
            ):
                get_mamba_attn_backend(mamba_type)
        else:
            expected = (
                QwenGDNAttentionBackend
                if model_type == "qwen3_5_text"
                else GDNAttentionBackend
            )
            assert get_mamba_attn_backend(mamba_type) is expected
    finally:
        _cached_get_mamba_attn_backend.cache_clear()


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("qwen3_5_text", MambaAttentionBackendEnum.QWEN_GDN_ATTN),
        ("qwen3_5_moe_text", MambaAttentionBackendEnum.GDN_ATTN),
        ("qwen3_next", MambaAttentionBackendEnum.GDN_ATTN),
    ],
)
def test_only_dense_qwen_gdn_uses_batch_invariant_backend(model_type, expected):
    layer = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type)
        )
    )

    assert QwenGatedDeltaNetAttention.mamba_type.__get__(layer) == expected


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        (80, False),
        (86, True),
        (89, True),
        (90, True),
        (100, False),
        (120, True),
    ],
)
def test_qwen_gdn_batch_invariance_capability_gate(
    monkeypatch: pytest.MonkeyPatch,
    capability: int,
    expected: bool,
):
    from vllm import platforms

    platform = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability=lambda candidate: candidate == capability,
    )
    monkeypatch.setattr(platforms, "current_platform", platform)

    assert QwenGDNAttentionBackend.supports_batch_invariance() is expected


@pytest.mark.parametrize(
    "layer_class, init_kwargs, expected_backend, expected_mamba_type",
    [
        (
            MambaMixer,
            dict(
                hidden_size=128,
                ssm_state_size=16,
                conv_kernel_size=4,
                intermediate_size=256,
                time_step_rank=8,
                use_conv_bias=True,
                use_bias=False,
                use_rms_norm=True,
            ),
            Mamba1AttentionBackend,
            MambaAttentionBackendEnum.MAMBA1,
        ),
        (
            MambaMixer2,
            dict(
                hidden_size=128,
                ssm_state_size=16,
                conv_kernel_size=4,
                intermediate_size=256,
                use_conv_bias=True,
                use_bias=False,
                n_groups=1,
                num_heads=8,
                head_dim=32,
            ),
            Mamba2AttentionBackend,
            MambaAttentionBackendEnum.MAMBA2,
        ),
        (
            MiniMaxText01LinearAttention,
            dict(
                config=SimpleNamespace(
                    hidden_size=256,
                    num_attention_heads=8,
                    head_dim=32,
                    num_hidden_layers=12,
                    block=64,
                ),
                prefix="layers.0.self_attn",
            ),
            LinearAttentionBackend,
            MambaAttentionBackendEnum.LINEAR,
        ),
        (
            ShortConv,
            dict(
                config=SimpleNamespace(conv_L_cache=32, conv_bias=True),
                dim=128,
                layer_idx=0,
            ),
            ShortConvAttentionBackend,
            MambaAttentionBackendEnum.SHORT_CONV,
        ),
    ],
)
def test_mamba_layers_get_attn_backend(
    default_vllm_config,
    dist_init,
    layer_class,
    init_kwargs,
    expected_backend,
    expected_mamba_type,
):
    """Test that Mamba-like layers return the correct attention backend."""
    if layer_class is MiniMaxText01LinearAttention:
        init_kwargs["vllm_config"] = default_vllm_config
    layer = layer_class(**init_kwargs)

    backend_class = layer.get_attn_backend()
    assert backend_class is expected_backend
    assert layer.mamba_type == expected_mamba_type


@pytest.mark.parametrize(
    "layer_class,expected_backend,expected_mamba_type",
    [
        (MambaMixer, Mamba1AttentionBackend, MambaAttentionBackendEnum.MAMBA1),
        (MambaMixer2, Mamba2AttentionBackend, MambaAttentionBackendEnum.MAMBA2),
        (
            MiniMaxText01LinearAttention,
            LinearAttentionBackend,
            MambaAttentionBackendEnum.LINEAR,
        ),
        (ShortConv, ShortConvAttentionBackend, MambaAttentionBackendEnum.SHORT_CONV),
    ],
)
def test_mamba_layers_have_unified_interface(
    layer_class, expected_backend, expected_mamba_type
):
    """Test that all Mamba layers have the unified get_attn_backend
    interface."""
    assert hasattr(layer_class, "get_attn_backend"), (
        f"{layer_class.__name__} should have get_attn_backend method"
    )
    assert hasattr(layer_class, "mamba_type"), (
        f"{layer_class.__name__} should have mamba_type property"
    )
