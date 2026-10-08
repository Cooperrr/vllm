# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn


@pytest.mark.parametrize(
    "capability,requested,expected",
    [
        (86, None, "triton"),
        (120, None, "triton"),
        (120, "auto", "triton"),
        (120, "triton", "triton"),
        (90, None, "flashinfer"),
        (90, "triton", "triton"),
        (100, None, "flashinfer"),
        (100, "triton", "triton"),
    ],
)
def test_gdn_prefill_backend_selection(monkeypatch, capability, requested, expected):
    """Check selection only; mocked hardware does not validate kernel numerics."""
    monkeypatch.setattr(gdn.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        gdn.current_platform, "is_device_capability", lambda cap: capability == cap
    )
    monkeypatch.setattr(
        gdn.current_platform,
        "is_device_capability_family",
        lambda cap: capability // 10 == cap // 10,
    )
    monkeypatch.setattr(gdn.current_platform, "get_cuda_runtime_major", lambda: 13)
    monkeypatch.setattr(gdn, "_is_libs_cu13_install_intact", lambda: True)
    config = SimpleNamespace(
        additional_config=(
            {} if requested is None else {"gdn_prefill_backend": requested}
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(linear_key_head_dim=128)
        ),
    )

    assert gdn._resolve_gdn_prefill_backend(config) == (
        requested or "auto",
        expected,
    )
