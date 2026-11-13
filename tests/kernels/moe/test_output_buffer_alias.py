# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real modular runner's default-off output alias contract."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe import modular_kernel as mk


@pytest.mark.parametrize(
    "enabled,kind",
    [
        (False, "valid"),
        (True, "valid"),
        (True, "shape"),
        (True, "dtype"),
        (True, "strided"),
    ],
)
def test_output_alias_contract(enabled, kind):
    assert mk.FusedMoEExpertsModular.supports_output_buffer_alias.fget(None) is False
    hidden = torch.arange(8, dtype=torch.float32).view(2, 4)
    fallback = torch.empty_like(hidden)
    alias = torch.full_like(hidden, -99)
    if kind == "shape":
        alias = torch.empty(4, 2)
    elif kind == "dtype":
        alias = torch.empty_like(hidden, dtype=torch.float64)
    elif kind == "strided":
        alias = torch.empty(4, 2).T

    class Experts:
        supports_output_buffer_alias = enabled
        a2_scale = None

        def moe_problem_size(self, *args):
            return 1, 2, 4, 4, 1

        def apply(self, *, output, hidden_states, **kwargs):
            output.copy_(hidden_states * 2)

    runner = SimpleNamespace(
        fused_experts=Experts(), _allocate_buffers=lambda *a: (None, None, fallback)
    )
    output = mk.FusedMoEKernelModularImpl._fused_experts(
        runner,
        hidden.dtype,
        hidden,
        None,
        hidden,
        hidden,
        torch.ones(2, 1),
        torch.zeros(2, 1, dtype=torch.int32),
        None,
        1,
        1,
        None,
        False,
        None,
        output_alias=alias,
    )
    selected = enabled and kind == "valid"
    assert output.data_ptr() == (alias if selected else fallback).data_ptr()
    torch.testing.assert_close(output, hidden * 2)
    hidden.add_(3)
    output = mk.FusedMoEKernelModularImpl._fused_experts(
        runner,
        hidden.dtype,
        hidden,
        None,
        hidden,
        hidden,
        torch.ones(2, 1),
        torch.zeros(2, 1, dtype=torch.int32),
        None,
        1,
        1,
        None,
        False,
        None,
        output_alias=alias,
    )
    torch.testing.assert_close(output, hidden * 2)
