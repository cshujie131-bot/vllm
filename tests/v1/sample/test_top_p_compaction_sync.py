# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-p scratch reuse must not change a fixed input's filtering result."""

import pytest
import torch

from vllm.triton_utils import HAS_TRITON
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton


@pytest.mark.skipif(
    not HAS_TRITON or not torch.cuda.is_available(),
    reason="Requires a CUDA-compatible Triton device",
)
@pytest.mark.parametrize("capture_graph", [False, True])
def test_top_p_compaction_is_repeatable(capture_graph):
    generator = torch.Generator(device="cpu").manual_seed(780)
    source = torch.randn((8, 163840), generator=generator, device="cpu").cuda()
    logits = source.clone()
    p = torch.tensor([0.95, 0.5] * 4, device="cuda")
    other = -source
    other_p = torch.full((8,), 0.95, device="cuda")

    def apply_filter():
        logits.copy_(source)
        apply_top_k_top_p_triton(logits, None, p)

    apply_filter()
    torch.cuda.synchronize()
    if capture_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            apply_filter()
        run = graph.replay
    else:
        run = apply_filter

    run()
    expected = logits.cpu()
    for _ in range(32):
        apply_top_k_top_p_triton(other.clone(), None, other_p)
        run()
        assert torch.equal(logits.cpu(), expected), (
            "top-p mask must not depend on scratch history"
        )
