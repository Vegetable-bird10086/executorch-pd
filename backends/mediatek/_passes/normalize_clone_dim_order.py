# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Optional

import torch
from torch.fx.immutable_collections import immutable_dict
from executorch.exir.dialects._ops import ops
from executorch.exir.dim_order_utils import get_memory_format


def normalized_clone(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """Return an importer-compatible clone without mutating the partition graph.

    Neuropilot's graph interface requires default dimension order. Preserve-format
    copies are supported only when their output also has default dimension order.
    Other layouts remain on the host, rather than silently changing their meaning.
    """
    if node.target != ops.edge.dim_order_ops._clone_dim_order.default:
        return None
    order = node.kwargs.get("dim_order")
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    if order is not None and list(order) != list(range(value.dim())):
        return None
    if order is None and tuple(value.dim_order()) != tuple(range(value.dim())):
        return None
    # ATen clone has no non_blocking argument. Do not discard an explicit request.
    if node.kwargs.get("non_blocking", False):
        return None
    clone = copy.copy(node)
    clone.target = ops.edge.aten.clone.default
    # This detached copy is only an importer query. The public kwargs setter
    # updates input users in the original graph, which must remain untouched.
    clone._kwargs = immutable_dict(
        memory_format=get_memory_format(None if order is None else list(order))
    )
    return clone


def normalize_clone_dim_order(graph_module: torch.fx.GraphModule) -> None:
    for node in graph_module.graph.nodes:
        clone = normalized_clone(node)
        if clone is not None:
            node.target = clone.target
            node.kwargs = clone.kwargs
    graph_module.graph.lint()
    graph_module.recompile()
