# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
from executorch import exir
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.dialects._ops import ops
from executorch.backends.mediatek.partitioner import NeuropilotPartitioner, NeuropilotOperatorsSupport
from executorch.backends.mediatek._passes.normalize_clone_dim_order import normalize_clone_dim_order, normalized_clone

class AttentionLayout(torch.nn.Module):
    def forward(self, x):
        x = (x + 1).transpose(1, 2)
        return x.reshape(1, 5, 12) * 2

class CloneTests(unittest.TestCase):
    def test_attention_partition_and_values(self):
        x = torch.randn(1, 3, 5, 4)
        ep = exir.to_edge(torch.export.export(AttentionLayout(), (x,))).exported_program()
        clones = [n for n in ep.graph.nodes if n.target == ops.edge.dim_order_ops._clone_dim_order.default]
        self.assertEqual(len(clones), 1)
        before = ep.graph_module.code
        users_before = {n.name: [u.name for u in n.users] for n in ep.graph.nodes}
        partitioner = NeuropilotPartitioner([CompileSpec('platform-config', b'mt6989')])
        self.assertEqual(len(partitioner.partition(ep).partition_tags), 1)
        self.assertEqual(ep.graph_module.code, before)
        self.assertEqual({n.name: [u.name for u in n.users] for n in ep.graph.nodes}, users_before)
        expected = ep.module()(x)
        normalize_clone_dim_order(ep.graph_module)
        torch.testing.assert_close(ep.module()(x), expected, rtol=0, atol=0)
        self.assertEqual(clones[0].target, ops.edge.aten.clone.default)

    def test_layouts_and_skip(self):
        for order, value, supported in [
            ([0, 1, 2, 3], torch.randn(1, 3, 5, 4), True),
            (None, torch.randn(1, 3, 5, 4), True),
            ([0, 2, 3, 1], torch.randn(1, 3, 5, 4).contiguous(memory_format=torch.channels_last), False),
            (None, torch.randn(1, 3, 5, 4).contiguous(memory_format=torch.channels_last), False),
            ([0, 1, 3, 2], torch.randn(1, 3, 5, 4), False),
        ]:
            with self.subTest(order=order, supported=supported):
                graph = torch.fx.Graph();x = graph.placeholder('x')
                n = graph.call_function(ops.edge.dim_order_ops._clone_dim_order.default, (x,), {'dim_order': order})
                n.meta['val'] = value
                self.assertEqual(normalized_clone(n) is not None, supported)
                self.assertEqual(NeuropilotOperatorsSupport().is_node_supported(None, n), supported)
                self.assertFalse(NeuropilotOperatorsSupport(op_names_to_skip={n.name}).is_node_supported(None, n))
                self.assertEqual(n.target, ops.edge.dim_order_ops._clone_dim_order.default)

if __name__ == '__main__':
    unittest.main()
