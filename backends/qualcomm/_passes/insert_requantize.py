# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List

import torch

from executorch.backends.qualcomm.utils.constants import (
    QCOM_QUANT_ATTRS,
    QCOM_QUANTIZED_IO,
    QCOM_REQUANTIZE,
)

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class InsertRequantize(ExportPass):
    """
    This pass inserts convert op for operators which have
    different quantization specs in input and activation.
    Convert OP is a specific op which helps to requantize in Qnn backend
    """

    # Storing ops that has multi output but run _single_output_annotation logic
    # instead of _multi_output_annotation. Ops might be added into this set because
    # we don't use the 2nd output, 2nd output is an integer, etc.
    multi_output_op_ignore_set = {
        exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
        exir_ops.edge.aten.topk.default,
    }

    def __init__(
        self,
    ):
        super(InsertRequantize, self).__init__()

    def _make_hashable(self, value):
        if isinstance(value, dict):
            return tuple(sorted(value.items()))
        return value

    def _invert_dict(self, requantize_dict):
        inverted_dict = defaultdict(list)
        for user_node_name, quant_attr in requantize_dict.items():
            hashable_quant_attr = self._make_hashable(quant_attr)
            inverted_dict[hashable_quant_attr].append(user_node_name)
        return inverted_dict

    def _quant_values_equal(self, lhs, rhs) -> bool:
        if lhs is None or rhs is None:
            return lhs is rhs
        if isinstance(lhs, torch.Tensor) or isinstance(rhs, torch.Tensor):
            try:
                return torch.equal(torch.as_tensor(lhs), torch.as_tensor(rhs))
            except (TypeError, ValueError, RuntimeError):
                return False
        if isinstance(lhs, dict) and isinstance(rhs, dict):
            return lhs.keys() == rhs.keys() and all(
                self._quant_values_equal(lhs[key], rhs[key]) for key in lhs
            )
        if isinstance(lhs, (list, tuple)) and isinstance(rhs, (list, tuple)):
            return len(lhs) == len(rhs) and all(
                self._quant_values_equal(left, right)
                for left, right in zip(lhs, rhs)
            )
        try:
            return bool(lhs == rhs)
        except (TypeError, ValueError):
            return False

    def _quant_attrs_equal(self, lhs: Dict, rhs: Dict) -> bool:
        def encoding_family(encoding):
            name = str(encoding)
            for family in ("per_tensor", "per_channel", "per_block"):
                if family in name:
                    return family
            return name

        if encoding_family(lhs.get("encoding")) != encoding_family(
            rhs.get("encoding")
        ):
            return False
        # Quantize and Dequantize overloads describe the same stored encoding
        # but use different operator objects, and Dequantize may additionally
        # expose an out_dtype=None schema field. Compare every numerical/storage
        # attribute while normalizing only those two representation details.
        ignored = {"encoding", "out_dtype"}
        for key in (lhs.keys() | rhs.keys()) - ignored:
            if not self._quant_values_equal(lhs.get(key), rhs.get(key)):
                return False
        return True

    def _insert_to_copy(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.node,
        quant_attr: Dict,
        user_nodes: List[str],
    ):
        source_quant_attr = node.meta.get(QCOM_QUANT_ATTRS)
        if source_quant_attr is not None and self._quant_attrs_equal(
            source_quant_attr, quant_attr
        ):
            # The annotation graph can carry an explicit requant edge even
            # when both sides resolve to exactly the same encoding. Emitting a
            # QNN Convert for that identity is unnecessary, and QAIRT 2.37's
            # HTP optimizer has no QNN_Convert_w_scale kernel for PCQ tensors.
            return
        with graph_module.graph.inserting_after(node):
            users = list(node.users.keys())
            inserted_n = graph_module.graph.create_node(
                "call_function",
                exir_ops.edge.aten._to_copy.default,
                (node,),
            )
            inserted_n.meta["val"] = node.meta["val"]
            inserted_n.meta[QCOM_QUANT_ATTRS] = quant_attr

            # create node and replace input
            if node.meta.get(QCOM_QUANTIZED_IO):
                inserted_n.meta[QCOM_QUANTIZED_IO] = node.meta[QCOM_QUANTIZED_IO]

            for user in filter(lambda u: u.name in user_nodes, users):
                user.replace_input_with(node, inserted_n)

    # TODO: Implement this function when we have an op with
    # multiple outputs that requires quant attributes.
    def _multi_output_annotation(self) -> None:
        raise NotImplementedError("requant is not implemented for multi output yet")

    def _single_output_annotation(
        self, gm: torch.fx.GraphModule, n: torch.fx.node
    ) -> None:
        # {user_node_name: quant_attr}
        requantize_dict = n.meta.pop(QCOM_REQUANTIZE)
        # {quant_attr: user_node_name_list}
        group_quant_attr_dict = self._invert_dict(requantize_dict)

        for hashable_quant_attr, user_nodes in group_quant_attr_dict.items():
            user_nodes_copy = user_nodes.copy()
            self._insert_to_copy(gm, n, dict(hashable_quant_attr), user_nodes_copy)

    def _insert(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for n in graph_module.graph.nodes:
            if QCOM_REQUANTIZE in n.meta:
                (
                    self._single_output_annotation(graph_module, n)
                    if isinstance(
                        n.meta["val"], torch._subclasses.fake_tensor.FakeTensor
                    )
                    or n.target in self.multi_output_op_ignore_set
                    else self._multi_output_annotation()
                )

    def call(self, graph_module: torch.fx.GraphModule):
        self._insert(graph_module)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
