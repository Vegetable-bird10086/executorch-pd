# Copyright (c) 2024 MediaTek Inc.
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import torch
from torch.fx import GraphModule
from torchao.quantization.pt2e.quantizer import Quantizer
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY
from torchao.quantization.pt2e import MinMaxObserver

from .._passes.decompose_scaled_dot_product_attention import (
    DecomposeScaledDotProductAttention,
)
from .annotator import annotate, OP_TO_ANNOTATOR
from .qconfig import get_quant_config, Precision


class NeuropilotQuantizer(Quantizer):

    def __init__(self):
        super().__init__()

        # TODO: Provide setter functions for those attributes
        self._precision = Precision.A8W8
        self._is_per_channel = True
        self._is_qat = False
        self._activation_observer_cls = MinMaxObserver
        self._skip_mlp_output_quantization = False
        self._module_name_precisions = {}

    def setup_precision(self, precision: Precision) -> None:
        self._precision = precision

    def set_activation_observer(self, observer_cls) -> None:
        self._activation_observer_cls = observer_cls

    def set_skip_mlp_output_quantization(self, enabled: bool = True) -> None:
        self._skip_mlp_output_quantization = enabled

    def set_module_name_precision(
        self, module_name: str, precision: Precision
    ) -> None:
        """Override affine weight precision for a named module.

        Module names are matched against the final component of the exported
        ``nn_module_stack`` path, so this remains stable when torch.export adds
        a root prefix such as ``L['self']``.
        """
        if not module_name or "." in module_name:
            raise ValueError("module_name must be one non-empty path component")
        self._module_name_precisions[module_name] = precision

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        model = DecomposeScaledDotProductAttention()(model).graph_module
        return model

    def annotate(self, model: GraphModule) -> GraphModule:
        self._annotate(model)
        return model

    def validate(self, model: GraphModule) -> None:
        pass

    def _annotate(self, gm: GraphModule) -> None:
        quant_config = get_quant_config(
            self._precision,
            self._is_per_channel,
            self._is_qat,
            self._activation_observer_cls,
        )
        self._annotate_module_name_overrides(gm)
        annotate(gm.graph, quant_config)
        if self._skip_mlp_output_quantization:
            self._remove_mlp_output_quantization(gm)

    def _annotate_module_name_overrides(self, gm: GraphModule) -> None:
        for module_name, precision in self._module_name_precisions.items():
            quant_config = get_quant_config(
                precision,
                self._is_per_channel,
                self._is_qat,
                self._activation_observer_cls,
            )
            matched = 0
            for node in gm.graph.nodes:
                if (
                    node.op != "call_function"
                    or node.target != torch.ops.aten.linear.default
                ):
                    continue
                paths = [
                    value[0]
                    for value in node.meta.get("nn_module_stack", {}).values()
                    if isinstance(value, tuple)
                ]
                if not paths:
                    continue
                path = paths[-1]
                if path == module_name or path.endswith(f".{module_name}"):
                    OP_TO_ANNOTATOR[node.target](node, quant_config)
                    matched += 1
            # A module override is absent from chunks that do not own it. The
            # tail chunk must still fail loudly if graph metadata changes.
            if matched == 0 and any(
                value[0] == module_name or value[0].endswith(f".{module_name}")
                for node in gm.graph.nodes
                for value in node.meta.get("nn_module_stack", {}).values()
                if isinstance(value, tuple)
            ):
                raise RuntimeError(
                    f"Module precision override matched no affine op: {module_name}"
                )

    @staticmethod
    def _remove_mlp_output_quantization(gm: GraphModule) -> None:
        skip_nodes = set()
        residuals = {}
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            paths = [
                value[0]
                for value in node.meta.get("nn_module_stack", {}).values()
                if isinstance(value, tuple)
            ]
            if not paths:
                continue
            path = paths[-1]
            if path.endswith(".mlp") and node.target == torch.ops.aten.mul.Tensor:
                skip_nodes.add(node)
            elif (
                path.endswith(".mlp.down_proj")
                and node.target == torch.ops.aten.linear.default
            ):
                skip_nodes.add(node)
            elif (
                path.startswith("layers.")
                and path.count(".") == 1
                and node.target == torch.ops.aten.add.Tensor
            ):
                residuals.setdefault(path, []).append(node)
        for nodes in residuals.values():
            skip_nodes.add(nodes[-1])
        for node in skip_nodes:
            annotation = node.meta.get(Q_ANNOTATION_KEY)
            if annotation is not None:
                annotation.output_qspec = None
