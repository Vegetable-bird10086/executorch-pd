# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import cast, Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import (
    QCOM_AXIS_ORDER,
    QCOM_DATA,
    QCOM_QUANT_ATTRS,
    QCOM_SCALES,
)

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpGather, OpStridedSlice, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class StrideSlice(NodeVisitor):
    target = ["aten.slice_copy.Tensor"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        tensor_type = PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE

        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            tensor_type,
            nodes_to_wrappers,
        )

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        dim = cast(int, node.args[1])
        if QCOM_AXIS_ORDER in node.meta:
            dim = node.meta[QCOM_AXIS_ORDER].index(dim)
        if dim < 0:
            dim = dim % len(input_tensor.shape)

        start = 0 if node.args[2] is None else cast(int, node.args[2])
        if start < 0:
            start = start % input_tensor.shape[dim]

        if len(node.args) > 3 and node.args[3] is not None:
            end = min(cast(int, node.args[3]), input_tensor.shape[dim])
            if end < 0:
                end = end % input_tensor.shape[dim]
        else:
            end = input_tensor.shape[dim]

        # QNN HTP rejects StridedSlice when its input carries a per-channel
        # encoding, even when the slice is on a different axis. Gather with a
        # contiguous static index vector is exactly equivalent to slice_copy
        # and preserves the input/output per-channel qparams. This path is
        # needed by MHA-to-SHA head splitting for token-axis A8 V-cache; the
        # established per-tensor and A16 StridedSlice paths remain unchanged.
        input_quant_attrs = input_node.meta.get(QCOM_QUANT_ATTRS) or {}
        if QCOM_SCALES in input_quant_attrs:
            step = cast(int, node.args[4]) if len(node.args) > 4 else 1
            indices = np.arange(start, end, step, dtype=np.int32)
            indices_name = f"{node.name}_slice_indices"
            indices_wrapper = PyQnnManager.TensorWrapper(
                indices_name,
                PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
                PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_INT_32,
                PyQnnManager.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED,
                {},
                1,
                [len(indices)],
                [],
                indices,
                True,
            )
            gather_op = PyQnnManager.PyQnnOpWrapper(
                node.name,
                QNN_OP_PACKAGE_NAME_QTI_AISW,
                OpGather.op_name,
            )
            gather_op.AddInputTensors([input_tensor_wrapper, indices_wrapper])
            gather_op.AddOutputTensors([output_tensor_wrapper])
            gather_op.AddScalarParam(
                OpGather.param_axis,
                PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_INT_32,
                {QCOM_DATA: np.int32(dim)},
            )
            return gather_op

        input_tensor_rank = len(input_tensor.shape)
        ranges = []
        for i in range(input_tensor_rank):
            if i == dim:
                # find step
                step = node.args[4] if len(node.args) > 4 else 1
                ranges.extend([start, end, step])
            else:
                ranges.extend([0, input_tensor.shape[i], 1])

        range_shape = [input_tensor_rank, 3]

        stride_slice_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpStridedSlice.op_name,
        )
        stride_slice_op.AddInputTensors([input_tensor_wrapper])
        stride_slice_op.AddOutputTensors([output_tensor_wrapper])

        stride_slice_op.AddTensorParam(
            OpStridedSlice.param_ranges,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            len(range_shape),
            range_shape,
            np.array(ranges, dtype=np.int32),
            True,
        )

        return stride_slice_op
