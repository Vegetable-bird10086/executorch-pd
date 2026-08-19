# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
from typing import Sequence

import torch
from executorch.backends.qualcomm.quantizer.quantizer import (
    get_16a8w_qnn_ptq_config,
    get_16a8w_qnn_qat_config,
    get_8a8w_qnn_ptq_config,
    get_8a8w_qnn_qat_config,
    get_ptq_per_channel_quant_config,
    QuantizationConfig,
)
from executorch.backends.qualcomm.quantizer.rules import (
    _is_float_tensor,
    Q_ANNOTATION_KEY,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node
from torchao.quantization.pt2e import MinMaxObserver, PerChannelMinMaxObserver
from torchao.quantization.pt2e.quantizer import (
    annotate_input_qspec_map,
    annotate_output_qspec,
    QuantizationAnnotation,
    QuantizationSpec,
    SharedQuantizationSpec,
)


def annotate_llm_residual_token_axis_a8(gm: torch.fx.GraphModule) -> None:
    """Give post-outlier prefill residuals one static A8 scale per token slot."""

    token_axis_qspec = QuantizationSpec(
        dtype=torch.uint8,
        quant_min=0,
        quant_max=255,
        qscheme=torch.per_channel_affine,
        ch_axis=1,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(eps=2**-12),
    )
    annotated = 0
    is_prefill_graph = False
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.add.Tensor:
            continue
        stack = node.meta.get("nn_module_stack", {})
        if not stack:
            continue
        deepest = str(list(stack.values())[-1][0])
        match = re.fullmatch(r"layers\.(\d+)", deepest)
        if match is None or int(match.group(1)) < 6:
            continue
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor) or value.dim() < 2:
            continue
        if value.shape[1] <= 1:
            # AR1 decode already has one independent token per invocation, so
            # token-axis quantization would be equivalent but needlessly changes
            # its QNN encoding type.
            continue
        is_prefill_graph = True
        annotation = node.meta.get(Q_ANNOTATION_KEY)
        if annotation is None:
            continue
        annotation.output_qspec = token_axis_qspec
        for user in node.users:
            user_annotation = user.meta.get(Q_ANNOTATION_KEY)
            if (
                user_annotation is not None
                and node in user_annotation.input_qspec_map
            ):
                user_annotation.input_qspec_map[node] = token_axis_qspec
        annotated += 1
    expected = 60 if is_prefill_graph else 0
    if annotated != expected:
        raise RuntimeError(
            f"Expected {expected} layer6-35 residual adds for token-axis A8, "
            f"annotated {annotated}"
        )


def annotate_llm_all_token_axis_a8(gm: torch.fx.GraphModule) -> None:
    """Use per-token A8 on layout-stable prefill compute and residual edges."""

    ar_len = int(os.environ.get("ET_QNN_A8_TOKEN_AXIS_AR", "128"))

    def token_qspec(node, forced_axis=None):
        if not isinstance(node, Node) or node.op == "get_attr":
            return None
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor) or value.dim() < 3:
            return None
        if forced_axis is None:
            axes = [axis for axis, size in enumerate(value.shape) if size == ar_len]
            if len(axes) != 1:
                return None
            axis = axes[0]
        else:
            axis = forced_axis
        # Do not reuse one QuantizationSpec across unrelated graph edges. PT2E
        # may then share a single observer through layout-changing consumers,
        # causing the same observer to see AR, hidden, and head dimensions.
        return QuantizationSpec(
            dtype=torch.uint8,
            quant_min=0,
            quant_max=255,
            qscheme=torch.per_channel_affine,
            ch_axis=axis,
            observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(
                eps=2**-12
            ),
        )

    layout_stable_ops = {
        torch.ops.aten.conv2d.default,
        torch.ops.aten.rms_norm.default,
        torch.ops.aten.mul.Tensor,
    }
    changed_outputs = 0
    changed_inputs = 0
    for node in gm.graph.nodes:
        annotation = node.meta.get(Q_ANNOTATION_KEY)
        if annotation is None:
            continue
        stack = node.meta.get("nn_module_stack", {})
        paths = [
            str(value[0]) if isinstance(value, (tuple, list)) and value else str(value)
            for value in stack.values()
        ]
        if not any(re.search(r"(?:^|\.)layers\.\d+(?:\.|$)", path) for path in paths):
            continue
        deepest = paths[-1] if paths else ""
        is_residual_add = (
            node.target == torch.ops.aten.add.Tensor
            and re.fullmatch(r"layers\.\d+", deepest) is not None
        )
        if node.target not in layout_stable_ops and not is_residual_add:
            continue

        is_conv = node.target == torch.ops.aten.conv2d.default
        # Conv outputs immediately enter permute/squeeze view chains. PT2E shares
        # their observer without remapping ch_axis, so keep those outputs
        # per-tensor until an explicit axis-remapping pass exists.
        output_spec = None if is_conv else token_qspec(node)
        if output_spec is not None:
            annotation.output_qspec = output_spec
            changed_outputs += 1
            for user in node.users:
                user_annotation = user.meta.get(Q_ANNOTATION_KEY)
                if (
                    user_annotation is not None
                    and node in user_annotation.input_qspec_map
                ):
                    user_annotation.input_qspec_map[node] = output_spec
                    changed_inputs += 1
        for input_node in list(annotation.input_qspec_map):
            input_spec = token_qspec(
                input_node, forced_axis=3 if is_conv and input_node.op != "get_attr" else None
            )
            if input_spec is not None:
                annotation.input_qspec_map[input_node] = input_spec
                changed_inputs += 1
                producer_annotation = input_node.meta.get(Q_ANNOTATION_KEY)
                if producer_annotation is not None:
                    producer_annotation.output_qspec = input_spec
                for sibling_user in input_node.users:
                    sibling_annotation = sibling_user.meta.get(Q_ANNOTATION_KEY)
                    if (
                        sibling_annotation is not None
                        and input_node in sibling_annotation.input_qspec_map
                    ):
                        sibling_annotation.input_qspec_map[input_node] = input_spec

    is_prefill_graph = changed_outputs > 0 or changed_inputs > 0
    if is_prefill_graph and (changed_outputs < 100 or changed_inputs < 350):
        raise RuntimeError(
            "Unexpectedly few all-token-axis A8 annotations: "
            f"outputs={changed_outputs}, inputs={changed_inputs}"
        )

    # QAIRT 2.37 rejects RmsNorm when any tensor uses a per-channel
    # quantization encoding. Keep only this operator's activation boundaries
    # per-tensor A8; adjacent supported Conv/attention chains can requantize
    # back to token-axis A8. Use distinct specs so unrelated layers/edges do
    # not share observers.
    rmsnorms = 0
    for node in gm.graph.nodes:
        if node.target != torch.ops.aten.rms_norm.default:
            continue
        annotation = node.meta.get(Q_ANNOTATION_KEY)
        if annotation is None:
            continue

        def per_tensor_qspec():
            return QuantizationSpec(
                dtype=torch.uint8,
                quant_min=0,
                quant_max=255,
                qscheme=torch.per_tensor_affine,
                observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
            )

        for input_node in list(annotation.input_qspec_map):
            if _is_float_tensor(input_node) and input_node.op != "get_attr":
                annotation.input_qspec_map[input_node] = per_tensor_qspec()
        output_spec = per_tensor_qspec()
        annotation.output_qspec = output_spec
        for user in node.users:
            user_annotation = user.meta.get(Q_ANNOTATION_KEY)
            if user_annotation is not None and node in user_annotation.input_qspec_map:
                user_annotation.input_qspec_map[node] = output_spec
        rmsnorms += 1
    if is_prefill_graph and rmsnorms != 145:
        raise RuntimeError(
            f"Expected 145 Qwen3-4B RMSNorm nodes for local per-tensor A8, got {rmsnorms}"
        )

    # QAIRT 2.37 permits at most one Conv input to use per-axis, block-wise, or
    # vector quantization. W4 block-wise weights already consume that slot, so
    # a token-axis activation input makes graph construction fail even though
    # the isolated capability query reports the Conv as supported. Keep the
    # producer/output path token-axis, but give every runtime Conv input its own
    # per-tensor A8 consumer qspec. PT2E then materializes the required
    # PCQ -> per-tensor requantization at the Conv boundary without collapsing
    # token-axis Conv outputs or terminal KV encodings.
    conv_runtime_inputs = 0
    for node in gm.graph.nodes:
        if node.target != torch.ops.aten.conv2d.default:
            continue
        stack = node.meta.get("nn_module_stack", {})
        deepest = str(list(stack.values())[-1][0]) if stack else ""
        if re.fullmatch(r"layers\.\d+\..+_conv", deepest) is None:
            continue
        annotation = node.meta.get(Q_ANNOTATION_KEY)
        if annotation is None:
            continue
        for input_node in list(annotation.input_qspec_map):
            if not _is_float_tensor(input_node) or input_node.op == "get_attr":
                continue
            annotation.input_qspec_map[input_node] = QuantizationSpec(
                dtype=torch.uint8,
                quant_min=0,
                quant_max=255,
                qscheme=torch.per_tensor_affine,
                observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
            )
            conv_runtime_inputs += 1
    if is_prefill_graph and conv_runtime_inputs != 252:
        raise RuntimeError(
            "Expected 252 Qwen3-4B Conv runtime inputs for local per-tensor A8, "
            f"got {conv_runtime_inputs}"
        )

    # The same QAIRT 2.37 advanced-encoding limit applies to binary elementwise
    # operators. Keep the first activation input token-axis and make additional
    # runtime tensor inputs per-tensor. This covers RoPE (Q/K times sin/cos),
    # its add/sub joins, and the FFN gate without collapsing the operator output
    # or the primary activation path back to per-tensor A8.
    binary_ops = {
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Tensor,
    }
    binary_secondary_inputs = 0
    for node in gm.graph.nodes:
        if node.target not in binary_ops:
            continue
        annotation = node.meta.get(Q_ANNOTATION_KEY)
        if annotation is None:
            continue
        runtime_inputs = [
            input_node
            for input_node in annotation.input_qspec_map
            if _is_float_tensor(input_node) and input_node.op != "get_attr"
        ]
        for input_node in runtime_inputs[1:]:
            annotation.input_qspec_map[input_node] = QuantizationSpec(
                dtype=torch.uint8,
                quant_min=0,
                quant_max=255,
                qscheme=torch.per_tensor_affine,
                observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
            )
            binary_secondary_inputs += 1
    if is_prefill_graph and binary_secondary_inputs < 200:
        raise RuntimeError(
            "Unexpectedly few binary secondary A8 inputs forced per-tensor: "
            f"got {binary_secondary_inputs}"
        )

    # QAIRT 2.37 Sigmoid has no per-channel quantized kernel. In Qwen's SiLU
    # branch, keep only Sigmoid locally per-tensor; the parallel gate activation
    # and the Mul output retain their token-axis encodings.
    sigmoid_nodes = 0
    for node in gm.graph.nodes:
        if node.target != torch.ops.aten.sigmoid.default:
            continue
        annotation = node.meta.get(Q_ANNOTATION_KEY)
        if annotation is None:
            continue

        def sigmoid_qspec():
            return QuantizationSpec(
                dtype=torch.uint8,
                quant_min=0,
                quant_max=255,
                qscheme=torch.per_tensor_affine,
                observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
            )

        for input_node in list(annotation.input_qspec_map):
            if _is_float_tensor(input_node) and input_node.op != "get_attr":
                annotation.input_qspec_map[input_node] = sigmoid_qspec()
        output_spec = sigmoid_qspec()
        annotation.output_qspec = output_spec
        for user in node.users:
            user_annotation = user.meta.get(Q_ANNOTATION_KEY)
            if user_annotation is not None and node in user_annotation.input_qspec_map:
                user_annotation.input_qspec_map[node] = output_spec
        sigmoid_nodes += 1
    if is_prefill_graph and sigmoid_nodes != 36:
        raise RuntimeError(
            f"Expected 36 Qwen3-4B Sigmoid nodes for local per-tensor A8, got {sigmoid_nodes}"
        )

    # Full graph construction is stricter than the isolated capability query:
    # QAIRT 2.37 rejects ElementWiseMultiply with even one PCQ tensor. Apply the
    # same local per-tensor contract to all binary elementwise inputs/outputs.
    # Token-axis encoding resumes at later supported producers; Conv/layout/KV
    # output paths are not globally collapsed.
    binary_nodes = 0
    for node in gm.graph.nodes:
        if node.target not in binary_ops:
            continue
        annotation = node.meta.get(Q_ANNOTATION_KEY)
        if annotation is None:
            continue

        def binary_qspec():
            return QuantizationSpec(
                dtype=torch.uint8,
                quant_min=0,
                quant_max=255,
                qscheme=torch.per_tensor_affine,
                observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
            )

        for input_node in list(annotation.input_qspec_map):
            if _is_float_tensor(input_node) and input_node.op != "get_attr":
                annotation.input_qspec_map[input_node] = binary_qspec()
        output_spec = binary_qspec()
        annotation.output_qspec = output_spec
        for user in node.users:
            user_annotation = user.meta.get(Q_ANNOTATION_KEY)
            if user_annotation is not None and node in user_annotation.input_qspec_map:
                user_annotation.input_qspec_map[node] = output_spec
        binary_nodes += 1
    if is_prefill_graph and binary_nodes < 250:
        raise RuntimeError(
            f"Unexpectedly few binary ops forced per-tensor A8: got {binary_nodes}"
        )


def annotate_llm_v_projection_token_axis_a8(gm: torch.fx.GraphModule) -> None:
    """Remap per-token A8 through Wv or every layer Conv output layout chain."""

    ar_len = int(os.environ.get("ET_QNN_A8_TOKEN_AXIS_AR", "128"))
    all_layer_convs = os.environ.get("ET_QNN_A8_ALL_CONV_OUTPUT_TOKEN_AXIS", "") == "1"

    def make_qspec(axis):
        return QuantizationSpec(
            dtype=torch.uint8,
            quant_min=0,
            quant_max=255,
            qscheme=torch.per_channel_affine,
            ch_axis=axis,
            observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(
                eps=2**-12
            ),
        )

    layout_ops = {
        torch.ops.aten.permute.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.transpose.int,
    }

    def remap_axis(node, input_axis):
        if node.target == torch.ops.aten.permute.default:
            dims = list(node.args[1])
            return dims.index(input_axis)
        if node.target == torch.ops.aten.transpose.int:
            dim0, dim1 = int(node.args[1]), int(node.args[2])
            if input_axis == dim0:
                return dim1
            if input_axis == dim1:
                return dim0
            return input_axis
        if node.target == torch.ops.aten.squeeze.dim:
            input_value = node.args[0].meta.get("val")
            rank = input_value.dim()
            dim = int(node.args[1]) % rank
            if dim == input_axis:
                raise RuntimeError("Wv layout chain unexpectedly squeezes token axis")
            return input_axis - 1 if dim < input_axis else input_axis
        if node.target in {
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
        }:
            shape = list(node.args[1])
            candidates = [i for i, size in enumerate(shape) if size == ar_len]
            if input_axis in candidates or shape[input_axis] == -1:
                return input_axis
            if len(candidates) == 1:
                return candidates[0]
            raise RuntimeError(
                f"Cannot remap Wv token axis through shape {shape} from {input_axis}"
            )
        raise RuntimeError(f"Unsupported Wv layout op {node.target}")

    chains = 0
    layout_nodes = 0
    for conv in gm.graph.nodes:
        if conv.target != torch.ops.aten.conv2d.default:
            continue
        stack = conv.meta.get("nn_module_stack", {})
        deepest = str(list(stack.values())[-1][0]) if stack else ""
        if all_layer_convs:
            if re.fullmatch(r"layers\.\d+\..+_conv", deepest) is None:
                continue
        elif re.fullmatch(r"layers\.\d+\.attention\.wv_conv", deepest) is None:
            continue
        annotation = conv.meta.get(Q_ANNOTATION_KEY)
        if annotation is None:
            continue
        axis = 3
        output_spec = make_qspec(axis)
        annotation.output_qspec = output_spec
        current = conv
        while True:
            users = list(current.users)
            if len(users) != 1 or users[0].target not in layout_ops:
                break
            layout = users[0]
            layout_annotation = layout.meta.get(Q_ANNOTATION_KEY)
            if layout_annotation is None:
                break
            layout_annotation.input_qspec_map[current] = output_spec
            axis = remap_axis(layout, axis)
            output_spec = make_qspec(axis)
            layout_annotation.output_qspec = output_spec
            current = layout
            layout_nodes += 1
        chains += 1

    expected_chains = 252 if all_layer_convs else 36
    valid_layout_count = (
        500 <= layout_nodes <= 900 if all_layer_convs else layout_nodes == 144
    )
    if chains not in (0, expected_chains) or (chains and not valid_layout_count):
        raise RuntimeError(
            "Unexpected Conv-output token-axis chain coverage: "
            f"chains={chains}, layout_nodes={layout_nodes}"
        )


def annotate_llm_prefill_kv_output_per_tensor_a8(
    gm: torch.fx.GraphModule,
) -> None:
    """Requantize token-axis prefill KV outputs to the AR1 per-tensor ABI.

    Per-token encodings are useful inside an AR128 prefill graph, but their 128
    scales cannot be represented by the existing PD handoff to an AR1 decode
    graph.  Keep the internal layout chain per-token and make only the terminal
    K/V graph outputs symmetric per-tensor U8.
    """

    ar_len = int(os.environ.get("ET_QNN_A8_TOKEN_AXIS_AR", "128"))
    def make_output_qspec():
        # A distinct spec is required per layer. Reusing a QuantizationSpec can
        # make PT2E share one observer across all 72 K/V ranges.
        return QuantizationSpec(
            dtype=torch.uint8,
            quant_min=0,
            quant_max=255,
            qscheme=torch.per_tensor_symmetric,
            observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
        )
    output_node = next((node for node in gm.graph.nodes if node.op == "output"), None)
    if output_node is None:
        raise RuntimeError("A8 KV boundary annotation found no graph output")

    candidates = []

    def collect(value):
        if isinstance(value, Node):
            tensor = value.meta.get("val")
            if (
                isinstance(tensor, torch.Tensor)
                and tensor.dim() == 4
                and tensor.shape[-2] == ar_len
                and tensor.shape[-1] == ar_len
            ):
                candidates.append(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    collect(output_node.args[0])
    # Qwen3-4B has 36 K and 36 V outputs. AR1 does not satisfy the two trailing
    # AR128 dimensions and therefore remains untouched by this annotation.
    if not candidates:
        return
    if len(candidates) != 72:
        raise RuntimeError(
            f"Expected 72 terminal AR128 K/V outputs, found {len(candidates)}"
        )
    replacements = {}
    with gm.graph.inserting_before(output_node):
        for node in candidates:
            annotation = node.meta.get(Q_ANNOTATION_KEY)
            if annotation is None or annotation.output_qspec is None:
                raise RuntimeError(f"KV output {node.name} has no output qspec")
            boundary = gm.graph.call_function(
                torch.ops.aten.clone.default,
                args=(node,),
            )
            boundary.meta = node.meta.copy()
            boundary.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                # Keep this as a detached float branch. Reusing the producer's
                # per-axis input qspec can make PT2E merge the branch observer
                # back into the node also consumed by attention.
                input_qspec_map={},
                output_qspec=make_output_qspec(),
                _annotated=True,
            )
            replacements[node] = boundary

    def replace(value):
        if isinstance(value, Node):
            return replacements.get(value, value)
        if isinstance(value, tuple):
            return tuple(replace(item) for item in value)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    output_node.args = (replace(output_node.args[0]),)
    gm.graph.lint()
    gm.recompile()


def annotate_eurobert(gm: torch.fx.GraphModule):
    """
    QNN does not support int32 -> signed 16bit quant
    We need to first annotate this to_fp node as 8bit quant, so it will perform requantize
    Final graph should look like: int32 -> convert -> cast -> matmul.args[1]

    """
    quantization_config_8a8w = get_8a8w_qnn_ptq_config()
    for node in gm.graph.nodes:
        # A little tricky here. This matmul node is wrapped inside a submodule after 1st torch.export.
        # There are actually 2 'to' op that is redundant.
        # It will look like: int64 -> to_fp -> to_fp -> matmul.args[1]
        # Draw out the graph after the 1st export will help visualize the submodule.

        if node.target == torch.ops.aten.matmul.default and node.args[1].args[0].args[
            0
        ].meta["val"].dtype in [torch.int64, torch.int32]:
            to_node = node.args[1]
            input_qspec_map = {}
            assert isinstance(to_node, Node)
            input_spec = quantization_config_8a8w.input_activation
            input_qspec_map[to_node] = input_spec
            to_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=quantization_config_8a8w.output_activation,
                _annotated=True,
            )


def annotate_mimi_decoder(gm: torch.fx.GraphModule):
    """
    The 1st transpose conv in mimi decoder is really sensitive to scale/offset in 16a8w, which causes execution failure.
    Annotate 1st transpose conv as 8a8w to prevent execution failure.
    """
    quantization_config_8a8w = get_8a8w_qnn_ptq_config()
    for node in gm.graph.nodes:
        if not _is_float_tensor(node):
            continue
        elif node.target == torch.ops.aten.conv_transpose1d.default:
            input_qspec_map = {}
            input_act = node.args[0]
            assert isinstance(input_act, Node)
            input_spec = quantization_config_8a8w.input_activation
            input_qspec_map[input_act] = input_spec

            weight = node.args[1]
            assert isinstance(weight, Node)
            input_qspec_map[weight] = quantization_config_8a8w.weight

            if len(node.args) > 2 and isinstance(node.args[2], Node):
                bias = node.args[2]
                input_qspec_map[bias] = quantization_config_8a8w.bias

            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=quantization_config_8a8w.output_activation,
                _annotated=True,
            )
            break


def annotate_kv_8bit(  # noqa: C901
    gm: torch.fx.GraphModule,
    is_qat=False,
    matmul_activation_bits=16,
) -> None:
    """
    This function is for static LLM models.
    By default this preserves the production 16a8w attention MatMul path.
    Experimental native-A8 recipes may request 8a8w MatMul annotations while
    retaining the same 8-bit KV-cache propagation rules.
    For k, we will tag such as the below, and
    for v, we will tag 8a until conv op.
                                                              q (16 bits) ──┬─> matmul op (16 bits)
                                       past k (8 bits) ┬─> cat op (8 bits) ─┘
    rotatary add (16 bits) ─┬> cat op (new k) (8 bits) ┘
    rotatary sub (16 bits) ─┘
    """

    def annotate_matmul(node: Node, quantization_config: QuantizationConfig):
        input_qspec_map = {}
        input_act = node.args[0]
        input_spec = quantization_config.input_activation
        input_qspec_map[input_act] = input_spec
        input_act1 = node.args[1]
        input_spec1 = quantization_config.weight
        input_qspec_map[input_act1] = input_spec1

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    def annotate_cat(node: Node, quantization_config: QuantizationConfig):
        input_nodes = node.args[0]

        first_input_node = input_nodes[0]
        input_qspec_map = {}
        input_qspec_map[first_input_node] = quantization_config.input_activation
        share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
            (first_input_node, node)
        )

        for input_node in input_nodes[1:]:
            if input_node not in input_qspec_map:
                input_qspec_map[input_node] = share_qparams_with_input_act0_qspec

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=share_qparams_with_input_act0_qspec,
            _annotated=True,
        )

    def annotate_rms_norm(node: Node, quantization_config: QuantizationConfig) -> None:
        act_node = node.args[0]
        weight_node = node.args[2]

        # TODO current only support 16a16w
        annotate_input_qspec_map(
            node,
            act_node,
            quantization_config.input_activation,
        )

        annotate_input_qspec_map(
            node,
            weight_node,
            quantization_config.input_activation,
        )
        annotate_output_qspec(node, quantization_config.output_activation)

    def annotate_single_in_single_out(
        node: Node, quantization_config: QuantizationConfig
    ) -> None:
        input_qspec_map = {}
        input_act = node.args[0]
        input_qspec_map[input_act] = quantization_config.input_activation

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    def annotate_single_in_share_out(
        node: Node, quantization_config: QuantizationConfig
    ) -> None:
        input_qspec_map = {}
        input_act = node.args[0]
        input_qspec_map[input_act] = quantization_config.input_activation

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=SharedQuantizationSpec((input_act, node)),
            _annotated=True,
        )

    def annotate_stack(node: Node, quantization_config: QuantizationConfig) -> None:
        input_nodes = node.args[0]

        first_input_node = input_nodes[0]
        input_qspec_map = {}
        input_qspec_map[first_input_node] = quantization_config.input_activation
        share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
            (first_input_node, node)
        )

        for input_node in input_nodes[1:]:
            if input_node not in input_qspec_map:
                input_qspec_map[input_node] = share_qparams_with_input_act0_qspec

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=share_qparams_with_input_act0_qspec,
            _annotated=True,
        )

    def annotate_matmul_input1(node: Node, is_qat: str):
        if is_qat:
            quantization_config_8a8w = get_8a8w_qnn_qat_config(
                act_symmetric=True, act_observer=MinMaxObserver
            )
        else:
            quantization_config_8a8w = get_8a8w_qnn_ptq_config(
                act_symmetric=True, act_observer=MinMaxObserver
            )
        while isinstance(node, Node) and node.op == "call_function":
            if node.target in [
                torch.ops.aten.select.int,
                torch.ops.aten.slice.Tensor,
            ]:
                annotate_single_in_single_out(node, quantization_config_8a8w)
                node = node.args[0]
            elif node.target in [
                torch.ops.aten.permute.default,
                torch.ops.aten.squeeze.dim,
                torch.ops.aten.transpose.int,
                torch.ops.aten.view.default,
                torch.ops.aten.reshape.default,
                torch.ops.aten.expand.default,
                torch.ops.aten.unsqueeze.default,
                torch.ops.aten.flatten.using_ints,
            ]:
                annotate_single_in_share_out(node, quantization_config_8a8w)
                node = node.args[0]
            elif node.target == torch.ops.aten.stack.default:
                annotate_stack(node, quantization_config_8a8w)
                node = node.args[0]
            elif node.target == torch.ops.aten.rms_norm.default:
                annotate_rms_norm(node, quantization_config_8a8w)
                node = node.args[0]
            elif node.target == torch.ops.aten.cat.default:
                annotate_cat(node, quantization_config_8a8w)
                # For v, we tag 8a until conv op.
                # For k, we tag 8a until add or sub op (rotatary embedding).
                # The arguments of cat op: (the past kv cache, the new kv cache)
                node = node.args[0][1]
            elif node.target in [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.sub.Tensor,
                torch.ops.aten.matmul.default,
                torch.ops.aten.conv2d.default,
            ]:
                break
            else:
                print(f"The node ({node}) is not expected in the input1 of the matmul")
                node = node.args[0]

    if matmul_activation_bits not in (8, 16):
        raise ValueError(
            "matmul_activation_bits must be 8 or 16, got "
            f"{matmul_activation_bits}"
        )
    if matmul_activation_bits == 8:
        quantization_config_matmul = (
            get_8a8w_qnn_qat_config(
                act_observer=MinMaxObserver,
            )
            if is_qat
            else get_8a8w_qnn_ptq_config(
                act_observer=MinMaxObserver,
            )
        )
    else:
        quantization_config_matmul = (
            get_16a8w_qnn_qat_config(
                act_observer=MinMaxObserver,
            )
            if is_qat
            else get_16a8w_qnn_ptq_config(
                act_observer=MinMaxObserver,
            )
        )

    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.matmul.default
            and all(arg.op == "call_function" for arg in node.args)
        ):
            # Only apply custom annotation on Q @ K^T @ V
            annotate_matmul(node, quantization_config_matmul)
            annotate_matmul_input1(node.args[1], is_qat=is_qat)


def annotate_kv_8bit_a8(gm: torch.fx.GraphModule, is_qat=False) -> None:
    """Annotate both attention MatMul activations and KV cache as 8 bit."""

    annotate_kv_8bit(
        gm,
        is_qat=is_qat,
        matmul_activation_bits=8,
    )


def custom_annotate_llama_matmul_16a8w(gm: torch.fx.GraphModule) -> None:  # noqa: C901
    """
    This function is specific for llama matmul op 16a8w.
    """

    def annotate_matmul(node: Node, quantization_config: QuantizationConfig):
        input_qspec_map = {}
        input_act = node.args[0]
        input_spec = quantization_config.input_activation
        input_qspec_map[input_act] = input_spec
        input_act1 = node.args[1]
        input_spec1 = quantization_config.weight
        input_qspec_map[input_act1] = input_spec1
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    def annotate_index_put(node: Node, quantization_config: QuantizationConfig) -> None:
        # Avoid annotating the input node because mutable buffers will be folded during the convert_pt2e process.
        value = node.args[2]

        input_qspec_map = {}
        input_qspec_map[value] = quantization_config.input_activation

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=SharedQuantizationSpec((value, node)),
            _annotated=True,
        )

    def annotate_single_in_single_out(
        node: Node, quantization_config: QuantizationConfig
    ) -> None:
        input_qspec_map = {}
        input_act = node.args[0]
        input_qspec_map[input_act] = quantization_config.input_activation
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    def annotate_cat(node: Node, quantization_config: QuantizationConfig):
        input_nodes = node.args[0]
        assert isinstance(input_nodes, Sequence)
        first_input_node = input_nodes[0]
        input_qspec_map = {}
        assert isinstance(first_input_node, Node)
        assert isinstance(node, Node)
        input_qspec_map[first_input_node] = quantization_config.input_activation
        share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
            (first_input_node, node)
        )
        for input_node in input_nodes[1:]:
            if input_node not in input_qspec_map:
                assert isinstance(input_node, Node)
                input_qspec_map[input_node] = share_qparams_with_input_act0_qspec
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=share_qparams_with_input_act0_qspec,
            _annotated=True,
        )

    def is_edge_condition(node: Node):
        if not isinstance(node, Node) or node.op != "call_function":
            return True
        return False

    def annotate_matmul_input1(node: Node, quantization_config: QuantizationConfig):
        if is_edge_condition(node):
            return
        if node.target in [
            torch.ops.aten.index_put.default,
            torch.ops.aten.index_put_.default,
        ]:
            annotate_index_put(node, quantization_config)
            annotate_matmul_input1(node.args[0], quantization_config)
        elif node.target == torch.ops.aten.cat.default:
            annotate_cat(node, quantization_config)
            # Expect that the inputs of the cat op are select ops
            for arg in node.args[0]:
                annotate_matmul_input1(arg, quantization_config)
        else:
            annotate_single_in_single_out(node, quantization_config)
            annotate_matmul_input1(node.args[0], quantization_config)

    # Annotate 16a8w for matmul op to get better performance
    quantization_config_16a8w = get_16a8w_qnn_ptq_config()
    # Annotate 8a8w for second input of matmul until past_kv_cache
    quantization_config_8a8w = get_8a8w_qnn_ptq_config(act_symmetric=True)
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.matmul.default:
            if "nn_module_stack" in node.meta:
                module_values_list = list(node.meta["nn_module_stack"].values())
                full_qualified_name = module_values_list[-1][0]
                if "SDPA" in full_qualified_name:
                    annotate_matmul(node, quantization_config_16a8w)
                    annotate_matmul_input1(node.args[1], quantization_config_8a8w)


def custom_annotate_llama_last_conv_16a8w(gm: torch.fx.GraphModule) -> None:
    def annotate_conv2d(node: Node, quantization_config: QuantizationConfig) -> None:
        input_qspec_map = {}
        input_act = node.args[0]
        input_spec = quantization_config.input_activation
        input_qspec_map[input_act] = input_spec

        weight = node.args[1]
        input_qspec_map[weight] = quantization_config.weight

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    quantization_config_16a8w_per_channel = get_ptq_per_channel_quant_config(
        torch.uint16, weight_dtype=torch.int8
    )
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.conv2d.default:
            if "nn_module_stack" in node.meta:
                module_values_list = list(node.meta["nn_module_stack"].values())
                full_qualified_name = module_values_list[0][0]
                if full_qualified_name == "L['self'].llama.output":
                    annotate_conv2d(
                        node, quantization_config=quantization_config_16a8w_per_channel
                    )


def custom_annotate_matmul_16a8w(gm: torch.fx.GraphModule):
    """
    Annotate matmul op with 16a8w quantization config
    """

    def annotate_matmul(node: Node, quantization_config: QuantizationConfig):
        input_qspec_map = {}
        input_act = node.args[0]
        input_spec = quantization_config.input_activation
        input_qspec_map[input_act] = input_spec
        input_act1 = node.args[1]
        input_spec1 = quantization_config.weight
        input_qspec_map[input_act1] = input_spec1
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    # Annotate 16a8w for matmul op to get better performance
    quantization_config_16a8w = get_16a8w_qnn_ptq_config()
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.matmul.default:
            annotate_matmul(node, quantization_config_16a8w)


def get_custom_quant_ios_dtype(
    cache_shape: torch.Size,
    node: torch.fx.Node,
    kv_dtype=torch.uint8,
    sharding_dtype=torch.uint16,
):
    """
    This function is specific for llama inputs and outputs
    """
    if node.op == "placeholder" and "attention_kv_cache_past_" in node.name:
        return kv_dtype

    # Tag index put node before copy node, because copy is a skipped node in qnn
    if (
        exir_ops.edge.aten.index_put.default == node.target
        and node.meta["val"].shape == cache_shape
    ):
        return kv_dtype

    # Tag sharding io
    if exir_ops.edge.llama.fallback.default in [
        u.target for u in list(node.users.keys())
    ] + [node.target]:
        return sharding_dtype

    # Tag index op as quantized tensors. It is caused by sharding
    if exir_ops.edge.aten.index.Tensor in [
        u.target for u in list(node.users.keys())
    ] + [node.target]:
        return sharding_dtype
