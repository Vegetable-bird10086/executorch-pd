# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
import logging
import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from executorch.backends.qualcomm.builders import node_visitor_manager
from executorch.backends.qualcomm.builders.node_visitor import (
    IncompleteQuantizationEncodingError,
    dq_ops,
    q_ops,
)
from executorch.backends.qualcomm.builders.utils import (
    is_mutable_buffer_input,
    is_parameter,
)
from executorch.backends.qualcomm.builders.qnn_constants import OpContextLoader
from executorch.backends.qualcomm.qnn_preprocess import QnnBackend
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_BYPASS_NODE,
    QCOM_DTYPE,
    QCOM_ENCODING,
    QCOM_QUANT_ATTRS,
    QCOM_QUANTIZED_IO,
    QCOM_SCALES,
    QCOM_ZERO_POINTS,
)

from executorch.backends.qualcomm.utils.qnn_manager_lifecycle import (
    get_current_qnn_manager,
)
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
)
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data, tag_mutated_buffer
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import OperatorSupportBase

from .common_defs import (
    allow_list_operator,
    constant_operator,
    not_supported_operator,
    to_be_implemented_operator,
)
from .utils import filter_fn, generate_qnn_executorch_option, get_skip_decomp_table

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QnnOperatorSupport(OperatorSupportBase):
    def __init__(
        self,
        edge_program: torch.export.ExportedProgram,
        compiler_specs,
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
    ):
        option = generate_qnn_executorch_option(compiler_specs)
        python_options = flatbuffer_to_option(option)
        self.node_visitors = node_visitor_manager.get_node_visitors(
            edge_program,
            op_package_infos=python_options.op_package_options.op_package_infos,
        )

        self.skip_node_op_set = skip_node_op_set
        self.skip_node_id_set = skip_node_id_set
        self.nodes_to_wrappers = defaultdict(dict)
        self.support_node_names = None
        prelower_limit = int(
            os.environ.get("ET_QNN_PRELOWER_SHARD_LIMIT", "0") or 0
        )
        if prelower_limit:
            prelower_start = int(
                os.environ.get("ET_QNN_PRELOWER_SHARD_START", "0") or 0
            )
            expected_total = int(
                os.environ.get("ET_QNN_PRELOWER_TOTAL_SHARDS", "0") or 0
            )
            segment = 0
            support_node_names = set()
            for graph_node in edge_program.graph_module.graph.nodes:
                if (
                    graph_node.op == "call_function"
                    and "llama.fallback" in str(graph_node.target)
                ):
                    segment += 1
                    continue
                if prelower_start <= segment < prelower_start + prelower_limit:
                    support_node_names.add(graph_node.name)

            # Lifted parameters and their Q/DQ preparation nodes are often
            # scheduled before the first fallback marker. Later bounded
            # segments still need those exact static dependencies. Add only
            # dependency subgraphs whose leaves are parameters/constants; do
            # not cross into runtime computation from another segment.
            static_dependency_cache = {}

            def is_static_dependency(node):
                cached = static_dependency_cache.get(node)
                if cached is not None:
                    return cached
                if node.op == "get_attr":
                    result = True
                elif node.op == "placeholder":
                    result = is_parameter(node, edge_program)
                elif node.op == "call_function":
                    inputs = list(node.all_input_nodes)
                    result = bool(inputs) and all(
                        is_static_dependency(input_node)
                        for input_node in inputs
                    )
                else:
                    result = False
                static_dependency_cache[node] = result
                return result

            selected_nodes = [
                node
                for node in edge_program.graph_module.graph.nodes
                if node.name in support_node_names
            ]
            pending = [
                input_node
                for node in selected_nodes
                for input_node in node.all_input_nodes
            ]
            while pending:
                dependency = pending.pop()
                if not is_static_dependency(dependency):
                    continue
                if dependency.op == "call_function":
                    if dependency.name in support_node_names:
                        continue
                    support_node_names.add(dependency.name)
                pending.extend(dependency.all_input_nodes)

            actual_total = segment + 1
            if expected_total and actual_total != expected_total:
                raise RuntimeError(
                    "fallback-boundary shard selection saw an unexpected "
                    f"segment count: actual={actual_total} "
                    f"expected={expected_total}"
                )
            self.support_node_names = support_node_names
            logger.warning(
                "QNN support filtering selected fallback segments "
                "start=%d count=%d total=%d nodes=%d",
                prelower_start,
                prelower_limit,
                actual_total,
                len(support_node_names),
            )
        self.qnn_manager = get_current_qnn_manager(
            python_options.backend_options.backend_type, compiler_specs
        )

    def is_node_supported(self, _, node: torch.fx.Node) -> bool:
        if node.op != "call_function" or node.target in not_supported_operator:
            return False

        if (
            self.support_node_names is not None
            and node.name not in self.support_node_names
        ):
            return False

        if node.target in to_be_implemented_operator:
            print(
                f"[QNN Partitioner Op Support]: {node.target.__name__} | Skipped, this op can be supported, please report an issue in https://github.com/pytorch/executorch/issues"
            )
            return False

        if (
            node.target in allow_list_operator
            # bypass if custom op appears
            or OpContextLoader.namespace == node.target.namespace
            # bypass dequantize op for parameters & buffers
            or node.meta.get(QCOM_BYPASS_NODE, False)
        ):
            return True

        if (
            node.name in self.skip_node_id_set
            or node.target.__name__ in self.skip_node_op_set
        ):
            print(f"[QNN Partitioner Op Support]: {node.target.__name__} | Skipped")
            return False

        quant_attrs = node.meta.get(QCOM_QUANT_ATTRS, {})
        if QCOM_SCALES in quant_attrs and (
            quant_attrs.get(QCOM_SCALES) is None
            or quant_attrs.get(QCOM_ZERO_POINTS) is None
        ):
            # An annotated per-channel node without observed qparams cannot be
            # represented by QNN. Treat it as a capability miss so the cached
            # repartitioner keeps it portable instead of constructing a
            # malformed encoding (or calling len(None)).
            print(
                f"[QNN Partitioner Op Support]: {node.target.__name__} | False "
                "(incomplete per-channel qparams)"
            )
            return False

        supported = False
        try:
            op_wrapper = self.node_visitors[node.target.__name__].define_node(
                node, self.nodes_to_wrappers
            )
        except IncompleteQuantizationEncodingError:
            # The effective encoding may come from QCOM_REQUANTIZE on another
            # node, so checking only node.meta[QCOM_QUANT_ATTRS] is insufficient.
            # Keep this op portable rather than handing malformed qparams to QNN.
            print(
                f"[QNN Partitioner Op Support]: {node.target.__name__} | False "
                "(incomplete effective per-channel qparams)"
            )
            return False
        if node.target in constant_operator:
            return True

        op_wrapper_list = []
        if isinstance(op_wrapper, List):
            op_wrapper_list.extend(op_wrapper)
        else:
            op_wrapper_list.append(op_wrapper)

        if op_wrapper is not None:
            supported = self.qnn_manager.IsNodeSupportedByBackend(
                [op_wrapper.GetOpWrapper() for op_wrapper in op_wrapper_list]
            )

        self.nodes_to_wrappers.clear()
        print(f"[QNN Partitioner Op Support]: {node.target.__name__} | {supported}")
        return supported


class QnnPartitioner(Partitioner):
    """
    QnnPartitioner identifies subgraphs that can be lowered to QNN backend, by tagging nodes for delegation,
    and manages special cases such as mutable buffers and consumed constants.
    """

    def __init__(
        self,
        compiler_specs: List[CompileSpec],
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
        skip_mutable_buffer: bool = False,
    ):
        """
        Args:
            compiler_specs (List[CompileSpec]): Backend compiler specifications.
            skip_node_id_set (set, optional): Set of node IDs to exclude from partitioning.
            skip_node_op_set (set, optional): Set of OpOverload to exclude from partitioning.
            skip_mutable_buffer (bool, optional): If True, mutable buffers are not delegated to QNN.
        """
        self.compiler_specs_snapshot = copy.deepcopy(compiler_specs)

        self.delegation_spec = DelegationSpec(
            QnnBackend.__name__, self.compiler_specs_snapshot
        )
        self.partition_tags: Dict[str, DelegationSpec] = {}
        self.skip_node_id_set = set() if skip_node_id_set is None else skip_node_id_set
        self.skip_node_op_set = set() if skip_node_op_set is None else skip_node_op_set
        self.skip_mutable_buffer = skip_mutable_buffer

    def generate_partitions(
        self, edge_program: torch.export.ExportedProgram
    ) -> List[Any]:
        self.op_support_checker = QnnOperatorSupport(
            edge_program,
            self.compiler_specs_snapshot,
            self.skip_node_id_set,
            self.skip_node_op_set,
        )
        partitions = generate_partitions_from_list_of_nodes(
            edge_program.graph_module,
            op_support=self.op_support_checker,
        )
        # Q/DQ nodes are useful inside delegated computation, but they must not
        # straddle a QNN/portable boundary. Removing them from an already formed
        # partition can create dependency cycles, so discover boundary Q/DQ and
        # re-run capability partitioning from cached support decisions. This
        # splits the graph correctly without repeating QNN validation.
        initially_supported = {
            node for partition in partitions for node in partition.nodes
        }
        blocked_qdq = set()

        class CachedOperatorSupport(OperatorSupportBase):
            def is_node_supported(self, _, node: torch.fx.Node) -> bool:
                return node in initially_supported and node not in blocked_qdq

        while True:
            newly_blocked = set()
            for partition in partitions:
                remaining = set(partition.nodes)
                while True:
                    boundary_qdq = {
                        node
                        for node in remaining
                        if node.op == "call_function"
                        and (node.target in q_ops or node.target in dq_ops)
                        and (
                            any(
                                inp.op == "call_function" and inp not in remaining
                                for inp in node.all_input_nodes
                            )
                            or any(user not in remaining for user in node.users)
                        )
                    }
                    if not boundary_qdq:
                        break
                    remaining.difference_update(boundary_qdq)
                    newly_blocked.update(boundary_qdq)

                if remaining and not any(
                    node.op == "call_function"
                    and node.target not in q_ops
                    and node.target not in dq_ops
                    for node in remaining
                ):
                    newly_blocked.update(remaining)

            newly_blocked.difference_update(blocked_qdq)
            if not newly_blocked:
                break
            blocked_qdq.update(newly_blocked)
            logger.info(
                "Repartitioning with boundary Q/DQ left portable: %s",
                sorted(node.name for node in newly_blocked),
            )
            partitions = generate_partitions_from_list_of_nodes(
                edge_program.graph_module,
                op_support=CachedOperatorSupport(),
            )

        # QNN cannot serialize a graph whose only external dependencies are
        # lifted constants. Portable fallbacks introduced above can isolate a
        # small constant-only shell (observed as view + dequantize), which QNN
        # otherwise rejects with "No graph inputs present". Leave only those
        # shells portable; partitions fed by user inputs, mutable buffers, or
        # another portable computation remain delegated.
        def has_runtime_input(partition: Partition) -> bool:
            partition_nodes = set(partition.nodes)
            for node in partition.nodes:
                for input_node in node.all_input_nodes:
                    if input_node in partition_nodes:
                        continue
                    if input_node.op != "placeholder":
                        return True
                    if not is_parameter(input_node, edge_program) or is_mutable_buffer_input(
                        input_node, edge_program
                    ):
                        return True
            return False

        constant_only = [p for p in partitions if not has_runtime_input(p)]
        if constant_only:
            logger.info(
                "Leaving %d constant-only QNN shell partition(s) portable: %s",
                len(constant_only),
                [sorted(node.name for node in p.nodes) for p in constant_only],
            )
            constant_only_ids = {id(p) for p in constant_only}
            partitions = [p for p in partitions if id(p) not in constant_only_ids]

        prelower_limit = int(
            os.environ.get("ET_QNN_PRELOWER_SHARD_LIMIT", "0") or 0
        )
        if prelower_limit:
            prelower_start = int(
                os.environ.get("ET_QNN_PRELOWER_SHARD_START", "0") or 0
            )
            expected_total = int(
                os.environ.get("ET_QNN_PRELOWER_TOTAL_SHARDS", "0") or 0
            )
            support_layer_filter = bool(
                os.environ.get("ET_QNN_PRELOWER_SHARD_LIMIT", "")
            )
            if support_layer_filter and len(partitions) != prelower_limit:
                raise RuntimeError(
                    "support-layer filtering produced an unexpected partition "
                    f"count: actual={len(partitions)} "
                    f"expected={prelower_limit}"
                )
            if (
                not support_layer_filter
                and expected_total
                and len(partitions) != expected_total
            ):
                raise RuntimeError(
                    "pre-lowering shard selection saw an unexpected partition "
                    f"count: actual={len(partitions)} expected={expected_total}"
                )
            if support_layer_filter:
                logger.warning(
                    "QNN fallback-segment filtering retained global shard "
                    "range start=%d count=%d",
                    prelower_start,
                    prelower_limit,
                )
                return partitions
            prelower_end = prelower_start + prelower_limit
            if (
                prelower_start < 0
                or prelower_limit <= 0
                or prelower_end > len(partitions)
            ):
                raise RuntimeError(
                    "invalid pre-lowering shard range: "
                    f"start={prelower_start} limit={prelower_limit} "
                    f"partitions={len(partitions)}"
                )
            selected = partitions[prelower_start:prelower_end]
            logger.warning(
                "Pre-lowering QNN shard selection retained partitions %s "
                "from total=%d",
                [partition.id for partition in selected],
                len(partitions),
            )
            partitions = selected

        return partitions

    def tag_nodes(
        self, partitions: List[Partition], edge_program: torch.export.ExportedProgram
    ) -> None:
        """
        Tags nodes in the given partitions and the edge program's graph with delegation tags for QNN partitioning.
        """
        quantized_pcq_outputs = 0
        for partition in partitions:
            partition_nodes = set(partition.nodes)
            for node in partition.nodes:
                delegation_tag = f"qnn_{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                self.partition_tags[delegation_tag] = self.delegation_spec
                # A PCQ DQ rejected by capability remains in the portable
                # graph.  InsertIOQDQ decides whether to synthesize a delegate
                # output DQ from the *producer's* QCOM_QUANT_ATTRS, so use that
                # same source of truth here.  Looking only for a direct PCQ DQ
                # user misses boundaries after QDQ folding / layout splitting
                # and lets InsertIOQDQ put an unsupported per-channel DQ back
                # inside the extracted QNN submodule.
                #
                # Restrict this to real partition outputs with a complete PCQ
                # encoding. BuildQuantIo gives the call_delegate output its
                # integer dtype, and the portable graph retains the exact
                # dequantization/layout path and axis qparams.
                outside_users = [
                    user for user in node.users if user not in partition_nodes
                ]
                quant_attrs = node.meta.get(QCOM_QUANT_ATTRS, {})
                encoding = quant_attrs.get(QCOM_ENCODING)
                scales = quant_attrs.get(QCOM_SCALES)
                zero_points = quant_attrs.get(QCOM_ZERO_POINTS)
                is_complete_pcq = (
                    (encoding in dq_ops or encoding in q_ops)
                    and "per_channel" in str(encoding)
                    and scales is not None
                    and zero_points is not None
                )
                if outside_users and is_complete_pcq:
                    quantized_dtype = quant_attrs.get(QCOM_DTYPE)
                    if quantized_dtype is None:
                        raise AssertionError(
                            f"PCQ partition output {node.name} has no quantized dtype"
                        )
                    node.meta[QCOM_QUANTIZED_IO] = quantized_dtype
                    quantized_pcq_outputs += 1

        if quantized_pcq_outputs:
            logger.info(
                "Keeping %d PCQ delegate output(s) quantized for portable DQ",
                quantized_pcq_outputs,
            )

        # need to take care of consumed constants
        consumed_constants = (
            *edge_program.graph_signature.inputs_to_buffers,
            *edge_program.graph_signature.inputs_to_parameters,
        )
        for node in edge_program.graph_module.graph.nodes:
            # find placeholders as lifted_constants
            if node.op != "placeholder" or len(node.users) != 0:
                continue

            if node.name in consumed_constants:
                # does no harm to merge them into last partition,
                # since they will all be removed in following stage
                node.meta["delegation_tag"] = delegation_tag

    # override
    def partition(self, edge_program: torch.export.ExportedProgram) -> PartitionResult:
        # Generate partitions by QNN op_support checker
        partitions = self.generate_partitions(edge_program)
        del self.op_support_checker

        # If partitions are found, handle tagging of nodes, constant data, and mutated buffers for delegation
        if len(partitions) != 0:
            self.tag_nodes(partitions, edge_program)
            tag_constant_data(edge_program)
            if not self.skip_mutable_buffer:
                logger.info(
                    "Qnn partitioner will delegate torch mutable buffer with the same I/O address during the runtime, "
                    "so if your model contains mutable buffer, "
                    "then you can get the better performance with skip_mutable_buffer=False. "
                    "If you encounter accuracy issue during the runtime, "
                    "then please set `skip_mutable_buffer=True` and try again."
                )
                tag_mutated_buffer(edge_program)

        return PartitionResult(
            tagged_exported_program=edge_program, partition_tags=self.partition_tags
        )

    # override
    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        """
        Determines which op should not be decomposed during partitioning.
        The list of operators is obtained from `get_skip_decomp_table()`.
        The filter function (`filter_fn`) can be used to further refine which nodes are not decomposed. (advanced use case)
        """
        do_not_decompose = get_skip_decomp_table()
        return (do_not_decompose, filter_fn)
