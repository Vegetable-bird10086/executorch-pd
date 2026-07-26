# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import base64
import hashlib
import json
import logging
import os
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, final, List

import torch  # noqa: F401
import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
from executorch.backends.qualcomm._passes.qnn_pass_manager import QnnPassManager
from executorch.backends.qualcomm.builders.node_visitor_manager import get_node_visitors
from executorch.backends.qualcomm.builders.qnn_constants import OpContextLoader
from executorch.backends.qualcomm.partition.utils import generate_qnn_executorch_option
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchOpPackageInfo,
)
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_AXIS_ORDER,
    QCOM_TENSOR_NAME,
)
from executorch.backends.qualcomm.utils.qnn_manager_lifecycle import (
    get_current_qnn_manager,
)
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from executorch.exir.backend.utils import DelegateMappingBuilder
from executorch.exir.debug_handle_utils import DEBUG_HANDLE_KEY
from torch.export.exported_program import ExportedProgram

DEFAULT_DEBUG_HANDLE = 65535
DEFAULT_GRAPH_NAME = "forward"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_QNN_TENSOR_QPARAMS_PATH_ENV = "ET_QNN_TENSOR_QPARAMS_PATH"
_QNN_TENSOR_QPARAMS_OVERRIDE_PATH_ENV = "ET_QNN_TENSOR_QPARAMS_OVERRIDE_PATH"
_QNN_PRESERVE_STATIC_S16_ENV = "ET_QNN_PRESERVE_STATIC_S16"
_QNN_LLAMA_QUANT_PROFILE_ENABLED_ENV = "ET_QNN_LLAMA_QUANT_PROFILE"
_QNN_LLAMA_QUANT_PROFILE_PROBE_PATH_ENV = (
    "ET_QNN_LLAMA_QUANT_PROFILE_PROBE_PATH"
)
_QNN_DIAGNOSTIC_SHARD_LIMIT_ENV = "ET_QNN_DIAGNOSTIC_SHARD_LIMIT"
_llama_quant_profile_batches: List[Dict[str, Any]] = []
_qnn_tensor_qparams_override_cache: Dict[str, Any] = {}


_QNN_DATA_TYPE_ELEMENT_BYTES = {
    "QNN_DATATYPE_BOOL_8": 1,
    "QNN_DATATYPE_INT_8": 1,
    "QNN_DATATYPE_UINT_8": 1,
    "QNN_DATATYPE_SFIXED_POINT_4": 1,
    "QNN_DATATYPE_UFIXED_POINT_4": 1,
    "QNN_DATATYPE_SFIXED_POINT_8": 1,
    "QNN_DATATYPE_UFIXED_POINT_8": 1,
    "QNN_DATATYPE_INT_16": 2,
    "QNN_DATATYPE_UINT_16": 2,
    "QNN_DATATYPE_FLOAT_16": 2,
    "QNN_DATATYPE_SFIXED_POINT_16": 2,
    "QNN_DATATYPE_UFIXED_POINT_16": 2,
    "QNN_DATATYPE_INT_32": 4,
    "QNN_DATATYPE_UINT_32": 4,
    "QNN_DATATYPE_FLOAT_32": 4,
    "QNN_DATATYPE_SFIXED_POINT_32": 4,
    "QNN_DATATYPE_UFIXED_POINT_32": 4,
    "QNN_DATATYPE_INT_64": 8,
    "QNN_DATATYPE_UINT_64": 8,
    "QNN_DATATYPE_FLOAT_64": 8,
}

_LLAMA_QNN_QUANT_PROFILE_CAPABILITIES = {
    "exact_f32_scale_bits": True,
    "raw_operator_parameters": True,
    "operation_tensor_sources": True,
    "embedded_static_affine_tensor_bytes": True,
    "blockwise_weight_scale_payload": True,
    "blockwise_weight_payload_digest": True,
    "structured_decoder_tensor_bindings": True,
    # The runtime profile has a complete index of every QNN U16 activation
    # tensor.  llama.cpp consumes this index instead of inferring qparams from
    # a parameter ordinal or a printed FX graph.
    "complete_u16_tensor_qparams": True,
}

_DECODER_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_DECODER_PROJECTION_SUFFIXES = {
    "self_attn.q_proj": "self_attn.q_proj",
    "self_attn.k_proj": "self_attn.k_proj",
    "self_attn.v_proj": "self_attn.v_proj",
    "self_attn.o_proj": "self_attn.o_proj",
    "mlp.gate_proj": "mlp.gate_proj",
    "mlp.up_proj": "mlp.up_proj",
    "mlp.down_proj": "mlp.down_proj",
    # The Qwen prefill wrappers use these names before the model is lowered.
    "attention.wq_conv": "self_attn.q_proj",
    "attention.wk_conv": "self_attn.k_proj",
    "attention.wv_conv": "self_attn.v_proj",
    "attention.wo_conv": "self_attn.o_proj",
    "feed_forward.w1_conv": "mlp.gate_proj",
    "feed_forward.w3_conv": "mlp.up_proj",
    "feed_forward.w2_conv": "mlp.down_proj",
}


def reset_llama_quant_profile_batches() -> None:
    """Discard in-process QNN profiles before a decoder export."""
    _llama_quant_profile_batches.clear()


def consume_llama_quant_profile_batches() -> List[Dict[str, Any]]:
    """Return profiles captured during one export and clear the in-process cache."""
    batches = list(_llama_quant_profile_batches)
    _llama_quant_profile_batches.clear()
    return batches


def _llama_quant_profile_enabled() -> bool:
    return os.environ.get(_QNN_LLAMA_QUANT_PROFILE_ENABLED_ENV, "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _environment_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _enum_name(value) -> str:
    """Return the stable suffix of a pybind enum's string representation."""
    return str(value).rsplit(".", maxsplit=1)[-1]


def _qnn_raw_bytes(value, attribute: str) -> bytes:
    """Read pybind byte fields exposed as either a method or a bytes property."""
    raw = getattr(value, attribute)
    if callable(raw):
        raw = raw()
    return bytes(raw)


def _serialize_scale_offset(entry) -> Dict[str, Any]:
    # QNN dequantizes as (code + offset) * scale.  Keep the original offset
    # and the derived zero point so consumers cannot silently swap conventions.
    scale = float(entry.scale)
    offset = int(entry.offset)
    return {
        "scale": scale,
        "scale_f32_le_hex": struct.pack("<f", scale).hex(),
        "offset": offset,
        "zero_point": -offset,
    }


def _serialize_qnn_tensor(
    tensor,
    *,
    include_blockwise_payload: bool = False,
    include_static_payload: bool = False,
) -> Dict[str, Any]:
    """Serialize QNN tensor metadata without changing graph construction."""
    if tensor.version != 2:
        return {"version": int(tensor.version), "unsupported_version": True}

    node = tensor.v2
    qparams = node.quantizeParams
    encoding = qparams.quantizationEncoding
    dimensions = [int(dim) for dim in node.dimensions]
    result: Dict[str, Any] = {
        "version": int(tensor.version),
        "name": node.name,
        "tensor_type": _enum_name(node.type),
        "data_type": _enum_name(node.dataType),
        "data_format": _enum_name(node.dataFormat),
        "dimensions": dimensions,
        "quantization_encoding_definition": _enum_name(qparams.encodingDefinition),
        "quantization_encoding": _enum_name(encoding),
    }

    is_blockwise = (
        encoding
        == PyQnnManager.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION
    )
    if (
        encoding
        == PyQnnManager.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET
    ):
        scale_offset = _serialize_scale_offset(qparams.scaleOffsetEncoding)
        # Preserve the ETDump sidecar schema while adding an exact f32 payload.
        result["scale"] = scale_offset["scale"]
        result["offset"] = scale_offset["offset"]
        result["scale_offset"] = scale_offset
    elif (
        encoding
        == PyQnnManager.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET
    ):
        axis_scale_offset = qparams.axisScaleOffsetEncoding
        scale_offsets = [
            _serialize_scale_offset(entry) for entry in axis_scale_offset.scaleOffset
        ]
        # Existing consumers read axis/scale_offsets directly.
        result["axis"] = int(axis_scale_offset.axis)
        result["scale_offsets"] = scale_offsets
        result["axis_scale_offset"] = {
            "axis": result["axis"],
            "num_scale_offsets": len(scale_offsets),
            "scale_offsets": scale_offsets,
        }
    elif (
        include_blockwise_payload
        and is_blockwise
    ):
        if not hasattr(qparams, "blockwiseExpansionEncoding"):
            raise RuntimeError(
                "QNN Python bindings do not expose BLOCKWISE_EXPANSION metadata; "
                "rebuild PyQnnManagerAdaptor before requesting a llama profile"
            )
        # QNN does not store an explicit scaleOffsets count.  The builder emits
        # one entry for each element along the blockwise axis.
        blockwise_header = qparams.blockwiseExpansionEncoding(0)
        axis = int(blockwise_header["axis"])
        if axis < 0 or axis >= len(dimensions):
            raise RuntimeError(
                f"Invalid QNN blockwise axis {axis} for tensor {node.name} "
                f"with dimensions {dimensions}"
            )
        num_scale_offsets = dimensions[axis]
        blockwise = qparams.blockwiseExpansionEncoding(num_scale_offsets)
        blocks = bytes(blockwise["blocksScale8"])
        num_blocks_per_axis = int(blockwise["numBlocksPerAxis"])
        block_scale_storage_type = _enum_name(blockwise["blockScaleStorageType"])
        if block_scale_storage_type not in {
            "QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8",
            "QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_16",
        }:
            raise RuntimeError(
                f"Unsupported QNN blockwise scale storage for {node.name}: "
                f"{block_scale_storage_type}"
            )
        block_scale_element_bytes = (
            2
            if block_scale_storage_type
            == "QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_16"
            else 1
        )
        expected_bytes = (
            num_scale_offsets * num_blocks_per_axis * block_scale_element_bytes
        )
        if len(blocks) != expected_bytes:
            raise RuntimeError(
                f"QNN blockwise scale payload mismatch for {node.name}: "
                f"expected {expected_bytes}, got {len(blocks)}"
            )
        result["blockwise_expansion"] = {
            "axis": axis,
            "num_scale_offsets": num_scale_offsets,
            "num_blocks_per_axis": num_blocks_per_axis,
            "block_scale_bitwidth": int(blockwise["blockScaleBitwidth"]),
            "block_scale_storage_type": block_scale_storage_type,
            "block_scale_element_bytes": block_scale_element_bytes,
            # The QNN builder concatenates all block-scale codes for channel 0,
            # then channel 1, and so on.  Each storage element holds one code;
            # it is not bit-packed even when block_scale_bitwidth is 4.
            "block_scale_layout": "scale_offset_index_major_then_block_index",
            "block_scale_code_storage": "one_unsigned_code_per_storage_element",
            "block_scale_effective_scale_formula": (
                "effective_scale=scale_offsets[scale_offset_index].scale*"
                "block_scale_code"
            ),
            "block_scale_storage_byte_order": (
                "little_endian" if block_scale_element_bytes == 2 else "not_applicable"
            ),
            "scale_offsets": [
                _serialize_scale_offset(entry)
                for entry in blockwise["scaleOffsets"]
            ],
            "block_scales_base64": base64.b64encode(blocks).decode("ascii"),
            "block_scales_bytes": len(blocks),
        }

    if include_static_payload and result["tensor_type"] == "QNN_TENSOR_TYPE_STATIC":
        if is_blockwise:
            # The source GGUF reconstructs this large static weight stream in
            # registers, so embed a digest rather than duplicating every weight
            # code in the JSON manifest.  The exact blockwise scale payload is
            # already retained above and is mandatory for re-quantization.
            raw = _qnn_raw_bytes(node, "rawDataBytes")
            if not raw:
                raise RuntimeError(
                    f"QNN static blockwise tensor {node.name} has no payload to fingerprint"
                )
            result["static_payload"] = {
                "storage": "external_gptq_int2_source_reconstruction",
                "qnn_payload_bytes": len(raw),
                "qnn_payload_sha256": hashlib.sha256(raw).hexdigest(),
            }
        else:
            raw = _qnn_raw_bytes(node, "rawDataBytes")
            data_type = result["data_type"]
            element_bytes = _QNN_DATA_TYPE_ELEMENT_BYTES.get(data_type)
            if element_bytes is None:
                raise RuntimeError(
                    f"Unsupported static QNN tensor datatype {data_type} for {node.name}"
                )
            expected_bytes = element_bytes
            for dimension in dimensions:
                expected_bytes *= dimension
            if len(raw) != expected_bytes:
                raise RuntimeError(
                    f"QNN static tensor payload mismatch for {node.name}: "
                    f"expected {expected_bytes}, got {len(raw)}"
                )
            result["static_payload"] = {
                "storage": "embedded_exact_bytes",
                "element_bytes": element_bytes,
                "data_le_base64": base64.b64encode(raw).decode("ascii"),
                "data_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }

    return result


def _serialize_qnn_param(param) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "name": param.name,
        "param_type": _enum_name(param.paramType),
    }
    if param.paramType == PyQnnManager.Qnn_ParamType_t.QNN_PARAMTYPE_SCALAR:
        scalar = param.scalarParam
        scalar_data_type = _enum_name(scalar.dataType)
        try:
            raw = _qnn_raw_bytes(scalar, "rawValueBytes")
        except RuntimeError as error:
            # QNN exposes certain scalar union sentinels without a numeric ABI
            # representation.  Preserve their datatype and the SDK reason; do
            # not invent bytes or weaken tensor-qparam validation.
            result["param_type"] = "QNN_PARAMTYPE_UNSUPPORTED_SCALAR"
            result["unsupported_scalar"] = {
                "data_type": scalar_data_type,
                "reason": str(error),
            }
            return result
        result["scalar"] = {
            "data_type": scalar_data_type,
            "value_le_base64": base64.b64encode(raw).decode("ascii"),
            "value_bytes": len(raw),
        }
    elif param.paramType == PyQnnManager.Qnn_ParamType_t.QNN_PARAMTYPE_TENSOR:
        tensor = param.tensorParam
        serialized_tensor = _serialize_qnn_tensor(
            tensor,
            include_blockwise_payload=True,
            include_static_payload=True,
        )
        if serialized_tensor.get("unsupported_version"):
            # Some QNN parameter unions advertise TENSOR while their tensor
            # payload is an unavailable ABI version.  Preserve this exact fact
            # without treating it as a named runtime tensor.
            result["param_type"] = "QNN_PARAMTYPE_UNSUPPORTED_TENSOR"
            result["unsupported_tensor"] = serialized_tensor
            return result
        result["tensor"] = serialized_tensor
        raw = _qnn_raw_bytes(tensor.v2, "rawDataBytes")
        result["tensor_data_base64"] = base64.b64encode(raw).decode("ascii")
        result["tensor_data_bytes"] = len(raw)
    return result


def _serialize_fx_argument(value: Any) -> Any:
    if isinstance(value, torch.fx.Node):
        return {"fx_node": value.name}
    if isinstance(value, torch.dtype):
        return {"torch_dtype": str(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_fx_argument(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _serialize_fx_argument(item) for key, item in value.items()
        }
    return {"repr": repr(value)}


def _module_paths_from_stack(module_stack: Any) -> List[str]:
    """Preserve module paths without relying on a Python dict repr as an ABI."""
    paths: List[str] = []
    if isinstance(module_stack, dict):
        for key, value in module_stack.items():
            if isinstance(key, str) and key:
                paths.append(key)
            if isinstance(value, (tuple, list)) and value:
                module_path = value[0]
                if isinstance(module_path, str) and module_path:
                    paths.append(module_path)
    elif isinstance(module_stack, (tuple, list)):
        for value in module_stack:
            if isinstance(value, str) and value:
                paths.append(value)
            elif isinstance(value, (tuple, list)) and value and isinstance(value[0], str):
                paths.append(value[0])
    elif isinstance(module_stack, str) and module_stack:
        paths.append(module_stack)
    return list(dict.fromkeys(paths))


def _decoder_binding(module_stack: Any) -> Dict[str, Any]:
    module_paths = _module_paths_from_stack(module_stack)
    layer_ids = sorted(
        {
            int(layer_id)
            for module_path in module_paths
            for layer_id in _DECODER_LAYER_PATTERN.findall(module_path)
        }
    )
    projection = None
    for module_path in module_paths:
        for suffix, canonical_name in _DECODER_PROJECTION_SUFFIXES.items():
            if module_path == suffix or module_path.endswith(f".{suffix}"):
                projection = canonical_name
                break
        if projection is not None:
            break
    return {
        "layer_ids": layer_ids,
        "module_paths": module_paths,
        "projection": projection,
    }


def _source_record(node) -> Dict[str, Any]:
    value = node.meta.get("val")
    if isinstance(value, torch.Tensor):
        value_meta: Any = {
            "shape": [str(dim) for dim in value.shape],
            "dtype": str(value.dtype),
        }
    elif isinstance(value, (list, tuple)):
        value_meta = [
            {"shape": [str(dim) for dim in item.shape], "dtype": str(item.dtype)}
            if isinstance(item, torch.Tensor)
            else str(type(item))
            for item in value
        ]
    else:
        value_meta = None

    tensor_names = node.meta.get(QCOM_TENSOR_NAME, {})
    if isinstance(tensor_names, dict):
        tensor_names = [str(name) for _, name in sorted(tensor_names.items())]
    else:
        tensor_names = [str(name) for name in tensor_names]
    axis_order = node.meta.get(QCOM_AXIS_ORDER)
    module_stack = node.meta.get("nn_module_stack", "")
    return {
        "fx_node_name": node.name,
        "fx_op": node.op,
        "fx_target": str(node.target),
        "qnn_tensor_names": tensor_names,
        "tensor": value_meta,
        "axis_order": None if axis_order is None else [int(axis) for axis in axis_order],
        "fx_args": _serialize_fx_argument(node.args),
        "fx_kwargs": _serialize_fx_argument(node.kwargs),
        "nn_module_stack": str(module_stack),
        "decoder_binding": _decoder_binding(module_stack),
        "stack_trace": str(node.meta.get("stack_trace", "")),
        "source_fn_stack": str(node.meta.get("source_fn_stack", "")),
    }


def _source_records_by_qnn_name(graph_module) -> Dict[str, Dict[str, Any]]:
    records = {}
    for node in graph_module.graph.nodes:
        if not hasattr(node, "meta"):
            continue
        record = _source_record(node)
        records.setdefault(node.name, record)
        for tensor_name in record["qnn_tensor_names"]:
            records.setdefault(tensor_name, record)
    return records


def _register_profile_tensor(
    tensors: Dict[str, Dict[str, Any]], tensor: Dict[str, Any]
) -> None:
    name = tensor.get("name")
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"QNN tensor profile has no valid name: {tensor}")
    existing = tensors.get(name)
    if existing is None:
        tensors[name] = tensor
    elif existing != tensor:
        raise RuntimeError(
            "inconsistent QNN tensor metadata for the same tensor name "
            f"{name}; refusing to discard qparams from a later operation"
        )


def _u16_tensor_index(
    tensors: Dict[str, Dict[str, Any]], operations: List[Dict[str, Any]],
    logical_tensors: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Index all QNN U16 ABI tensors without approximating their qparams.

    The top-level tensor map remains the single source of exact qparams.  This
    compact index gives llama.cpp the producer/consumer path and decoder source
    records needed to select a qparam record at runtime.
    """
    index: Dict[str, Dict[str, Any]] = {
        name: {
            "name": name,
            "data_type": tensor["data_type"],
            "quantization_encoding": tensor["quantization_encoding"],
            "operation_uses": [],
            "sources": [],
        }
        for name, tensor in tensors.items()
        if tensor.get("data_type") == "QNN_DATATYPE_UFIXED_POINT_16"
    }

    def add_source(tensor_name: str, source: Any) -> None:
        if tensor_name not in index or not isinstance(source, dict):
            return
        fx_node_name = source.get("fx_node_name")
        if not isinstance(fx_node_name, str) or not fx_node_name:
            return
        if all(
            existing.get("fx_node_name") != fx_node_name
            for existing in index[tensor_name]["sources"]
        ):
            index[tensor_name]["sources"].append(source)

    for tensor_name, source in logical_tensors.items():
        add_source(tensor_name, source)

    for operation in operations:
        op_name = operation["name"]
        for role, names, sources in (
            ("input", operation["inputs"], operation["input_sources"]),
            ("output", operation["outputs"], operation["output_sources"]),
        ):
            for position, tensor_name in enumerate(names):
                if tensor_name not in index:
                    continue
                index[tensor_name]["operation_uses"].append(
                    {
                        "operation_name": op_name,
                        "role": role,
                        "position": position,
                    }
                )
                add_source(tensor_name, sources.get(tensor_name))
                add_source(tensor_name, operation.get("source"))
        for position, param in enumerate(operation["params"]):
            if param.get("param_type") != "QNN_PARAMTYPE_TENSOR":
                continue
            tensor_name = param.get("tensor", {}).get("name")
            if tensor_name not in index:
                continue
            index[tensor_name]["operation_uses"].append(
                {
                    "operation_name": op_name,
                    "role": "tensor_param",
                    "position": position,
                }
            )
            add_source(tensor_name, operation.get("source"))

    for entry in index.values():
        entry["decoder_bindings"] = [
            source["decoder_binding"] for source in entry["sources"]
        ]
    return index


def _build_llama_quant_profile(py_op_wrappers, graph_module, scope: str) -> Dict[str, Any]:
    source_records = _source_records_by_qnn_name(graph_module)
    tensors: Dict[str, Dict[str, Any]] = {}
    operations = []
    for py_op_wrapper in py_op_wrappers:
        op_config = py_op_wrapper.GetOpWrapper().GetOpConfig()
        inputs = [
            _serialize_qnn_tensor(
                tensor,
                include_blockwise_payload=True,
                include_static_payload=True,
            )
            for tensor in op_config["inputTensors"]
        ]
        outputs = [
            _serialize_qnn_tensor(
                tensor,
                include_blockwise_payload=True,
                include_static_payload=True,
            )
            for tensor in op_config["outputTensors"]
        ]
        for tensor in [*inputs, *outputs]:
            _register_profile_tensor(tensors, tensor)
        params = [_serialize_qnn_param(param) for param in op_config["params"]]
        for param in params:
            if param.get("param_type") == "QNN_PARAMTYPE_TENSOR":
                _register_profile_tensor(tensors, param["tensor"])
        operations.append(
            {
                "name": op_config["name"],
                "package_name": op_config["packageName"],
                "type_name": op_config["typeName"],
                "inputs": [tensor.get("name") for tensor in inputs],
                "outputs": [tensor.get("name") for tensor in outputs],
                "params": params,
                "source": source_records.get(op_config["name"]),
                # Tensor names are the ABI used by the QNN graph.  Keep their
                # FX origin beside each operation so llama.cpp can bind a QNN
                # qparam record to a decoder tensor without relying on an
                # exporter-specific ordinal such as frozen_param18.
                "input_sources": {
                    tensor["name"]: source_records.get(tensor["name"])
                    for tensor in inputs
                    if tensor["name"] in source_records
                },
                "output_sources": {
                    tensor["name"]: source_records.get(tensor["name"])
                    for tensor in outputs
                    if tensor["name"] in source_records
                },
            }
        )

    logical_tensors = {
        name: source_records[name]
        for name in tensors
        if name in source_records
    }
    u16_tensor_index = _u16_tensor_index(tensors, operations, logical_tensors)
    return {
        "schema_version": 2,
        "format": "llama-qnn-quant-profile-v2",
        "scope": scope,
        "quantization_formula": "real=(integer_code+offset)*scale",
        "capabilities": dict(_LLAMA_QNN_QUANT_PROFILE_CAPABILITIES),
        "tensors": tensors,
        "operations": operations,
        "logical_tensors": logical_tensors,
        "u16_tensor_index": u16_tensor_index,
    }


def _validate_scale_offset_payload(payload: Dict[str, Any], location: str) -> None:
    required = {"scale", "scale_f32_le_hex", "offset", "zero_point"}
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"missing scale/offset fields at {location}: {sorted(missing)}")
    raw_scale = bytes.fromhex(str(payload["scale_f32_le_hex"]))
    if len(raw_scale) != 4:
        raise RuntimeError(f"non-f32 scale payload at {location}")
    exact_scale = struct.unpack("<f", raw_scale)[0]
    if float(payload["scale"]) != exact_scale:
        raise RuntimeError(f"lossy scale serialization at {location}")
    if int(payload["zero_point"]) != -int(payload["offset"]):
        raise RuntimeError(f"inconsistent QNN offset convention at {location}")


def _validate_decoder_binding(
    source: Dict[str, Any], location: str, *, require_module_path: bool
) -> None:
    if not isinstance(source, dict):
        raise RuntimeError(f"missing decoder source binding at {location}")
    fx_node_name = source.get("fx_node_name")
    if not isinstance(fx_node_name, str) or not fx_node_name:
        raise RuntimeError(f"invalid FX node name at {location}")
    binding = source.get("decoder_binding")
    if not isinstance(binding, dict):
        raise RuntimeError(f"missing structured decoder binding at {location}")
    module_paths = binding.get("module_paths")
    layer_ids = binding.get("layer_ids")
    projection = binding.get("projection")
    if not isinstance(module_paths, list) or any(
        not isinstance(path, str) or not path for path in module_paths
    ):
        raise RuntimeError(f"invalid decoder module paths at {location}")
    if require_module_path and not module_paths:
        raise RuntimeError(f"missing decoder module path at {location}")
    if not isinstance(layer_ids, list) or any(
        not isinstance(layer_id, int) or layer_id < 0 for layer_id in layer_ids
    ):
        raise RuntimeError(f"invalid decoder layer IDs at {location}")
    if projection is not None and projection not in set(
        _DECODER_PROJECTION_SUFFIXES.values()
    ):
        raise RuntimeError(f"invalid decoder projection binding at {location}")


def _validate_static_payload(
    tensor_name: str, tensor: Dict[str, Any], dimensions: List[int]
) -> None:
    if tensor.get("tensor_type") != "QNN_TENSOR_TYPE_STATIC":
        if "static_payload" in tensor:
            raise RuntimeError(
                f"non-static QNN tensor {tensor_name} unexpectedly has static payload"
            )
        return

    static_payload = tensor.get("static_payload")
    if not isinstance(static_payload, dict):
        raise RuntimeError(f"static QNN tensor {tensor_name} lacks exact payload metadata")
    encoding = tensor.get("quantization_encoding")
    if encoding == "QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION":
        if static_payload.get("storage") != "external_gptq_int2_source_reconstruction":
            raise RuntimeError(
                f"static blockwise QNN tensor {tensor_name} has an invalid payload contract"
            )
        payload_bytes = int(static_payload.get("qnn_payload_bytes", -1))
        payload_sha256 = static_payload.get("qnn_payload_sha256")
        if payload_bytes <= 0 or not isinstance(payload_sha256, str) or len(payload_sha256) != 64:
            raise RuntimeError(
                f"static blockwise QNN tensor {tensor_name} has an invalid payload digest"
            )
        try:
            int(payload_sha256, 16)
        except ValueError as error:
            raise RuntimeError(
                f"static blockwise QNN tensor {tensor_name} has a non-hex payload digest"
            ) from error
        return

    if static_payload.get("storage") != "embedded_exact_bytes":
        raise RuntimeError(
            f"static QNN tensor {tensor_name} must embed its exact raw bytes"
        )
    data_type = tensor.get("data_type")
    expected_element_bytes = _QNN_DATA_TYPE_ELEMENT_BYTES.get(data_type)
    element_bytes = int(static_payload.get("element_bytes", -1))
    if expected_element_bytes is None or element_bytes != expected_element_bytes:
        raise RuntimeError(
            f"static QNN tensor {tensor_name} has an invalid element size for {data_type}"
        )
    try:
        raw = base64.b64decode(static_payload.get("data_le_base64", ""), validate=True)
    except ValueError as error:
        raise RuntimeError(f"invalid static QNN payload encoding at {tensor_name}") from error
    expected_bytes = element_bytes
    for dimension in dimensions:
        expected_bytes *= dimension
    if len(raw) != expected_bytes or int(static_payload.get("data_bytes", -1)) != expected_bytes:
        raise RuntimeError(
            f"static QNN payload size mismatch at {tensor_name}: "
            f"expected {expected_bytes}, got {len(raw)}"
        )
    payload_sha256 = static_payload.get("sha256")
    if payload_sha256 != hashlib.sha256(raw).hexdigest():
        raise RuntimeError(f"static QNN payload digest mismatch at {tensor_name}")


def _validate_profile_tensor(
    tensor_name: str,
    tensor: Dict[str, Any],
    logical_tensors: Dict[str, Dict[str, Any]],
) -> tuple[bool, str]:
    location = f"tensor {tensor_name}"
    if tensor.get("version") != 2 or tensor.get("name") != tensor_name:
        raise RuntimeError(f"invalid QNN tensor profile entry for {tensor_name}")
    dimensions = tensor.get("dimensions")
    if not isinstance(dimensions, list) or any(int(dim) <= 0 for dim in dimensions):
        raise RuntimeError(f"invalid QNN dimensions at {location}: {dimensions}")
    for metadata_name in ("tensor_type", "data_type", "data_format"):
        metadata_value = tensor.get(metadata_name)
        if not isinstance(metadata_value, str) or not metadata_value:
            raise RuntimeError(f"missing QNN {metadata_name} metadata at {location}")
    encoding_definition = tensor.get("quantization_encoding_definition")
    if not isinstance(encoding_definition, str) or not encoding_definition:
        raise RuntimeError(f"missing QNN quantization definition at {location}")
    encoding = tensor.get("quantization_encoding")
    if not isinstance(encoding, str) or not encoding:
        raise RuntimeError(f"missing QNN quantization encoding at {location}")
    _validate_static_payload(tensor_name, tensor, dimensions)

    if encoding == "QNN_QUANTIZATION_ENCODING_UNDEFINED":
        return False, encoding
    if encoding == "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET":
        _validate_scale_offset_payload(tensor.get("scale_offset", {}), location)
    elif encoding == "QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET":
        payload = tensor.get("axis_scale_offset", {})
        axis = int(payload.get("axis", -1))
        if axis < 0 or axis >= len(dimensions):
            raise RuntimeError(f"invalid QNN axis at {location}: {axis}")
        scale_offsets = payload.get("scale_offsets", [])
        if int(payload.get("num_scale_offsets", -1)) != len(scale_offsets):
            raise RuntimeError(f"missing axis qparam count at {location}")
        if len(scale_offsets) != dimensions[axis]:
            raise RuntimeError(
                f"axis qparam count mismatch at {location}: "
                f"expected {dimensions[axis]}, got {len(scale_offsets)}"
            )
        for index, scale_offset in enumerate(scale_offsets):
            _validate_scale_offset_payload(
                scale_offset, f"{location} axis scale_offset[{index}]"
            )
    elif encoding == "QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION":
        payload = tensor.get("blockwise_expansion", {})
        axis = int(payload.get("axis", -1))
        if axis < 0 or axis >= len(dimensions):
            raise RuntimeError(f"invalid QNN blockwise axis at {location}: {axis}")
        num_scale_offsets = int(payload.get("num_scale_offsets", -1))
        num_blocks_per_axis = int(payload.get("num_blocks_per_axis", -1))
        if num_scale_offsets != dimensions[axis] or num_blocks_per_axis <= 0:
            raise RuntimeError(f"invalid QNN blockwise dimensions at {location}")
        scale_offsets = payload.get("scale_offsets", [])
        if len(scale_offsets) != num_scale_offsets:
            raise RuntimeError(f"missing QNN blockwise scale offsets at {location}")
        for index, scale_offset in enumerate(scale_offsets):
            _validate_scale_offset_payload(
                scale_offset, f"{location} blockwise scale_offset[{index}]"
            )
        try:
            block_scales = base64.b64decode(
                payload.get("block_scales_base64", ""), validate=True
            )
        except ValueError as error:
            raise RuntimeError(f"invalid block scale payload at {location}") from error
        storage_type = payload.get("block_scale_storage_type")
        if storage_type not in {
            "QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8",
            "QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_16",
        }:
            raise RuntimeError(
                f"unsupported QNN block scale storage at {location}: {storage_type}"
            )
        element_bytes = int(payload.get("block_scale_element_bytes", -1))
        expected_element_bytes = (
            2
            if storage_type
            == "QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_16"
            else 1
        )
        if element_bytes != expected_element_bytes:
            raise RuntimeError(f"invalid QNN block scale element size at {location}")
        if payload.get("block_scale_layout") != (
            "scale_offset_index_major_then_block_index"
        ):
            raise RuntimeError(f"missing QNN block scale layout at {location}")
        if payload.get("block_scale_code_storage") != (
            "one_unsigned_code_per_storage_element"
        ):
            raise RuntimeError(f"missing QNN block scale code storage at {location}")
        if payload.get("block_scale_effective_scale_formula") != (
            "effective_scale=scale_offsets[scale_offset_index].scale*"
            "block_scale_code"
        ):
            raise RuntimeError(
                f"missing QNN block scale effective-scale formula at {location}"
            )
        expected_byte_order = "little_endian" if element_bytes == 2 else "not_applicable"
        if payload.get("block_scale_storage_byte_order") != expected_byte_order:
            raise RuntimeError(f"invalid QNN block scale byte order at {location}")
        expected_block_bytes = num_scale_offsets * num_blocks_per_axis * element_bytes
        if len(block_scales) != expected_block_bytes or int(
            payload.get("block_scales_bytes", -1)
        ) != expected_block_bytes:
            raise RuntimeError(
                f"QNN block scale payload mismatch at {location}: "
                f"expected {expected_block_bytes}, got {len(block_scales)}"
            )
        if int(payload.get("block_scale_bitwidth", 0)) <= 0:
            raise RuntimeError(f"incomplete QNN blockwise metadata at {location}")
        # Lowering may create static blockwise helper constants without an FX
        # decoder source.  They are retained in the ABI profile but are not
        # llama.cpp weight bindings.  Validate a decoder binding only when QNN
        # actually carries one; U16 activations remain mandatory elsewhere.
        source = logical_tensors.get(tensor_name)
        if (
            tensor.get("tensor_type") == "QNN_TENSOR_TYPE_STATIC"
            and isinstance(source, dict)
            and source.get("decoder_binding", {}).get("module_paths")
        ):
            _validate_decoder_binding(
                source,
                f"static QNN blockwise tensor {tensor_name}",
                require_module_path=True,
            )
    else:
        raise RuntimeError(f"unhandled QNN quantization encoding {encoding} at {location}")
    return True, encoding


def _validate_u16_tensor_index(
    tensors: Dict[str, Dict[str, Any]],
    operations: List[Dict[str, Any]],
    u16_tensor_index: Dict[str, Dict[str, Any]],
) -> None:
    expected_names = {
        tensor_name
        for tensor_name, tensor in tensors.items()
        if tensor.get("data_type") == "QNN_DATATYPE_UFIXED_POINT_16"
    }
    if set(u16_tensor_index) != expected_names:
        raise RuntimeError(
            "QNN U16 index does not cover exactly the U16 tensor ABI: "
            f"expected={sorted(expected_names)} actual={sorted(u16_tensor_index)}"
        )

    expected_uses: Dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    for operation in operations:
        op_name = operation["name"]
        for role, tensor_names in (
            ("input", operation["inputs"]),
            ("output", operation["outputs"]),
        ):
            for position, tensor_name in enumerate(tensor_names):
                if tensor_name in expected_names:
                    expected_uses[tensor_name].add((op_name, role, position))
        for position, param in enumerate(operation["params"]):
            if param.get("param_type") != "QNN_PARAMTYPE_TENSOR":
                continue
            tensor_name = param["tensor"]["name"]
            if tensor_name in expected_names:
                expected_uses[tensor_name].add((op_name, "tensor_param", position))

    for tensor_name, entry in u16_tensor_index.items():
        tensor = tensors[tensor_name]
        if not isinstance(entry, dict):
            raise RuntimeError(f"QNN U16 index entry is malformed: {tensor_name}")
        if entry.get("name") != tensor_name:
            raise RuntimeError(f"QNN U16 index has a mismatched name: {tensor_name}")
        if entry.get("data_type") != tensor.get("data_type") or entry.get(
            "quantization_encoding"
        ) != tensor.get("quantization_encoding"):
            raise RuntimeError(
                f"QNN U16 index loses qparam identity for tensor {tensor_name}"
            )
        uses = entry.get("operation_uses")
        if not isinstance(uses, list):
            raise RuntimeError(f"QNN U16 index lacks operation uses: {tensor_name}")
        actual_uses = set()
        for use in uses:
            if not isinstance(use, dict):
                raise RuntimeError(f"invalid QNN U16 operation use: {tensor_name}")
            op_name = use.get("operation_name")
            role = use.get("role")
            position = use.get("position")
            if (
                not isinstance(op_name, str)
                or role not in {"input", "output", "tensor_param"}
                or not isinstance(position, int)
                or position < 0
            ):
                raise RuntimeError(f"invalid QNN U16 operation use: {tensor_name}")
            actual_uses.add((op_name, role, position))
        if actual_uses != expected_uses[tensor_name]:
            raise RuntimeError(
                f"QNN U16 index has incomplete operation coverage for {tensor_name}"
            )
        sources = entry.get("sources")
        bindings = entry.get("decoder_bindings")
        if not isinstance(sources, list) or not isinstance(bindings, list):
            raise RuntimeError(f"QNN U16 index lacks decoder sources: {tensor_name}")
        for source in sources:
            _validate_decoder_binding(
                source,
                f"QNN U16 tensor {tensor_name}",
                require_module_path=False,
            )
        if bindings != [source["decoder_binding"] for source in sources]:
            raise RuntimeError(
                f"QNN U16 index has inconsistent decoder bindings for {tensor_name}"
            )


def _validate_llama_quant_profile(profile: Dict[str, Any]) -> None:
    if profile.get("schema_version") != 2 or profile.get("format") != (
        "llama-qnn-quant-profile-v2"
    ):
        raise RuntimeError("unsupported llama QNN quantization profile schema")
    if profile.get("capabilities") != _LLAMA_QNN_QUANT_PROFILE_CAPABILITIES:
        raise RuntimeError("incomplete llama QNN quantization profile capabilities")
    tensors = profile.get("tensors")
    operations = profile.get("operations")
    logical_tensors = profile.get("logical_tensors")
    u16_tensor_index = profile.get("u16_tensor_index")
    if not isinstance(tensors, dict) or not isinstance(operations, list):
        raise RuntimeError("QNN tensor/operation profile fields have invalid types")
    if not isinstance(logical_tensors, dict):
        raise RuntimeError("QNN logical_tensors profile field must be a dictionary")
    if not isinstance(u16_tensor_index, dict):
        raise RuntimeError("QNN U16 tensor profile index must be a dictionary")
    unexpected_logical_tensors = set(logical_tensors).difference(tensors)
    if unexpected_logical_tensors:
        raise RuntimeError(
            "QNN logical source mappings reference absent tensors: "
            f"{sorted(unexpected_logical_tensors)}"
        )
    for tensor_name, source in logical_tensors.items():
        _validate_decoder_binding(
            source,
            f"logical QNN tensor {tensor_name}",
            require_module_path=False,
        )
    encoding_counts: Dict[str, int] = {}
    quantized_tensor_count = 0
    parameter_count = 0

    for tensor_name, tensor in tensors.items():
        quantized, encoding = _validate_profile_tensor(
            tensor_name, tensor, logical_tensors
        )
        encoding_counts[encoding] = encoding_counts.get(encoding, 0) + 1
        quantized_tensor_count += int(quantized)

    op_names = set()
    for operation in operations:
        op_name = operation.get("name")
        if not isinstance(op_name, str) or not op_name or op_name in op_names:
            raise RuntimeError(f"invalid or duplicate QNN operation name: {op_name}")
        op_names.add(op_name)
        inputs = operation.get("inputs", [])
        outputs = operation.get("outputs", [])
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            raise RuntimeError(f"invalid tensor list in QNN operation {op_name}")
        for tensor_name in [*inputs, *outputs]:
            if tensor_name not in tensors:
                raise RuntimeError(
                    f"QNN operation {op_name} references missing tensor {tensor_name}"
                )
        for field_name, tensor_names in (("input_sources", inputs), ("output_sources", outputs)):
            sources = operation.get(field_name)
            if not isinstance(sources, dict):
                raise RuntimeError(
                    f"QNN operation {op_name} lacks {field_name} tensor bindings"
                )
            unexpected_sources = set(sources).difference(tensor_names)
            if unexpected_sources:
                raise RuntimeError(
                    f"QNN operation {op_name} has invalid {field_name}: "
                    f"{sorted(unexpected_sources)}"
                )
            for tensor_name, source in sources.items():
                _validate_decoder_binding(
                    source,
                    f"QNN operation {op_name} tensor {tensor_name}",
                    require_module_path=False,
                )
        for param in operation.get("params", []):
            parameter_count += 1
            param_type = param.get("param_type")
            if param_type == "QNN_PARAMTYPE_SCALAR":
                payload = param.get("scalar", {})
                try:
                    raw = base64.b64decode(payload.get("value_le_base64", ""), validate=True)
                except ValueError as error:
                    raise RuntimeError(
                        f"invalid scalar parameter payload in QNN op {op_name}"
                    ) from error
                if len(raw) != int(payload.get("value_bytes", -1)):
                    raise RuntimeError(
                        f"scalar parameter length mismatch in QNN op {op_name}"
                    )
            elif param_type == "QNN_PARAMTYPE_UNSUPPORTED_SCALAR":
                payload = param.get("unsupported_scalar")
                if (
                    not isinstance(payload, dict)
                    or not isinstance(payload.get("data_type"), str)
                    or not payload["data_type"]
                    or not isinstance(payload.get("reason"), str)
                    or not payload["reason"]
                ):
                    raise RuntimeError(
                        f"invalid unsupported scalar parameter metadata in QNN op {op_name}"
                    )
            elif param_type == "QNN_PARAMTYPE_TENSOR":
                tensor = param.get("tensor", {})
                tensor_name = tensor.get("name", "<param>")
                if tensor_name not in tensors or tensors[tensor_name] != tensor:
                    raise RuntimeError(
                        f"QNN tensor parameter {tensor_name} is absent from the tensor ABI"
                    )
                try:
                    raw = base64.b64decode(
                        param.get("tensor_data_base64", ""), validate=True
                    )
                except ValueError as error:
                    raise RuntimeError(
                        f"invalid tensor parameter payload in QNN op {op_name}"
                    ) from error
                if len(raw) != int(param.get("tensor_data_bytes", -1)):
                    raise RuntimeError(
                        f"tensor parameter length mismatch in QNN op {op_name}"
                    )
            elif param_type == "QNN_PARAMTYPE_UNSUPPORTED_TENSOR":
                tensor = param.get("unsupported_tensor")
                if not isinstance(tensor, dict) or not tensor.get("unsupported_version"):
                    raise RuntimeError(
                        f"invalid unsupported QNN tensor parameter in op {op_name}"
                    )
            else:
                raise RuntimeError(
                    f"unhandled QNN parameter type {param_type} in op {op_name}"
                )

    _validate_u16_tensor_index(tensors, operations, u16_tensor_index)

    profile["coverage"] = {
        "tensor_count": len(tensors),
        "quantized_tensor_count": quantized_tensor_count,
        "unquantized_tensor_count": len(tensors) - quantized_tensor_count,
        "operation_count": len(operations),
        "parameter_count": parameter_count,
        "tensor_qparam_encodings": encoding_counts,
        "logical_tensor_count": len(logical_tensors),
        "logical_tensor_with_decoder_module_count": sum(
            bool(source["decoder_binding"]["module_paths"])
            for source in logical_tensors.values()
        ),
        "u16_tensor_count": len(u16_tensor_index),
        "u16_tensor_with_decoder_source_count": sum(
            bool(entry["sources"]) for entry in u16_tensor_index.values()
        ),
        "static_tensor_count": sum(
            tensor.get("tensor_type") == "QNN_TENSOR_TYPE_STATIC"
            for tensor in tensors.values()
        ),
        "embedded_static_tensor_count": sum(
            tensor.get("static_payload", {}).get("storage") == "embedded_exact_bytes"
            for tensor in tensors.values()
        ),
        "embedded_static_payload_bytes": sum(
            int(tensor.get("static_payload", {}).get("data_bytes", 0))
            for tensor in tensors.values()
        ),
        "blockwise_static_tensor_count": sum(
            tensor.get("static_payload", {}).get("storage")
            == "external_gptq_int2_source_reconstruction"
            for tensor in tensors.values()
        ),
    }


def _dump_llama_quant_profile(py_op_wrappers, graph_module, scope: str) -> None:
    if not _llama_quant_profile_enabled():
        return
    profile = _build_llama_quant_profile(py_op_wrappers, graph_module, scope)
    _validate_llama_quant_profile(profile)
    probe_destination = os.environ.get(_QNN_LLAMA_QUANT_PROFILE_PROBE_PATH_ENV)
    if probe_destination:
        destination = Path(probe_destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.name}.tmp")
        temporary.write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        logger.info(
            "Captured and validated llama QNN shard profile probe: scope=%s path=%s",
            scope,
            destination,
        )
        raise RuntimeError(
            f"QNN shard profile probe completed successfully: {destination}"
        )
    _llama_quant_profile_batches.append(profile)
    logger.info(
        "Captured llama QNN quantization profile: scope=%s tensors=%d operations=%d",
        scope,
        len(profile["tensors"]),
        len(profile["operations"]),
    )


def _load_qnn_tensor_qparams_override_shard(path: Path, scope: str) -> Dict[str, Any]:
    cache_key = str(path.resolve())
    payload = _qnn_tensor_qparams_override_cache.get(cache_key)
    if payload is None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _qnn_tensor_qparams_override_cache[cache_key] = payload

    candidates: List[Dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("scope"), str):
        candidates.append(payload)
    if isinstance(payload, dict):
        direct_profile = payload.get("llama_qnn_quant_profile")
        if isinstance(direct_profile, dict):
            candidates.extend(direct_profile.get("shards", []))
        for graph in payload.get("graphs", {}).values():
            if not isinstance(graph, dict):
                continue
            profile = graph.get("llama_qnn_quant_profile")
            if isinstance(profile, dict):
                candidates.extend(profile.get("shards", []))

    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("scope") == scope
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "QNN qparam override profile must contain exactly one shard for "
            f"scope {scope!r}; found {len(matches)} in {path}"
        )
    tensors = matches[0].get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise RuntimeError(f"QNN qparam override shard {scope} has no tensor ABI")
    return matches[0]


def _apply_qnn_tensor_qparams_override(py_op_wrappers, scope: str) -> None:
    override_path = os.environ.get(_QNN_TENSOR_QPARAMS_OVERRIDE_PATH_ENV)
    if not override_path:
        return

    path = Path(override_path)
    if not path.is_file():
        raise RuntimeError(f"QNN qparam override profile does not exist: {path}")
    shard = _load_qnn_tensor_qparams_override_shard(path, scope)
    expected_tensors = shard["tensors"]
    preserve_static_s16 = _environment_flag_enabled(_QNN_PRESERVE_STATIC_S16_ENV)
    preserved_names = {
        name
        for name, tensor in expected_tensors.items()
        if preserve_static_s16
        and tensor.get("data_type") == "QNN_DATATYPE_SFIXED_POINT_16"
        and tensor.get("static_payload", {}).get("storage")
        == "embedded_exact_bytes"
    }

    wrappers_by_name: Dict[str, Dict[int, Any]] = defaultdict(dict)
    for py_op_wrapper in py_op_wrappers:
        op_wrapper = py_op_wrapper.GetOpWrapper()
        for tensor_wrapper in [
            *op_wrapper.GetInputTensors(),
            *op_wrapper.GetOutputTensors(),
        ]:
            wrappers_by_name[tensor_wrapper.GetName()][id(tensor_wrapper)] = tensor_wrapper

    missing = set(expected_tensors).difference(wrappers_by_name)
    if missing:
        raise RuntimeError(
            f"QNN qparam override tensor ABI is missing {len(missing)} tensors in "
            f"{scope}: {sorted(missing)[:8]}"
        )

    expected_scale_offset_names = {
        name
        for name, tensor in expected_tensors.items()
        if tensor.get("quantization_encoding")
        == "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET"
    }
    actual_scale_offset_names = {
        name
        for name, wrappers in wrappers_by_name.items()
        if any(
            _enum_name(wrapper.GetQuantizationEncoding())
            == "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET"
            for wrapper in wrappers.values()
        )
    }
    if actual_scale_offset_names != expected_scale_offset_names:
        raise RuntimeError(
            "QNN SCALE_OFFSET tensor ABI differs from the override profile: "
            f"missing={sorted(expected_scale_offset_names - actual_scale_offset_names)[:8]} "
            f"unexpected={sorted(actual_scale_offset_names - expected_scale_offset_names)[:8]}"
        )

    static_payload_count = 0
    static_payload_bytes = 0
    wrapper_update_count = 0
    for name, expected in expected_tensors.items():
        expected_dims = [int(value) for value in expected.get("dimensions", [])]
        expected_data_type = expected.get("data_type")
        expected_encoding = expected.get("quantization_encoding")
        scale_offset = expected.get("scale_offset")
        static_payload = expected.get("static_payload")

        if expected_encoding == "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET":
            if not isinstance(scale_offset, dict):
                raise RuntimeError(f"QNN override tensor {name} lacks scale/offset metadata")
            _validate_scale_offset_payload(scale_offset, f"override tensor {name}")
            scale = struct.unpack(
                "<f", bytes.fromhex(str(scale_offset["scale_f32_le_hex"]))
            )[0]
            offset = int(scale_offset["offset"])
        else:
            scale = None
            offset = None

        replacement_bytes = None
        if isinstance(static_payload, dict) and name not in preserved_names:
            storage = static_payload.get("storage")
            if storage == "embedded_exact_bytes":
                replacement_bytes = base64.b64decode(
                    static_payload.get("data_le_base64", ""), validate=True
                )
                if len(replacement_bytes) != int(static_payload.get("data_bytes", -1)):
                    raise RuntimeError(
                        f"QNN override static tensor {name} has an invalid byte count"
                    )
                if hashlib.sha256(replacement_bytes).hexdigest() != static_payload.get(
                    "sha256"
                ):
                    raise RuntimeError(
                        f"QNN override static tensor {name} has an invalid digest"
                    )
            elif storage != "external_gptq_int2_source_reconstruction":
                raise RuntimeError(
                    f"QNN override tensor {name} has unsupported static storage {storage}"
                )

        for wrapper in wrappers_by_name[name].values():
            actual_dims = [int(value) for value in wrapper.GetDims()]
            actual_data_type = _enum_name(wrapper.GetDataType())
            actual_encoding = _enum_name(wrapper.GetQuantizationEncoding())
            if (
                actual_dims != expected_dims
                or actual_data_type != expected_data_type
                or actual_encoding != expected_encoding
            ):
                raise RuntimeError(
                    f"QNN override ABI mismatch for {name}: "
                    f"expected=({expected_dims},{expected_data_type},{expected_encoding}) "
                    f"actual=({actual_dims},{actual_data_type},{actual_encoding})"
                )
            if name in preserved_names:
                if not wrapper.IsTensorStatic():
                    raise RuntimeError(
                        f"preserved QNN S16 tensor {name} is not static in the graph"
                    )
                current_bytes = bytes(wrapper.GetStaticDataBytes())
                if len(current_bytes) % 2:
                    raise RuntimeError(
                        f"preserved QNN S16 tensor {name} has an odd byte count"
                    )
                current_values = struct.unpack(
                    f"<{len(current_bytes) // 2}h", current_bytes
                )
                if current_values and (
                    min(current_values) < -0x8000
                    or max(current_values) > 0x7F7F
                ):
                    raise RuntimeError(
                        f"preserved QNN S16 tensor {name} exceeds the HTP range "
                        "[-0x8000, 0x7F7F]"
                    )
                continue
            if scale is not None:
                wrapper.SetScaleOffsetQuantizeParams(scale, offset)
                wrapper_update_count += 1
            if replacement_bytes is not None:
                if not wrapper.IsTensorStatic():
                    raise RuntimeError(
                        f"QNN override payload tensor {name} is not static in the graph"
                    )
                wrapper.SetStaticDataBytes(replacement_bytes)
                if bytes(wrapper.GetStaticDataBytes()) != replacement_bytes:
                    raise RuntimeError(
                        f"QNN override static tensor {name} failed byte verification"
                    )

        if replacement_bytes is not None:
            static_payload_count += 1
            static_payload_bytes += len(replacement_bytes)

    observed: Dict[str, Dict[str, Any]] = {}
    for py_op_wrapper in py_op_wrappers:
        op_config = py_op_wrapper.GetOpWrapper().GetOpConfig()
        for tensor in [*op_config["inputTensors"], *op_config["outputTensors"]]:
            serialized = _serialize_qnn_tensor(tensor)
            if serialized.get("name") in expected_scale_offset_names:
                _register_profile_tensor(observed, serialized)

    if set(observed) != expected_scale_offset_names:
        raise RuntimeError(
            "QNN qparam override verification did not observe the complete tensor ABI"
        )
    for name in expected_scale_offset_names:
        if name in preserved_names:
            continue
        expected = expected_tensors[name]
        actual = observed[name]
        if actual.get("scale_offset") != expected.get("scale_offset"):
            raise RuntimeError(
                f"QNN qparam override did not reproduce exact f32 bits for {name}: "
                f"expected={expected.get('scale_offset')} actual={actual.get('scale_offset')}"
            )

    logger.info(
        "Applied and verified QNN qparam override: scope=%s tensors=%d "
        "wrapper_updates=%d static_payloads=%d static_bytes=%d "
        "preserved_static_s16=%d profile=%s",
        scope,
        len(expected_scale_offset_names),
        wrapper_update_count,
        static_payload_count,
        static_payload_bytes,
        len(preserved_names),
        path,
    )


def _dump_qnn_tensor_qparams(py_op_wrappers, scope: str) -> None:
    """Optionally persist compact QNN compile metadata for ETDump analysis."""
    destination = os.environ.get(_QNN_TENSOR_QPARAMS_PATH_ENV)
    if not destination:
        return

    tensors = {}
    operations = []
    for py_op_wrapper in py_op_wrappers:
        op_config = py_op_wrapper.GetOpWrapper().GetOpConfig()
        inputs = [_serialize_qnn_tensor(tensor) for tensor in op_config["inputTensors"]]
        outputs = [_serialize_qnn_tensor(tensor) for tensor in op_config["outputTensors"]]
        for tensor in [*inputs, *outputs]:
            if "name" in tensor:
                tensors.setdefault(tensor["name"], tensor)
        operations.append(
            {
                "name": op_config["name"],
                "package_name": op_config["packageName"],
                "type_name": op_config["typeName"],
                "inputs": [tensor.get("name") for tensor in inputs],
                "outputs": [tensor.get("name") for tensor in outputs],
            }
        )

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema_version": 1, "batches": []}
    if path.exists():
        try:
            document = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("Ignoring invalid existing QNN tensor metadata: %s", path)

    document.setdefault("schema_version", 1)
    document.setdefault("batches", []).append(
        {"scope": scope, "tensors": tensors, "operations": operations}
    )
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    logger.info(
        "Wrote QNN tensor quantization metadata: path=%s scope=%s tensors=%d",
        path,
        scope,
        len(tensors),
    )


@final
class QnnBackend(BackendDetails):
    @staticmethod
    def _build_op_wrappers(
        edge_program: ExportedProgram,
        enable_tensor_dump: bool,
        op_package_infos: List[QnnExecuTorchOpPackageInfo],
        use_mha2sha: bool,
    ):
        for node in edge_program.graph_module.graph.nodes:
            if hasattr(node, "meta"):
                # pop certain keys in meta for not affecting the passes in compilation
                node.meta.pop(QCOM_AXIS_ORDER, "")
        # QNN Delegate Specific Passes
        graph_module = QnnPassManager().transform_for_preprocess_pipeline(
            edge_program, use_mha2sha=use_mha2sha
        )
        assert graph_module is not None

        nodes_to_wrappers = defaultdict(dict)
        node_visitors = get_node_visitors(
            edge_program,
            enable_tensor_dump=enable_tensor_dump,
            op_package_infos=op_package_infos,
        )
        py_op_wrapper_list = []
        for node in graph_module.graph.nodes:
            if node.op == "call_function":
                logger.info(f"Visiting: {node}, {node.target.__name__}")
                if node.target.__name__ in node_visitors:
                    py_op_wrapper = node_visitors[node.target.__name__].define_node(
                        node, nodes_to_wrappers
                    )
                    if py_op_wrapper is not None:
                        if isinstance(py_op_wrapper, List):
                            py_op_wrapper_list.extend(py_op_wrapper)
                        else:
                            py_op_wrapper_list.append(py_op_wrapper)
                else:
                    err_msg = (
                        f"For {node}, {node.op}:{node.target.__name__} "
                        "is not supported in Qnn Delegate"
                    )
                    try:
                        context_loader_target = eval(
                            f"torch.ops.{OpContextLoader.namespace}.{node.target.__name__}",
                            globals().update(torch.__dict__),
                        )
                        assert node.target == context_loader_target, err_msg
                        # if graph has context binary loader node, return directly
                        return node.meta[OpContextLoader.meta_ctx_bin]
                    except:
                        raise RuntimeError(err_msg)

            elif node.op in [
                "get_attr",
                "placeholder",
                "output",
            ]:
                continue
            else:
                raise RuntimeError(f"{node.op} is not supported in Qnn")

        return py_op_wrapper_list, graph_module

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        option = generate_qnn_executorch_option(compile_specs)
        obj_options = flatbuffer_to_option(option)
        qnn_manager = get_current_qnn_manager(
            obj_options.backend_options.backend_type, compile_specs
        )
        qnn_manager.InitContext([DEFAULT_GRAPH_NAME])
        build_result = QnnBackend._build_op_wrappers(
            edge_program,
            qnn_manager.IsTensorDump(),
            obj_options.op_package_options.op_package_infos,
            obj_options.use_mha2sha,
        )
        if isinstance(build_result, bytes):
            qnn_manager.DestroyContext()
            return PreprocessResult(processed_bytes=build_result, debug_handle_map={})
        py_op_wrapper_list, profile_graph_module = build_result
        _apply_qnn_tensor_qparams_override(py_op_wrapper_list, "preprocess")
        _dump_qnn_tensor_qparams(py_op_wrapper_list, "preprocess")
        _dump_llama_quant_profile(
            py_op_wrapper_list, profile_graph_module, "preprocess"
        )

        qnn_context_binary = qnn_manager.Compile(
            qnn_manager.GetGraphNames(),
            [[py_op_wrapper.GetOpWrapper() for py_op_wrapper in py_op_wrapper_list]],
        )

        if obj_options.saver:
            exit(
                f"Record all QNN API calls from saver backend at: {obj_options.saver_output_dir}"
            )
        assert len(qnn_context_binary) != 0, "Failed to generate Qnn context binary."
        qnn_manager.DestroyContext()
        # For now, debug_handle_map is not used by QNN ExecuTorch
        return PreprocessResult(
            processed_bytes=bytes(qnn_context_binary),
            debug_handle_map={},
        )

    @staticmethod
    def preprocess_multimethod(  # noqa: C901
        edge_programs: Dict[str, List[ExportedProgram]],
        compile_specs: Dict[str, List[List[CompileSpec]]],
    ) -> PreprocessResult:
        # TODO: refactor QnnManager to consume multiple compile_spec
        # take first compile_specs here for the same partitions
        graph_names = list(edge_programs.keys())
        compile_spec = list(compile_specs.values())[0][0]
        option = flatbuffer_to_option(compile_spec[0].value)
        # check if each graph has equal number of partitions
        num_sub_graphs = set()
        for edge_program in edge_programs.values():
            num_sub_graphs.add(len(edge_program))
        # this constraint is dedicated to weight-sharing scenario
        assert (
            len(num_sub_graphs) == 1
        ), "Only graphs with the same number of partitions could be used"

        all_processed_results = {key: [] for key in edge_programs.keys()}
        num_sub_graphs = next(iter(num_sub_graphs))
        qnn_manager = get_current_qnn_manager(
            option.backend_options.backend_type, compile_spec
        )
        diagnostic_shard_limit = 0
        diagnostic_shard_limit_value = os.environ.get(
            _QNN_DIAGNOSTIC_SHARD_LIMIT_ENV, ""
        )
        if diagnostic_shard_limit_value:
            try:
                diagnostic_shard_limit = int(diagnostic_shard_limit_value)
            except ValueError as error:
                raise RuntimeError(
                    f"{_QNN_DIAGNOSTIC_SHARD_LIMIT_ENV} must be a positive integer"
                ) from error
            if diagnostic_shard_limit <= 0:
                raise RuntimeError(
                    f"{_QNN_DIAGNOSTIC_SHARD_LIMIT_ENV} must be a positive integer"
                )
            diagnostic_shard_limit = min(
                diagnostic_shard_limit, num_sub_graphs
            )
            logger.warning(
                "QNN diagnostic export will compile only the first %d/%d shard(s)",
                diagnostic_shard_limit,
                num_sub_graphs,
            )

        compile_sub_graphs = diagnostic_shard_limit or num_sub_graphs
        for i in range(compile_sub_graphs):
            # e.g. 2 methods (x, y) with 3 subgraphs(partitions)
            #      > context_binary_0: [x.subgraph_0, y.subgraph_0]
            #      > context_binary_1: [x.subgraph_1, y.subgraph_1]
            #      > context_binary_2: [x.subgraph_2, y.subgraph_2]
            # Each context binary owns its delegate mapping.  Sharded exports
            # have independently captured FX graphs, so their node names and
            # debug handles may restart at the same values.
            debug_handle_builder = DelegateMappingBuilder(generated_identifiers=False)
            qnn_manager.InitContext(graph_names)
            py_op_wrapper_list, ctx_binary_list = [], []
            for j, (graph_name, programs) in enumerate(edge_programs.items()):
                logger.info(f"Processing Method({j}): ({i+1}/{num_sub_graphs})")
                build_result = QnnBackend._build_op_wrappers(
                    programs[i],
                    qnn_manager.IsTensorDump(),
                    option.op_package_options.op_package_infos,
                    option.use_mha2sha,
                )
                if qnn_manager.IsTensorDump():
                    for node in programs[i].graph.nodes:
                        if handle_id := node.meta.get(DEBUG_HANDLE_KEY):
                            debug_handle_builder.insert_delegate_mapping_entry(
                                handles=handle_id,
                                identifier=f"{graph_name}.context_{i}.{node.name}",
                            )
                if isinstance(build_result, bytes):
                    ctx_binary_list.append(build_result)
                else:
                    py_op_wrappers, profile_graph_module = build_result
                    scope = f"{graph_name}.context_{i}"
                    _apply_qnn_tensor_qparams_override(py_op_wrappers, scope)
                    _dump_qnn_tensor_qparams(py_op_wrappers, scope)
                    _dump_llama_quant_profile(
                        py_op_wrappers, profile_graph_module, scope
                    )
                    py_op_wrapper_list.append(
                        [
                            py_op_wrapper.GetOpWrapper()
                            for py_op_wrapper in py_op_wrappers
                        ]
                    )

            if len(py_op_wrapper_list) == len(edge_programs.values()):
                qnn_context_binary = qnn_manager.Compile(
                    graph_names, py_op_wrapper_list
                )
                if option.saver:
                    # TODO: Currently, only the first method is saved. Update this logic if saving multiple methods becomes necessary in the future.
                    exit(
                        f"Record all QNN API calls from saver backend at: {option.saver_output_dir}"
                    )
                assert (
                    len(qnn_context_binary) != 0
                ), "Failed to generate Qnn context binary."
                qnn_manager.DestroyContext()
                # methods should share the same context binary for current partition
                for key in edge_programs.keys():
                    all_processed_results[key].append(
                        PreprocessResult(
                            processed_bytes=bytes(qnn_context_binary),
                            debug_handle_map=debug_handle_builder.get_delegate_mapping(),
                        )
                    )
            elif len(ctx_binary_list) == len(edge_programs.values()):
                for i, key in enumerate(edge_programs.keys()):
                    all_processed_results[key].append(
                        PreprocessResult(
                            processed_bytes=ctx_binary_list[i],
                            debug_handle_map=debug_handle_builder.get_delegate_mapping(),
                        )
                    )
            else:
                raise RuntimeError("Hybrid compilation is not supported")

        if diagnostic_shard_limit:
            # Backend lowering requires one result for every partition in the
            # tagged graph. Only the first results are real QNN contexts; the
            # remaining opaque delegates exist solely to satisfy the exported
            # program verifier and are never serialized by the diagnostic
            # shard exporter.
            for key, results in all_processed_results.items():
                for shard_index in range(len(results), num_sub_graphs):
                    results.append(
                        PreprocessResult(
                            processed_bytes=(
                                "ET_QNN_DIAGNOSTIC_UNCOMPILED_SHARD_"
                                f"{shard_index}"
                            ).encode("ascii"),
                            debug_handle_map={},
                        )
                    )
            logger.warning(
                "Inserted non-serializable diagnostic placeholders for %d "
                "uncompiled QNN shard(s)",
                num_sub_graphs - compile_sub_graphs,
            )

        return all_processed_results
