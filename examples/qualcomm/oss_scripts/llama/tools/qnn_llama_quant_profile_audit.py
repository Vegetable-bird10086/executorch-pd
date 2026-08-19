#!/usr/bin/env python3
"""Audit the QNN profile required to reproduce prefill arithmetic in llama.cpp.

This tool deliberately does not export, rebuild, or modify a PTE.  It verifies
that a v2 profile contains the bytes and qparams that a decoder implementation
needs before attempting numerical tensor comparisons on device.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROFILE_FORMAT = "llama-qnn-quant-profile-v2"
PROFILE_CAPABILITIES = {
    "exact_f32_scale_bits": True,
    "raw_operator_parameters": True,
    "operation_tensor_sources": True,
    "embedded_static_affine_tensor_bytes": True,
    "blockwise_weight_scale_payload": True,
    "blockwise_weight_payload_digest": True,
    "structured_decoder_tensor_bindings": True,
    "complete_u16_tensor_qparams": True,
}
DATA_TYPE_ELEMENT_BYTES = {
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


class AuditError(ValueError):
    pass


def _validate_decoder_binding(
    source: Any, location: str, *, require_module_path: bool
) -> None:
    if not isinstance(source, dict):
        raise AuditError(f"{location} has no decoder source binding")
    if not isinstance(source.get("fx_node_name"), str) or not source["fx_node_name"]:
        raise AuditError(f"{location} has no FX node name")
    binding = source.get("decoder_binding")
    if not isinstance(binding, dict):
        raise AuditError(f"{location} lacks structured decoder binding")
    module_paths = binding.get("module_paths")
    layer_ids = binding.get("layer_ids")
    projection = binding.get("projection")
    if not isinstance(module_paths, list) or any(
        not isinstance(path, str) or not path for path in module_paths
    ):
        raise AuditError(f"{location} has invalid decoder module paths")
    if require_module_path and not module_paths:
        raise AuditError(f"{location} has no decoder module path")
    if not isinstance(layer_ids, list) or any(
        not isinstance(layer_id, int) or layer_id < 0 for layer_id in layer_ids
    ):
        raise AuditError(f"{location} has invalid decoder layer IDs")
    if projection is not None and projection not in {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }:
        raise AuditError(f"{location} has an invalid decoder projection")


def _decode_base64(value: Any, location: str) -> bytes:
    if not isinstance(value, str):
        raise AuditError(f"{location} is not a base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as error:
        raise AuditError(f"{location} is not valid base64") from error


def _validate_scale_offset(value: Any, location: str) -> None:
    if not isinstance(value, dict):
        raise AuditError(f"{location} is missing")
    required = {"scale", "scale_f32_le_hex", "offset", "zero_point"}
    missing = required.difference(value)
    if missing:
        raise AuditError(f"{location} lacks {sorted(missing)}")
    try:
        raw_scale = bytes.fromhex(str(value["scale_f32_le_hex"]))
    except ValueError as error:
        raise AuditError(f"{location} has an invalid scale byte string") from error
    if len(raw_scale) != 4:
        raise AuditError(f"{location} must retain an IEEE-754 f32 scale")
    if float(value["scale"]) != struct.unpack("<f", raw_scale)[0]:
        raise AuditError(f"{location} lost exact f32 scale bits")
    if int(value["zero_point"]) != -int(value["offset"]):
        raise AuditError(f"{location} has inconsistent QNN offset/zero point")


def _validate_static_payload(tensor: Dict[str, Any], location: str) -> str | None:
    if tensor.get("tensor_type") != "QNN_TENSOR_TYPE_STATIC":
        if "static_payload" in tensor:
            raise AuditError(f"{location} is non-static but has a static payload")
        return None

    payload = tensor.get("static_payload")
    if not isinstance(payload, dict):
        raise AuditError(f"{location} lacks a static payload contract")
    encoding = tensor.get("quantization_encoding")
    if encoding == "QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION":
        if payload.get("storage") != "external_gptq_int2_source_reconstruction":
            raise AuditError(f"{location} has no GPTQ source reconstruction contract")
        digest = payload.get("qnn_payload_sha256")
        if int(payload.get("qnn_payload_bytes", 0)) <= 0:
            raise AuditError(f"{location} has no QNN static-weight byte count")
        if not isinstance(digest, str) or len(digest) != 64:
            raise AuditError(f"{location} has no QNN static-weight digest")
        try:
            int(digest, 16)
        except ValueError as error:
            raise AuditError(f"{location} has a non-hex QNN static-weight digest") from error
        return "gptq_source_reconstruction"

    if payload.get("storage") != "embedded_exact_bytes":
        raise AuditError(f"{location} must embed its exact static bytes")
    raw = _decode_base64(payload.get("data_le_base64"), f"{location}.data_le_base64")
    dimensions = tensor.get("dimensions")
    if not isinstance(dimensions, list) or any(int(dimension) <= 0 for dimension in dimensions):
        raise AuditError(f"{location} has invalid dimensions")
    data_type = tensor.get("data_type")
    expected_element_bytes = DATA_TYPE_ELEMENT_BYTES.get(data_type)
    if expected_element_bytes is None:
        raise AuditError(f"{location} uses unsupported static datatype {data_type}")
    if int(payload.get("element_bytes", -1)) != expected_element_bytes:
        raise AuditError(f"{location} has an invalid static element size")
    expected_bytes = expected_element_bytes
    for dimension in dimensions:
        expected_bytes *= int(dimension)
    if len(raw) != expected_bytes or int(payload.get("data_bytes", -1)) != expected_bytes:
        raise AuditError(f"{location} static byte count does not match its shape")
    if payload.get("sha256") != hashlib.sha256(raw).hexdigest():
        raise AuditError(f"{location} static byte digest does not match")
    return "embedded_exact_bytes"


def _validate_blockwise(tensor: Dict[str, Any], location: str) -> None:
    payload = tensor.get("blockwise_expansion")
    if not isinstance(payload, dict):
        raise AuditError(f"{location} lacks blockwise metadata")
    dimensions = tensor.get("dimensions")
    axis = int(payload.get("axis", -1))
    if not isinstance(dimensions, list) or axis < 0 or axis >= len(dimensions):
        raise AuditError(f"{location} has an invalid blockwise axis")
    num_scale_offsets = int(payload.get("num_scale_offsets", -1))
    num_blocks_per_axis = int(payload.get("num_blocks_per_axis", -1))
    if num_scale_offsets != int(dimensions[axis]) or num_blocks_per_axis <= 0:
        raise AuditError(f"{location} has invalid blockwise dimensions")
    scales = payload.get("scale_offsets")
    if not isinstance(scales, list) or len(scales) != num_scale_offsets:
        raise AuditError(f"{location} has incomplete blockwise scale offsets")
    for index, scale in enumerate(scales):
        _validate_scale_offset(scale, f"{location}.scale_offsets[{index}]")
    storage_type = payload.get("block_scale_storage_type")
    expected_element_bytes = {
        "QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8": 1,
        "QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_16": 2,
    }.get(storage_type)
    if expected_element_bytes is None:
        raise AuditError(f"{location} has unsupported block scale storage")
    if int(payload.get("block_scale_element_bytes", -1)) != expected_element_bytes:
        raise AuditError(f"{location} has invalid block scale element bytes")
    raw_codes = _decode_base64(
        payload.get("block_scales_base64"), f"{location}.block_scales_base64"
    )
    expected_bytes = num_scale_offsets * num_blocks_per_axis * expected_element_bytes
    if len(raw_codes) != expected_bytes or int(payload.get("block_scales_bytes", -1)) != expected_bytes:
        raise AuditError(f"{location} has incomplete block scale codes")
    required = {
        "block_scale_layout": "scale_offset_index_major_then_block_index",
        "block_scale_code_storage": "one_unsigned_code_per_storage_element",
        "block_scale_effective_scale_formula": (
            "effective_scale=scale_offsets[scale_offset_index].scale*block_scale_code"
        ),
        "block_scale_storage_byte_order": (
            "little_endian" if expected_element_bytes == 2 else "not_applicable"
        ),
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise AuditError(f"{location} has an incompatible {field}")


def _validate_tensor(tensor_name: str, tensor: Any) -> Tuple[str, str | None]:
    location = f"tensor {tensor_name}"
    if not isinstance(tensor, dict) or tensor.get("name") != tensor_name:
        raise AuditError(f"{location} is malformed")
    encoding = tensor.get("quantization_encoding")
    if not isinstance(encoding, str):
        raise AuditError(f"{location} lacks a quantization encoding")
    static_storage = _validate_static_payload(tensor, location)
    if encoding == "QNN_QUANTIZATION_ENCODING_UNDEFINED":
        return encoding, static_storage
    if encoding == "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET":
        _validate_scale_offset(tensor.get("scale_offset"), location)
    elif encoding == "QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET":
        axis_payload = tensor.get("axis_scale_offset")
        if not isinstance(axis_payload, dict):
            raise AuditError(f"{location} lacks axis qparams")
        scales = axis_payload.get("scale_offsets")
        if not isinstance(scales, list) or not scales:
            raise AuditError(f"{location} has no axis scale offsets")
        for index, scale in enumerate(scales):
            _validate_scale_offset(scale, f"{location}.axis_scale_offsets[{index}]")
    elif encoding == "QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION":
        _validate_blockwise(tensor, location)
    else:
        raise AuditError(f"{location} has unsupported qparam encoding {encoding}")
    return encoding, static_storage


def _validate_params(operation: Dict[str, Any], location: str) -> int:
    params = operation.get("params")
    if not isinstance(params, list):
        raise AuditError(f"{location} parameters are not a list")
    for param_index, param in enumerate(params):
        param_location = f"{location}.params[{param_index}]"
        if not isinstance(param, dict):
            raise AuditError(f"{param_location} is malformed")
        param_type = param.get("param_type")
        if param_type == "QNN_PARAMTYPE_SCALAR":
            scalar = param.get("scalar")
            if not isinstance(scalar, dict):
                raise AuditError(f"{param_location} lacks scalar bytes")
            raw = _decode_base64(scalar.get("value_le_base64"), param_location)
            if len(raw) != int(scalar.get("value_bytes", -1)):
                raise AuditError(f"{param_location} scalar byte count does not match")
        elif param_type == "QNN_PARAMTYPE_UNSUPPORTED_SCALAR":
            scalar = param.get("unsupported_scalar")
            if (
                not isinstance(scalar, dict)
                or not isinstance(scalar.get("data_type"), str)
                or not scalar["data_type"]
                or not isinstance(scalar.get("reason"), str)
                or not scalar["reason"]
            ):
                raise AuditError(f"{param_location} has invalid unsupported scalar metadata")
        elif param_type == "QNN_PARAMTYPE_TENSOR":
            parameter_tensor = param.get("tensor")
            if not isinstance(parameter_tensor, dict):
                raise AuditError(f"{param_location} lacks tensor metadata")
            tensor_name = parameter_tensor.get("name")
            if not isinstance(tensor_name, str) or not tensor_name:
                raise AuditError(f"{param_location} tensor has no name")
            _validate_tensor(tensor_name, parameter_tensor)
            raw = _decode_base64(param.get("tensor_data_base64"), param_location)
            if len(raw) != int(param.get("tensor_data_bytes", -1)):
                raise AuditError(f"{param_location} tensor byte count does not match")
        elif param_type == "QNN_PARAMTYPE_UNSUPPORTED_TENSOR":
            tensor = param.get("unsupported_tensor")
            if not isinstance(tensor, dict) or not tensor.get("unsupported_version"):
                raise AuditError(f"{param_location} has invalid unsupported tensor metadata")
        else:
            raise AuditError(f"{param_location} has unsupported QNN parameter type")
    return len(params)


def _validate_recipe(recipe: Any) -> int:
    if not isinstance(recipe, dict):
        raise AuditError("profile lacks a GPTQ source recipe")
    if recipe.get("schema_version") != 1 or recipe.get("format") != "llama-gptq-source-recipe-v1":
        raise AuditError("profile has an unsupported GPTQ source recipe")
    source_bits = int(recipe.get("source_weight_bits", 0))
    group_size = int(recipe.get("group_size", 0))
    if source_bits != 2 or group_size <= 0 or group_size % 32 != 0:
        raise AuditError("profile source recipe is not GS32-aligned GPTQ INT2")
    if recipe.get("gguf_weight_type") != f"GPTQ2_{group_size}":
        raise AuditError("profile GGUF type does not match the declared group size")
    if recipe.get("qweight_mode") not in {"qweight_minus_qzeros", "qweight"}:
        raise AuditError("profile has an unsupported GPTQ qweight mode")
    contract = recipe.get("qnn_weight_code_contract")
    expected = {
        "schema_version": 1,
        "source_group_size": group_size,
        "source_group_code_bytes": group_size // 4,
        "source_metadata_bytes": 4,
        "source_code_packing": "four_int2_codes_per_byte_lsb_first",
        "source_group_metadata": "fp16_le_scale_then_fp16_le_zero_bias",
        "source_zero_point_formula": "clamp(round(zero_bias/max(scale,0.0001)),0,3)",
        "source_code_bits": 2,
        "qnn_code_bits": 4,
        "qnn_code_formula": "qnn_signed_code=source_code-source_zero_point",
        "qnn_code_storage": "two_complement_int4",
        "qnn_pte_layout": "gs32_64_rows",
        "decode_reconstruction": (
            "expand_gptq2_codes_in_registers_then_apply_qnn_blockwise_scale"
        ),
    }
    if contract != expected:
        raise AuditError("profile has an incompatible GPTQ-to-QNN reconstruction contract")
    return group_size


def audit_profile(profile: Any, selected_shards: Iterable[int] | None = None) -> Dict[str, Any]:
    if not isinstance(profile, dict):
        raise AuditError("llama_qnn_quant_profile is not an object")
    if profile.get("schema_version") != 2 or profile.get("format") != PROFILE_FORMAT:
        raise AuditError("profile is not llama-qnn-quant-profile-v2")
    if profile.get("capabilities") != PROFILE_CAPABILITIES:
        raise AuditError("profile lacks required exact-decode capabilities")
    group_size = _validate_recipe(profile.get("gptq_source_recipe"))
    shards = profile.get("shards")
    if not isinstance(shards, list) or not shards:
        raise AuditError("profile has no shards")
    selected = set(range(len(shards))) if selected_shards is None else set(selected_shards)
    if not selected.issubset(range(len(shards))):
        raise AuditError(f"requested shard outside profile range: {sorted(selected)}")

    report_shards: List[Dict[str, Any]] = []
    total_op_types: Counter[str] = Counter()
    total_static_storage: Counter[str] = Counter()
    total_encodings: Counter[str] = Counter()
    total_params = 0
    total_unquantized_uses = 0

    for shard_index in sorted(selected):
        shard = shards[shard_index]
        if not isinstance(shard, dict):
            raise AuditError(f"shard {shard_index} is not an object")
        if shard.get("capabilities") != PROFILE_CAPABILITIES:
            raise AuditError(f"shard {shard_index} lacks required exact-decode capabilities")
        tensors = shard.get("tensors")
        operations = shard.get("operations")
        logical_tensors = shard.get("logical_tensors")
        u16_tensor_index = shard.get("u16_tensor_index")
        if (
            not isinstance(tensors, dict)
            or not isinstance(operations, list)
            or not isinstance(logical_tensors, dict)
            or not isinstance(u16_tensor_index, dict)
        ):
            raise AuditError(f"shard {shard_index} has invalid tensor or operation lists")
        indexed_data_types = {
            entry.get("data_type")
            for entry in u16_tensor_index.values()
            if isinstance(entry, dict)
        }
        if indexed_data_types not in (
            {"QNN_DATATYPE_UFIXED_POINT_8"},
            {"QNN_DATATYPE_UFIXED_POINT_16"},
        ):
            raise AuditError(
                f"shard {shard_index} has an invalid primary activation index: "
                f"{sorted(str(value) for value in indexed_data_types)}"
            )
        activation_data_type = next(iter(indexed_data_types))
        activation_bits = DATA_TYPE_ELEMENT_BYTES[activation_data_type] * 8
        residual_u16_names = {
            tensor_name
            for tensor_name, tensor in tensors.items()
            if isinstance(tensor, dict)
            and tensor.get("data_type") == "QNN_DATATYPE_UFIXED_POINT_16"
        }
        if activation_bits == 8 and residual_u16_names:
            preview = sorted(residual_u16_names)[:8]
            raise AuditError(
                f"shard {shard_index} declares an A8 ABI but retains "
                f"{len(residual_u16_names)} UFIXED16 tensor(s), e.g. {preview}"
            )
        tensor_encodings: Counter[str] = Counter()
        static_storage: Counter[str] = Counter()
        for tensor_name, tensor in tensors.items():
            encoding, storage = _validate_tensor(tensor_name, tensor)
            tensor_encodings[encoding] += 1
            if storage is not None:
                static_storage[storage] += 1
            if (
                tensor.get("tensor_type") == "QNN_TENSOR_TYPE_STATIC"
                and encoding == "QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION"
            ):
                source = logical_tensors.get(tensor_name)
                # QNN may introduce blockwise helper constants with no decoder
                # source.  They are not llama.cpp bindings; validate only the
                # subset that actually maps to a decoder tensor.
                if (
                    isinstance(source, dict)
                    and source.get("decoder_binding", {}).get("module_paths")
                ):
                    _validate_decoder_binding(
                        source,
                        f"shard {shard_index} static tensor {tensor_name}",
                        require_module_path=True,
                    )
        if not set(logical_tensors).issubset(tensors):
            raise AuditError(f"shard {shard_index} has logical tensors outside QNN ABI")
        for tensor_name, source in logical_tensors.items():
            _validate_decoder_binding(
                source,
                f"shard {shard_index} logical tensor {tensor_name}",
                require_module_path=False,
            )

        op_types: Counter[str] = Counter()
        params = 0
        unquantized_uses = 0
        expected_u16_uses: Dict[str, set[Tuple[str, str, int]]] = {}
        for operation_index, operation in enumerate(operations):
            location = f"shard {shard_index}.operations[{operation_index}]"
            if not isinstance(operation, dict):
                raise AuditError(f"{location} is malformed")
            name = operation.get("name")
            type_name = operation.get("type_name")
            inputs = operation.get("inputs")
            outputs = operation.get("outputs")
            if not isinstance(name, str) or not isinstance(type_name, str):
                raise AuditError(f"{location} lacks a QNN operation identity")
            if not isinstance(inputs, list) or not isinstance(outputs, list) or not outputs:
                raise AuditError(f"{location} has invalid QNN input/output ABI")
            for direction, tensor_names in (("inputs", inputs), ("outputs", outputs)):
                bindings = operation.get(f"{direction[:-1]}_sources")
                if not isinstance(bindings, dict):
                    raise AuditError(f"{location} lacks {direction} source bindings")
                if not set(bindings).issubset(tensor_names):
                    raise AuditError(f"{location} has source bindings outside {direction}")
                for tensor_position, tensor_name in enumerate(tensor_names):
                    if tensor_name not in tensors:
                        raise AuditError(f"{location} references absent tensor {tensor_name}")
                    if tensors[tensor_name].get("data_type") == activation_data_type:
                        expected_u16_uses.setdefault(tensor_name, set()).add(
                            (name, direction[:-1], tensor_position)
                        )
                    if tensors[tensor_name].get("quantization_encoding") == (
                        "QNN_QUANTIZATION_ENCODING_UNDEFINED"
                    ):
                        unquantized_uses += 1
                for tensor_name, source in bindings.items():
                    _validate_decoder_binding(
                        source,
                        f"{location} {direction} tensor {tensor_name}",
                        require_module_path=False,
                    )
            params += _validate_params(operation, location)
            for param_index, param in enumerate(operation.get("params", [])):
                if param.get("param_type") != "QNN_PARAMTYPE_TENSOR":
                    continue
                tensor_name = param.get("tensor", {}).get("name")
                if tensors.get(tensor_name, {}).get("data_type") == activation_data_type:
                    expected_u16_uses.setdefault(tensor_name, set()).add(
                        (name, "tensor_param", param_index)
                    )
            op_types[type_name] += 1

        expected_u16_names = {
            tensor_name
            for tensor_name, tensor in tensors.items()
            if tensor.get("data_type") == activation_data_type
        }
        if set(u16_tensor_index) != expected_u16_names:
            raise AuditError(
                f"shard {shard_index} activation index does not cover the "
                f"QNN A{activation_bits} ABI"
            )
        for tensor_name, entry in u16_tensor_index.items():
            location = f"shard {shard_index} u16 tensor {tensor_name}"
            if not isinstance(entry, dict) or entry.get("name") != tensor_name:
                raise AuditError(f"{location} has an invalid identity")
            tensor = tensors[tensor_name]
            if entry.get("data_type") != tensor.get("data_type") or entry.get(
                "quantization_encoding"
            ) != tensor.get("quantization_encoding"):
                raise AuditError(f"{location} loses its qparam identity")
            uses = entry.get("operation_uses")
            if not isinstance(uses, list):
                raise AuditError(f"{location} lacks operation uses")
            actual_uses = set()
            for use in uses:
                if (
                    not isinstance(use, dict)
                    or not isinstance(use.get("operation_name"), str)
                    or use.get("role") not in {"input", "output", "tensor_param"}
                    or not isinstance(use.get("position"), int)
                    or use["position"] < 0
                ):
                    raise AuditError(f"{location} has an invalid operation use")
                actual_uses.add(
                    (use["operation_name"], use["role"], use["position"])
                )
            if actual_uses != expected_u16_uses.get(tensor_name, set()):
                raise AuditError(f"{location} has incomplete operation coverage")
            sources = entry.get("sources")
            bindings = entry.get("decoder_bindings")
            if not isinstance(sources, list) or not isinstance(bindings, list):
                raise AuditError(f"{location} lacks decoder source records")
            for source in sources:
                _validate_decoder_binding(source, location, require_module_path=False)
            if bindings != [source["decoder_binding"] for source in sources]:
                raise AuditError(f"{location} has inconsistent decoder bindings")

        total_op_types.update(op_types)
        total_static_storage.update(static_storage)
        total_encodings.update(tensor_encodings)
        total_params += params
        total_unquantized_uses += unquantized_uses
        report_shards.append(
            {
                "index": shard_index,
                "layer_start": shard.get("layer_start"),
                "layer_end_exclusive": shard.get("layer_end_exclusive"),
                "tensor_count": len(tensors),
                "operation_count": len(operations),
                "parameter_count": params,
                "logical_tensor_count": len(logical_tensors),
                "activation_bits": activation_bits,
                "activation_tensor_count": len(u16_tensor_index),
                "u8_activation_tensor_count": (
                    len(u16_tensor_index) if activation_bits == 8 else 0
                ),
                "u16_tensor_count": (
                    len(u16_tensor_index) if activation_bits == 16 else 0
                ),
                "u16_tensor_with_decoder_source_count": sum(
                    bool(entry["sources"]) for entry in u16_tensor_index.values()
                ),
                "logical_tensor_with_decoder_module_count": sum(
                    bool(source["decoder_binding"]["module_paths"])
                    for source in logical_tensors.values()
                ),
                "tensor_qparam_encodings": dict(sorted(tensor_encodings.items())),
                "static_storage": dict(sorted(static_storage.items())),
                "qnn_operation_types": dict(sorted(op_types.items())),
                "unquantized_tensor_uses": unquantized_uses,
            }
        )

    return {
        "schema_version": 1,
        "format": "llama-qnn-quant-profile-audit-v1",
        "result": "pass",
        "source_weight_bits": 2,
        "source_group_size": group_size,
        "checked_shards": sorted(selected),
        "shards": report_shards,
        "totals": {
            "operation_types": dict(sorted(total_op_types.items())),
            "static_storage": dict(sorted(total_static_storage.items())),
            "tensor_qparam_encodings": dict(sorted(total_encodings.items())),
            "parameter_count": total_params,
            "unquantized_tensor_uses": total_unquantized_uses,
        },
        "next_verification": (
            "Compare each reconstructed GPTQ source block against the recorded "
            "qnn_payload_sha256, then compare QNN and llama.cpp intermediate tensors."
        ),
    }


def _load_profile(manifest_path: Path, graph_name: str) -> Dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    graph = payload.get("graphs", {}).get(graph_name)
    if not isinstance(graph, dict):
        raise AuditError(f"manifest has no graph {graph_name!r}")
    return graph.get("llama_qnn_quant_profile")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--graph", default="prefill_forward")
    parser.add_argument(
        "--shard",
        type=int,
        action="append",
        help="Audit only one or more shard indices. Defaults to every shard.",
    )
    parser.add_argument("--out", type=Path, help="Write JSON report to this path.")
    args = parser.parse_args()

    report = audit_profile(_load_profile(args.manifest, args.graph), args.shard)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
