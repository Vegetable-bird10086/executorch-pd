#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

from qnn_llama_quant_profile_bin import convert_profile


FIXED_POINT_DATA_TYPES = {
    "QNN_DATATYPE_UFIXED_POINT_8",
    "QNN_DATATYPE_UFIXED_POINT_16",
    "QNN_DATATYPE_UFIXED_POINT_32",
    "QNN_DATATYPE_SFIXED_POINT_8",
    "QNN_DATATYPE_SFIXED_POINT_16",
    "QNN_DATATYPE_SFIXED_POINT_32",
}
MAX_AUXILIARY_STATIC_PAYLOAD_BYTES = 1 << 20

# FX assigns these suffix counters from layer zero.  A retained tail graph is
# re-exported as a smaller GraphModule, so its counters restart at zero even
# though decoder bindings keep the original layer ids.  Normalize only the FX
# lookup key; QNN operation and tensor names must remain exactly as exported.
INDEXED_FX_LAYER_STRIDES = {
    "aten_rms_norm_default": 4,
    "aten_matmul_default": 4,
    "aten_mul_tensor": 10,
    "aten_slice_copy_tensor": 4,
    "aten_sub_tensor": 2,
    "aten_add_tensor": 5,
    "aten_convolution_default": 7,
    "aten_div_tensor": 1,
    "aten_amin_default": 1,
    "aten_where_self": 1,
    "aten__softmax_default": 1,
    "aten_sigmoid_default": 1,
}


def normalize_supplemental_fx_names(shard):
    layer_start = int(shard["layer_start"])
    if layer_start <= 0:
        return 0
    normalized = 0
    for operation in shard["operations"]:
        source = operation["source"]
        binding_layers = sorted(set(source["decoder_binding"]["layer_ids"]))
        if len(binding_layers) != 1 or binding_layers[0] < layer_start:
            continue
        fx_name = source["fx_node_name"]
        for stem, stride in INDEXED_FX_LAYER_STRIDES.items():
            match = re.fullmatch(
                rf"{re.escape(stem)}(?:_(\d+))?(_h_\d+)?", fx_name
            )
            if match is None:
                continue
            local_index = int(match.group(1) or 0)
            global_floor = stride * layer_start
            if stem == "aten_mul_tensor":
                # The full graph consumes suffix zero for its input activation
                # scale.  A retained tail starts directly at layer RoPE, so all
                # of its local Mul suffixes are one lower than the full graph.
                global_floor += 1
            if local_index >= global_floor:
                break
            global_index = local_index + global_floor
            source["fx_node_name"] = (
                stem
                + (f"_{global_index}" if global_index else "")
                + (match.group(2) or "")
            )
            normalized += 1
            break
    return normalized


def is_fixed_point_tensor(tensor):
    return (
        tensor.get("data_type") in FIXED_POINT_DATA_TYPES
        and tensor.get("quantization_encoding") in {
            "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET",
            "QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET",
        }
    )


def compact_tensor(tensor):
    result = {
        "data_type": tensor["data_type"],
        "dimensions": tensor["dimensions"],
        "quantization_encoding": tensor["quantization_encoding"],
    }
    encoding = tensor["quantization_encoding"]
    if encoding == "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET":
        result["scale_offset"] = tensor["scale_offset"]
    elif encoding == "QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET":
        result["axis_scale_offset"] = tensor["axis_scale_offset"]
    elif encoding == "QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION":
        # Decode must use the exact QNN block scales, not the source GGUF FP16
        # scales that were quantized to produce them.  The INT4 weight payload
        # remains external; this only retains the compact per-block scale ABI.
        result["blockwise_expansion"] = tensor["blockwise_expansion"]
    payload = tensor.get("static_payload")
    embed_static_payload = (
        payload is not None
        and is_fixed_point_tensor(tensor)
        and (
            tensor["data_type"] == "QNN_DATATYPE_UFIXED_POINT_16"
            or payload.get("data_bytes", 0) <= MAX_AUXILIARY_STATIC_PAYLOAD_BYTES
        )
    )
    if embed_static_payload:
        if (
            payload.get("storage") != "embedded_exact_bytes"
            or payload.get("element_bytes") not in (1, 2, 4)
            or payload.get("data_bytes", 0) <= 0
            or not payload.get("data_le_base64")
        ):
            raise ValueError(
                f"static tensor {tensor.get('name', '<unnamed>')} has an invalid exact payload"
            )
        result["static_payload"] = {
            "storage": payload["storage"],
            "element_bytes": payload["element_bytes"],
            "data_le_base64": payload["data_le_base64"],
            "data_bytes": payload["data_bytes"],
            "sha256": payload["sha256"],
        }
    return result


def operation_tensor_uses(shard):
    uses = {}
    bindings = {}
    for operation in shard["operations"]:
        binding = compact_decoder_binding(operation["source"]["decoder_binding"])
        for role in ("inputs", "outputs"):
            singular_role = "input" if role == "inputs" else "output"
            for position, name in enumerate(operation[role]):
                uses.setdefault(name, []).append(
                    {
                        "operation_name": operation["name"],
                        "role": singular_role,
                        "position": position,
                    }
                )
                if binding not in bindings.setdefault(name, []):
                    bindings[name].append(binding)
    return uses, bindings


def compact_decoder_binding(binding):
    return {
        "layer_ids": binding["layer_ids"],
        "module_paths": [
            path for path in binding["module_paths"] if path.startswith("layers.")
        ],
        "projection": binding["projection"],
    }


def compact_operation(operation):
    source = operation["source"]
    result = {
        "name": operation["name"],
        "type_name": operation["type_name"],
        "inputs": operation["inputs"],
        "outputs": operation["outputs"],
        "source": {
            "fx_node_name": source["fx_node_name"],
            "decoder_binding": compact_decoder_binding(source["decoder_binding"]),
        },
    }
    if operation["type_name"] == "RmsNorm":
        fx_args = source.get("fx_args", [])
        if not fx_args or not isinstance(fx_args[-1], (int, float)):
            raise ValueError(f"RmsNorm operation {operation['name']} lacks epsilon")
        result["epsilon"] = fx_args[-1]
    return result


def compact_manifest(manifest):
    graph = manifest["graphs"]["prefill_forward"]
    profile = graph["llama_qnn_quant_profile"]
    compact_shards = []
    for shard in profile["shards"]:
        u16_names = set(shard["u16_tensor_index"])
        uses, bindings = operation_tensor_uses(shard)
        referenced_names = set(uses)
        quantized_names = u16_names | {
            name
            for name in referenced_names
            if name in shard["tensors"]
            and is_fixed_point_tensor(shard["tensors"][name])
        }
        blockwise_names = {
            name
            for operation in shard["operations"]
            if operation["type_name"] == "Conv2d"
            for name in operation["inputs"][1:2]
        }
        tensor_names = quantized_names | blockwise_names
        compact_shards.append(
            {
                "scope": shard["scope"],
                "layer_start": shard["layer_start"],
                "layer_end_exclusive": shard["layer_end_exclusive"],
                "capabilities": shard["capabilities"],
                "tensors": {
                    name: compact_tensor(shard["tensors"][name])
                    for name in tensor_names
                },
                "u16_tensor_index": {
                    name: {
                        "name": name,
                        "data_type": entry["data_type"],
                        "quantization_encoding": entry["quantization_encoding"],
                        "operation_uses": [],
                        "decoder_bindings": [],
                    }
                    for name, entry in shard["u16_tensor_index"].items()
                },
                "quantized_tensor_index": {
                    name: {
                        "name": name,
                        "data_type": shard["tensors"][name]["data_type"],
                        "quantization_encoding": shard["tensors"][name]["quantization_encoding"],
                        "operation_uses": uses.get(name, []),
                        "decoder_bindings": bindings.get(name, []),
                    }
                    for name in quantized_names
                },
                "operations": [
                    compact_operation(operation)
                    for operation in shard["operations"]
                ],
            }
        )
    compact_profile = {
        "schema_version": profile["schema_version"],
        "format": profile["format"],
        "capabilities": profile["capabilities"],
        "quantization_formula": profile["quantization_formula"],
        "gptq_source_recipe": profile["gptq_source_recipe"],
        "shards": compact_shards,
    }
    return {
        "schema_version": manifest.get("schema_version"),
        "num_decoder_layers": manifest["num_decoder_layers"],
        "graphs": {
            "prefill_forward": {
                "llama_qnn_quant_profile": compact_profile,
            }
        },
    }



def validate_compact_manifest(manifest, compact):
    source_profile = manifest["graphs"]["prefill_forward"]["llama_qnn_quant_profile"]
    runtime_profile = compact["graphs"]["prefill_forward"]["llama_qnn_quant_profile"]
    for field in (
        "schema_version",
        "format",
        "capabilities",
        "quantization_formula",
        "gptq_source_recipe",
    ):
        if runtime_profile[field] != source_profile[field]:
            raise ValueError(f"runtime profile changed {field}")

    source_shards = source_profile["shards"]
    runtime_shards = runtime_profile["shards"]
    if len(runtime_shards) != len(source_shards):
        raise ValueError("runtime profile changed shard count")

    statistics = {
        "shards": len(source_shards),
        "u16_tensors": 0,
        "operations": 0,
        "u16_operands": 0,
        "linear_weight_tensors": 0,
        "static_u16_tensors": 0,
        "static_u16_bytes": 0,
        "quantized_tensors": 0,
        "non_u16_quantized_tensors": 0,
        "static_quantized_tensors": 0,
        "static_quantized_bytes": 0,
    }
    for shard_index, (source_shard, runtime_shard) in enumerate(
        zip(source_shards, runtime_shards)
    ):
        for field in (
            "scope",
            "layer_start",
            "layer_end_exclusive",
            "capabilities",
        ):
            if runtime_shard[field] != source_shard[field]:
                raise ValueError(f"shard {shard_index} changed {field}")

        u16_names = set(source_shard["u16_tensor_index"])
        uses, bindings = operation_tensor_uses(source_shard)
        quantized_names = u16_names | {
            name
            for name in uses
            if name in source_shard["tensors"]
            and is_fixed_point_tensor(source_shard["tensors"][name])
        }
        linear_weight_names = {
            operation["inputs"][1]
            for operation in source_shard["operations"]
            if operation["type_name"] == "Conv2d" and len(operation["inputs"]) > 1
        }
        retained_names = quantized_names | linear_weight_names
        if set(runtime_shard["tensors"]) != retained_names:
            raise ValueError(f"shard {shard_index} changed retained tensor set")
        if set(runtime_shard["u16_tensor_index"]) != u16_names:
            raise ValueError(f"shard {shard_index} changed U16 tensor index")
        if set(runtime_shard["quantized_tensor_index"]) != quantized_names:
            raise ValueError(f"shard {shard_index} changed quantized tensor index")

        for tensor_name in quantized_names:
            source_tensor = source_shard["tensors"][tensor_name]
            expected_index = {
                "name": tensor_name,
                "data_type": source_tensor["data_type"],
                "quantization_encoding": source_tensor["quantization_encoding"],
                "operation_uses": uses[tensor_name],
                "decoder_bindings": bindings[tensor_name],
            }
            if runtime_shard["quantized_tensor_index"][tensor_name] != expected_index:
                raise ValueError(
                    f"shard {shard_index} changed quantized tensor index entry {tensor_name}"
                )

        for tensor_name in retained_names:
            expected_tensor = compact_tensor(source_shard["tensors"][tensor_name])
            if runtime_shard["tensors"][tensor_name] != expected_tensor:
                raise ValueError(
                    f"shard {shard_index} changed tensor qparams for {tensor_name}"
                )
            payload = expected_tensor.get("static_payload")
            if payload is not None:
                if tensor_name in quantized_names:
                    statistics["static_quantized_tensors"] += 1
                    statistics["static_quantized_bytes"] += payload["data_bytes"]
                    if source_shard["tensors"][tensor_name]["data_type"] == \
                            "QNN_DATATYPE_UFIXED_POINT_16":
                        statistics["static_u16_tensors"] += 1
                        statistics["static_u16_bytes"] += payload["data_bytes"]

        source_operations = source_shard["operations"]
        runtime_operations = runtime_shard["operations"]
        if len(runtime_operations) != len(source_operations):
            raise ValueError(f"shard {shard_index} changed operation count")
        for operation_index, (source_operation, runtime_operation) in enumerate(
            zip(source_operations, runtime_operations)
        ):
            if runtime_operation != compact_operation(source_operation):
                raise ValueError(
                    f"shard {shard_index} changed operation {operation_index}"
                )
            statistics["u16_operands"] += sum(
                tensor_name in u16_names
                for tensor_name in (
                    source_operation["inputs"] + source_operation["outputs"]
                )
            )

        statistics["u16_tensors"] += len(u16_names)
        statistics["quantized_tensors"] += len(quantized_names)
        statistics["non_u16_quantized_tensors"] += len(quantized_names - u16_names)
        statistics["operations"] += len(source_operations)
        statistics["linear_weight_tensors"] += len(linear_weight_names)

    return statistics


def apply_supplemental_profile_shards(manifest, supplemental):
    if supplemental.get("num_decoder_layers") != manifest.get("num_decoder_layers"):
        raise ValueError("supplemental manifest changed decoder layer count")
    profile = manifest["graphs"]["prefill_forward"]["llama_qnn_quant_profile"]
    supplemental_profile = supplemental["graphs"]["prefill_forward"][
        "llama_qnn_quant_profile"
    ]
    for field in (
        "schema_version",
        "format",
        "capabilities",
        "quantization_formula",
        "gptq_source_recipe",
    ):
        if supplemental_profile[field] != profile[field]:
            raise ValueError(f"supplemental manifest changed {field}")
    shards_by_scope = {shard["scope"]: shard for shard in profile["shards"]}
    replaced_scopes = []
    for shard in supplemental_profile["shards"]:
        normalize_supplemental_fx_names(shard)
        scope = shard["scope"]
        original = shards_by_scope.get(scope)
        if original is None:
            raise ValueError(f"supplemental manifest has unknown scope {scope}")
        for field in ("layer_start", "layer_end_exclusive", "capabilities"):
            if shard[field] != original[field]:
                raise ValueError(f"supplemental scope {scope} changed {field}")
        shards_by_scope[scope] = shard
        replaced_scopes.append(scope)
    if not replaced_scopes:
        raise ValueError("supplemental manifest contains no profile shards")
    profile["shards"] = [shards_by_scope[shard["scope"]] for shard in profile["shards"]]
    return replaced_scopes


def main():
    parser = argparse.ArgumentParser(
        description="Strip a full QNN shard manifest to its exact runtime activation ABI."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--supplemental-manifest",
        action="append",
        default=[],
        type=Path,
        help=(
            "replace matching profile shard scopes before compaction; useful when "
            "a no-output graph prunes decode-only projections"
        ),
    )
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--bin-out",
        type=Path,
        help="V5 sidecar metadata (default: JSON output with .meta suffix)",
    )
    parser.add_argument(
        "--bin-converter",
        help="qnn-u16-profile-convert executable",
    )
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8") as source:
        manifest = json.load(source)
    replaced_scopes = []
    for supplemental_path in args.supplemental_manifest:
        with supplemental_path.open("r", encoding="utf-8") as source:
            supplemental = json.load(source)
        replaced_scopes.extend(
            apply_supplemental_profile_shards(manifest, supplemental)
        )
    compact = compact_manifest(manifest)
    statistics = validate_compact_manifest(manifest, compact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as destination:
        json.dump(compact, destination, separators=(",", ":"), sort_keys=True)
        destination.write("\n")
    binary_out = args.bin_out or args.out.with_suffix(".meta")
    convert_profile(args.out, args.gguf, binary_out, args.bin_converter)
    payload_out = binary_out.with_suffix(".bin")

    print(
        "QNN runtime activation profile: "
        f"verified={json.dumps(statistics, sort_keys=True)} "
        f"supplemental_scopes={json.dumps(replaced_scopes)} "
        f"shards={len(compact['graphs']['prefill_forward']['llama_qnn_quant_profile']['shards'])} "
        f"json_bytes={args.out.stat().st_size} json_out={args.out} "
        f"meta_bytes={binary_out.stat().st_size} meta_out={binary_out} "
        f"payload_bytes={payload_out.stat().st_size} payload_out={payload_out}"
    )


if __name__ == "__main__":
    main()
