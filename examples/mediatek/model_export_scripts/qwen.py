# Copyright (c) MediaTek Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import argparse
import hashlib
import json
import re
import struct
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_flatten

from aot_utils.llm_utils.preformatter import Preformatter
from aot_utils.llm_utils.sanity_checks import (
    check_all_chunks_same_num_layer,
    check_between_inclusive,
    check_exist,
    check_ext,
    check_old_arg,
    check_shapes,
    check_supported_model,
    check_supported_tokenizer,
    check_tokenizer_exist,
    check_weights_exist,
)
from aot_utils.llm_utils.utils import (
    dump_embedding_lut_for_cmdline,
    generate_mask,
    get_dest_path,
    get_dirname,
    get_embedding_layer,
    get_exp_name,
    get_export_shapes,
    get_master_rot_emb,
    get_normalized_config,
    load_checkpoints,
    resolve_model_classes,
)
from datasets import load_dataset
from executorch import exir
from executorch.backends.mediatek import (
    NeuropilotPartitioner,
    NeuropilotQuantizer,
    Precision,
)
from executorch.exir.backend.backend_api import (
    MethodProgramsPartitionerSpec,
    to_backend,
)
from executorch.exir.backend.backend_details import CompileSpec
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from tqdm import tqdm

warnings.filterwarnings("ignore")

PLATFORM_CONFIGS = {
    "DX3": b"mt6989",
    "DX4": b"mt6991",
    "DX5": b"mt6993",
}


class ShardedLMHead(torch.nn.Module):
    """Split a large vocabulary projection into explicit output-channel shards."""

    def __init__(self, linear, shard_size):
        super().__init__()
        self.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(part.detach(), requires_grad=False)
                for part in linear.weight.split(shard_size, dim=0)
            ]
        )
        if linear.bias is None:
            self.biases = None
        else:
            self.biases = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(part.detach(), requires_grad=False)
                    for part in linear.bias.split(shard_size, dim=0)
                ]
            )

    def forward(self, inputs):
        outputs = []
        for index, weight in enumerate(self.weights):
            bias = None if self.biases is None else self.biases[index]
            outputs.append(F.linear(inputs, weight, bias))
        return tuple(outputs)


def get_argument_parser():
    parser = argparse.ArgumentParser(
        description="Run Export to ET for suppoorted LLM models.", allow_abbrev=False
    )
    parser.add_argument(
        "config",
        type=str,
        help="[Required] Model config json file. "
        "Model config must be in same directory as all model weight bins and tokenizer files.",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="A16W8",
        choices=["A16W4", "A16W8", "A16W16", "A8W4", "A8W8"],
        help="Precision to quantize entire model to.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="DX4",
        choices=list(PLATFORM_CONFIGS),
        help="Chip model of the inference device. "
        "DX3 for Dimensity 9300, DX4 for Dimensity 9400, "
        "DX5 for Dimensity 9500.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Write PTE artifacts to this directory instead of pte/<experiment>.",
    )
    parser.add_argument(
        "--dump-qweights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save every converted 2D int8 qweight and a manifest before MTK "
            "compilation. These files are used to locate and strip packed "
            "weights from the resulting PTE."
        ),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        help="Calibration dataset name or path to dataset. Defaults to None to use random inputs",
    )
    parser.add_argument(
        "-n",
        "--num_chunks",
        type=int,
        default=4,
        help="Number of chunks to cut the model into. Defaults to 4.",
    )
    parser.add_argument(
        "--export-chunk",
        type=int,
        default=None,
        help="Export only one zero-based chunk for diagnostics. Defaults to all chunks.",
    )
    parser.add_argument(
        "-r",
        "--response_cap",
        type=int,
        default=9,
        help="Max Number of Response Tokens to save during calibration. Defaults to 9.",
    )
    parser.add_argument(
        "--preformatter",
        type=str,
        default=None,
        help="Preformatter Template to use to wrap input with. Defaults to None.",
    )
    parser.add_argument(
        "-shapes",
        nargs="+",
        help="[Required] Expected input shapes to reconfigure TFLites to. Space separated list of "
        "shapes in the format: xtyc (e.g. 32t512c)",
    )
    parser.add_argument(
        "--import-forever",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep Neuron input/output imports in the APUSys cache. Defaults to disabled. "
        "Enable only for a long-lived process that loads the model once.",
    )
    parser.add_argument(
        "--layer-debug",
        action="store_true",
        help="Export every decoder-layer hidden state as an extra float output. "
        "Diagnostic only; disabled by default.",
    )
    parser.add_argument(
        "--operator-debug-layer",
        type=int,
        default=None,
        help="Expose selected post-QDQ operator boundaries for one global decoder "
        "layer. Diagnostic only; disabled by default.",
    )
    parser.add_argument(
        "--tail-debug",
        action="store_true",
        help="Expose the final-norm tensor consumed by lm_head. Diagnostic only.",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=["full", "prompt-only"],
        default="full",
        help="Calibrate all collected prompt/decode batches or only the prompt batch.",
    )
    parser.add_argument(
        "--calibration-response-steps",
        type=int,
        default=0,
        help="With prompt-only calibration, additionally calibrate the first N "
        "real decode batches. Zero preserves prompt-only behavior.",
    )
    parser.add_argument(
        "--skip-mlp-output-quantization",
        action="store_true",
        help="Keep MLP multiply/down-projection/residual outputs in float. Diagnostic mixed precision.",
    )
    parser.add_argument(
        "--lm-head-precision",
        choices=["same", "A16W8", "A16W16"],
        default="same",
        help="Override lm_head precision while leaving decoder precision unchanged.",
    )
    parser.add_argument(
        "--lm-head-shard-size",
        type=int,
        default=0,
        help="Split lm_head output rows into explicit linear shards of this size; 0 disables.",
    )
    parser.add_argument(
        "--prefill-no-lm-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Export the final Prefill chunk with KV outputs only. This excludes "
            "final norm/lm_head and lets dead-code elimination prune the unused "
            "final hidden-state branch. Embedding remains in the external LUT."
        ),
    )
    parser.add_argument(
        "--compiler-accuracy-mode",
        action="store_true",
        help="Pass Neuropilot --opt-accuracy so compilation prioritizes accuracy over performance.",
    )
    parser.add_argument(
        "--compiler-opt-level",
        choices=[0, 1, 2, 3],
        type=int,
        default=3,
        help="Neuropilot global optimization level. Defaults to 3.",
    )
    parser.add_argument(
        "--converter-dump",
        action="store_true",
        help="Dump converter MLIR and quantization parameters beside the PTE for diagnostics.",
    )
    parser.add_argument(
        "--mtk-gguf-qparams-dir",
        type=str,
        default=None,
        help=(
            "Directory produced by mtk_gguf_requantize.py. Replace converted "
            "decoder qweight/scale/zero_point with GGUF-derived MTK per-channel qparams."
        ),
    )

    return parser


# flake8: noqa: F405
def args_sanity_checks(args):
    check_old_arg(args.config)
    check_exist(args.config, "Config file")
    check_ext(args.config, ".json", "Config file")
    config = get_normalized_config(args.config)

    weight_dir = get_dirname(args.config)
    check_tokenizer_exist(weight_dir)
    check_weights_exist(weight_dir)

    check_supported_model(config)
    check_supported_tokenizer(config)

    if args.preformatter is not None:
        check_exist(args.preformatter, "Preformatter json file")
        check_ext(args.preformatter, ".json", "preformatter")

    if args.dataset is not None:
        check_exist(args.dataset)

    debug_modes = sum(
        (args.layer_debug, args.operator_debug_layer is not None, args.tail_debug)
    )
    if debug_modes > 1:
        raise ValueError(
            "--layer-debug, --operator-debug-layer, and --tail-debug are mutually exclusive"
        )
    if args.operator_debug_layer is not None:
        check_between_inclusive(
            args.operator_debug_layer,
            0,
            config.num_hidden_layers - 1,
            "operator_debug_layer",
        )

    check_between_inclusive(args.num_chunks, 1, config.num_hidden_layers, "num_chunks")
    if args.export_chunk is not None:
        check_between_inclusive(
            args.export_chunk, 0, args.num_chunks - 1, "export_chunk"
        )
    if args.lm_head_shard_size < 0:
        raise ValueError("--lm-head-shard-size must be 0 or a positive integer")
    if args.prefill_no_lm_head and (
        args.lm_head_precision != "same" or args.lm_head_shard_size
    ):
        raise ValueError(
            "--prefill-no-lm-head requires --lm-head-precision=same and "
            "--lm-head-shard-size=0"
        )
    if args.calibration_response_steps < 0:
        raise ValueError("--calibration-response-steps must be non-negative")
    if args.calibration_mode != "prompt-only" and args.calibration_response_steps:
        raise ValueError(
            "--calibration-response-steps requires --calibration-mode=prompt-only"
        )
    if args.calibration_response_steps and args.dataset is None:
        raise ValueError("--calibration-response-steps requires a calibration dataset")
    if args.calibration_response_steps > args.response_cap:
        raise ValueError(
            "--calibration-response-steps cannot exceed --response_cap"
        )

    check_shapes(args.shapes)


def print_args(args, exp_name):
    print("Please check if all arguments are correct:")
    print(f"Config file:                  {args.config}")
    print(
        "Output pte folder:            "
        f"{args.output_folder or os.path.join('pte', exp_name)}"
    )
    print(f"Dump converted qweights:      {args.dump_qweights}")
    print(f"Quantization precision:       {args.precision}")
    print(f"Preformatter:                 {args.preformatter}")
    print(f"Calibration Dataset:          {args.dataset}")
    print(f"Max Response Tokens:          {args.response_cap}")
    print(f"Number of chunks:             {args.num_chunks}")
    print(f"Export only chunk:            {args.export_chunk}")
    print(f"Export shape(s):              {args.shapes}")
    print(f"Platform:                     {args.platform}")
    print(f"ImportForever:                {args.import_forever}")
    print(f"Layer debug outputs:          {args.layer_debug}")
    print(f"Operator debug layer:         {args.operator_debug_layer}")
    print(f"Tail debug output:            {args.tail_debug}")
    print(f"Calibration mode:             {args.calibration_mode}")
    print(f"Calibration response steps:   {args.calibration_response_steps}")
    print(f"Skip MLP output QDQ:          {args.skip_mlp_output_quantization}")
    print(f"LM head precision:            {args.lm_head_precision}")
    print(f"LM head shard size:           {args.lm_head_shard_size}")
    print(f"Exclude Prefill lm_head:      {args.prefill_no_lm_head}")
    print(f"Compiler accuracy mode:       {args.compiler_accuracy_mode}")
    print(f"Compiler optimization level:  {args.compiler_opt_level}")
    print(f"Converter diagnostic dump:    {args.converter_dump}")
    print(f"GGUF-derived MTK qparams:      {args.mtk_gguf_qparams_dir}")
    print()


def apply_preformatter(inp, preformatter=None):
    formatted_text = preformatter.generate_prompt(inp["text"])
    inp["text"] = formatted_text
    print(f"Formatted Prompt:\n{formatted_text}")
    return inp


def tokenize_dataset(inp, tokenizer):
    text = inp["text"]
    inp_encoded = tokenizer(text, return_tensors="pt")  # dict
    inp_encoded.pop("attention_mask")
    inp_encoded = inp_encoded["input_ids"]
    inp_encoded = inp_encoded.to(torch.int32)
    inp["input_ids"] = inp_encoded
    inp.pop("text")
    return inp


def reset_cache(
    num_chunks, num_key_value_heads, num_blocks_per_chunk, head_dim, max_cache_size
):
    cache = []
    for i in range(num_chunks):
        curr_chunk_cache = torch.zeros(
            (
                2 * num_blocks_per_chunk[i],
                num_key_value_heads,
                max_cache_size,  # generate fixed cache as torch dynamic shape cannot handle 2 dynamic dim
                head_dim,
            ),
            dtype=torch.float32,
        )
        cache.append(curr_chunk_cache)
    return cache


def forward_and_save(
    models,
    hidden_state,
    cache,
    mask,
    pos_emb,
    model_input_dict,
    num_blocks_per_chunk,
    batch_name,
):
    for chunk_idx in range(len(models)):
        cache_in = cache[chunk_idx]

        try:
            model_input_dict[str(chunk_idx)] = {
                **model_input_dict[str(chunk_idx)],
                batch_name: {
                    "hidden_state": hidden_state,
                    "mask": mask,
                    "pos_emb": pos_emb,
                    "cache": cache_in,
                },
            }
        except:
            model_input_dict[str(chunk_idx)] = {
                batch_name: {
                    "hidden_state": hidden_state,
                    "mask": mask,
                    "pos_emb": pos_emb,
                    "cache": cache_in,
                }
            }
        with torch.no_grad():
            model_out = models[chunk_idx](
                hidden_state, mask, pos_emb, *torch.split(cache_in, 1, dim=0)
            )
        hidden_state = model_out[0]
        cache[chunk_idx] = torch.cat(
            model_out[1 : 1 + 2 * num_blocks_per_chunk[chunk_idx]], dim=0
        ).clone()
    return hidden_state, cache


def prepare_model_inputs(
    inp,
    models,
    embedding_layer,
    master_rot_emb,
    num_blocks_per_chunk,
    num_key_value_heads,
    head_dim,
    max_cache_size,
    eos_token_id_tensor,
    response_cap,
):
    model_input_dict = {str(i): None for i in range(len(models))}
    input_ids = inp.pop("input_ids")
    hidden_state = embedding_layer(torch.tensor(input_ids))
    input_length = hidden_state.shape[1]
    # Assume fixed cache size
    mask = generate_mask(max_cache_size, 0, input_length, input_length)
    pos_emb = master_rot_emb[:, :, :input_length, :]
    # cache shape: num chunks of 2*num_block, num kv heads, c, head dim
    cache = reset_cache(
        len(models), num_key_value_heads, num_blocks_per_chunk, head_dim, max_cache_size
    )  # empty kv
    logits, cache = forward_and_save(
        models,
        hidden_state,
        cache,
        mask,
        pos_emb,
        model_input_dict,
        num_blocks_per_chunk,
        "prompt",
    )
    next_token_logits = logits[:, -1, :]  # last layer logits
    next_token = torch.argmax(next_token_logits, dim=-1)
    response_count = 0
    seq_length = input_length
    while True:
        curr_input_id = next_token[:, None].to(torch.int32)
        input_length = curr_input_id.shape[1]
        hidden_state = embedding_layer(curr_input_id)
        mask = generate_mask(max_cache_size, seq_length, input_length, input_length)
        pos_emb = master_rot_emb[:, :, seq_length : seq_length + input_length, :]
        logits, cache = forward_and_save(
            models,
            hidden_state,
            cache,
            mask,
            pos_emb,
            model_input_dict,
            num_blocks_per_chunk,
            f"response{response_count}",
        )
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        if next_token == eos_token_id_tensor:
            print(f"Found EOS on batch: {response_count}")
            break

        response_count += 1
        seq_length += input_length
        if response_count == response_cap:
            break

    return model_input_dict


def calibrate_model(
    model, cal_dataset, chunk_idx: str, prompt_only=False, response_steps=0
):
    with torch.no_grad():
        for inp in tqdm(cal_dataset, desc="Calibrating Model: "):
            # pass prompt and response
            if prompt_only:
                batches = ["prompt"] + [
                    f"response{index}" for index in range(response_steps)
                ]
            else:
                batches = inp[chunk_idx].keys()
            for batch in tqdm(batches, desc="Batch: "):
                if batch in inp[chunk_idx] and inp[chunk_idx][batch] is not None:
                    inputs_embeds = torch.tensor(inp[chunk_idx][batch]["hidden_state"])
                    mask = torch.tensor(inp[chunk_idx][batch]["mask"])
                    pos_emb = torch.tensor(inp[chunk_idx][batch]["pos_emb"])
                    cache = torch.tensor(inp[chunk_idx][batch]["cache"])
                    model(inputs_embeds, mask, pos_emb, *torch.split(cache, 1, dim=0))


def append_layer_debug_outputs(converted_graph, num_blocks):
    """Expose last-token decoder residuals after QDQ conversion.

    Slicing before the diagnostic output boundary prevents an outlier in an
    earlier prompt token from dominating the single A16 output scale.
    """
    candidates = {}
    for node in converted_graph.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.add.Tensor:
            continue
        stack = node.meta.get("nn_module_stack", {})
        paths = [value[0] for value in stack.values() if isinstance(value, tuple)]
        for layer_idx in range(num_blocks):
            layer_path = f"layers.{layer_idx}"
            # Each decoder has two residual adds at exactly the layer scope;
            # the last one is the post-MLP hidden state.
            if paths and paths[-1] == layer_path:
                candidates[layer_idx] = node
    missing = sorted(set(range(num_blocks)) - set(candidates))
    if missing:
        raise RuntimeError(f"Unable to locate residual outputs for layers: {missing}")
    post_qdq = {}
    for layer_idx, residual in candidates.items():
        for quantized in residual.users:
            if "quantize_per_tensor" not in str(quantized.target):
                continue
            for dequantized in quantized.users:
                if "dequantize_per_tensor" in str(dequantized.target):
                    post_qdq[layer_idx] = dequantized
                    break
            if layer_idx in post_qdq:
                break
        # Mixed-precision diagnostics may intentionally remove the residual
        # output QDQ boundary. In that case expose the floating residual itself.
        post_qdq.setdefault(layer_idx, residual)
    output_node = next(node for node in converted_graph.graph.nodes if node.op == "output")
    outputs = output_node.args[0]
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    with converted_graph.graph.inserting_before(output_node):
        layer_outputs = tuple(
            converted_graph.graph.call_function(
                torch.ops.aten.slice.Tensor,
                args=(post_qdq[i], 1, -1, 9223372036854775807),
            )
            for i in range(num_blocks)
        )
    output_node.args = (tuple(outputs) + layer_outputs,)
    # torch.export GraphModules reconstruct the original tuple via _out_spec.
    # Keep that schema in sync with the appended flat tensor outputs.
    _, out_spec = tree_flatten(tuple(range(len(outputs) + num_blocks)))
    converted_graph.graph._codegen.pytree_info = (
        converted_graph.graph._codegen.pytree_info._replace(out_spec=out_spec)
    )
    converted_graph.graph.lint()
    converted_graph.recompile()
    return converted_graph


def _module_paths(node):
    stack = node.meta.get("nn_module_stack", {})
    return [value[0] for value in stack.values() if isinstance(value, tuple)]


def _resolve_graph_attr(graph_module, target):
    value = graph_module
    for part in str(target).split("."):
        value = getattr(value, part)
    return value


def _resolve_constant_argument(graph_module, value):
    if isinstance(value, torch.fx.Node):
        if value.op != "get_attr":
            raise ValueError(f"non-constant FX argument: {value}")
        return _resolve_graph_attr(graph_module, value.target)
    if isinstance(value, tuple):
        return tuple(_resolve_constant_argument(graph_module, item) for item in value)
    if isinstance(value, list):
        return [_resolve_constant_argument(graph_module, item) for item in value]
    if isinstance(value, dict):
        return {
            key: _resolve_constant_argument(graph_module, item)
            for key, item in value.items()
        }
    return value


def _qweight_consumer_paths(node):
    """Find the closest Linear module paths reached from a qweight get_attr."""
    pending = [(node, 0)]
    visited = {node}
    matches = []
    while pending:
        current, depth = pending.pop(0)
        if depth > 8:
            continue
        if current is not node and current.op == "call_function":
            target = str(current.target)
            paths = _module_paths(current)
            if paths and any(name in target for name in ("linear", "addmm", "mm")):
                matches.extend(paths)
                continue
        for user in current.users:
            if user not in visited:
                visited.add(user)
                pending.append((user, depth + 1))
    return list(dict.fromkeys(matches))


def dump_converted_qweights(converted_graph, output_folder, chunk_idx):
    """Persist the exact integer tensors that the MTK converter will pack."""
    qweight_dir = os.path.join(output_folder, "qweights", f"chunk_{chunk_idx:02d}")
    os.makedirs(qweight_dir, exist_ok=True)
    candidates = []
    for ordinal, node in enumerate(converted_graph.graph.nodes):
        value = None
        attr = None
        consumer_root = node
        if node.op == "get_attr":
            folded = _resolve_graph_attr(converted_graph, node.target)
            if (
                isinstance(folded, torch.Tensor)
                and folded.ndim == 2
                and folded.dtype == torch.int8
            ):
                value = folded
                attr = str(node.target)
        elif node.op == "call_function" and "quantize_per_channel" in str(node.target):
            source = node.args[0] if node.args else None
            if not isinstance(source, torch.fx.Node) or source.op != "get_attr":
                continue
            source_tensor = _resolve_graph_attr(converted_graph, source.target)
            if not isinstance(source_tensor, torch.Tensor) or source_tensor.ndim != 2:
                continue
            args = _resolve_constant_argument(converted_graph, node.args)
            kwargs = _resolve_constant_argument(converted_graph, node.kwargs)
            value = node.target(*args, **kwargs)
            attr = str(source.target)
        if value is None:
            continue
        candidates.append((ordinal, consumer_root, attr, value))

    records = []
    seen = set()
    for ordinal, consumer_root, attr, value in candidates:
        digest = hashlib.sha256(
            value.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        identity = (attr, tuple(value.shape), digest)
        if identity in seen:
            continue
        seen.add(identity)
        minimum = int(value.min())
        maximum = int(value.max())
        precision = "int4" if minimum >= -8 and maximum <= 7 else "int8"
        filename = f"qweight_{len(records):03d}.npy"
        path = os.path.join(qweight_dir, filename)
        np.save(path, value.detach().cpu().contiguous().numpy())
        records.append(
            {
                "ordinal": ordinal,
                "attr": attr,
                "consumer_module_paths": _qweight_consumer_paths(consumer_root),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "minimum": minimum,
                "maximum": maximum,
                "precision": precision,
                "qweight_sha256": digest,
                "path": filename,
            }
        )
    if not records:
        raise RuntimeError(f"No converted qweights found in chunk {chunk_idx}")
    manifest_path = os.path.join(qweight_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"schema_version": 1, "chunk_index": chunk_idx, "records": records},
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    print(f"Saved {len(records)} converted qweights -> {manifest_path}")


def _set_graph_attr(graph_module, target, value):
    owner = graph_module
    parts = str(target).split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    current = getattr(owner, parts[-1])
    value = value.to(device=current.device, dtype=current.dtype).contiguous()
    if isinstance(current, torch.nn.Parameter):
        value = torch.nn.Parameter(value, requires_grad=False)
    setattr(owner, parts[-1], value)


def _nearest_dequantize_per_channel(node):
    pending = [(node, 0)]
    visited = {node}
    while pending:
        current, depth = pending.pop(0)
        if depth > 4:
            continue
        if current is not node and current.op == "call_function" and (
            "dequantize_per_channel" in str(current.target)
        ):
            return current
        for user in current.users:
            if user not in visited:
                visited.add(user)
                pending.append((user, depth + 1))
    return None


def replace_converted_qparams_from_gguf(
    converted_graph, qparams_dir, chunk_idx, layers_per_chunk
):
    """Apply deterministic GGUF->MTK per-channel qparams before lowering."""
    op_map = {
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
        "mlp.gate_proj": "ffn_gate",
        "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
    }
    replaced = []
    seen = set()
    for node in converted_graph.graph.nodes:
        weight_node = None
        consumer_root = node
        folded_qweight = None
        scale_node = None
        zero_node = None
        if node.op == "get_attr":
            candidate = _resolve_graph_attr(converted_graph, node.target)
            if (
                isinstance(candidate, torch.Tensor)
                and candidate.ndim == 2
                and candidate.dtype == torch.int8
            ):
                weight_node = node
                folded_qweight = candidate
                dequant = _nearest_dequantize_per_channel(node)
                if dequant is not None and len(dequant.args) >= 3:
                    scale_node, zero_node = dequant.args[1], dequant.args[2]
        elif node.op == "call_function" and "quantize_per_channel" in str(
            node.target
        ) and "dequantize" not in str(node.target):
            source_node = node.args[0] if node.args else None
            if isinstance(source_node, torch.fx.Node) and source_node.op == "get_attr":
                candidate = _resolve_graph_attr(converted_graph, source_node.target)
                if isinstance(candidate, torch.Tensor) and candidate.ndim == 2:
                    resolved_args = _resolve_constant_argument(
                        converted_graph, node.args
                    )
                    resolved_kwargs = _resolve_constant_argument(
                        converted_graph, node.kwargs
                    )
                    folded_qweight = node.target(*resolved_args, **resolved_kwargs)
                    weight_node = source_node
                    if len(node.args) >= 3:
                        scale_node, zero_node = node.args[1], node.args[2]
        if weight_node is None or str(weight_node.target) in seen:
            continue
        if not (
            isinstance(folded_qweight, torch.Tensor)
            and folded_qweight.ndim == 2
            and folded_qweight.dtype == torch.int8
        ):
            continue
        paths = _qweight_consumer_paths(consumer_root)
        logical = None
        for path in reversed(paths):
            match = re.search(
                r"layers\.(\d+)\.(self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
                r"mlp\.(?:gate_proj|up_proj|down_proj))$",
                path,
            )
            if match:
                logical = (int(match.group(1)), match.group(2))
                break
        if logical is None:
            continue
        local_layer, local_op = logical
        global_layer = chunk_idx * layers_per_chunk + local_layer
        source_path = os.path.join(
            qparams_dir, f"blk.{global_layer}.{op_map[local_op]}.weight.npz"
        )
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"missing GGUF-derived MTK qparams: {source_path}")
        source = np.load(source_path)
        qweight = torch.from_numpy(source["qweight"])
        scale = torch.from_numpy(source["scale"])
        zero_point = torch.from_numpy(source["zero_point"])
        if tuple(qweight.shape) != tuple(folded_qweight.shape):
            raise RuntimeError(
                f"{source_path}: qweight shape {tuple(qweight.shape)} != "
                f"converted target {tuple(folded_qweight.shape)}"
            )
        if not (
            isinstance(scale_node, torch.fx.Node)
            and scale_node.op == "get_attr"
            and isinstance(zero_node, torch.fx.Node)
            and zero_node.op == "get_attr"
        ):
            raise RuntimeError(
                f"non-attribute MTK qparams for {weight_node.target}: "
                f"scale={scale_node}, zero={zero_node}"
            )
        scale_target = _resolve_graph_attr(converted_graph, scale_node.target)
        zero_target = _resolve_graph_attr(converted_graph, zero_node.target)
        if tuple(scale.shape) != tuple(scale_target.shape) or tuple(
            zero_point.shape
        ) != tuple(zero_target.shape):
            raise RuntimeError(
                f"MTK qparam shape mismatch for {weight_node.target}: source "
                f"scale={tuple(scale.shape)} zero={tuple(zero_point.shape)}, target "
                f"scale={tuple(scale_target.shape)} zero={tuple(zero_target.shape)}"
            )
        current_weight = _resolve_graph_attr(converted_graph, weight_node.target)
        if current_weight.dtype == torch.int8:
            replacement_weight = qweight
        else:
            replacement_weight = (
                qweight.to(torch.float32) - zero_point.to(torch.float32)[:, None]
            ) * scale.to(torch.float32)[:, None]
        _set_graph_attr(converted_graph, weight_node.target, replacement_weight)
        _set_graph_attr(converted_graph, scale_node.target, scale)
        _set_graph_attr(converted_graph, zero_node.target, zero_point)
        seen.add(str(weight_node.target))
        replaced.append(
            {
                "global_layer": global_layer,
                "op": local_op,
                "weight_attr": str(weight_node.target),
                "scale_attr": str(scale_node.target),
                "zero_point_attr": str(zero_node.target),
                "source": source_path,
            }
        )
    if not replaced:
        raise RuntimeError("No MTK decoder qparams were replaced from GGUF")
    print(
        f"Replaced {len(replaced)} MTK qweight/scale/zero-point triplets "
        f"from {qparams_dir}"
    )
    return replaced


def _post_qdq(node):
    """Return the dequantized consumer when node has an immediate QDQ boundary."""
    for quantized in node.users:
        if "quantize_per_tensor" not in str(quantized.target):
            continue
        for dequantized in quantized.users:
            if "dequantize_per_tensor" in str(dequantized.target):
                return dequantized
    return node


def append_operator_debug_outputs(converted_graph, layer_idx):
    """Expose key operator boundaries for one local decoder layer.

    The returned tensors are sliced to the final query token.  Linear/norm
    boundaries use their post-QDQ value when PT2E inserted one, matching what
    the following delegated operator consumes.
    """
    layer_path = f"layers.{layer_idx}"
    nodes = list(converted_graph.graph.nodes)

    def in_layer(node):
        return any(
            path == layer_path or path.startswith(layer_path + ".")
            for path in _module_paths(node)
        )

    def exact_suffix(node, suffix):
        paths = _module_paths(node)
        return bool(paths) and paths[-1] == f"{layer_path}.{suffix}"

    def tensor_rank(node):
        value = node.meta.get("val")
        return getattr(value, "ndim", None)

    def target_has(node, *names):
        target = str(node.target)
        return any(name in target for name in names)

    selected = []

    def add(label, candidates, use_post_qdq=True, last=True):
        candidates = [node for node in candidates if tensor_rank(node) in (3, 4)]
        if not candidates:
            raise RuntimeError(
                f"Unable to locate operator-debug tap {label} in {layer_path}"
            )
        node = candidates[-1] if last else candidates[0]
        value = node.meta.get("val")
        selected.append(
            (
                label,
                _post_qdq(node) if use_post_qdq else node,
                value.ndim,
                [str(dim) for dim in value.shape],
                node.name,
                str(node.target),
            )
        )

    compute_nodes = [
        node
        for node in nodes
        if node.op == "call_function"
        and in_layer(node)
        and "quantize_per_tensor" not in str(node.target)
        and "dequantize_per_tensor" not in str(node.target)
    ]

    add("input_norm", [n for n in compute_nodes if exact_suffix(n, "input_norm")])
    for name in ("q_proj", "k_proj", "v_proj"):
        add(
            name,
            [
                n
                for n in compute_nodes
                if exact_suffix(n, f"self_attn.{name}")
                and target_has(n, "linear", "addmm", "mm")
            ],
        )
    # Qwen3 applies per-head Q/K RMSNorm.
    for name in ("q_norm", "k_norm"):
        matches = [
            n for n in compute_nodes if exact_suffix(n, f"self_attn.{name}")
        ]
        if matches:
            add(name, matches)

    attn_nodes = [
        n
        for n in compute_nodes
        if exact_suffix(n, "self_attn") and tensor_rank(n) == 4
    ]
    rope_adds = [n for n in attn_nodes if target_has(n, "aten.add.Tensor")]
    if len(rope_adds) < 3:
        raise RuntimeError(
            f"Expected Q/K RoPE adds and mask add in {layer_path}, got {len(rope_adds)}"
        )
    add("q_rope", [rope_adds[0]])
    add("k_rope", [rope_adds[1]])
    matmuls = [n for n in attn_nodes if target_has(n, "matmul")]
    if len(matmuls) < 2:
        raise RuntimeError(f"Expected two attention matmuls in {layer_path}")
    add("qk_matmul", [matmuls[0]])
    add("mask_add", [rope_adds[-1]])
    add(
        "softmax",
        [n for n in attn_nodes if target_has(n, "softmax")],
        use_post_qdq=True,
    )
    add("av_matmul", [matmuls[-1]])
    add(
        "o_proj",
        [
            n
            for n in compute_nodes
            if exact_suffix(n, "self_attn.o_proj")
            and target_has(n, "linear", "addmm", "mm")
        ],
    )

    residual_adds = [
        n
        for n in compute_nodes
        if _module_paths(n)
        and _module_paths(n)[-1] == layer_path
        and target_has(n, "aten.add.Tensor")
    ]
    if len(residual_adds) != 2:
        raise RuntimeError(
            f"Expected two residual adds in {layer_path}, got {len(residual_adds)}"
        )
    add("attn_residual", [residual_adds[0]])
    add(
        "post_attention_norm",
        [n for n in compute_nodes if exact_suffix(n, "post_attention_norm")],
    )
    for name in ("gate_proj", "up_proj"):
        add(
            name,
            [
                n
                for n in compute_nodes
                if exact_suffix(n, f"mlp.{name}")
                and target_has(n, "linear", "addmm", "mm")
            ],
        )
    mlp_nodes = [n for n in compute_nodes if exact_suffix(n, "mlp")]
    add("silu", [n for n in mlp_nodes if target_has(n, "silu", "sigmoid")])
    add("gate_mul_up", [n for n in mlp_nodes if target_has(n, "aten.mul.Tensor")])
    add(
        "down_proj",
        [
            n
            for n in compute_nodes
            if exact_suffix(n, "mlp.down_proj")
            and target_has(n, "linear", "addmm", "mm")
        ],
    )
    add("mlp_residual", [residual_adds[1]])

    output_node = next(node for node in nodes if node.op == "output")
    outputs = output_node.args[0]
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    with converted_graph.graph.inserting_before(output_node):
        debug_outputs = []
        manifest = []
        for output_index, (
            label,
            node,
            rank,
            source_shape,
            source_node,
            source_target,
        ) in enumerate(selected):
            token_dim = 1 if rank == 3 else 2
            sliced = converted_graph.graph.call_function(
                torch.ops.aten.slice.Tensor,
                args=(node, token_dim, -1, 9223372036854775807),
            )
            debug_outputs.append(sliced)
            manifest.append(
                {
                    "index": output_index,
                    "name": label,
                    "source_node": source_node,
                    "source_target": source_target,
                    "source_shape": source_shape,
                    "token_dim": token_dim,
                }
            )
    output_node.args = (tuple(outputs) + tuple(debug_outputs),)
    _, out_spec = tree_flatten(tuple(range(len(outputs) + len(debug_outputs))))
    converted_graph.graph._codegen.pytree_info = (
        converted_graph.graph._codegen.pytree_info._replace(out_spec=out_spec)
    )
    converted_graph.graph.lint()
    converted_graph.recompile()
    return converted_graph, manifest


def append_tail_debug_output(converted_graph):
    """Expose the exact final-norm activation consumed by lm_head."""
    lm_heads = []
    for node in converted_graph.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.linear.default:
            continue
        paths = _module_paths(node)
        if paths and (paths[-1] == "lm_head" or paths[-1].endswith(".lm_head")):
            lm_heads.append(node)
    if len(lm_heads) != 1:
        raise RuntimeError(f"Expected one lm_head linear, got {len(lm_heads)}")
    final_norm = lm_heads[0].args[0]
    if not isinstance(final_norm, torch.fx.Node):
        raise RuntimeError("lm_head input is not an FX node")
    output_node = next(
        node for node in converted_graph.graph.nodes if node.op == "output"
    )
    outputs = output_node.args[0]
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    with converted_graph.graph.inserting_before(output_node):
        last_token_norm = converted_graph.graph.call_function(
            torch.ops.aten.slice.Tensor,
            args=(final_norm, 1, -1, 9223372036854775807),
        )
    output_node.args = (tuple(outputs) + (last_token_norm,),)
    _, out_spec = tree_flatten(tuple(range(len(outputs) + 1)))
    converted_graph.graph._codegen.pytree_info = (
        converted_graph.graph._codegen.pytree_info._replace(out_spec=out_spec)
    )
    converted_graph.graph.lint()
    converted_graph.recompile()
    return converted_graph


def export_to_et_ir(
    output_folder,
    exp_name,
    model,
    precision,
    max_num_token,
    max_cache_size,
    chunk_idx,
    export_shapes,
    platform_b,
    cal_dataset=None,
    import_forever=False,
    layer_debug=False,
    operator_debug_local_layer=None,
    tail_debug=False,
    prompt_only_calibration=False,
    skip_mlp_output_quantization=False,
    lm_head_precision="same",
    compiler_accuracy_mode=False,
    compiler_opt_level=3,
    converter_dump=False,
    calibration_response_steps=0,
    dump_qweights=False,
    mtk_gguf_qparams_dir=None,
    layers_per_chunk=0,
):
    print(f"Exporting Chunk {chunk_idx} to PTE")
    example_inputs, dynamic_shapes = model.get_example_inputs(
        max_num_token, max_cache_size, True
    )
    print("Getting pre autograd ATen Dialect Graph")
    pre_autograd_aten_dialect = torch.export.export(
        model, example_inputs, dynamic_shapes=dynamic_shapes, strict=True
    ).module()  # NOTE: Will be replaced with export
    quantizer = NeuropilotQuantizer()
    quantizer.setup_precision(getattr(Precision, precision))
    quantizer.set_skip_mlp_output_quantization(skip_mlp_output_quantization)
    if lm_head_precision != "same":
        quantizer.set_module_name_precision(
            "lm_head", getattr(Precision, lm_head_precision)
        )
    prepared_graph = prepare_pt2e(pre_autograd_aten_dialect, quantizer)
    # at this point quant min max are inf
    if cal_dataset is not None:
        calibrate_model(
            prepared_graph,
            cal_dataset,
            str(chunk_idx),
            prompt_only=prompt_only_calibration,
            response_steps=calibration_response_steps,
        )
    else:
        prepared_graph(*example_inputs)  # dummy calibration
    converted_graph = convert_pt2e(prepared_graph, fold_quantize=False)
    if mtk_gguf_qparams_dir is not None:
        replaced = replace_converted_qparams_from_gguf(
            converted_graph, mtk_gguf_qparams_dir, chunk_idx, layers_per_chunk
        )
        replace_dir = os.path.join(output_folder, "gguf_qparam_replacement")
        os.makedirs(replace_dir, exist_ok=True)
        with open(
            os.path.join(replace_dir, f"chunk_{chunk_idx:02d}.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(replaced, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    if dump_qweights:
        dump_converted_qweights(converted_graph, output_folder, chunk_idx)
    if layer_debug:
        converted_graph = append_layer_debug_outputs(converted_graph, model.num_blocks)
    if operator_debug_local_layer is not None:
        converted_graph, manifest = append_operator_debug_outputs(
            converted_graph, operator_debug_local_layer
        )
        os.makedirs(output_folder, exist_ok=True)
        with open(
            os.path.join(output_folder, f"operator_debug_chunk_{chunk_idx}.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Operator debug outputs: {len(manifest)}")
    if tail_debug:
        converted_graph = append_tail_debug_output(converted_graph)

    method_to_edge_program = {}
    method_to_partitioner = {}
    edge_compile_config = exir.EdgeCompileConfig(_check_ir_validity=False)

    model_shared_key_name = f"{exp_name}_{chunk_idx}"

    # Fixed Shape Export Here
    for shape, ntok_and_cache in export_shapes.items():
        model_fname = f"{exp_name}_{shape}_{chunk_idx}"
        example_inputs = model.get_example_inputs(*ntok_and_cache)
        print(f"Getting ATen Dialect Graph for {exp_name} {shape} chunk {chunk_idx}")
        aten_dialect: exir.ExportedProgram = torch.export.export(
            converted_graph, example_inputs, strict=True
        )

        method_to_edge_program[f"{model_fname}"] = exir.to_edge(
            aten_dialect
        ).exported_program()
        del aten_dialect

        compile_spec = [
            CompileSpec("gno", b"LTS"),
            CompileSpec("gno-exp", b""),
            CompileSpec("gno-non-4d-tiling", b""),
            CompileSpec("ImportForever", struct.pack("?", import_forever)),
            CompileSpec("platform-config", platform_b),
            CompileSpec("ExtractSharedBlobKey", model_shared_key_name.encode()),
        ]
        if compiler_accuracy_mode:
            compile_spec.append(CompileSpec("opt-accuracy", b""))
        if compiler_opt_level != 3:
            compile_spec.append(CompileSpec("opt", str(compiler_opt_level).encode()))
        if converter_dump:
            converter_dump_dir = os.path.abspath(
                os.path.join(output_folder, "converter_dump", model_fname)
            )
            compile_spec.append(
                CompileSpec("ConverterDumpDir", converter_dump_dir.encode())
            )
        method_to_partitioner[f"{model_fname}"] = NeuropilotPartitioner(compile_spec)

    print("Delegating Edge Program to Neuropilot Backend")
    delegated_program = to_backend(
        MethodProgramsPartitionerSpec(method_to_edge_program, method_to_partitioner)
    )

    edge_manager = exir.EdgeProgramManager(
        delegated_program, compile_config=edge_compile_config
    )
    del delegated_program

    print("Transforming delegated program to executorch backend")
    executorch_program = edge_manager.to_executorch(
        config=exir.ExecutorchBackendConfig(
            memory_planning_pass=exir.passes.MemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
            extract_delegate_segments=True,
        )
    )
    del edge_manager
    print(f"\n Model Size: {len(executorch_program.buffer)}")

    dest_path = get_dest_path(output_folder, exp_name, None, chunk_idx)
    print(f"{exp_name} ET Model chunk {chunk_idx} Dest: {dest_path}\n")
    os.makedirs(dest_path.rsplit("/", 1)[0], exist_ok=True)
    with open(dest_path, "wb") as file:
        file.write(executorch_program.buffer)


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    args_sanity_checks(args)
    if args.dataset is None:
        exp_name = f"{get_exp_name(args.config)}_{args.precision}_dummy_cal_{args.num_chunks}_chunks"
    else:
        exp_name = (
            f"{get_exp_name(args.config)}_{args.precision}_{args.num_chunks}_chunks"
        )
    if args.layer_debug:
        exp_name += "_layer_debug_last_token"
    if args.operator_debug_layer is not None:
        exp_name += f"_operator_debug_layer_{args.operator_debug_layer}"
    if args.tail_debug:
        exp_name += "_tail_debug"
    if args.export_chunk is not None:
        exp_name += f"_chunk_{args.export_chunk}_only"
    if args.skip_mlp_output_quantization:
        exp_name += "_skip_mlp_qdq"
    if args.lm_head_precision != "same":
        exp_name += f"_lm_head_{args.lm_head_precision}"
    if args.lm_head_shard_size:
        exp_name += f"_lm_head_shard{args.lm_head_shard_size}"
        exp_name += "_separate_outputs"
    exports_final_chunk = (
        args.export_chunk is None or args.export_chunk == args.num_chunks - 1
    )
    if args.prefill_no_lm_head and exports_final_chunk:
        exp_name += "_no_lm_head"
    if args.compiler_accuracy_mode:
        exp_name += "_compiler_accuracy"
    if args.compiler_opt_level != 3:
        exp_name += f"_opt{args.compiler_opt_level}"
    if args.converter_dump:
        exp_name += "_converter_dump"
    if args.dataset is not None and args.calibration_mode == "prompt-only":
        if args.calibration_response_steps:
            exp_name += f"_prompt_plus_{args.calibration_response_steps}_decode"
        else:
            exp_name += "_prompt_only"
    platform_b = PLATFORM_CONFIGS[args.platform]
    print_args(args, exp_name)

    config, weight_dir, tokenizer_class, chunk_class = resolve_model_classes(
        args.config
    )
    tokenizer = tokenizer_class.from_pretrained(weight_dir)
    if args.preformatter is not None:
        preformatter = Preformatter(args.preformatter)

    head_dim = int(config.head_dim)

    # Evenly distribute the layers across chunks.
    num_blocks_per_chunk = [
        (config.num_hidden_layers // args.num_chunks)
        + (i < (config.num_hidden_layers % args.num_chunks))
        for i in range(args.num_chunks)
    ]
    check_all_chunks_same_num_layer(num_blocks_per_chunk)  # noqa: F405

    output_folder = args.output_folder or os.path.join("pte", exp_name)

    # Load all collected checkpoint files into one giant state_dict
    state_dict = load_checkpoints(weight_dir)

    dump_embedding_lut_for_cmdline(weight_dir, state_dict, config)

    export_shapes, max_num_token, max_cache_size = get_export_shapes(args.shapes)
    print(f"export shapes: {export_shapes}")
    print(f"Max Num Token: {max_num_token}")
    print(f"Max Cache Size: {max_cache_size}")

    if args.dataset is not None:
        embedding_layer = get_embedding_layer(config, weight_dir, state_dict)

    # Instantiate model chunks
    print("Instantiating submodels")
    models = []
    for chunk_idx, num_blocks in enumerate(num_blocks_per_chunk):
        chunk = chunk_class(
            config,
            num_blocks,
            chunk_idx=chunk_idx,
            dtype=torch.float32,
            include_tail=(chunk_idx == args.num_chunks - 1),
            jit_trace=True,
        )
        chunk = chunk.load_weights(state_dict, sum(num_blocks_per_chunk[:chunk_idx]))
        models.append(chunk)

    cal_dataset = None
    if args.dataset is not None:
        cal_dataset = load_dataset("text", data_files=args.dataset, split="train")
        master_rot_emb = get_master_rot_emb(config, dtype=torch.float32)
        if args.preformatter is not None:
            cal_dataset = cal_dataset.map(
                apply_preformatter, fn_kwargs={"preformatter": preformatter}
            )
        cal_dataset = cal_dataset.map(
            tokenize_dataset, fn_kwargs={"tokenizer": tokenizer}
        )
        print("Preparing Model Calibration Inputs...")
        cal_dataset = cal_dataset.map(
            prepare_model_inputs,
            fn_kwargs={
                "models": models,
                "embedding_layer": embedding_layer,
                "master_rot_emb": master_rot_emb,
                "num_blocks_per_chunk": num_blocks_per_chunk,
                "num_key_value_heads": config.num_key_value_heads,
                "head_dim": head_dim,
                "max_cache_size": max_cache_size,
                "eos_token_id_tensor": torch.tensor(tokenizer.eos_token_id),
                "response_cap": args.response_cap,
            },
        )

    # Keep calibration-data collection on the original logits model so prompt
    # plus decode-token calibration can select real response tokens. Replace
    # the exported final chunk with a KV-only graph only after inputs are ready.
    if args.prefill_no_lm_head and exports_final_chunk:
        # Calibration input preparation needs the original final logits to
        # collect optional response tokens. Recreate only the exported final
        # chunk afterwards, with no final norm or lm_head parameters at all.
        final_chunk_idx = args.num_chunks - 1
        final_chunk_start = sum(num_blocks_per_chunk[:-1])
        final_chunk = chunk_class(
            config,
            num_blocks_per_chunk[-1],
            chunk_idx=final_chunk_idx,
            dtype=torch.float32,
            include_tail=False,
            jit_trace=True,
        )
        final_chunk = final_chunk.load_weights(
            load_checkpoints(weight_dir), final_chunk_start
        )
        final_chunk.prefill_no_output = True
        models[-1] = final_chunk
    elif args.lm_head_shard_size:
        models[-1].lm_head = ShardedLMHead(
            models[-1].lm_head, args.lm_head_shard_size
        )

    for chunk_idx, chunk in enumerate(models):
        if args.export_chunk is not None and chunk_idx != args.export_chunk:
            continue
        chunk_start = sum(num_blocks_per_chunk[:chunk_idx])
        operator_debug_local_layer = None
        if (
            args.operator_debug_layer is not None
            and chunk_start
            <= args.operator_debug_layer
            < chunk_start + num_blocks_per_chunk[chunk_idx]
        ):
            operator_debug_local_layer = args.operator_debug_layer - chunk_start
        export_to_et_ir(
            output_folder,
            exp_name,
            chunk,
            args.precision,
            max_num_token,
            max_cache_size,
            chunk_idx,
            export_shapes,
            platform_b,
            cal_dataset,
            args.import_forever,
            args.layer_debug,
            operator_debug_local_layer,
            args.tail_debug,
            args.calibration_mode == "prompt-only",
            args.skip_mlp_output_quantization,
            args.lm_head_precision,
            args.compiler_accuracy_mode,
            args.compiler_opt_level,
            args.converter_dump,
            args.calibration_response_steps,
            args.dump_qweights,
            args.mtk_gguf_qparams_dir,
            num_blocks_per_chunk[chunk_idx],
        )


if __name__ == "__main__":
    main()
