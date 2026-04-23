# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import inspect
import json
import logging
import math
import os
import types

from collections import deque
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch

from executorch.backends.qualcomm._passes import FoldQDQ, I64toI32, TagQuantIO
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm._passes.utils import (
    get_passes_dependency_for_capture_program,
)
from executorch.backends.qualcomm.builders.utils import is_graph_output
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)
from executorch.backends.qualcomm.utils.utils import (
    convert_linear_to_conv2d,
    to_edge_transform_and_lower_to_qnn,
    update_spill_fill_size,
)
from executorch.devtools.backend_debug import print_delegation_info
from executorch.examples.models.llama.hf_download import (
    download_and_convert_hf_checkpoint,
)
from executorch.examples.models.llama.source_transformation.quantize import (
    get_quant_embedding_transform,
)
from executorch.examples.qualcomm.oss_scripts.llama import (
    LLM_VARIANT_ARCHS,
    LLMModelConfig,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    AUDIO_ENCODER,
    DECODE_QDQ_FILENAME,
    DECODER_GRAPH_NAMES,
    TEXT_DECODER,
    TEXT_ENCODER,
    TOK_EMBEDDING,
    TOK_EMBEDDING_GRAPH_NAMES,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_utils import (
    graph_module_inference,
)
from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_quant_recipe import (
    EncoderQuantRecipe,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.embedding import (
    TokenEmbedding,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaModel,
    ModelArgs,
)
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
    StaticLLMQuantRecipe,
)
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.base_component import (
    Component,
    get_model_specific_kwargs,
    log_info,
    Mode,
    process_model_args,
    Processor,
    Request,
)
from executorch.examples.qualcomm.utils import make_quantizer
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.extension.llm.custom_ops import model_sharding
from executorch.extension.llm.export.builder import DType
from torchao.prototype.spinquant import apply_spinquant
from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from transformers import AutoModel


class TextDecoder(Component):

    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
        mode: Mode,
        apply_embedding: bool = False,
    ):
        self.control_args = control_args
        self.config = config
        self.mode = mode
        self.passes_job = get_capture_program_passes()
        self.dep_table = get_passes_dependency_for_capture_program()
        self.meta = {}
        self.kv_quant_attrs_sidecar = {"output": []}
        self.quant_recipe: StaticLLMQuantRecipe = (
            self.config.quant_recipe(True) if self.config.quant_recipe else None
        )

        # For multimodal embedding
        self.apply_embedding = apply_embedding
        self.tok_embedding_passes_job = (
            get_capture_program_passes() if apply_embedding else None
        )
        self.tok_embedding_dep_table = (
            get_passes_dependency_for_capture_program() if apply_embedding else None
        )

        # load static llama model args
        params_path = (
            config.params_path if control_args.params is None else control_args.params
        )
        with open(params_path) as f:
            self.model_args = process_model_args(
                control_args, ModelArgs(**json.load(f)), self.quant_recipe, config, mode
            )
        # prepare instance
        self.tok_embedding, self.decoder = self._prepare_model()

        # check if sharding required
        if self.decoder and self.config.num_sharding > 1:
            SplitGraph, setting = model_sharding.get_split_graph_pass(
                self.meta["get_n_layers"],
                shares=self.config.num_sharding,
            )
            self.passes_job[SplitGraph] = setting
            self.dep_table[SplitGraph] = [FoldQDQ]
            self.dep_table[TagQuantIO] = [SplitGraph]

    def _prepare_model(self):  # noqa: C901
        if (instance := self._get_model_instance()) is None:
            return None, None
        tok_embedding, decoder = instance
        # load parameters for HF models
        if self.control_args.checkpoint is None:
            checkpoint = download_and_convert_hf_checkpoint(
                self.config.repo_id,
                self.config.convert_weights.__func__,
            )
            state_dict = torch.load(
                checkpoint, weights_only=True, map_location="cpu", mmap=True
            )
            if self.control_args.decoder_model in {
                "gemma-2b",
                "gemma2-2b",
                "gemma3-1b",
            }:
                for k, v in state_dict.items():
                    if "norm" not in k:
                        continue
                    # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
                    # See https://github.com/huggingface/transformers/pull/29402
                    state_dict[k] = v.float() + torch.ones(v.shape, dtype=torch.float32)
        else:
            state_dict = torch.load(
                self.control_args.checkpoint,
                weights_only=True,
                map_location="cpu",
                mmap=True,
            )
            if "model" in state_dict:
                state_dict = state_dict["model"]

            if self.control_args.decoder_model == "stories260k":
                state_dict = {
                    k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
                }

        # change to HF weight to improve the performance of RoPE in HTP backend.
        if self.config.transform_weight:

            def permute(w, heads, partial_rotary_dim):
                dim_0 = w.size(0)
                dim_1 = w.size(1)
                transformed_weight = (
                    w.view(
                        heads, -1, dim_0 // heads // 2 // partial_rotary_dim, 2, dim_1
                    )
                    .transpose(2, 3)
                    .reshape(dim_0, dim_1)
                )
                return transformed_weight

            # TODO: handle cases where input size isn't divisible.
            partial_rotary_dim = int(1 // self.model_args.partial_rotary_factor)
            for layer_i in range(decoder.n_layers):
                state_dict[f"layers.{layer_i}.attention.wq.weight"] = permute(
                    state_dict[f"layers.{layer_i}.attention.wq.weight"],
                    decoder.n_heads,
                    partial_rotary_dim,
                )
                state_dict[f"layers.{layer_i}.attention.wk.weight"] = permute(
                    state_dict[f"layers.{layer_i}.attention.wk.weight"],
                    decoder.n_kv_heads,
                    partial_rotary_dim,
                )

        decoder.load_state_dict(state_dict, strict=True, assign=True)

        # apply spin quant if required
        if any([self.config.r1, self.config.r2]):
            decoder.config = types.SimpleNamespace(
                dim=decoder.dim,
                head_dim=decoder.dim // decoder.n_heads,
                n_local_heads=decoder.n_heads,
                intermediate_size=4 * decoder.dim,
            )
            apply_spinquant(
                decoder,
                use_r1=self.config.r1,
                use_r2=self.config.r2,
                use_r4=False,
                pretrained_rotation_path=None,
                qkv_split=True,
            )

        # perform model transformation
        for layer in decoder.layers:
            if getattr(layer.attention, "prepare_attention_conv", None):
                layer.attention.prepare_attention_conv()
            if getattr(layer.feed_forward, "prepare_feedfoward_conv", None):
                layer.feed_forward.prepare_feedfoward_conv()

        decoder = convert_linear_to_conv2d(decoder)

        # check dtype override
        if self.control_args.dtype_override is not None:
            dtype_override = DType[self.control_args.dtype_override]
            decoder = decoder.to(dtype_override.to_torch_dtype())

        # check embedding fallback
        if self.control_args.embedding_quantize:
            decoder = get_quant_embedding_transform(
                embedding_quantize=self.control_args.embedding_quantize
            )(decoder)
            self.passes_job[I64toI32][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
                "skip_node"
            ] = {"tokens"}
            if self.apply_embedding:
                tok_embedding = get_quant_embedding_transform(
                    embedding_quantize=self.control_args.embedding_quantize
                )(tok_embedding)
                self.tok_embedding_passes_job[I64toI32][
                    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY
                ]["skip_node"] = {"tokens"}

        if tok_embedding is not None:
            tok_embedding = tok_embedding.eval()

        return tok_embedding, decoder.eval()

    def _get_model_instance(self) -> LlamaModel:
        if self.mode == Mode.PREFILL and self.control_args.model_mode == "kv":
            return None
        use_i64_token = self.control_args.embedding_quantize is not None

        # get embedding model
        tok_embedding = None
        if self.apply_embedding:
            auto_model = AutoModel.from_pretrained(
                self.config.repo_id, _attn_implementation="eager"
            )
            tok_embedding = TokenEmbedding(
                auto_model.get_input_embeddings().to(torch.float32),
                self.model_args.max_batch_size,
                self.model_args.ar_len,
                self.model_args.vocab_size,
                self.model_args.dim,
                use_i64_token,
            )
        # get decoder model
        if self.control_args.decoder_model in {"gemma-2b", "gemma3-1b"}:
            # For gemma, we have preprocessed the weight of rmsnorm
            self.model_args.norm_type = "rmsnorm"

        decoder: LlamaModel = LLM_VARIANT_ARCHS.get(
            self.control_args.decoder_model, LlamaModel
        )(
            self.model_args,
            ar_len=self.model_args.ar_len,
            output_new_cache_only=True,
            output_cache=True,
            use_i64_token=use_i64_token,
            **get_model_specific_kwargs(self.control_args, self.config),
        )

        self.meta = decoder.get_metadata()
        # get example input
        self.example_input = decoder.get_example_inputs()
        self.get_example_inputs = decoder.get_example_inputs
        self.export_input = (
            self.example_input[0],  # tokens or hidden_states
            *self.example_input[1],  # attn_mask
            *((self.example_input[2],) if decoder.use_kv_cache else []),  # pos_ids
            *(self.example_input[3] if decoder.use_kv_cache else []),  # k_caches
            *(self.example_input[4] if decoder.use_kv_cache else []),  # v_caches
        )
        self.io_shape = {
            # logit output
            (
                decoder.max_batch_size,
                decoder.ar_len,
                decoder.vocab_size,
            ),
        }
        # shape of k caches and v caches
        self.kv_cache_shape = {
            # single head, kv input
            (self.meta["get_head_dim"], self.meta["get_max_context_len"]),
            (self.meta["get_max_context_len"], self.meta["get_head_dim"]),
            # single head, kv output
            (self.meta["get_head_dim"], self.meta["get_ar_len"]),
            (self.meta["get_ar_len"], self.meta["get_head_dim"]),
        }

        if self.apply_embedding:
            self.tok_embedding_export_input = (
                tok_embedding.get_example_input()
            )  # tokens

        return tok_embedding, decoder

    def _save_logits_quant_attrs(self):
        for node in self.decoder.graph.nodes:
            if node.op == "output":
                for output_node in node.args[0]:
                    if (
                        output_node.target
                        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                    ):
                        source_node = output_node.args[0].args[0]
                        if source_node.meta["val"].size() in self.io_shape:
                            self.meta["get_logits_scale"] = output_node.args[1]
                            self.meta["get_logits_zero_point"] = output_node.args[2]
                            break

    def _json_safe_value(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            return value.item() if value.numel() == 1 else value.tolist()
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return item()
            except Exception:
                pass
        return str(value)

    def _classify_kv_output_kind(self, node) -> str:
        stack_trace = str(node.meta.get("stack_trace", ""))
        if "k = k.transpose(2, 3)" in stack_trace:
            return "k"
        if "v = v.view(" in stack_trace and "transpose(1, 2)" in stack_trace:
            return "v"
        raise RuntimeError(
            f"Unable to classify KV output node {node.name} from stack_trace: {stack_trace}"
        )

    def _save_output_kv_cache_quant_attrs(self):
        output_records = []
        explicit_output = {"combined": [], "k": [], "v": []}
        k_layer_idx = 0
        v_layer_idx = 0
        for node in self.decoder.graph.nodes:
            if not is_graph_output(node):
                continue
            cache_output_node = node.args[0].args[0]
            if cache_output_node.meta["val"].size()[-2:] in self.kv_cache_shape:
                # [QCOM_SCALE, QCOM_ZERO_POINT, QCOM_QUANT_MIN, QCOM_QUANT_MAX, QCOM_DTYPE]
                # This meta is for attention sink feature
                self.meta[f"get_kv_output_{len(output_records)}_quant_attr"] = [
                    node.args[1],
                    node.args[2],
                    node.args[3],
                    node.args[4],
                    str(node.args[5]),
                ]
                record = {
                    "index": len(output_records),
                    "node_name": str(cache_output_node.name),
                    "node_target": str(cache_output_node.target),
                    "scale": self._json_safe_value(node.args[1]),
                    "zero_point": self._json_safe_value(node.args[2]),
                    "quant_min": self._json_safe_value(node.args[3]),
                    "quant_max": self._json_safe_value(node.args[4]),
                    "dtype": self._json_safe_value(node.args[5]),
                    "stack_trace": str(cache_output_node.meta.get("stack_trace", "")),
                    "source_fn_stack": str(
                        cache_output_node.meta.get("source_fn_stack", "")
                    ),
                }
                output_records.append(record)
                explicit_output["combined"].append(record)

                kind = self._classify_kv_output_kind(cache_output_node)
                if kind == "k":
                    explicit_output["k"].append(
                        {
                            **record,
                            "layer_index": k_layer_idx,
                        }
                    )
                    k_layer_idx += 1
                else:
                    explicit_output["v"].append(
                        {
                            **record,
                            "layer_index": v_layer_idx,
                        }
                    )
                    v_layer_idx += 1
        if output_records:
            n_layers = self.meta.get("get_n_layers")
            assert len(explicit_output["k"]) == n_layers, (
                f"Expected {n_layers} K quant attrs, got {len(explicit_output['k'])}"
            )
            assert len(explicit_output["v"]) == n_layers, (
                f"Expected {n_layers} V quant attrs, got {len(explicit_output['v'])}"
            )
            self.kv_quant_attrs_sidecar["output"] = explicit_output

    def _dump_kv_quant_attrs_json(self):
        if not self.kv_quant_attrs_sidecar["output"]:
            return
        artifact_dir = getattr(self.control_args, "artifact", ".")
        os.makedirs(artifact_dir, exist_ok=True)
        payload = {
            "format_version": "kv-quant-attrs-v1",
            "mode": self.mode.name.lower(),
            "model_mode": getattr(self.control_args, "model_mode", None),
            "n_layers": self.meta.get("get_n_layers"),
            "head_dim": self.meta.get("get_head_dim"),
            "ar_len": self.meta.get("get_ar_len"),
            "max_context_len": self.meta.get("get_max_context_len"),
            "kv_io_bit_width": self.meta.get("get_kv_io_bit_width"),
            "output": self.kv_quant_attrs_sidecar["output"],
        }
        json_path = os.path.join(
            artifact_dir, f"{self.mode.name.lower()}_kv_quant_attrs.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        logging.info("Saved KV quant attrs JSON to %s", json_path)

    def _tag_ios(self, node, fixed_point_type):
        atten_mask_shape = {
            (
                self.meta["get_max_batch_size"],
                self.meta["get_ar_len"],
                self.meta["get_max_context_len"],
            ),
        }

        freq_shape = {
            (self.meta["get_ar_len"], self.meta["get_head_dim"] // 2),
        }

        freq_op = {
            exir_ops.edge.aten.select.int,
        }
        quant_io_type = None

        if node.op == "placeholder":
            if (
                len(users := list(node.users)) == 1
                and users[0].meta["val"].size()[-2:] in self.kv_cache_shape
            ):
                quant_io_type = fixed_point_type["kv_type"]
            elif node.meta["val"].size() in self.io_shape:
                quant_io_type = fixed_point_type["io_type"]
            elif node.meta["val"].size() in atten_mask_shape:
                quant_io_type = fixed_point_type["io_type"]

        if is_graph_output(node):
            if node.meta["val"].size()[-2:] in self.kv_cache_shape:
                quant_io_type = fixed_point_type["kv_type"]
            elif node.meta["val"].size() in self.io_shape:
                quant_io_type = fixed_point_type["io_type"]

        # tag sharding io
        if exir_ops.edge.llama.fallback.default in [
            u.target for u in list(node.users.keys())
        ] + [node.target]:
            quant_io_type = fixed_point_type["io_type"]

        # tag select op as quantized tensors for freq_sin and freq_cos. It is caused by sharding
        if node.target in freq_op and node.meta["val"].size() in freq_shape:
            quant_io_type = fixed_point_type["io_type"]

        return quant_io_type
    def _check_annotation(self, gm, keyword: str):
        print(f"\n========== CHECK ANNOTATION: {keyword} ==========\n")

        for node in gm.graph.nodes:
            if keyword in str(node):
                print(f"[NODE] {node.name}")
                print(f"  target: {node.target}")
                print(f"  has annotation: {'quantization_annotation' in node.meta}")
                if "quantization_annotation" in node.meta:
                    print(f"  annotation: {node.meta['quantization_annotation']}")
                print()
    def _dump_one_param(self, gm, name: str):
        print(f"\n========== PARAM: {name} ==========\n")

        # Read the original float parameter first.
        if name in dict(gm.named_parameters()):
            p = dict(gm.named_parameters())[name]
            print("[FLOAT PARAM]")
            print(f"  shape={tuple(p.shape)} dtype={p.dtype}")
            print(f"  min={p.min().item():.6f} max={p.max().item():.6f}")
        else:
            print("param not found in named_parameters")

        print("\n--- Related Graph Nodes ---")

        # Find related graph nodes.
        for node in gm.graph.nodes:
            if name.split(".weight")[0] in str(node):
                print(f"[NODE] {node.name}")
                print(f"  op: {node.op}")
                print(f"  target: {node.target}")
                print(f"  args: {node.args}")
                print()

    def _matches_quant_keyword(self, text: str, keyword: str | None) -> bool:
        return keyword is None or keyword in text

    def _tensor_brief(self, x: torch.Tensor):
        try:
            shape = tuple(x.shape)
            dtype = x.dtype
            if x.numel() == 0:
                return f"tensor(shape={shape}, dtype={dtype}, empty)"
            xf = x.detach()
            if xf.is_floating_point():
                mn = xf.min().item()
                mx = xf.max().item()
                return f"tensor(shape={shape}, dtype={dtype}, min={mn:.6f}, max={mx:.6f})"
            mn = xf.min().item()
            mx = xf.max().item()
            return f"tensor(shape={shape}, dtype={dtype}, min={mn}, max={mx})"
        except Exception as e:
            return f"tensor(<failed to summarize: {e}>)"

    def _obj_brief(self, obj):
        if isinstance(obj, torch.nn.Parameter):
            return f"Parameter({self._tensor_brief(obj.detach())})"
        if isinstance(obj, torch.Tensor):
            return self._tensor_brief(obj)
        return repr(obj)

    def _resolve_attr(self, root, target: str):
        obj = root
        for part in target.split('.'):
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    def _is_quant_related_node(self, node) -> bool:
        tgt = str(node.target)
        name = str(node.name)
        txt = f"{tgt} {name}".lower()
        keys = [
            'quant',
            'dequant',
            'scale',
            'zero_point',
            'qint8',
            'quint8',
            'int8',
            'uint8',
            'quantized_decomposed',
        ]
        return any(k in txt for k in keys)

    def _is_possible_qweight_obj(self, obj) -> bool:
        if isinstance(obj, torch.nn.Parameter):
            obj = obj.detach()
        if not isinstance(obj, torch.Tensor):
            return False
        return obj.dtype in (
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
        )

    def _format_node(self, node) -> str:
        try:
            return f"{node.op}:{node.name} target={node.target}"
        except Exception:
            return str(node)

    def _dump_users(self, node, logger, max_depth=2, indent='    '):
        q = deque([(node, 0)])
        seen = set()
        while q:
            cur, depth = q.popleft()
            if cur in seen or depth > max_depth:
                continue
            seen.add(cur)
            prefix = indent * depth
            if depth > 0:
                logger.info('%s[user] %s', prefix, self._format_node(cur))
            if depth < max_depth:
                for user in cur.users:
                    q.append((user, depth + 1))

    def _dump_quant_info(self, gm, tag: str, keyword: str | None = None):
        logger = getattr(self, 'logger', None)
        if logger is None:
            logger = logging.getLogger(__name__)

        logger.info('========== %s ==========', tag)
        logger.info('Quant info filter: %s', keyword)

        logger.info('----- Module attributes -----')
        for name, mod in gm.named_modules():
            if not self._matches_quant_keyword(name, keyword):
                continue

            msg = f'[module] {name}'
            if hasattr(mod, 'weight') and isinstance(
                mod.weight, (torch.Tensor, torch.nn.Parameter)
            ):
                weight = (
                    mod.weight.detach()
                    if isinstance(mod.weight, torch.nn.Parameter)
                    else mod.weight
                )
                msg += f' | weight={self._tensor_brief(weight)}'
            if hasattr(mod, 'bias') and mod.bias is not None:
                bias = (
                    mod.bias.detach()
                    if isinstance(mod.bias, torch.nn.Parameter)
                    else mod.bias
                )
                msg += f' | bias={self._tensor_brief(bias)}'
            logger.info(msg)

        logger.info('----- Parameters and buffers -----')
        for name, param in gm.named_parameters(recurse=True):
            if self._matches_quant_keyword(name, keyword):
                logger.info('[parameter] %s | %s', name, self._tensor_brief(param.detach()))
        for name, buffer in gm.named_buffers(recurse=True):
            if (
                self._matches_quant_keyword(name, keyword)
                or 'scale' in name
                or 'zero_point' in name
            ):
                logger.info('[buffer] %s | %s', name, self._tensor_brief(buffer))

        logger.info('----- Graph get_attr / quant nodes -----')
        get_attr_map = {}
        for node in gm.graph.nodes:
            if node.op == 'get_attr':
                try:
                    get_attr_map[node] = self._resolve_attr(gm, node.target)
                except Exception as e:
                    get_attr_map[node] = e

        found_any = False
        for node in gm.graph.nodes:
            text = f'{node.name} {node.target} {node.op}'
            should_print = False

            if self._matches_quant_keyword(text, keyword):
                should_print = True
            elif node.op == 'get_attr':
                obj = get_attr_map.get(node)
                if not isinstance(obj, Exception):
                    if self._is_possible_qweight_obj(obj):
                        should_print = True
                    elif isinstance(obj, torch.Tensor):
                        target = str(node.target).lower()
                        if 'scale' in target or 'zero_point' in target:
                            should_print = True
            elif self._is_quant_related_node(node):
                should_print = True

            if not should_print:
                continue

            found_any = True
            if node.op == 'get_attr':
                obj = get_attr_map.get(node)
                if isinstance(obj, Exception):
                    logger.info('[get_attr] %s -> failed to fetch: %s', node.target, obj)
                else:
                    kinds = []
                    if self._is_possible_qweight_obj(obj):
                        kinds.append('possible_qweight')
                    target = str(node.target).lower()
                    if 'scale' in target:
                        kinds.append('scale')
                    if 'zero_point' in target:
                        kinds.append('zero_point')
                    kind_str = f" ({', '.join(kinds)})" if kinds else ''
                    logger.info(
                        '[get_attr]%s %s -> %s',
                        kind_str,
                        node.target,
                        self._obj_brief(obj),
                    )
                    self._dump_users(node, logger, max_depth=2)
            else:
                logger.info(
                    '[node] op=%s name=%s target=%s args=%s kwargs=%s',
                    node.op,
                    node.name,
                    node.target,
                    node.args,
                    node.kwargs,
                )

        if not found_any:
            logger.info('No quant-related graph nodes matched.')

    def _node_module_stack_text(self, node) -> str:
        try:
            stack = node.meta.get("nn_module_stack")
            if stack is None:
                return ""
            return str(stack)
        except Exception:
            return ""

    def _extract_attr_value(self, gm, arg):
        if hasattr(arg, "op") and getattr(arg, "op", None) == "get_attr":
            try:
                return self._resolve_attr(gm, arg.target)
            except Exception as e:
                return e
        return arg

    def _force_fill_get_attr_tensor(
        self,
        gm,
        attr_name: str,
        fill_value: int = -1,
        transform_mode: str = "negate_zero",
    ):
        logger = getattr(self, "logger", None)
        if logger is None:
            logger = logging.getLogger(__name__)

        if not hasattr(gm, attr_name):
            logger.info("[force_fill] attr %s not found on module", attr_name)
            return False

        obj = getattr(gm, attr_name)
        tensor = None
        if isinstance(obj, torch.nn.Parameter):
            tensor = obj.data
        elif isinstance(obj, torch.Tensor):
            tensor = obj

        if tensor is None:
            logger.info("[force_fill] attr %s is not tensor-like: %s", attr_name, type(obj))
            return False

        with torch.no_grad():
            before_min = tensor.min().item() if tensor.numel() > 0 else None
            before_max = tensor.max().item() if tensor.numel() > 0 else None
            before_dtype = tensor.dtype

            if tensor.ndim < 1:
                logger.info(
                    "[force_fill] attr %s ndim=%d < 1, skip last-row update",
                    attr_name,
                    tensor.ndim,
                )
                return False

            if tensor.size(0) < 1:
                logger.info(
                    "[force_fill] attr %s shape=%s has no rows, skip update",
                    attr_name,
                    tuple(tensor.shape),
                )
                return False

            target = tensor[-1, ...]
            before_block = target.detach().cpu().clone()

            # Print original last-row fragment before modification.
            before_show = before_block.reshape(-1)[:32]
            logger.info(
                "[force_fill] %s original last_row first_32 before update:\n%s",
                attr_name,
                before_show.tolist(),
            )

            # Save original last row before modification.
            artifact_dir = getattr(self.control_args, "artifact", ".")
            os.makedirs(artifact_dir, exist_ok=True)
            mode_name = getattr(getattr(self, "mode", None), "name", "unknown").lower()
            safe_attr_name = "".join(
                c if c.isalnum() or c in "._-" else "_" for c in str(attr_name)
            )
            original_dump_path = os.path.join(
                artifact_dir,
                f"force_fill_original_last_row_{mode_name}_{safe_attr_name}.pt",
            )
            try:
                torch.save(target.detach().cpu().clone(), original_dump_path)
                logger.info(
                    "[force_fill] %s saved original block to %s",
                    attr_name,
                    original_dump_path,
                )
            except Exception as e:
                logger.warning(
                    "[force_fill] %s failed to save original block: %s",
                    attr_name,
                    e,
                )

            # Update rule requested by user:
            # modify only the last row: negate values and map 0 -> -1.
            updated = -target
            updated = torch.where(updated == 0, torch.full_like(updated, -1), updated)
            target.copy_(updated)

            zero_count = int(target.eq(0).sum().item()) if target.numel() > 0 else 0

            after_min = tensor.min().item() if tensor.numel() > 0 else None
            after_max = tensor.max().item() if tensor.numel() > 0 else None

        logger.info(
            "[force_fill] %s updated last_row with negate_and_zero_to_minus1 | dtype=%s shape=%s updated_rows=%d zero_count=%d | before[min=%s,max=%s] after[min=%s,max=%s]",
            attr_name,
            before_dtype,
            tuple(tensor.shape),
            1,
            zero_count,
            before_min,
            before_max,
            after_min,
            after_max,
        )
        return True

    def _iter_upstream_nodes(self, root_node):
        visited = set()
        q = deque([root_node])
        while q:
            cur = q.popleft()
            if cur in visited or not hasattr(cur, "op"):
                continue
            visited.add(cur)
            yield cur

            def _enqueue(item):
                if hasattr(item, "op"):
                    q.append(item)
                elif isinstance(item, (list, tuple)):
                    for v in item:
                        _enqueue(v)
                elif isinstance(item, dict):
                    for v in item.values():
                        _enqueue(v)

            _enqueue(getattr(cur, "args", ()))
            _enqueue(getattr(cur, "kwargs", {}))

    def _is_dequant_node(self, node) -> bool:
        return (
            hasattr(node, "op")
            and node.op == "call_function"
            and (
                "dequantize_affine" in str(node.target)
                or "dequantize_per_channel" in str(node.target)
                or "dequantize_per_tensor" in str(node.target)
                or "dequantize" in str(node.target)
            )
        )

    def _iter_node_args(self, obj):
        if hasattr(obj, "op"):
            yield obj
            return
        if isinstance(obj, (list, tuple)):
            for v in obj:
                yield from self._iter_node_args(v)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                yield from self._iter_node_args(v)

    def _get_weight_candidate_roots(self, node):
        if node.op == "call_function":
            if node.target == torch.ops.aten.conv2d.default and len(node.args) >= 2:
                return [node.args[1]]
            if node.target == torch.ops.aten.embedding.default and len(node.args) >= 1:
                return [node.args[0]]
            if node.target == torch.ops.aten.rms_norm.default:
                roots = []
                if len(node.args) >= 3:
                    roots.append(node.args[2])
                if "weight" in node.kwargs:
                    roots.append(node.kwargs["weight"])
                if roots:
                    return roots

        roots = [node]
        roots.extend(list(self._iter_node_args(node.args)))
        roots.extend(list(self._iter_node_args(node.kwargs)))
        return roots

    def _arg_to_attr_name(self, arg) -> Optional[str]:
        if hasattr(arg, "op") and getattr(arg, "op", None) == "get_attr":
            return str(arg.target)
        return None

    def _pick_scale_zero_from_dequant(self, dequant_node) -> Tuple[Optional[str], Optional[str]]:
        args = list(getattr(dequant_node, "args", ()))
        tgt = str(dequant_node.target)

        scale_attr = None
        zero_attr = None

        if "dequantize_affine" in tgt:
            # Common signatures:
            # (input, block_size, scale, zero_point, ...)
            # (input, scale, zero_point, ...)
            if len(args) >= 4:
                scale_attr = self._arg_to_attr_name(args[2])
                zero_attr = self._arg_to_attr_name(args[3])
                if scale_attr is None and len(args) >= 3:
                    scale_attr = self._arg_to_attr_name(args[1])
                    zero_attr = self._arg_to_attr_name(args[2])
            elif len(args) >= 3:
                scale_attr = self._arg_to_attr_name(args[1])
                zero_attr = self._arg_to_attr_name(args[2])
        else:
            # dequantize_per_tensor / per_channel style
            if len(args) >= 2:
                scale_attr = self._arg_to_attr_name(args[1])
            if len(args) >= 3:
                zero_attr = self._arg_to_attr_name(args[2])

        return scale_attr, zero_attr

    def _find_qweight_and_scale_attrs_for_layer(
        self, gm, layer_keyword: str
    ) -> List[Tuple[str, Optional[str]]]:
        refs = []
        seen = set()
        for node in gm.graph.nodes:
            module_text = self._node_module_stack_text(node)
            node_text = f"{node.name} {node.target} {node.op} {module_text}"
            if layer_keyword not in node_text:
                continue

            candidate_roots = self._get_weight_candidate_roots(node)
            for root in candidate_roots:
                if not hasattr(root, "op"):
                    continue
                for up in self._iter_upstream_nodes(root):
                    if not self._is_dequant_node(up) or len(up.args) == 0:
                        continue
                    qweight_attr = self._arg_to_attr_name(up.args[0])
                    if qweight_attr is None:
                        continue
                    scale_attr, _ = self._pick_scale_zero_from_dequant(up)
                    key = (qweight_attr, scale_attr)
                    if key in seen:
                        continue
                    seen.add(key)
                    refs.append(key)

        return refs

    def _find_qweight_scale_zero_attrs_for_layer(
        self, gm, layer_keyword: str
    ) -> List[Tuple[str, Optional[str], Optional[str]]]:
        refs = []
        seen = set()
        for node in gm.graph.nodes:
            module_text = self._node_module_stack_text(node)
            node_text = f"{node.name} {node.target} {node.op} {module_text}"
            if layer_keyword not in node_text:
                continue

            candidate_roots = self._get_weight_candidate_roots(node)
            for root in candidate_roots:
                if not hasattr(root, "op"):
                    continue
                for up in self._iter_upstream_nodes(root):
                    if not self._is_dequant_node(up) or len(up.args) == 0:
                        continue
                    qweight_attr = self._arg_to_attr_name(up.args[0])
                    if qweight_attr is None:
                        continue
                    scale_attr, zero_attr = self._pick_scale_zero_from_dequant(up)
                    key = (qweight_attr, scale_attr, zero_attr)
                    if key in seen:
                        continue
                    seen.add(key)
                    refs.append(key)

        return refs

    def _log_dequant_nodes_for_layer(self, gm, layer_keyword: str) -> None:
        logger = getattr(self, "logger", None)
        if logger is None:
            logger = logging.getLogger(__name__)

        hit = 0
        for node in gm.graph.nodes:
            module_text = self._node_module_stack_text(node)
            node_text = f"{node.name} {node.target} {node.op} {module_text}"
            if layer_keyword not in node_text:
                continue
            if not self._is_dequant_node(node):
                continue

            hit += 1
            logger.info(
                "[qat_post_replace][dequant] layer=%s idx=%d name=%s target=%s args=%s kwargs=%s",
                layer_keyword,
                hit,
                node.name,
                node.target,
                node.args,
                node.kwargs,
            )

        if hit == 0:
            logger.info(
                "[qat_post_replace][dequant] layer=%s no dequant node matched",
                layer_keyword,
            )

    def _dump_dequantized_weight_for_attrs(
        self,
        gm,
        *,
        layer_keyword: str,
        q_attr: str,
        s_attr: Optional[str],
        z_attr: Optional[str],
    ) -> None:
        logger = getattr(self, "logger", None)
        if logger is None:
            logger = logging.getLogger(__name__)

        try:
            q_obj = self._resolve_attr(gm, q_attr)
        except Exception as e:
            logger.warning(
                "[qat_post_replace][dequant_w] layer=%s failed to resolve q_attr=%s: %s",
                layer_keyword,
                q_attr,
                e,
            )
            return

        q_tensor = q_obj.detach() if isinstance(q_obj, torch.nn.Parameter) else q_obj
        if not isinstance(q_tensor, torch.Tensor):
            logger.warning(
                "[qat_post_replace][dequant_w] layer=%s q_attr=%s is not tensor-like",
                layer_keyword,
                q_attr,
            )
            return

        s_tensor = None
        if s_attr is not None:
            try:
                s_obj = self._resolve_attr(gm, s_attr)
                s_tensor = s_obj.detach() if isinstance(s_obj, torch.nn.Parameter) else s_obj
                if not isinstance(s_tensor, torch.Tensor):
                    s_tensor = None
            except Exception as e:
                logger.warning(
                    "[qat_post_replace][dequant_w] layer=%s failed to resolve s_attr=%s: %s",
                    layer_keyword,
                    s_attr,
                    e,
                )

        z_tensor = None
        if z_attr is not None:
            try:
                z_obj = self._resolve_attr(gm, z_attr)
                z_tensor = z_obj.detach() if isinstance(z_obj, torch.nn.Parameter) else z_obj
                if not isinstance(z_tensor, torch.Tensor):
                    z_tensor = None
            except Exception as e:
                logger.warning(
                    "[qat_post_replace][dequant_w] layer=%s failed to resolve z_attr=%s: %s",
                    layer_keyword,
                    z_attr,
                    e,
                )

        q_for_dequant = q_tensor.to(torch.float32)
        if q_for_dequant.ndim >= 4 and tuple(q_for_dequant.shape[-2:]) == (1, 1):
            q_for_dequant = q_for_dequant.squeeze(-1).squeeze(-1)

        if q_for_dequant.ndim != 2:
            logger.warning(
                "[qat_post_replace][dequant_w] layer=%s unsupported q shape=%s (expect 2D or 4D 1x1)",
                layer_keyword,
                tuple(q_for_dequant.shape),
            )
            return

        q_target_shape = tuple(q_for_dequant.shape)

        def _expand_to_q_shape(name: str, tensor: Optional[torch.Tensor], fill: float) -> torch.Tensor:
            if tensor is None:
                return torch.full_like(q_for_dequant, fill_value=fill, dtype=torch.float32)

            t = tensor.to(torch.float32)
            if t.ndim >= 4 and tuple(t.shape[-2:]) == (1, 1):
                t = t.squeeze(-1).squeeze(-1)

            # already aligned
            if t.ndim == 2 and tuple(t.shape) == q_target_shape:
                return t.contiguous()

            # (out, groups) -> (out, in)
            if t.ndim == 2 and t.shape[0] == q_target_shape[0]:
                groups = int(t.shape[1])
                if groups > 0:
                    group_size = int(math.ceil(q_target_shape[1] / groups))
                    expanded = t.repeat_interleave(group_size, dim=1)[:, : q_target_shape[1]]
                    return expanded.contiguous()

            # (groups, out) -> (out, in)
            if t.ndim == 2 and t.shape[1] == q_target_shape[0]:
                groups = int(t.shape[0])
                if groups > 0:
                    t_t = t.transpose(0, 1).contiguous()
                    group_size = int(math.ceil(q_target_shape[1] / groups))
                    expanded = t_t.repeat_interleave(group_size, dim=1)[:, : q_target_shape[1]]
                    return expanded.contiguous()

            # (out,) -> (out, in)
            if t.ndim == 1 and t.shape[0] == q_target_shape[0]:
                return t.unsqueeze(1).expand(q_target_shape[0], q_target_shape[1]).contiguous()

            # (in,) -> (out, in)
            if t.ndim == 1 and t.shape[0] == q_target_shape[1]:
                return t.unsqueeze(0).expand(q_target_shape[0], q_target_shape[1]).contiguous()

            logger.warning(
                "[qat_post_replace][dequant_w] layer=%s unable to align %s shape=%s to q shape=%s",
                layer_keyword,
                name,
                tuple(t.shape),
                q_target_shape,
            )
            raise RuntimeError(f"unable to align {name}")

        if s_tensor is not None:
            s_for_dequant = _expand_to_q_shape("s", s_tensor, fill=1.0)
        else:
            s_for_dequant = _expand_to_q_shape("s", None, fill=1.0)

        if z_tensor is not None:
            z_for_dequant = _expand_to_q_shape("z", z_tensor, fill=0.0)
        else:
            z_for_dequant = _expand_to_q_shape("z", None, fill=0.0)

        try:
            dequant_w = (q_for_dequant - z_for_dequant) * s_for_dequant
        except Exception as e:
            logger.warning(
                "[qat_post_replace][dequant_w] layer=%s broadcast failed q=%s s=%s z=%s err=%s",
                layer_keyword,
                tuple(q_for_dequant.shape),
                tuple(s_for_dequant.shape),
                tuple(z_for_dequant.shape),
                e,
            )
            return

        dequant_w_cpu = dequant_w.detach().cpu().contiguous()
        artifact_dir = getattr(self.control_args, "artifact", ".")
        dump_dir = os.path.join(artifact_dir, "qat_dequant_weight_dump")
        os.makedirs(dump_dir, exist_ok=True)

        layer_safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in layer_keyword)
        q_safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in q_attr)
        pt_path = os.path.join(dump_dir, f"{layer_safe}__{q_safe}__dequant.pt")
        w8_path = os.path.join(dump_dir, f"{layer_safe}__{q_safe}__dequant_first8x8.json")
        q8_path = os.path.join(dump_dir, f"{layer_safe}__{q_safe}__qweight_first8x8.json")
        s8_path = os.path.join(dump_dir, f"{layer_safe}__{q_safe}__scale_first8x8.json")
        z8_path = os.path.join(dump_dir, f"{layer_safe}__{q_safe}__qzero_first8x8.json")

        torch.save(
            {
                "layer": layer_keyword,
                "q_attr": q_attr,
                "s_attr": s_attr,
                "z_attr": z_attr,
                "q": q_for_dequant.detach().cpu().contiguous(),
                "s": s_for_dequant.detach().cpu().contiguous(),
                "z": z_for_dequant.detach().cpu().contiguous(),
                "s_raw": s_tensor.detach().cpu().contiguous() if isinstance(s_tensor, torch.Tensor) else None,
                "z_raw": z_tensor.detach().cpu().contiguous() if isinstance(z_tensor, torch.Tensor) else None,
                "dequant_weight": dequant_w_cpu,
            },
            pt_path,
        )

        def _first_8x8(t: torch.Tensor) -> List[List[float]]:
            t_cpu = t.detach().cpu()
            if t_cpu.ndim == 0:
                return [[float(t_cpu.item())]]
            if t_cpu.ndim == 1:
                cols = min(8, int(t_cpu.shape[0]))
                return [t_cpu[:cols].to(torch.float32).tolist()]
            rows = min(8, int(t_cpu.shape[0]))
            cols = min(8, int(t_cpu.shape[1]))
            return t_cpu[:rows, :cols].to(torch.float32).tolist()

        q_8x8 = _first_8x8(q_for_dequant)
        s_8x8 = _first_8x8(s_for_dequant)
        z_8x8 = _first_8x8(z_for_dequant)
        w_8x8 = _first_8x8(dequant_w_cpu)

        with open(q8_path, "w", encoding="utf-8") as f:
            json.dump(q_8x8, f, ensure_ascii=False)
        with open(s8_path, "w", encoding="utf-8") as f:
            json.dump(s_8x8, f, ensure_ascii=False)
        with open(z8_path, "w", encoding="utf-8") as f:
            json.dump(z_8x8, f, ensure_ascii=False)
        with open(w8_path, "w", encoding="utf-8") as f:
            json.dump(w_8x8, f, ensure_ascii=False)

        logger.info("[qat_post_replace][dequant_w][8x8] qweight=%s", q_8x8)
        logger.info("[qat_post_replace][dequant_w][8x8] scale=%s", s_8x8)
        logger.info("[qat_post_replace][dequant_w][8x8] qzero=%s", z_8x8)
        logger.info("[qat_post_replace][dequant_w][8x8] weight=%s", w_8x8)

        w_min = dequant_w_cpu.min().item() if dequant_w_cpu.numel() > 0 else None
        w_max = dequant_w_cpu.max().item() if dequant_w_cpu.numel() > 0 else None
        logger.info(
            "[qat_post_replace][dequant_w] layer=%s q_attr=%s s_attr=%s z_attr=%s q_shape=%s s_shape=%s z_shape=%s w_shape=%s min=%s max=%s",
            layer_keyword,
            q_attr,
            s_attr,
            z_attr,
            tuple(q_for_dequant.shape),
            tuple(s_for_dequant.shape),
            tuple(z_for_dequant.shape),
            tuple(dequant_w_cpu.shape),
            w_min,
            w_max,
        )
        logger.info(
            "[qat_post_replace][dequant_w] saved tensor=%s first8x8[w]=%s first8x8[qweight]=%s first8x8[scale]=%s first8x8[qzero]=%s",
            pt_path,
            w8_path,
            q8_path,
            s8_path,
            z8_path,
        )

    def _log_dequant_node_and_dump_weight_for_attrs(
        self,
        gm,
        *,
        layer_keyword: str,
        q_attr: str,
        s_attr: Optional[str],
        z_attr: Optional[str],
    ) -> None:
        logger = getattr(self, "logger", None)
        if logger is None:
            logger = logging.getLogger(__name__)

        hit = 0
        for node in gm.graph.nodes:
            if not self._is_dequant_node(node) or len(getattr(node, "args", ())) == 0:
                continue

            q_attr_node = self._arg_to_attr_name(node.args[0])
            s_attr_node, z_attr_node = self._pick_scale_zero_from_dequant(node)
            if q_attr_node != q_attr:
                continue
            if s_attr is not None and s_attr_node != s_attr:
                continue

            hit += 1
            logger.info(
                "[qat_post_replace][dequant] layer=%s idx=%d name=%s target=%s q_attr=%s s_attr=%s z_attr=%s args=%s kwargs=%s",
                layer_keyword,
                hit,
                node.name,
                node.target,
                q_attr_node,
                s_attr_node,
                z_attr_node,
                node.args,
                node.kwargs,
            )

        if hit == 0:
            logger.info(
                "[qat_post_replace][dequant] layer=%s no dequant node matched by attrs q=%s s=%s z=%s",
                layer_keyword,
                q_attr,
                s_attr,
                z_attr,
            )

        self._dump_dequantized_weight_for_attrs(
            gm,
            layer_keyword=layer_keyword,
            q_attr=q_attr,
            s_attr=s_attr,
            z_attr=z_attr,
        )

    def _verify_qat_chain_alignment_for_attr(
        self,
        gm,
        *,
        layer_keyword: str,
        source_base: str,
        q_attr: str,
        s_attr: Optional[str],
        z_attr: Optional[str],
        q_src: torch.Tensor,
        z_src: Optional[torch.Tensor],
        s_src: torch.Tensor,
        qweight_mode: str,
        bits_hint: int,
        group_size_hint: int,
        zero_points_src: Optional[torch.Tensor] = None,
    ) -> None:
        logger = getattr(self, "logger", None)
        if logger is None:
            logger = logging.getLogger(__name__)

        q_obj = self._resolve_attr(gm, q_attr)
        q_target = q_obj.detach() if isinstance(q_obj, torch.nn.Parameter) else q_obj
        if not isinstance(q_target, torch.Tensor):
            logger.warning(
                "[qat_verify] q attr %s is not tensor-like (layer=%s)",
                q_attr,
                layer_keyword,
            )
            return

        q_expected = self._prepare_qat_qweight_for_target(
            q_packed=q_src,
            z_packed=z_src,
            target=q_target,
            qweight_mode=qweight_mode,
            bits_hint=bits_hint,
            group_size_hint=group_size_hint,
        )
        if q_expected is None:
            logger.warning(
                "[qat_verify] failed to prepare expected qweight for %s (%s)",
                layer_keyword,
                q_attr,
            )
            return
        q_expected_aligned, q_align_mode = self._align_tensor_to_target_shape(
            q_expected,
            q_target,
            return_mode=True,
        )
        if q_expected_aligned is None:
            logger.warning(
                "[qat_verify] qweight shape mismatch for %s attr=%s expected=%s target=%s",
                layer_keyword,
                q_attr,
                tuple(q_expected.shape),
                tuple(q_target.shape),
            )
            return
        q_expected_cast = q_expected_aligned.to(dtype=q_target.dtype, device=q_target.device)
        q_diff_max = (q_target.to(torch.float32) - q_expected_cast.to(torch.float32)).abs().max().item()

        s_diff_max = None
        s_align_mode = None
        s_min = None
        s_max = None
        s_neg = None
        if s_attr is not None:
            s_obj = self._resolve_attr(gm, s_attr)
            s_target = s_obj.detach() if isinstance(s_obj, torch.nn.Parameter) else s_obj
            if isinstance(s_target, torch.Tensor):
                s_expected = self._prepare_qat_scale_for_target(s_src, s_target)
                s_expected_aligned, s_align_mode = self._align_tensor_to_target_shape(
                    s_expected,
                    s_target,
                    return_mode=True,
                )
                if s_expected_aligned is not None:
                    s_expected_cast = s_expected_aligned.to(dtype=s_target.dtype, device=s_target.device)
                    s_diff_max = (
                        (s_target.to(torch.float32) - s_expected_cast.to(torch.float32))
                        .abs()
                        .max()
                        .item()
                    )
                    s_min = s_target.min().item() if s_target.numel() > 0 else None
                    s_max = s_target.max().item() if s_target.numel() > 0 else None
                    s_neg = int((s_target < 0).sum().item()) if s_target.numel() > 0 else 0

        z_nonzero = None
        z_shape = None
        if z_attr is not None and hasattr(gm, z_attr):
            z_obj = self._resolve_attr(gm, z_attr)
            z_target = z_obj.detach() if isinstance(z_obj, torch.nn.Parameter) else z_obj
            if isinstance(z_target, torch.Tensor):
                z_nonzero = int((z_target != 0).sum().item()) if z_target.numel() > 0 else 0
                z_shape = tuple(z_target.shape)

        logger.info(
            "[qat_verify][layer] target=%s source=%s q_attr=%s s_attr=%s z_attr=%s | q_diff_max=%s s_diff_max=%s | align[q=%s,s=%s] | s[min=%s,max=%s,neg=%s] z[shape=%s,nonzero=%s]",
            layer_keyword,
            source_base,
            q_attr,
            s_attr,
            z_attr,
            q_diff_max,
            s_diff_max,
            q_align_mode,
            s_align_mode,
            s_min,
            s_max,
            s_neg,
            z_shape,
            z_nonzero,
        )
        if q_align_mode == "numel_reshape" or s_align_mode == "numel_reshape":
            logger.warning(
                "[qat_verify][layout] numel-based reshape used for layer=%s (q_align=%s s_align=%s); this may hide layout mismatch",
                layer_keyword,
                q_align_mode,
                s_align_mode,
            )

        self._verify_qat_weight_reconstruction(
            layer_keyword=layer_keyword,
            source_base=source_base,
            q_src=q_src,
            z_src=z_src,
            s_src=s_src,
            qweight_mode=qweight_mode,
            bits_hint=bits_hint,
            group_size_hint=group_size_hint,
            zero_points_src=zero_points_src,
        )

    def _align_tensor_to_target_shape(
        self,
        src: torch.Tensor,
        target: torch.Tensor,
        *,
        return_mode: bool = False,
    ):
        mode = "none"
        if tuple(src.shape) == tuple(target.shape):
            mode = "exact"
            return (src, mode) if return_mode else src
        if src.numel() == target.numel():
            try:
                mode = "numel_reshape"
                aligned = src.reshape(target.shape)
                return (aligned, mode) if return_mode else aligned
            except Exception:
                pass
        if target.ndim >= 2 and tuple(target.shape[-2:]) == (1, 1):
            if tuple(src.shape) == tuple(target.shape[:-2]):
                mode = "unsqueeze_hw"
                aligned = src.unsqueeze(-1).unsqueeze(-1)
                return (aligned, mode) if return_mode else aligned
            # common scale layout: src=(groups,out), target=(out,groups,1,1)
            if src.ndim == 2 and tuple((src.shape[1], src.shape[0])) == tuple(
                target.shape[:-2]
            ):
                mode = "transpose_unsqueeze_hw"
                aligned = src.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
                return (aligned, mode) if return_mode else aligned
        # common scale layout without trailing spatial dims: src=(groups,out), target=(out,groups)
        if src.ndim == 2 and target.ndim == 2 and tuple((src.shape[1], src.shape[0])) == tuple(
            target.shape
        ):
            mode = "transpose_2d"
            aligned = src.transpose(0, 1)
            return (aligned, mode) if return_mode else aligned
        return (None, mode) if return_mode else None

    def _normalize_qat_scale_shape(
        self,
        s_src: torch.Tensor,
        *,
        num_groups: int,
        out_features: int,
    ) -> Optional[torch.Tensor]:
        if s_src.ndim != 2:
            return None
        if tuple(s_src.shape) == (num_groups, out_features):
            return s_src
        if tuple(s_src.shape) == (out_features, num_groups):
            return s_src.transpose(0, 1).contiguous()
        return None

    def _effqat_expand_group_rows(
        self,
        values: torch.Tensor,
        *,
        in_features: int,
        group_size: int,
    ) -> Tuple[torch.Tensor, int]:
        if group_size <= 0:
            return values, 0
        remainder = in_features % group_size
        if remainder == 0:
            return values, 0
        padded_rows = group_size - remainder
        if values.numel() == 0:
            return values, padded_rows
        pad_block = torch.zeros(
            (padded_rows, values.shape[1]),
            dtype=values.dtype,
            device=values.device,
        )
        return torch.cat([values, pad_block], dim=0), padded_rows

    def _effqat_reconstruct_weight(
        self,
        *,
        q_unpacked: torch.Tensor,
        z_unpacked: torch.Tensor,
        s_group: torch.Tensor,
        in_features: int,
        out_features: int,
        group_size: int,
        use_pad_unpad: bool,
    ) -> torch.Tensor:
        clip_min = float(getattr(self.control_args, "qat_scale_clip_min", 1e-4))
        clip_max = float(getattr(self.control_args, "qat_scale_clip_max", 1e4))
        if clip_max <= clip_min:
            clip_max = max(clip_min * 10.0, 1.0)

        s_group = torch.clamp(s_group, min=clip_min, max=clip_max).to(torch.float32)
        q_unpacked = q_unpacked.to(torch.float32)
        z_unpacked = z_unpacked.to(torch.float32)

        if use_pad_unpad:
            q_work, padded_rows = self._effqat_expand_group_rows(
                q_unpacked,
                in_features=in_features,
                group_size=group_size,
            )
            rows_needed = q_work.shape[0]
            groups_needed = rows_needed // max(1, group_size)

            z_work = z_unpacked
            s_work = s_group
            if z_work.shape[0] < groups_needed:
                z_pad = torch.zeros(
                    (groups_needed - z_work.shape[0], z_work.shape[1]),
                    dtype=z_work.dtype,
                    device=z_work.device,
                )
                z_work = torch.cat([z_work, z_pad], dim=0)
            if s_work.shape[0] < groups_needed:
                s_pad = torch.zeros(
                    (groups_needed - s_work.shape[0], s_work.shape[1]),
                    dtype=s_work.dtype,
                    device=s_work.device,
                )
                s_work = torch.cat([s_work, s_pad], dim=0)

            z_work = z_work[:groups_needed, :]
            s_work = s_work[:groups_needed, :]

            w = (
                (q_work.view(-1, group_size, out_features) - z_work.view(-1, 1, out_features))
                * s_work.view(-1, 1, out_features)
            ).reshape(rows_needed, out_features)
            if padded_rows > 0:
                w = w[:-padded_rows, :]
            return w.contiguous()

        z_expand = z_unpacked.repeat_interleave(group_size, dim=0)
        s_expand = s_group.repeat_interleave(group_size, dim=0)
        if z_expand.shape[0] < in_features:
            pad_rows = in_features - z_expand.shape[0]
            z_expand = torch.cat(
                [
                    z_expand,
                    torch.zeros(
                        (pad_rows, z_expand.shape[1]),
                        dtype=z_expand.dtype,
                        device=z_expand.device,
                    ),
                ],
                dim=0,
            )
        if s_expand.shape[0] < in_features:
            pad_rows = in_features - s_expand.shape[0]
            s_expand = torch.cat(
                [
                    s_expand,
                    torch.zeros(
                        (pad_rows, s_expand.shape[1]),
                        dtype=s_expand.dtype,
                        device=s_expand.device,
                    ),
                ],
                dim=0,
            )
        return ((q_unpacked - z_expand[:in_features, :]) * s_expand[:in_features, :]).contiguous()

    def _unpack_cols_reference(
        self,
        packed: torch.Tensor,
        bits: int,
        rows: int,
        cols: int,
    ) -> torch.Tensor:
        pack_factor = self._pack_factor(bits)
        shifts = torch.arange(
            0,
            pack_factor * bits,
            bits,
            dtype=torch.int64,
            device=packed.device,
        )
        unpacked = (
            packed.to(torch.int64).unsqueeze(-1)
            >> shifts.view(1, 1, pack_factor)
        ) & ((1 << bits) - 1)
        return unpacked.reshape(rows, -1)[:, :cols].to(torch.float32).contiguous()

    def _verify_qat_weight_reconstruction(
        self,
        *,
        layer_keyword: str,
        source_base: str,
        q_src: torch.Tensor,
        z_src: Optional[torch.Tensor],
        s_src: torch.Tensor,
        qweight_mode: str,
        bits_hint: int,
        group_size_hint: int,
        zero_points_src: Optional[torch.Tensor],
    ) -> None:
        logger = getattr(self, "logger", None)
        if logger is None:
            logger = logging.getLogger(__name__)

        if z_src is None:
            logger.info(
                "[qat_verify][w_ref] skip layer=%s source=%s because z_src is missing",
                layer_keyword,
                source_base,
            )
            return

        if q_src.ndim != 2 or z_src.ndim != 2:
            logger.warning(
                "[qat_verify][w_ref] skip layer=%s invalid q/z dims q=%s z=%s",
                layer_keyword,
                tuple(q_src.shape),
                tuple(z_src.shape),
            )
            return

        out_features = int(q_src.shape[1])
        num_groups = int(z_src.shape[0])
        group_size = group_size_hint if group_size_hint > 0 else 32
        in_features = int(num_groups * group_size)
        bits = self._infer_bits_from_packed_rows(
            in_features,
            int(q_src.shape[0]),
            bits_hint,
        )
        in_features = int(q_src.shape[0] * self._pack_factor(bits))

        q_unpacked = self._unpack_rows(q_src, bits, in_features, out_features)
        z_unpacked = self._unpack_cols(z_src, bits, num_groups, out_features)
        z_unpacked_ref = self._unpack_cols_reference(z_src, bits, num_groups, out_features)
        z_unpack_diff = (z_unpacked - z_unpacked_ref).abs().max().item() if z_unpacked.numel() > 0 else 0.0

        maxq = 2**bits - 1
        z_rounded_from_unpacked = self._rounded_zero_points_from_unpacked(z_unpacked, bits)
        z_round_diff = None
        if zero_points_src is not None and isinstance(zero_points_src, torch.Tensor):
            if zero_points_src.ndim == 2:
                z_zero_points = zero_points_src
                if tuple(z_zero_points.shape) == (out_features, num_groups):
                    z_zero_points = z_zero_points.transpose(0, 1).contiguous()
                if tuple(z_zero_points.shape) == tuple(z_unpacked.shape):
                    z_rounded_ref = torch.clamp(torch.round(z_zero_points), 0, maxq).to(torch.float32)
                    z_round_diff = (
                        z_unpacked.to(torch.float32) - z_rounded_ref.to(torch.float32)
                    ).abs().max().item()

        s_group = self._normalize_qat_scale_shape(
            s_src,
            num_groups=num_groups,
            out_features=out_features,
        )
        if s_group is None:
            logger.warning(
                "[qat_verify][w_ref] skip layer=%s invalid scale shape=%s expected (%d,%d) or (%d,%d)",
                layer_keyword,
                tuple(s_src.shape),
                num_groups,
                out_features,
                out_features,
                num_groups,
            )
            return

        w_ref = self._effqat_reconstruct_weight(
            q_unpacked=q_unpacked,
            z_unpacked=z_rounded_from_unpacked,
            s_group=s_group,
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
            use_pad_unpad=True,
        )

        w_mine = self._effqat_reconstruct_weight(
            q_unpacked=q_unpacked,
            z_unpacked=z_rounded_from_unpacked if qweight_mode == "qweight_minus_qzeros" else torch.zeros_like(z_rounded_from_unpacked),
            s_group=s_group,
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
            use_pad_unpad=False,
        )

        w_diff_max = (w_ref - w_mine).abs().max().item() if w_ref.numel() > 0 else 0.0
        q_min = q_unpacked.min().item() if q_unpacked.numel() > 0 else None
        q_max = q_unpacked.max().item() if q_unpacked.numel() > 0 else None
        z_min = z_unpacked.min().item() if z_unpacked.numel() > 0 else None
        z_max = z_unpacked.max().item() if z_unpacked.numel() > 0 else None

        logger.info(
            "[qat_verify][reconstruct] target=%s source=%s | bits=%d group=%d in=%d out=%d groups=%d | z_unpack_diff=%s z_round_diff=%s | q[min=%s,max=%s] z[min=%s,max=%s] | W_mine_vs_ref_max=%s",
            layer_keyword,
            source_base,
            bits,
            group_size,
            in_features,
            out_features,
            num_groups,
            z_unpack_diff,
            z_round_diff,
            q_min,
            q_max,
            z_min,
            z_max,
            w_diff_max,
        )

        if z_unpack_diff > 0:
            logger.warning(
                "[qat_verify][reconstruct] z unpack mismatch detected for layer=%s diff=%s",
                layer_keyword,
                z_unpack_diff,
            )
        if w_diff_max > 0:
            logger.warning(
                "[qat_verify][reconstruct] W reconstruction mismatch for layer=%s max_diff=%s",
                layer_keyword,
                w_diff_max,
            )

    def _pack_factor(self, bits: int) -> int:
        if bits not in {2, 3, 4, 8}:
            raise NotImplementedError(f"unsupported bits={bits}")
        return 32 // bits

    def _rounded_zero_points_from_unpacked(self, z_unpacked: torch.Tensor, bits: int) -> torch.Tensor:
        maxq = 2**bits - 1
        return torch.clamp(torch.round(z_unpacked), min=0, max=maxq)

    def _unpack_rows(self, packed: torch.Tensor, bits: int, rows: int, cols: int) -> torch.Tensor:
        pack_factor = self._pack_factor(bits)
        packed = packed.to(torch.int64)
        unpacked = torch.zeros((rows, cols), dtype=torch.int64, device=packed.device)
        maxq = 2**bits - 1
        for offset in range(pack_factor):
            if offset >= rows:
                break
            row_indices = torch.arange(offset, rows, pack_factor, device=packed.device)
            if row_indices.numel() == 0:
                continue
            packed_rows = row_indices // pack_factor
            unpacked[row_indices] = (packed[packed_rows] >> (bits * offset)) & maxq
        return unpacked.to(torch.float16)

    def _unpack_cols(self, packed: torch.Tensor, bits: int, rows: int, cols: int) -> torch.Tensor:
        pack_factor = self._pack_factor(bits)
        packed = packed.to(torch.int64)
        unpacked = torch.zeros((rows, cols), dtype=torch.int64, device=packed.device)
        maxq = 2**bits - 1
        for offset in range(pack_factor):
            if offset >= cols:
                break
            col_indices = torch.arange(offset, cols, pack_factor, device=packed.device)
            if col_indices.numel() == 0:
                continue
            packed_cols = col_indices // pack_factor
            unpacked[:, col_indices] = (packed[:, packed_cols] >> (bits * offset)) & maxq
        return unpacked.to(torch.float16)

    def _infer_bits_from_packed_rows(
        self, in_features: int, packed_rows: int, fallback_bits: int
    ) -> int:
        for bits in (2, 3, 4, 8):
            pack_factor = self._pack_factor(bits)
            if math.ceil(in_features / pack_factor) == packed_rows:
                return bits
        return fallback_bits

    def _infer_group_size_from_qparams(
        self,
        target: torch.Tensor,
        z_packed: Optional[torch.Tensor],
        s_src: Optional[torch.Tensor],
        fallback_group_size: int,
    ) -> int:
        if target.ndim >= 4 and tuple(target.shape[-2:]) == (1, 1):
            out_features = int(target.shape[0])
            in_features = int(target.shape[1])
        elif target.ndim == 2:
            out_features = int(target.shape[0])
            in_features = int(target.shape[1])
        else:
            return fallback_group_size

        candidates = []
        if isinstance(z_packed, torch.Tensor) and z_packed.ndim == 2:
            candidates.append(z_packed)
        if isinstance(s_src, torch.Tensor) and s_src.ndim == 2:
            candidates.append(s_src)

        for candidate in candidates:
            shape = tuple(candidate.shape)
            if shape[0] > 0 and shape[1] == out_features:
                num_groups = int(shape[0])
            elif shape[1] > 0 and shape[0] == out_features:
                num_groups = int(shape[1])
            else:
                continue
            if num_groups > 0:
                inferred = int(math.ceil(in_features / num_groups))
                if inferred > 0:
                    return inferred

        return fallback_group_size

    def _prepare_qat_qweight_for_target(
        self,
        q_packed: torch.Tensor,
        z_packed: Optional[torch.Tensor],
        target: torch.Tensor,
        qweight_mode: str,
        bits_hint: int,
        group_size_hint: int,
    ) -> Optional[torch.Tensor]:
        if target.ndim >= 4 and tuple(target.shape[-2:]) == (1, 1):
            out_features = int(target.shape[0])
            in_features = int(target.shape[1])
        elif target.ndim == 2:
            out_features = int(target.shape[0])
            in_features = int(target.shape[1])
        else:
            return None

        if q_packed.ndim != 2:
            return None

        bits = self._infer_bits_from_packed_rows(in_features, int(q_packed.shape[0]), bits_hint)
        q_unpacked = self._unpack_rows(q_packed, bits, in_features, out_features).to(torch.float16)

        if qweight_mode == "qweight_minus_qzeros" and z_packed is not None:
            num_groups = int(z_packed.shape[0])
            z_unpacked = self._unpack_cols(z_packed, bits, num_groups, out_features)
            z_unpacked = self._rounded_zero_points_from_unpacked(
                z_unpacked.to(torch.float32), bits
            ).to(torch.float16)
            group_size = group_size_hint if group_size_hint > 0 else max(1, in_features // max(1, num_groups))
            z_expanded = z_unpacked.repeat_interleave(group_size, dim=0)
            if z_expanded.shape[0] < in_features:
                pad_rows = in_features - z_expanded.shape[0]
                z_expanded = torch.cat(
                    [z_expanded, z_expanded[-1:, :].repeat(pad_rows, 1)], dim=0
                )
            z_expanded = z_expanded[:in_features, :]
            q_unpacked = q_unpacked - z_expanded

        q_target = q_unpacked.transpose(0, 1).contiguous()  # (out, in)
        if target.ndim >= 4 and tuple(target.shape[-2:]) == (1, 1):
            q_target = q_target.unsqueeze(-1).unsqueeze(-1)
        return q_target

    def _prepare_qat_scale_for_target(self, s_src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        clip_min = float(getattr(self.control_args, "qat_scale_clip_min", 1e-4))
        clip_max = float(getattr(self.control_args, "qat_scale_clip_max", 1e4))
        if clip_max <= clip_min:
            clip_max = max(clip_min * 10.0, 1.0)
        s_src = torch.clamp(s_src, min=clip_min, max=clip_max)

        if target.ndim == 1:
            if s_src.ndim == 2 and tuple(s_src.shape) == (1, target.shape[0]):
                return s_src.squeeze(0).contiguous()
            if s_src.ndim == 2 and tuple(s_src.shape) == (target.shape[0], 1):
                return s_src.squeeze(1).contiguous()

        if target.ndim >= 4 and tuple(target.shape[-2:]) == (1, 1):
            if s_src.ndim == 2 and tuple((s_src.shape[1], s_src.shape[0])) == tuple(
                target.shape[:-2]
            ):
                return s_src.transpose(0, 1).unsqueeze(-1).unsqueeze(-1).contiguous()
        if target.ndim == 2 and s_src.ndim == 2 and tuple((s_src.shape[1], s_src.shape[0])) == tuple(
            target.shape
        ):
            return s_src.transpose(0, 1).contiguous()
        return s_src

    def _replace_attr_tensor(
        self,
        gm,
        attr_name: str,
        src_tensor: torch.Tensor,
        *,
        debug: bool = False,
        context: str = "",
    ) -> bool:
        logger = getattr(self, "logger", None)
        if logger is None:
            logger = logging.getLogger(__name__)

        if not hasattr(gm, attr_name):
            logger.warning("[qat_replace] attr %s not found", attr_name)
            return False

        obj = getattr(gm, attr_name)
        target = obj.data if isinstance(obj, torch.nn.Parameter) else obj
        if not isinstance(target, torch.Tensor):
            logger.warning("[qat_replace] attr %s is not tensor-like", attr_name)
            return False

        if tuple(src_tensor.shape) == tuple(target.shape):
            aligned = src_tensor
        elif (
            src_tensor.ndim == 2
            and target.ndim == 4
            and tuple(target.shape[-2:]) == (1, 1)
            and tuple(src_tensor.shape) == tuple(target.shape[:-2])
        ):
            aligned = src_tensor.unsqueeze(-1).unsqueeze(-1)
        else:
            msg = (
                "[qat_replace] shape mismatch attr=%s src=%s target=%s context=%s; "
                "only exact match or 2D->4D (with trailing 1x1) expansion is allowed"
            )
            logger.error(msg, attr_name, tuple(src_tensor.shape), tuple(target.shape), context)
            raise ValueError(
                msg
                % (attr_name, tuple(src_tensor.shape), tuple(target.shape), context)
            )

        before_min = target.min().item() if target.numel() > 0 else None
        before_max = target.max().item() if target.numel() > 0 else None
        aligned = aligned.to(dtype=target.dtype, device=target.device)
        with torch.no_grad():
            target.copy_(aligned)

        after_min = target.min().item() if target.numel() > 0 else None
        after_max = target.max().item() if target.numel() > 0 else None
        if debug:
            logger.info(
                "[qat_replace][attr] %s | context=%s | src_shape=%s src_dtype=%s | target_shape=%s target_dtype=%s | before[min=%s,max=%s] after[min=%s,max=%s]",
                attr_name,
                context,
                tuple(src_tensor.shape),
                src_tensor.dtype,
                tuple(target.shape),
                target.dtype,
                before_min,
                before_max,
                after_min,
                after_max,
            )
        else:
            logger.info(
                "[qat_replace][attr] %s replaced (context=%s)",
                attr_name,
                context,
            )
        return True

    def _tensor_fingerprint(self, tensor: torch.Tensor, sample_cap: int = 4096) -> Dict[str, Any]:
        det = tensor.detach()
        numel = int(det.numel())
        if numel == 0:
            return {
                "shape": tuple(det.shape),
                "dtype": str(det.dtype),
                "numel": 0,
                "sample_numel": 0,
                "sample_min": None,
                "sample_max": None,
                "sample_mean": None,
                "sample_std": None,
                "sample_sum": None,
            }

        flat = det.reshape(-1)
        if numel > sample_cap:
            stride = max(1, numel // sample_cap)
            sample = flat[::stride][:sample_cap]
        else:
            sample = flat
        sample_f = sample.to(torch.float32)

        return {
            "shape": tuple(det.shape),
            "dtype": str(det.dtype),
            "numel": numel,
            "sample_numel": int(sample_f.numel()),
            "sample_min": float(sample_f.min().item()),
            "sample_max": float(sample_f.max().item()),
            "sample_mean": float(sample_f.mean().item()),
            "sample_std": float(sample_f.std(unbiased=False).item()),
            "sample_sum": float(sample_f.sum().item()),
        }

    def _snapshot_attr_fingerprints(
        self,
        gm,
        attr_names: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        snapshots: Dict[str, Dict[str, Any]] = {}
        for attr_name in attr_names:
            if not hasattr(gm, attr_name):
                snapshots[attr_name] = {"missing": True}
                continue
            try:
                obj = self._resolve_attr(gm, attr_name)
            except Exception as error:
                snapshots[attr_name] = {"error": str(error)}
                continue
            tensor = obj.detach() if isinstance(obj, torch.nn.Parameter) else obj
            if not isinstance(tensor, torch.Tensor):
                snapshots[attr_name] = {"type": str(type(obj)), "not_tensor": True}
                continue
            snapshots[attr_name] = self._tensor_fingerprint(tensor)
        return snapshots

    def _diff_attr_fingerprints(
        self,
        baseline: Dict[str, Dict[str, Any]],
        current: Dict[str, Dict[str, Any]],
        *,
        stage: str,
    ) -> None:
        logger = getattr(self, "logger", None)
        if logger is None:
            logger = logging.getLogger(__name__)

        total = len(baseline)
        missing = 0
        changed = 0
        unchanged = 0

        changed_examples = []
        for attr_name, base_fp in baseline.items():
            cur_fp = current.get(attr_name, {"missing": True})
            if base_fp.get("missing", False) or cur_fp.get("missing", False):
                if cur_fp.get("missing", False):
                    missing += 1
                continue
            if base_fp.get("error") or cur_fp.get("error"):
                changed += 1
                if len(changed_examples) < 5:
                    changed_examples.append((attr_name, base_fp, cur_fp))
                continue

            same_shape = base_fp.get("shape") == cur_fp.get("shape")
            same_dtype = base_fp.get("dtype") == cur_fp.get("dtype")
            sum_delta = abs(base_fp.get("sample_sum", 0.0) - cur_fp.get("sample_sum", 0.0))
            mean_delta = abs(base_fp.get("sample_mean", 0.0) - cur_fp.get("sample_mean", 0.0))
            min_delta = abs(base_fp.get("sample_min", 0.0) - cur_fp.get("sample_min", 0.0))
            max_delta = abs(base_fp.get("sample_max", 0.0) - cur_fp.get("sample_max", 0.0))

            if same_shape and same_dtype and sum_delta < 1e-6 and mean_delta < 1e-6 and min_delta < 1e-6 and max_delta < 1e-6:
                unchanged += 1
            else:
                changed += 1
                if len(changed_examples) < 5:
                    changed_examples.append((attr_name, base_fp, cur_fp))

        logger.info(
            "[qat_backend_check][%s] attrs_total=%d unchanged=%d changed=%d missing=%d",
            stage,
            total,
            unchanged,
            changed,
            missing,
        )

        if missing == total and total > 0:
            logger.warning(
                "[qat_backend_check][%s] all replaced attrs are absent in this graph module; likely folded/snapshotted into delegated blobs",
                stage,
            )

        for attr_name, base_fp, cur_fp in changed_examples:
            logger.info(
                "[qat_backend_check][%s][changed] attr=%s base=%s current=%s",
                stage,
                attr_name,
                base_fp,
                cur_fp,
            )

    def _log_lowered_module_summary(self, gm, stage: str) -> None:
        logger = getattr(self, "logger", None)
        if logger is None:
            logger = logging.getLogger(__name__)

        lowered = []
        for name, module in gm.named_modules():
            type_name = type(module).__name__
            if "LoweredBackendModule" in type_name:
                processed = getattr(module, "processed_bytes", None)
                processed_size = len(processed) if isinstance(processed, (bytes, bytearray)) else None
                lowered.append((name, type_name, processed_size))

        logger.info(
            "[qat_backend_check][%s] lowered_modules=%d",
            stage,
            len(lowered),
        )
        for name, type_name, processed_size in lowered[:8]:
            logger.info(
                "[qat_backend_check][%s] lowered_module name=%s type=%s processed_bytes=%s",
                stage,
                name,
                type_name,
                processed_size,
            )

    def _run_qat_backend_usage_check(self, lowered_gm, stage: str) -> None:
        attr_names = getattr(self, "_qat_replaced_attr_names", None)
        logger = getattr(self, "logger", None)
        if logger is None:
            logger = logging.getLogger(__name__)

        if not attr_names:
            logger.info(
                "[qat_backend_check][%s] skipped: no replaced attrs recorded",
                stage,
            )
            return

        logger.info(
            "[qat_backend_check][%s] start: replaced_attrs=%d",
            stage,
            len(attr_names),
        )

        baseline = getattr(self, "_qat_replaced_attr_fingerprints", None)
        if baseline is None:
            baseline = self._snapshot_attr_fingerprints(self.decoder, attr_names)
            self._qat_replaced_attr_fingerprints = baseline

        current = self._snapshot_attr_fingerprints(lowered_gm, attr_names)
        self._diff_attr_fingerprints(baseline, current, stage=stage)
        self._log_lowered_module_summary(lowered_gm, stage)

    def _replace_qparams_from_qat_checkpoint(self, gm):
        logger = getattr(self, "logger", None)
        if logger is None:
            logger = logging.getLogger(__name__)

        qat_path = getattr(self.control_args, "replace_with_qat_checkpoint", None)
        if not qat_path:
            return

        qweight_mode = getattr(
            self.control_args,
            "qat_qweight_mode",
            "qweight_minus_qzeros",
        )
        bits_hint = int(getattr(self.control_args, "qat_source_wbits", 2))
        group_size_hint = int(getattr(self.control_args, "qat_source_group_size", 32))
        debug = getattr(self.control_args, "qat_replace_debug", False)
        verify_chain = getattr(self.control_args, "qat_verify_chain", False)

        try:
            from safetensors.torch import load_file

            qat_sd = load_file(qat_path)
        except Exception as e:
            logger.warning("[qat_replace] failed to load checkpoint %s: %s", qat_path, e)
            return

        layer_count = int(self.meta.get("get_n_layers", 28))
        target_map = {
            "attention.wq_conv": "self_attn.q_proj",
            "attention.wk_conv": "self_attn.k_proj",
            "attention.wv_conv": "self_attn.v_proj",
            "attention.wo_conv": "self_attn.o_proj",
            "feed_forward.w1_conv": "mlp.gate_proj",
            "feed_forward.w3_conv": "mlp.up_proj",
            "feed_forward.w2_conv": "mlp.down_proj",
        }
        extra_targets = {
            "output.conv": "lm_head",
        }

        replaced_q = 0
        replaced_s = 0
        missed = 0
        matched = 0
        replaced_attr_names = set()

        for layer_i in range(layer_count):
            for local_key, qat_local in target_map.items():
                layer_keyword = f"layers.{layer_i}.{local_key}"
                refs = self._find_qweight_scale_zero_attrs_for_layer(gm, layer_keyword)
                if not refs:
                    continue
                matched += 1

                base = f"model.layers.{layer_i}.{qat_local}"
                qkey = f"{base}.qweight"
                zkey = f"{base}.qzeros"
                skey = f"{base}.scales"
                zpkey = f"{base}.zero_points"

                logger.info(
                    "[qat_replace][layer] target=%s source_base=%s refs=%d",
                    layer_keyword,
                    base,
                    len(refs),
                )

                if qkey not in qat_sd or skey not in qat_sd:
                    logger.warning(
                        "[qat_replace] missing source tensors for %s (need %s and %s)",
                        layer_keyword,
                        qkey,
                        skey,
                    )
                    missed += 1
                    continue

                q_src = qat_sd[qkey]
                if qweight_mode == "qweight_minus_qzeros":
                    if zkey in qat_sd:
                        logger.info(
                            "[qat_replace][layer] qweight mode=%s use (%s - %s)",
                            qweight_mode,
                            qkey,
                            zkey,
                        )
                    else:
                        logger.warning(
                            "[qat_replace] qweight_minus_qzeros requested but zkey missing for %s; fallback to qweight",
                            layer_keyword,
                        )
                        logger.info(
                            "[qat_replace][layer] qweight mode fallback to %s",
                            qkey,
                        )
                else:
                    logger.info(
                        "[qat_replace][layer] qweight mode=%s use %s",
                        qweight_mode,
                        qkey,
                    )

                s_src = qat_sd[skey]
                s_min = s_src.min().item() if s_src.numel() > 0 else None
                s_max = s_src.max().item() if s_src.numel() > 0 else None
                s_neg = int((s_src < 0).sum().item()) if s_src.numel() > 0 else 0
                logger.info(
                    "[qat_replace][layer] scale source=%s shape=%s dtype=%s min=%s max=%s neg_count=%d",
                    skey,
                    tuple(s_src.shape),
                    s_src.dtype,
                    s_min,
                    s_max,
                    s_neg,
                )

                for q_attr, s_attr, z_attr in refs:
                    context = f"{layer_keyword} <- {base}"
                    q_target_obj = self._resolve_attr(gm, q_attr)
                    q_target_tensor = (
                        q_target_obj.detach()
                        if isinstance(q_target_obj, torch.nn.Parameter)
                        else q_target_obj
                    )
                    inferred_group_size = group_size_hint
                    if isinstance(q_target_tensor, torch.Tensor):
                        inferred_group_size = self._infer_group_size_from_qparams(
                            q_target_tensor,
                            qat_sd.get(zkey, None),
                            s_src,
                            group_size_hint,
                        )
                    q_src_for_target = q_src
                    if isinstance(q_target_tensor, torch.Tensor):
                        q_src_prepared = self._prepare_qat_qweight_for_target(
                            q_packed=q_src,
                            z_packed=qat_sd.get(zkey, None),
                            target=q_target_tensor,
                            qweight_mode=qweight_mode,
                            bits_hint=bits_hint,
                            group_size_hint=inferred_group_size,
                        )
                        if q_src_prepared is not None:
                            q_src_for_target = q_src_prepared

                    if self._replace_attr_tensor(
                        gm,
                        q_attr,
                        q_src_for_target,
                        debug=debug,
                        context=context,
                    ):
                        replaced_q += 1
                        replaced_attr_names.add(q_attr)
                    if s_attr is not None:
                        s_target_obj = self._resolve_attr(gm, s_attr)
                        s_target_tensor = (
                            s_target_obj.detach()
                            if isinstance(s_target_obj, torch.nn.Parameter)
                            else s_target_obj
                        )
                        s_src_for_target = s_src
                        if isinstance(s_target_tensor, torch.Tensor):
                            s_src_for_target = self._prepare_qat_scale_for_target(
                                s_src, s_target_tensor
                            )
                        if self._replace_attr_tensor(
                            gm,
                            s_attr,
                            s_src_for_target,
                            debug=debug,
                            context=context,
                        ):
                            replaced_s += 1
                            replaced_attr_names.add(s_attr)

                    if verify_chain:
                        self._verify_qat_chain_alignment_for_attr(
                            gm,
                            layer_keyword=layer_keyword,
                            source_base=base,
                            q_attr=q_attr,
                            s_attr=s_attr,
                            z_attr=z_attr,
                            q_src=q_src,
                            z_src=qat_sd.get(zkey, None),
                            s_src=s_src,
                            qweight_mode=qweight_mode,
                            bits_hint=bits_hint,
                            group_size_hint=inferred_group_size,
                            zero_points_src=qat_sd.get(zpkey, None),
                        )

                    if layer_i == 0:
                        self._log_dequant_node_and_dump_weight_for_attrs(
                            gm,
                            layer_keyword=layer_keyword,
                            q_attr=q_attr,
                            s_attr=s_attr,
                            z_attr=z_attr,
                        )

        for layer_keyword, base in extra_targets.items():
            refs = self._find_qweight_scale_zero_attrs_for_layer(gm, layer_keyword)
            if not refs:
                continue
            matched += 1

            qkey = f"{base}.qweight"
            zkey = f"{base}.qzeros"
            skey = f"{base}.scales"
            zpkey = f"{base}.zero_points"

            logger.info(
                "[qat_replace][layer] target=%s source_base=%s refs=%d",
                layer_keyword,
                base,
                len(refs),
            )

            if qkey not in qat_sd or skey not in qat_sd:
                logger.warning(
                    "[qat_replace] missing source tensors for %s (need %s and %s)",
                    layer_keyword,
                    qkey,
                    skey,
                )
                missed += 1
                continue

            q_src = qat_sd[qkey]
            logger.info(
                "[qat_replace][layer] qweight mode=lm_head_symmetric_override use %s and ignore %s",
                qkey,
                zkey,
            )

            s_src = qat_sd[skey]
            s_min = s_src.min().item() if s_src.numel() > 0 else None
            s_max = s_src.max().item() if s_src.numel() > 0 else None
            s_neg = int((s_src < 0).sum().item()) if s_src.numel() > 0 else 0
            logger.info(
                "[qat_replace][layer] scale source=%s shape=%s dtype=%s min=%s max=%s neg_count=%d",
                skey,
                tuple(s_src.shape),
                s_src.dtype,
                s_min,
                s_max,
                s_neg,
            )

            for q_attr, s_attr, z_attr in refs:
                context = f"{layer_keyword} <- {base}"
                q_target_obj = self._resolve_attr(gm, q_attr)
                q_target_tensor = (
                    q_target_obj.detach()
                    if isinstance(q_target_obj, torch.nn.Parameter)
                    else q_target_obj
                )
                inferred_group_size = group_size_hint
                if isinstance(q_target_tensor, torch.Tensor):
                    inferred_group_size = self._infer_group_size_from_qparams(
                        q_target_tensor,
                        qat_sd.get(zkey, None),
                        s_src,
                        group_size_hint,
                    )
                q_src_for_target = q_src
                if isinstance(q_target_tensor, torch.Tensor):
                    q_src_prepared = self._prepare_qat_qweight_for_target(
                        q_packed=q_src,
                        z_packed=None,
                        target=q_target_tensor,
                        qweight_mode="qweight_only",
                        bits_hint=bits_hint,
                        group_size_hint=inferred_group_size,
                    )
                    if q_src_prepared is not None:
                        q_src_for_target = q_src_prepared - 128.0

                if self._replace_attr_tensor(
                    gm,
                    q_attr,
                    q_src_for_target,
                    debug=debug,
                    context=context,
                ):
                    replaced_q += 1
                    replaced_attr_names.add(q_attr)
                if s_attr is not None:
                    s_target_obj = self._resolve_attr(gm, s_attr)
                    s_target_tensor = (
                        s_target_obj.detach()
                        if isinstance(s_target_obj, torch.nn.Parameter)
                        else s_target_obj
                    )
                    s_src_for_target = s_src
                    if isinstance(s_target_tensor, torch.Tensor):
                        s_src_for_target = self._prepare_qat_scale_for_target(
                            s_src, s_target_tensor
                        )
                    if self._replace_attr_tensor(
                        gm,
                        s_attr,
                        s_src_for_target,
                        debug=debug,
                        context=context,
                    ):
                        replaced_s += 1
                        replaced_attr_names.add(s_attr)

                if verify_chain:
                    self._verify_qat_chain_alignment_for_attr(
                        gm,
                        layer_keyword=layer_keyword,
                        source_base=base,
                        q_attr=q_attr,
                        s_attr=s_attr,
                        z_attr=z_attr,
                        q_src=q_src,
                        z_src=None,
                        s_src=s_src,
                        qweight_mode="qweight_only",
                        bits_hint=bits_hint,
                        group_size_hint=inferred_group_size,
                        zero_points_src=None,
                    )

                self._log_dequant_node_and_dump_weight_for_attrs(
                    gm,
                    layer_keyword=layer_keyword,
                    q_attr=q_attr,
                    s_attr=s_attr,
                    z_attr=z_attr,
                )

        logger.info(
            "[qat_replace] completed mode=%s matched_layers=%d qweight_replaced=%d scale_replaced=%d missed_layers=%d debug=%s",
            qweight_mode,
            matched,
            replaced_q,
            replaced_s,
            missed,
            debug,
        )

        self._qat_replaced_attr_names = sorted(replaced_attr_names)
        self._qat_replaced_attr_fingerprints = self._snapshot_attr_fingerprints(
            gm,
            self._qat_replaced_attr_names,
        )

    def _find_qweight_attr_names_for_layer(
        self,
        gm,
        layer_keyword: str,
        log_prefix: str = "qweight_scan",
    ) -> List[str]:
        logger = getattr(self, "logger", None)
        if logger is None:
            logger = logging.getLogger(__name__)

        def _iter_node_args(obj):
            if hasattr(obj, "op"):
                yield obj
                return
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    yield from _iter_node_args(v)
                return
            if isinstance(obj, dict):
                for v in obj.values():
                    yield from _iter_node_args(v)

        def _is_dequant_node(node) -> bool:
            return (
                hasattr(node, "op")
                and node.op == "call_function"
                and (
                    "dequantize_affine" in str(node.target)
                    or "dequantize_per_channel" in str(node.target)
                    or "dequantize_per_tensor" in str(node.target)
                    or "dequantize" in str(node.target)
                )
            )

        def _add_attr_from_get_attr_node(get_attr_node, context: str):
            if not hasattr(get_attr_node, "op") or get_attr_node.op != "get_attr":
                return
            attr_name = str(get_attr_node.target)
            attr_name_lower = attr_name.lower()
            if "zero_point" in attr_name_lower or "scale" in attr_name_lower:
                logger.info(
                    "[%s] skip non-qweight attr=%s for layer=%s",
                    log_prefix,
                    attr_name,
                    layer_keyword,
                )
                return

            resolved = self._extract_attr_value(gm, get_attr_node)
            if isinstance(resolved, Exception):
                return
            if not self._is_possible_qweight_obj(resolved):
                return

            tensor = resolved.detach() if isinstance(resolved, torch.nn.Parameter) else resolved
            if not isinstance(tensor, torch.Tensor):
                return
            # Keep 1D/2D+ weight-like tensors, but skip scalar metadata.
            if tensor.ndim < 1:
                return

            if attr_name in seen:
                return
            seen.add(attr_name)
            attr_names.append(attr_name)
            logger.info(
                "[%s] matched layer=%s %s -> qweight attr=%s (%s)",
                log_prefix,
                layer_keyword,
                context,
                attr_name,
                self._tensor_brief(tensor),
            )

        attr_names = []
        seen = set()

        def _get_weight_candidate_roots(node):
            # Prefer operator-specific weight argument locations so we do not
            # accidentally traverse activation branches (which can bring in
            # unrelated large tensors like token embeddings).
            if node.op == "call_function":
                if node.target == torch.ops.aten.conv2d.default and len(node.args) >= 2:
                    return [node.args[1]]
                if node.target == torch.ops.aten.embedding.default and len(node.args) >= 1:
                    return [node.args[0]]
                if node.target == torch.ops.aten.rms_norm.default:
                    roots = []
                    # aten.rms_norm.default(input, normalized_shape, weight, eps)
                    if len(node.args) >= 3:
                        roots.append(node.args[2])
                    # Some versions may carry weight in kwargs.
                    if "weight" in node.kwargs:
                        roots.append(node.kwargs["weight"])
                    if roots:
                        return roots

            # Fallback: broader traversal for unsupported ops.
            roots = [node]
            roots.extend(list(_iter_node_args(node.args)))
            roots.extend(list(_iter_node_args(node.kwargs)))
            return roots

        for node in gm.graph.nodes:
            module_text = self._node_module_stack_text(node)
            node_text = f"{node.name} {node.target} {node.op} {module_text}"
            if layer_keyword not in node_text:
                continue

            candidate_roots = _get_weight_candidate_roots(node)

            for root in candidate_roots:
                if not hasattr(root, "op"):
                    continue
                for up in self._iter_upstream_nodes(root):
                    if up.op == "get_attr":
                        _add_attr_from_get_attr_node(
                            up,
                            context=f"node={node.name}",
                        )
                        continue

                    if not _is_dequant_node(up) or len(up.args) == 0:
                        continue

                    qarg = up.args[0]
                    _add_attr_from_get_attr_node(
                        qarg,
                        context=f"node={node.name} dequant={up.target}",
                    )

        return attr_names

    def _force_fill_qweight_by_layer_keyword(
        self,
        gm,
        layer_keyword: str,
        fill_value: int = -1,
        require_unique: bool = False,
    ) -> bool:
        logger = getattr(self, "logger", None)
        if logger is None:
            logger = logging.getLogger(__name__)

        attr_names = self._find_qweight_attr_names_for_layer(
            gm,
            layer_keyword,
            log_prefix="force_fill_by_layer",
        )
        if len(attr_names) == 0:
            logger.info(
                "[force_fill_by_layer] no quantized qweight attr found for keyword=%s",
                layer_keyword,
            )
            return False

        if require_unique and len(attr_names) != 1:
            logger.warning(
                "[force_fill_by_layer] require_unique=True but found %d attrs for %s: %s",
                len(attr_names),
                layer_keyword,
                attr_names,
            )
            return False

        layer_keyword_lower = str(layer_keyword).lower()
        transform_mode = (
            "norm_u16_invert"
            if "norm" in layer_keyword_lower
            else (
                "uint16_invert"
                if "tok_embeddings" in layer_keyword_lower
                else "negate_zero"
            )
        )

        ok = False
        for attr_name in attr_names:
            ok = self._force_fill_get_attr_tensor(
                gm,
                attr_name=attr_name,
                fill_value=fill_value,
                transform_mode=transform_mode,
            ) or ok
        return ok

    def _target_qweight_keywords(self) -> List[str]:
        keywords = ["tok_embeddings"]
        for layer_i in range(28):
            prefix = f"layers.{layer_i}"
            keywords.extend(
                [
                    f"{prefix}.attention.wq_conv",
                    f"{prefix}.attention.wk_conv",
                    f"{prefix}.attention.wv_conv",
                    f"{prefix}.attention.wo_conv",
                    f"{prefix}.feed_forward.w1_conv",
                    f"{prefix}.feed_forward.w3_conv",
                    f"{prefix}.feed_forward.w2_conv",
                ]
            )
        keywords.append("output.conv")
        return keywords

    def _dump_qweights_for_keywords(self, gm, keywords: List[str], tag: str):
        logger = getattr(self, "logger", None)
        if logger is None:
            logger = logging.getLogger(__name__)

        def _safe_name(text: str) -> str:
            s = str(text)
            return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)

        artifact_dir = getattr(self.control_args, "artifact", ".")
        os.makedirs(artifact_dir, exist_ok=True)
        mode_name = getattr(getattr(self, "mode", None), "name", "unknown").lower()

        logger.info("========== %s ==========" , tag)

        dumped = 0
        for keyword in keywords:
            attr_names = self._find_qweight_attr_names_for_layer(
                gm,
                keyword,
                log_prefix="dump_qweights_scan",
            )
            if not attr_names:
                logger.info("[dump_qweights] no qweight attr found for keyword=%s", keyword)
                continue

            tensors = {}
            meta = {
                "mode": mode_name,
                "keyword": keyword,
                "attrs": [],
            }

            for attr_name in attr_names:
                try:
                    resolved = self._resolve_attr(gm, attr_name)
                except Exception as e:
                    logger.warning(
                        "[dump_qweights] failed to resolve attr=%s for keyword=%s: %s",
                        attr_name,
                        keyword,
                        e,
                    )
                    continue

                tensor = resolved.detach() if isinstance(resolved, torch.nn.Parameter) else resolved
                if not isinstance(tensor, torch.Tensor):
                    continue
                if not self._is_possible_qweight_obj(tensor):
                    continue

                t = tensor.detach().cpu().contiguous()
                tensors[attr_name] = t
                meta["attrs"].append(
                    {
                        "attr_name": attr_name,
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "numel": int(t.numel()),
                    }
                )

            if not tensors:
                logger.info(
                    "[dump_qweights] keyword=%s matched attrs but no dumpable qweight tensors",
                    keyword,
                )
                continue

            stem = f"layer_qweights_{mode_name}_{_safe_name(keyword)}"
            pt_path = os.path.join(artifact_dir, f"{stem}.pt")
            json_path = os.path.join(artifact_dir, f"{stem}.json")

            torch.save(tensors, pt_path)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            dumped += 1
            logger.info(
                "[dump_qweights] keyword=%s saved attrs=%s -> %s",
                keyword,
                [x["attr_name"] for x in meta["attrs"]],
                pt_path,
            )

        logger.info("[dump_qweights] completed: dumped %d keyword files", dumped)

    def _dump_single_layer_quant_info(self, gm, keyword: str, tag: str):
        logger = getattr(self, "logger", None)
        if logger is None:
            logger = logging.getLogger(__name__)

        def _json_safe(val):
            if isinstance(val, torch.Tensor):
                return self._tensor_brief(val)
            if isinstance(val, torch.nn.Parameter):
                return self._tensor_brief(val.detach())
            if isinstance(val, torch.dtype):
                return str(val)
            if isinstance(val, (str, int, float, bool)) or val is None:
                return val
            if isinstance(val, (list, tuple)):
                return [_json_safe(v) for v in val]
            if isinstance(val, dict):
                return {str(k): _json_safe(v) for k, v in val.items()}
            return str(val)

        def _safe_name(text: str) -> str:
            s = str(text)
            return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)

        def _is_dequant_node(node) -> bool:
            return (
                hasattr(node, "op")
                and node.op == "call_function"
                and (
                    "dequantize_affine" in str(node.target)
                    or "dequantize_per_channel" in str(node.target)
                    or "dequantize_per_tensor" in str(node.target)
                    or "dequantize" in str(node.target)
                )
            )

        def _iter_node_args(obj):
            if hasattr(obj, "op"):
                yield obj
                return
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    yield from _iter_node_args(v)
                return
            if isinstance(obj, dict):
                for v in obj.values():
                    yield from _iter_node_args(v)

        def _get_weight_candidate_roots(node):
            if node.op == "call_function":
                if node.target == torch.ops.aten.conv2d.default and len(node.args) >= 2:
                    return [node.args[1]]
                if node.target == torch.ops.aten.embedding.default and len(node.args) >= 1:
                    return [node.args[0]]
                if node.target == torch.ops.aten.rms_norm.default:
                    roots = []
                    # aten.rms_norm.default(input, normalized_shape, weight, eps)
                    if len(node.args) >= 3:
                        roots.append(node.args[2])
                    if "weight" in node.kwargs:
                        roots.append(node.kwargs["weight"])
                    if roots:
                        return roots

            roots = [node]
            roots.extend(list(_iter_node_args(node.args)))
            roots.extend(list(_iter_node_args(node.kwargs)))
            return roots

        def _parse_dequant_args(weight_node, quant_items, quant_tensors, arg_order):
            logger.info(
                "[target_weight] dequant_node=%s args=%s kwargs=%s",
                weight_node.target,
                weight_node.args,
                weight_node.kwargs,
            )

            tgt = str(weight_node.target)

            def _label_for_arg(idx, arg):
                if idx == 0:
                    return "qweight"

                # quantized_decomposed.dequantize_per_tensor.default
                # Signature: (input, scale, zero_point, quant_min, quant_max, dtype)
                if "dequantize_per_tensor" in tgt:
                    return {
                        1: "scale",
                        2: "zero_point",
                        3: "quant_min",
                        4: "quant_max",
                        5: "dtype",
                    }.get(idx, f"arg{idx}")

                # torchao.dequantize_affine commonly has block_size in arg1,
                # then scale/zero_point/dtype/min/max variants by backend.
                if "dequantize_affine" in tgt:
                    if idx == 1 and not hasattr(arg, "op"):
                        return "block_size_or_scale"
                    return {
                        2: "scale",
                        3: "zero_point",
                        4: "dtype",
                        5: "quant_min",
                        6: "quant_max",
                    }.get(idx, f"arg{idx}")

                # quantized_decomposed.dequantize_per_channel.default
                if "dequantize_per_channel" in tgt:
                    return {
                        1: "scale",
                        2: "zero_point",
                        3: "axis",
                        4: "quant_min",
                        5: "quant_max",
                        6: "dtype",
                    }.get(idx, f"arg{idx}")

                return f"arg{idx}"

            for idx, arg in enumerate(weight_node.args):
                resolved = self._extract_attr_value(gm, arg)
                label = _label_for_arg(idx, arg)

                unique_label = label
                dup_idx = 1
                while unique_label in quant_items:
                    unique_label = f"{label}_{dup_idx}"
                    dup_idx += 1

                item = {"arg_index": idx, "summary": self._obj_brief(resolved)}
                if hasattr(arg, "op") and getattr(arg, "op", None) == "get_attr":
                    item["source"] = "get_attr"
                    item["attr_name"] = str(arg.target)
                    logger.info(
                        "[target_weight] %s %s -> %s",
                        unique_label,
                        arg.target,
                        self._obj_brief(resolved),
                    )
                    if (
                        unique_label.startswith("qweight")
                        and str(arg.target) == "_frozen_param2"
                        and isinstance(resolved, torch.Tensor)
                    ):
                        qweight_tensor = resolved.detach().cpu().squeeze(-1).squeeze(-1)
                        weight_slice = qweight_tensor[:32, :32]

                        logger.info("[target_weight] qweight %s first_32x32=", arg.target)
                        for row_idx, row in enumerate(weight_slice):
                            row_data = row.tolist()
                            logger.info("第 %2d 行: %s", row_idx, row_data)
                        logger.info(
                            "[target_weight] qweight %s first_256_flat=%s",
                            arg.target,
                            qweight_tensor.reshape(-1)[:256].tolist(),
                        )
                else:
                    item["source"] = "literal"
                    logger.info(
                        "[target_weight] %s -> %s",
                        unique_label,
                        self._obj_brief(resolved),
                    )

                if isinstance(resolved, torch.Tensor):
                    t = resolved.detach().cpu().contiguous()
                    quant_tensors[unique_label] = t
                    item["shape"] = list(t.shape)
                    item["dtype"] = str(t.dtype)
                    if t.numel() > 0:
                        item["min"] = t.min().item()
                        item["max"] = t.max().item()
                else:
                    item["value"] = _json_safe(resolved)

                quant_items[unique_label] = item
                arg_order.append(unique_label)

        def _parse_get_attr_node(attr_node, quant_items, quant_tensors, arg_order):
            obj = self._extract_attr_value(gm, attr_node)
            label = f"attr_{_safe_name(str(attr_node.target))}"
            unique_label = label
            dup_idx = 1
            while unique_label in quant_items:
                unique_label = f"{label}_{dup_idx}"
                dup_idx += 1

            logger.info(
                "[target_weight] direct_get_attr %s -> %s",
                attr_node.target,
                self._obj_brief(obj),
            )
            item = {
                "source": "get_attr",
                "attr_name": str(attr_node.target),
                "summary": self._obj_brief(obj),
            }
            if isinstance(obj, torch.Tensor):
                t = obj.detach().cpu().contiguous()
                quant_tensors[unique_label] = t
                item["shape"] = list(t.shape)
                item["dtype"] = str(t.dtype)
                if t.numel() > 0:
                    item["min"] = t.min().item()
                    item["max"] = t.max().item()
            else:
                item["value"] = _json_safe(obj)

            quant_items[unique_label] = item
            arg_order.append(unique_label)

        logger.info("========== %s ==========", tag)
        logger.info("Target quant layer: %s", keyword)

        matched = False
        dumped = False
        for node in gm.graph.nodes:
            module_text = self._node_module_stack_text(node)
            node_text = f"{node.name} {node.target} {node.op} {module_text}"
            if keyword not in node_text:
                continue

            matched = True
            logger.info("[target_node] op=%s name=%s target=%s", node.op, node.name, node.target)
            logger.info("[target_node] module_stack=%s", module_text)

            quant_items = {}
            quant_tensors = {}
            arg_order = []
            seen_get_attr = set()
            seen_dequant = set()

            candidate_roots = _get_weight_candidate_roots(node)

            for root in candidate_roots:
                if not hasattr(root, "op"):
                    continue
                for up in self._iter_upstream_nodes(root):
                    if up.op == "get_attr":
                        if up in seen_get_attr:
                            continue
                        seen_get_attr.add(up)
                        _parse_get_attr_node(up, quant_items, quant_tensors, arg_order)
                    elif _is_dequant_node(up):
                        if up in seen_dequant:
                            continue
                        seen_dequant.add(up)
                        _parse_dequant_args(up, quant_items, quant_tensors, arg_order)

            if not arg_order:
                logger.info(
                    "[target_node] no upstream get_attr/dequant weight sources found for %s",
                    node.name,
                )
                continue

            artifact_dir = getattr(self.control_args, "artifact", ".")
            os.makedirs(artifact_dir, exist_ok=True)
            mode_name = getattr(getattr(self, "mode", None), "name", "unknown").lower()
            stem = (
                f"layer_quant_dump_{mode_name}_{_safe_name(keyword)}_{_safe_name(node.name)}"
            )
            json_path = os.path.join(artifact_dir, f"{stem}.json")
            pt_path = os.path.join(artifact_dir, f"{stem}.pt")

            payload = {
                "tag": tag,
                "mode": mode_name,
                "keyword": keyword,
                "target_node": {
                    "op": str(node.op),
                    "name": str(node.name),
                    "target": str(node.target),
                    "module_stack": module_text,
                },
                "arg_order": arg_order,
                "quant_items": quant_items,
                "tensor_dump_file": pt_path,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            torch.save(quant_tensors, pt_path)
            logger.info("[target_weight] saved quant metadata -> %s", json_path)
            logger.info("[target_weight] saved quant tensors -> %s", pt_path)
            dumped = True

        if not matched:
            logger.info("No target graph node matched keyword %s.", keyword)
        elif not dumped:
            logger.info("Matched keyword %s but found no dumpable weight tensors.", keyword)
    def _calibrate(
        self,
        model,
        tokenizer,
        event,
        user_calibration_data,
        tok_embedding=None,
        intermediate_outputs=None,
    ):
        """
        Calibrate the model using either task-based evaluation or prompt-based inference.

        This method performs Post-Training Quantization (PTQ) calibration by running inference
        on the model with either:
        1. Task-based datasets by lm_eval for text-only models in perplexity evaluation
        2. User-provided prompts for both text-only and multimodal models

        Args:
            model: The decoder model to calibrate (GraphModule after prepare_pt2e)
            tokenizer: Tokenizer for encoding text inputs
            event: Event name for logging (e.g., "prepare_pt2e", "convert_pt2e")
            tok_embedding: Optional text embedding module (required only for multimodal models)
            intermediate_outputs: Optional pre-computed embeddings from vision/audio encoder
                                 (required only for multimodal models)
        """
        # Determine if this is a multimodal model
        is_multimodal = tok_embedding is not None

        # Determine if task-based calibration is requested
        has_task_calibration = self.control_args.tasks is not None

        # Task-based calibration: Only for text-only LLMs
        # Multimodal models (VLMs) cannot use task-based evaluation currently.
        if has_task_calibration and not is_multimodal:
            lookahead_config = getattr(self.model_args, "lookahead_config", None)
            if lookahead_config is None and isinstance(self.model_args, dict):
                lookahead_config = self.model_args.get("lookahead_config", None)

            graph_module_inference(
                use_kv_cache=self.meta["get_use_kv_cache"],
                get_example_inputs=self.get_example_inputs,
                module=model,
                tokenizer=tokenizer,
                ar_len=self.meta["get_ar_len"],
                max_seq_len=self.meta["get_max_context_len"],
                tasks=self.control_args.tasks,
                tasks_limit=self.control_args.limit,
                num_fewshot=self.control_args.num_fewshot,
                use_i64_token=self.control_args.embedding_quantize is not None,
                event_name=f"{event}_tasks",
                seq_mse_candidates=self.config.seq_mse_candidates,
            )

        # prepare lookahead config if applicable
        lookahead_config = (
            (self.control_args.window, self.control_args.ngram, self.control_args.gcap)
            if (
                self.mode == Mode.DECODE and self.control_args.model_mode == "lookahead"
            )
            else None
        )
        # check user's prompt which helps calibrate special token
        for turn in zip(intermediate_outputs, user_calibration_data):
            hidden_states, prompt = turn
            graph_module_inference(
                use_kv_cache=self.meta["get_use_kv_cache"],
                get_example_inputs=self.get_example_inputs,
                hidden_states=hidden_states,  # hidden_states for multimodal
                module=model,
                tok_embedding=tok_embedding,
                image_token_id=self.meta.get("image_token_id", None),
                tokenizer=tokenizer,
                ar_len=self.meta["get_ar_len"],
                max_seq_len=self.meta["get_max_context_len"],
                prompt=prompt,
                use_i64_token=self.control_args.embedding_quantize is not None,
                event_name=f"{event}_prompt",
                lookahead_config=lookahead_config,
            )

    def _run_post_replace_smoke_test(self, model):
        if not getattr(self.control_args, "qat_post_replace_smoke_test", False):
            return

        logger = getattr(self, "logger", None)
        if logger is None:
            logger = logging.getLogger(__name__)

        def _iter_tensors(obj):
            if isinstance(obj, torch.Tensor):
                yield obj
                return
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    yield from _iter_tensors(item)
                return
            if isinstance(obj, dict):
                for item in obj.values():
                    yield from _iter_tensors(item)

        logger.info("[qat_smoke] running post-replace smoke inference")
        try:
            try:
                from torchao.quantization.pt2e import move_exported_model_to_eval

                move_exported_model_to_eval(model)
            except Exception:
                try:
                    from torchao.quantization.pt2e import allow_exported_model_train_eval

                    allow_exported_model_train_eval(model)
                    model.eval()
                except Exception:
                    pass

            with torch.no_grad():
                output = model(*self.export_input)

            out_tensors = list(_iter_tensors(output))
            if not out_tensors:
                logger.warning("[qat_smoke] no tensor outputs found in model output type=%s", type(output))
                return

            total_nan = 0
            total_inf = 0
            for idx, tensor in enumerate(out_tensors):
                det = tensor.detach()
                nan_count = int(torch.isnan(det).sum().item())
                inf_count = int(torch.isinf(det).sum().item())
                total_nan += nan_count
                total_inf += inf_count
                if idx < 4:
                    tmin = det.min().item() if det.numel() > 0 else None
                    tmax = det.max().item() if det.numel() > 0 else None
                    logger.info(
                        "[qat_smoke][out%d] shape=%s dtype=%s min=%s max=%s nan=%d inf=%d",
                        idx,
                        tuple(det.shape),
                        det.dtype,
                        tmin,
                        tmax,
                        nan_count,
                        inf_count,
                    )

            logger.info(
                "[qat_smoke] done outputs=%d total_nan=%d total_inf=%d",
                len(out_tensors),
                total_nan,
                total_inf,
            )
        except Exception as error:
            logger.exception("[qat_smoke] failed post-replace inference: %s", error)

    def _run_post_replace_text_generation(self, model, tokenizer):
        if not getattr(self.control_args, "qat_post_replace_generate_text", False):
            return

        logger = getattr(self, "logger", None)
        if logger is None:
            logger = logging.getLogger(__name__)

        prompt = getattr(self.control_args, "qat_post_replace_generate_prompt", None)
        if not prompt:
            prompt = getattr(self.control_args, "prompt", None)
        if isinstance(prompt, (list, tuple)):
            if len(prompt) == 0:
                prompt = None
            else:
                logger.info(
                    "[qat_generate] prompt is %s with len=%d, using the first entry for generation",
                    type(prompt).__name__,
                    len(prompt),
                )
                prompt = prompt[0]
        if not prompt:
            logger.warning(
                "[qat_generate] skip post-replace generation: empty prompt (set --qat_post_replace_generate_prompt or --prompt)"
            )
            return

        logger.info("[qat_generate] running post-replace text generation")
        try:
            try:
                from torchao.quantization.pt2e import move_exported_model_to_eval

                move_exported_model_to_eval(model)
            except Exception:
                pass

            max_seq_len = getattr(
                self.control_args,
                "qat_post_replace_generate_max_seq_len",
                None,
            )
            if max_seq_len is None:
                max_seq_len = self.meta["get_max_context_len"]

            lookahead_config = getattr(self.model_args, "lookahead_config", None)
            if lookahead_config is None and isinstance(self.model_args, dict):
                lookahead_config = self.model_args.get("lookahead_config", None)

            graph_module_inference(
                use_kv_cache=self.meta["get_use_kv_cache"],
                get_example_inputs=self.get_example_inputs,
                module=model,
                tokenizer=tokenizer,
                ar_len=self.meta["get_ar_len"],
                max_seq_len=max_seq_len,
                prompt=prompt,
                tok_embedding=self.tok_embedding,
                image_token_id=self.meta.get("image_token_id", None),
                use_i64_token=self.control_args.embedding_quantize is not None,
                event_name="qat_post_replace_generate",
                lookahead_config=lookahead_config,
            )
            logger.info("[qat_generate] done")
        except Exception as error:
            logger.exception("[qat_generate] failed post-replace generation: %s", error)

    @log_info
    def quantize(self, request: Request):  # noqa: C901
        if self.quant_recipe is None:
            return

        if self.decoder is None or (
            self.apply_embedding and self.tok_embedding is None
        ):
            return

        # check bit width graph io
        fixed_point_type = {"kv_type": torch.float32, "io_type": torch.float32}
        if self.quant_recipe.get_kv_io_bit_width() == 8:
            fixed_point_type["kv_type"] = torch.uint8
        elif self.quant_recipe.get_kv_io_bit_width() == 16:
            fixed_point_type["kv_type"] = torch.uint16
        else:
            raise RuntimeError(
                f"unknown kv io bit width {self.quant_recipe.get_kv_io_bit_width()}"
            )

        if self.quant_recipe.get_logits_output_bit_width() == 16:
            fixed_point_type["io_type"] = torch.uint16
        else:
            raise RuntimeError(
                f"unknown logits io bit width {self.quant_recipe.get_logits_output_bit_width()}"
            )

        data = request.method_data[TEXT_DECODER]
        audio_turns = request.method_data[
            AUDIO_ENCODER
        ].calibration_data.intermediate_outputs
        vision_turns = request.method_data[
            VISION_ENCODER
        ].calibration_data.intermediate_outputs
        if audio_turns is None:
            audio_turns = [[] for _ in range(len(data.calibration_data.datasets))]
        if vision_turns is None:
            vision_turns = [[] for _ in range(len(data.calibration_data.datasets))]
        intermediate_outputs = [
            [*audio_turn, *vision_turn]
            for audio_turn, vision_turn in zip(audio_turns, vision_turns)
        ]

        quantizer = make_quantizer(backend=data.backend, soc_model=data.soc_model)
        quantizer.set_recipe(self.quant_recipe.recipe)

        tok_embedding_quantizer = make_quantizer(
            quant_dtype=QuantDtype.use_16a8w,
            per_channel_conv=True,
            per_channel_linear=True,
            act_observer=MinMaxObserver,
            backend=data.backend,
            soc_model=data.soc_model,
        )

        with torch.no_grad():
            # prepare tok embedding model for ptq
            if self.apply_embedding:
                self.tok_embedding = torch.export.export(
                    self.tok_embedding,
                    self.tok_embedding.get_example_input(),
                    strict=True,
                ).module()

            # prepare decoder model for ptq
            self.decoder = torch.export.export(
                self.decoder, self.export_input, strict=True
            ).module()

            self.decoder = prepare_pt2e(self.decoder, quantizer)
            print("\n=== After prepare_pt2e for decoder ===")
            if getattr(self.control_args, "dump_quant_info", False):
                self._dump_quant_info(
                    self.decoder,
                    tag=f"decoder after prepare_pt2e ({self.mode.name.lower()})",
                    keyword=getattr(self.control_args, "dump_quant_filter", None),
                )
            if self.apply_embedding:
                self.tok_embedding = prepare_pt2e(
                    self.tok_embedding, tok_embedding_quantizer
                )
                if getattr(self.control_args, "dump_quant_info", False):
                    self._dump_quant_info(
                        self.tok_embedding,
                        tag=(
                            "tok_embedding after prepare_pt2e "
                            f"({self.mode.name.lower()})"
                        ),
                        keyword=getattr(self.control_args, "dump_quant_filter", None),
                    )

            # start calibration (only for kv mode or prefill mode without kv cache)
            if self.mode == Mode.DECODE or not self.model_args.use_kv_cache:
                self._calibrate(
                    model=self.decoder,
                    tokenizer=data.tokenizer,
                    event="prepare_pt2e",
                    user_calibration_data=data.calibration_data.datasets,
                    tok_embedding=self.tok_embedding,
                    intermediate_outputs=intermediate_outputs,
                )
            else:
                # one dummy inference to remove affine observer
                # error happened in convert_pt2e
                self.decoder(*self.export_input)

            self.decoder = convert_pt2e(self.decoder)
            print("\n=== After convert_pt2e for decoder ===")

            # Optional: replace quantized qweight/scale from user-provided QAT checkpoint.
            self._replace_qparams_from_qat_checkpoint(self.decoder)

            # Optional: run one forward smoke test right after QAT replacement,
            # before backend lowering/compilation.
            self._run_post_replace_smoke_test(self.decoder)

            # Optional: run one full text generation right after QAT replacement,
            # before backend lowering/compilation.
            self._run_post_replace_text_generation(self.decoder, data.tokenizer)

            if getattr(self.control_args, "dump_quant_info", False):
                dump_filter = getattr(self.control_args, "dump_quant_filter", None)
                self._dump_quant_info(
                    self.decoder,
                    tag=f"decoder after convert_pt2e ({self.mode.name.lower()})",
                    keyword=dump_filter,
                )
                if dump_filter:
                    self._dump_single_layer_quant_info(
                        self.decoder,
                        keyword=dump_filter,
                        tag=(
                            f"decoder target layer quant info "
                            f"({self.mode.name.lower()})"
                        ),
                    )

            # Optional experiment: force fill target quantized qweight right after
            # convert_pt2e, before any backend lowering.
            enable_force_fill = getattr(
                self.control_args,
                "enable_force_fill_qweight",
                False,
            )
            layer_keyword = getattr(
                self.control_args,
                "force_fill_layer_keyword",
                None,
            )
            fill_value = getattr(self.control_args, "force_fill_qweight_value", -1)
            fallback_attr = getattr(
                self.control_args,
                "force_fill_qweight_attr",
                None,
            )
            require_unique = getattr(
                self.control_args,
                "force_fill_require_unique",
                False,
            )
            try:
                fill_value = int(fill_value)
            except (TypeError, ValueError):
                logging.warning(
                    "Invalid force_fill_qweight_value=%s, fallback to -1",
                    fill_value,
                )
                fill_value = -1

            did_fill = False
            if enable_force_fill and layer_keyword:
                did_fill = self._force_fill_qweight_by_layer_keyword(
                    self.decoder,
                    layer_keyword=layer_keyword,
                    fill_value=fill_value,
                    require_unique=require_unique,
                )
                if not did_fill:
                    logging.warning(
                        "post-convert force_fill by layer keyword failed for mode=%s "
                        "keyword=%s; fallback attr=%s",
                        self.mode.name.lower(),
                        layer_keyword,
                        fallback_attr,
                    )

            if enable_force_fill and not did_fill and fallback_attr:
                self._force_fill_get_attr_tensor(
                    self.decoder,
                    attr_name=fallback_attr,
                    fill_value=fill_value,
                )

            if getattr(self.control_args, "dump_quant_info", False) or getattr(
                self.control_args,
                "save_layer_qweights",
                False,
            ):
                is_hybrid = getattr(self.control_args, "model_mode", None) == "hybrid"
                if is_hybrid and self.mode == Mode.PREFILL:
                    logging.info(
                        "[dump_qweights] skip prefill dump in hybrid mode; "
                        "use decode dump as canonical weights"
                    )
                else:
                    self._dump_qweights_for_keywords(
                        self.decoder,
                        keywords=self._target_qweight_keywords(),
                        tag=f"decoder qweight dump by keyword ({self.mode.name.lower()})",
                    )


            # Saving Decode QDQ Model EP for SQNR evaluation
            if self.mode == Mode.DECODE:
                qdq_ep = torch.export.export(
                    self.decoder, self.export_input, strict=True
                )
                qdq_ep_path = f"{self.control_args.artifact}/{DECODE_QDQ_FILENAME}"
                torch.export.save(qdq_ep, qdq_ep_path)
                logging.info(f"QDQ EP saved to {qdq_ep_path}")

            if self.apply_embedding:
                self.tok_embedding = convert_pt2e(self.tok_embedding)
                if getattr(self.control_args, "dump_quant_info", False):
                    self._dump_quant_info(
                        self.tok_embedding,
                        tag=(
                            "tok_embedding after convert_pt2e "
                            f"({self.mode.name.lower()})"
                        ),
                        keyword=getattr(self.control_args, "dump_quant_filter", None),
                    )

            if self.control_args.verbose and self.mode == Mode.DECODE:
                if self.apply_embedding:
                    qdq_intermediate_outputs = request.method_data[
                        VISION_ENCODER
                    ].calibration_data.qdq_intermediate_outputs
                self._calibrate(
                    model=self.decoder,
                    tokenizer=data.tokenizer,
                    event="convert_pt2e",
                    user_calibration_data=data.calibration_data.datasets,
                    tok_embedding=self.tok_embedding,
                    intermediate_outputs=qdq_intermediate_outputs,
                )

        # save logit's quantization attributes to meta
        self._save_logits_quant_attrs()

        # save output KV cache's quantization attributes to meta for attention sink
        self._save_output_kv_cache_quant_attrs()
        self._dump_kv_quant_attrs_json()

        # setup quantized IO
        self.passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
        self.passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
            "get_quant_io_dtype_fn"
        ] = partial(self._tag_ios, fixed_point_type=fixed_point_type)
        if self.tok_embedding_passes_job is not None:
            self.tok_embedding_passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
            self.tok_embedding_passes_job[TagQuantIO][
                QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY
            ]["get_quant_io_dtype_fn"] = partial(
                self._tag_ios, fixed_point_type=fixed_point_type
            )


class HybridTextDecoder(Component):
    @log_info
    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
        apply_embedding: bool = False,
    ):
        self.decode = TextDecoder(
            control_args,
            config,
            Mode.DECODE,
            apply_embedding=apply_embedding,
        )
        self.prefill = TextDecoder(
            control_args,
            config,
            Mode.PREFILL,
            apply_embedding=apply_embedding,
        )
        self.control_args = control_args
        self.config = config
        self.set_next(self.decode).set_next(self.prefill)

        self.apply_embedding = apply_embedding

    def _encoding_override(self, decode_model, prefill_model):  # noqa: C901
        pbq_target = {
            torch.ops.torchao.dequantize_affine,
            torch.ops.torchao.quantize_affine,
        }
        pcq_target = {
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.quantized_decomposed.quantize_per_channel.default,
        }
        ptq_target = {
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        }
        qdq_target = pbq_target | pcq_target | ptq_target

        def compare_nodes(decode_node, prefill_node):
            def info(node):
                return node.name + (
                    str(node.meta["nn_module_stack"].values())
                    if node.op == "call_function"
                    else ""
                )

            assert info(decode_node) == info(
                prefill_node
            ), f"found unmatched order for ops: {decode_node} va {prefill_node}"

        def resolve_param_target(node):
            return (
                node
                if node.op == "call_function" and node.target not in qdq_target
                else resolve_param_target(list(node.users)[0])
            )

        def activation_override(decode_node, prefill_node):
            for decode_user, prefill_user in zip(
                list(decode_node.users), list(prefill_node.users)
            ):
                assert decode_user.target == prefill_user.target, (
                    "found unmatched targets: "
                    f"{decode_user.target} vs {prefill_user.target}"
                )
                if decode_user.target in qdq_target:
                    prefill_user.args = (prefill_user.args[0], *decode_user.args[1:])
                    activation_override(decode_user, prefill_user)

        def parameter_override(decode_node, prefill_node):
            setattr(
                prefill_model,
                prefill_node.target,
                getattr(decode_model, decode_node.target),
            )
            # scale / zero point are part of op's attributes
            if list(decode_node.users)[0].target in ptq_target:
                activation_override(decode_node, prefill_node)

        # copy encoding for hybrid mode
        parameters = [
            {
                n: resolve_param_target(n)
                for n in model.graph.nodes
                if n.op == "get_attr"
            }
            for model in (decode_model, prefill_model)
        ]
        activations = [
            [
                n
                for n in model.graph.nodes
                if n.target not in qdq_target
                and n.op in {"call_function", "placeholder"}
            ]
            for model in (decode_model, prefill_model)
        ]
        # check topology order by node name & nn_module_stack
        for act_decode, act_prefill in zip(*activations):
            compare_nodes(act_decode, act_prefill)

        for op_decode, op_prefill in zip(*[p.values() for p in parameters]):
            compare_nodes(op_decode, op_prefill)
        # perform encoding override
        for act_decode, act_prefill in zip(*activations):
            activation_override(act_decode, act_prefill)

        for param_decode, param_prefill in zip(*[p.keys() for p in parameters]):
            parameter_override(param_decode, param_prefill)

        prefill_model.recompile()

    @log_info
    def compile(self, request: Request):  # noqa: C901
        # perform encoding override for hybrid mode
        # ---
        # theoretically decode & prefill model should share the same encoding
        # given that they are using the identical calibration dataset.
        #
        # however, pytorch will use different computaion kernels for different
        # workloads (AR1 vs ARN) which will introduce some numerical discrepancy.
        #
        # here we use a mechanism to make sure the encoding align correctly and
        # save AoT quantization time as well.
        # ---
        if self.prefill.decoder is not None and self.prefill.model_args.use_kv_cache:
            self._encoding_override(
                decode_model=self.decode.decoder,
                prefill_model=self.prefill.decoder,
            )
            if self.apply_embedding:
                self._encoding_override(
                    decode_model=self.decode.tok_embedding,
                    prefill_model=self.prefill.tok_embedding,
                )
            # TextDecoder.quantize() dumps prefill KV attrs before hybrid
            # encoding override runs. Re-export the prefill sidecar here so
            # PD handoff uses the final encoding that will actually be lowered.
            self.prefill.kv_quant_attrs_sidecar = {"output": []}
            self.prefill._save_output_kv_cache_quant_attrs()
            self.prefill._dump_kv_quant_attrs_json()

        # prepare lowering tok_embedding if applicable
        if self.apply_embedding:
            tok_embedding_data = request.method_data[TOK_EMBEDDING]
            models = [
                d for d in [self.decode, self.prefill] if d.tok_embedding is not None
            ]
            tok_embedding_example_inputs = [
                m.tok_embedding_export_input for m in models if m is not None
            ]  # tokens
            tok_embedding_graph_names = TOK_EMBEDDING_GRAPH_NAMES[: len(models)]

        # prepare lowering decoder
        data = request.method_data[TEXT_DECODER]
        models = [d for d in [self.decode, self.prefill] if d.decoder is not None]
        example_inputs = [m.export_input for m in models if m is not None]
        graph_names = DECODER_GRAPH_NAMES[: len(models)]

        # start lowering
        if self.apply_embedding:
            tok_embedding_edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                module=dict(
                    zip(
                        tok_embedding_graph_names,
                        [model.tok_embedding for model in models],
                    )
                ),
                inputs=dict(
                    zip(tok_embedding_graph_names, tok_embedding_example_inputs)
                ),
                compiler_specs=dict(
                    zip(tok_embedding_graph_names, tok_embedding_data.compile_spec)
                ),
                dep_table=dict(
                    zip(
                        tok_embedding_graph_names,
                        [model.tok_embedding_dep_table for model in models],
                    )
                ),
                passes_job=dict(
                    zip(
                        tok_embedding_graph_names,
                        [model.tok_embedding_passes_job for model in models],
                    )
                ),
            )
            if self.control_args.verbose:
                for ep in tok_embedding_edge_prog_mgr._edge_programs.values():
                    print_delegation_info(ep.graph_module)

            executorch_config = ExecutorchBackendConfig(
                # For shared buffer, user must pass the memory address
                # which is allocated by RPC memory to executor runner
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False,
                    alloc_graph_output=False,
                ),
            )
            tok_embedding_exec_prog_mgr = tok_embedding_edge_prog_mgr.to_executorch(
                executorch_config
            )
            data = request.method_data[TOK_EMBEDDING]
            with open(
                f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
            ) as file:
                tok_embedding_exec_prog_mgr.write_to_file(file)

        # for node in model.decoder.graph.nodes:
        #     print(node.op, node.target, node.args, node.meta.get("val", None))
        # exit()

        # decoder lowering
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module=dict(zip(graph_names, [model.decoder for model in models])),
            inputs=dict(zip(graph_names, example_inputs)),
            compiler_specs=dict(zip(graph_names, data.compile_spec)),
            constant_methods={**self.decode.meta},
            dep_table=dict(zip(graph_names, [model.dep_table for model in models])),
            passes_job=dict(zip(graph_names, [model.passes_job for model in models])),
            skip_node_op_set={"llama.fallback.default"},
        )

        if getattr(self.control_args, "replace_with_qat_checkpoint", None) and getattr(
            self.control_args, "qat_verify_chain", False
        ):
            for graph_name, model in zip(graph_names, models):
                try:
                    lowered_gm = edge_prog_mgr.exported_program(graph_name).graph_module
                    model._run_qat_backend_usage_check(
                        lowered_gm,
                        stage=f"post_lowering:{graph_name}",
                    )
                except Exception as error:
                    logging.warning(
                        "[qat_backend_check] failed on graph=%s: %s",
                        graph_name,
                        error,
                    )

        if self.config.num_sharding > 1:
            for graph_name in graph_names:
                update_spill_fill_size(edge_prog_mgr.exported_program(graph_name))

        if self.control_args.verbose:
            for ep in edge_prog_mgr._edge_programs.values():
                print_delegation_info(ep.graph_module)

        executorch_config = ExecutorchBackendConfig(
            # For shared buffer, user must pass the memory address
            # which is allocated by RPC memory to executor runner
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
        )
        exec_prog_mgr = edge_prog_mgr.to_executorch(executorch_config)
        data = request.method_data[TEXT_DECODER]
        with open(
            f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
        ) as file:
            exec_prog_mgr.write_to_file(file)


class Modality(Component):
    def __init__(
        self, control_args: argparse.Namespace, config: LLMModelConfig, modality
    ):
        self.control_args = control_args
        self.model = None
        self.modality = modality
        repo_id = config.repo_id

        if config := getattr(config, modality, None):
            if modality == TEXT_ENCODER or modality == AUDIO_ENCODER:
                raise NotImplementedError(f"{modality} is under development")

            auto_model = AutoModel.from_pretrained(
                repo_id, _attn_implementation="eager"
            )
            # Create an instance of the config class since it has init=False
            self.model = config().create_encoder(auto_model.config)
            # set strict to false to simplify parameter loading for non-text models
            auto_model = auto_model.eval()
            self.model = self.model.eval()
            self.model.load_state_dict(auto_model.state_dict(), strict=False)
            self.example_input = self.model.get_example_inputs()

            # set quant recipe
            self.quant_recipe: EncoderQuantRecipe = (
                config.quant_recipe(True) if config.quant_recipe else None
            )

    def compile(self, request: Request):
        if self.model is None:
            return

        request_data = request.method_data[self.modality]
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module=self.model,
            inputs=self.example_input,
            compiler_specs=request_data.compile_spec,
        )
        if self.control_args.verbose:
            print_delegation_info(edge_prog_mgr.exported_program().graph_module)

        exec_prog_mgr = edge_prog_mgr.to_executorch(ExecutorchBackendConfig())
        data = request.method_data[self.modality]
        with open(
            f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
        ) as file:
            exec_prog_mgr.write_to_file(file)

    def _calibrate(self, model, calibration_datasets):
        outputs = []
        for turn in calibration_datasets:
            outputs_each_turn = [model(*data) for data in turn]
            outputs.append(outputs_each_turn)
        return outputs

    def quantize(self, request: Request):
        if self.model is None:
            return

        request_data = request.method_data[self.modality]
        calibration_datasets = request_data.calibration_data.datasets

        with torch.no_grad():
            self.model = torch.export.export(self.model, self.example_input).module()

            if request_data.skip_quantize:
                logging.info(f"skipping encoder quantization for {self.modality}")
                intermediate_outputs = self._calibrate(self.model, calibration_datasets)
                request_data.calibration_data.intermediate_outputs = (
                    intermediate_outputs
                )
                return

            quantizer = make_quantizer(
                backend=request_data.backend, soc_model=request_data.soc_model
            )
            quantizer.set_recipe(self.quant_recipe.recipe)
            self.model = prepare_pt2e(self.model, quantizer)

            # start calibration
            intermediate_outputs = self._calibrate(self.model, calibration_datasets)
            request_data.calibration_data.intermediate_outputs = intermediate_outputs

            self.model = convert_pt2e(self.model)

            # QDQ inference
            if self.control_args.verbose:
                qdq_intermediate_outputs = self._calibrate(
                    self.model, calibration_datasets
                )
                request_data.calibration_data.qdq_intermediate_outputs = (
                    qdq_intermediate_outputs
                )


class MultiModalManager(Component):
    def __init__(self, control_args: argparse.Namespace, config: LLMModelConfig):
        self.audio_encoder = Modality(
            control_args,
            config,
            AUDIO_ENCODER,
        )
        self.text_encoder = Modality(
            control_args,
            config,
            TEXT_ENCODER,
        )
        self.vision_encoder = Modality(
            control_args,
            config,
            VISION_ENCODER,
        )
        self.text_decoder = HybridTextDecoder(
            control_args,
            config,
            apply_embedding=self.audio_encoder.model or self.vision_encoder.model,
        )
        self._modalities = [
            AUDIO_ENCODER,
            TEXT_ENCODER,
            VISION_ENCODER,
            TOK_EMBEDDING,
            TEXT_DECODER,
        ]
        # build dependency chain
        self.set_next(self.vision_encoder).set_next(self.audio_encoder).set_next(
            self.text_decoder
        )

    def process(self, request: Request) -> Request:
        Processor.process(self, request)

    @log_info
    def compile(
        self,
        compile_specs: Dict[str, List[CompileSpec]],
        pte_filenames: Dict[str, str],
    ):
        compile_request = Request(
            inspect.currentframe().f_code.co_name,
            {
                m: Request.Data(
                    compile_spec=compile_specs[m],
                    pte_filename=pte_filenames[m],
                )
                for m in self._modalities
            },
        )
        self.process(compile_request)

    @log_info
    def quantize(
        self,
        calibration_data: Dict[str, List[Any]],
        skip_quantize: Dict[str, bool],
        tokenizer,
        backend,
        soc_model,
    ):
        quantize_request = Request(
            inspect.currentframe().f_code.co_name,
            {
                m: Request.Data(
                    calibration_data=Request.CalibrationData(
                        datasets=calibration_data[m]
                    ),
                    skip_quantize=skip_quantize.get(m, False),
                    tokenizer=tokenizer,
                    backend=backend,
                    soc_model=soc_model,
                )
                for m in self._modalities
            },
        )
        self.process(quantize_request)
