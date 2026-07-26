#!/usr/bin/env python3
"""
ExecuTorch QNN .pte 基于 QAT checkpoint 重建量化权重的逆序实时删除与索引生成

Canonical location:
- examples/qualcomm/oss_scripts/llama/tools/pte_qat_checkpoint_reverse_strip.py

功能:
- 输入 old.pte 与 QAT safetensors checkpoint
- 直接按 llm_wrappers.py 中的 QAT 替换逻辑重建导出后实际写入 PTE 的量化权重
- 对 output.conv 按 int8 per-channel 的 64 行 block 逐块匹配删除
- 对 28 层 decoder 线性层按 int4pack block 匹配删除
- 记录所有待删块的源区间后，按保留区间重建 stripped.pte
- 输出:
  1) stripped.pte
  2) index.json
  3) report.txt

当前范围:
- 处理 output.conv
- 处理 28 层 decoder 线性层:
  attention.wq/wk/wv/wo_conv + feed_forward.w1/w3/w2_conv
- 不处理 tok_embeddings
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors.torch import load_file


FORWARD_LAYER_OPS = [
    "attention.wq_conv",
    "attention.wk_conv",
    "attention.wv_conv",
    "attention.wo_conv",
    "feed_forward.w1_conv",
    "feed_forward.w3_conv",
    "feed_forward.w2_conv",
]
REVERSE_LAYER_OPS = list(reversed(FORWARD_LAYER_OPS))
TARGET_MAP = {
    "attention.wq_conv": "self_attn.q_proj",
    "attention.wk_conv": "self_attn.k_proj",
    "attention.wv_conv": "self_attn.v_proj",
    "attention.wo_conv": "self_attn.o_proj",
    "feed_forward.w1_conv": "mlp.gate_proj",
    "feed_forward.w3_conv": "mlp.up_proj",
    "feed_forward.w2_conv": "mlp.down_proj",
}
BINARY_INDEX_MAGIC = 0x49515445
BINARY_INDEX_VERSION = 2
BINARY_INDEX_HEADER_SIZE = 32
BINARY_INDEX_RECORD_SIZE = 20
WEIGHT_KIND_OUTPUT_CONV = 0
WEIGHT_KIND_DECODER_LINEAR = 1
INVALID_LAYER_ID = 0xFF
INVALID_OP_ID = 0xFF
DEFAULT_NUM_LAYERS = 28
DEFAULT_OLD_PTE = "/root/autodl-tmp/executorch/llama_qnn/hybrid_llama_qnn.pte"

_LLAMA_QNN_QUANT_PROFILE_CAPABILITIES = {
    "exact_f32_scale_bits": True,
    "raw_operator_parameters": True,
    "operation_tensor_sources": True,
    "embedded_static_affine_tensor_bytes": True,
    "blockwise_weight_scale_payload": True,
    "blockwise_weight_payload_digest": True,
    "structured_decoder_tensor_bindings": True,
    "complete_u16_tensor_qparams": True,
}

def _gptq2_qnn_weight_code_contract(group_size: int) -> Dict[str, Any]:
    return {
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


@dataclass
class QATWeightSpec:
    name: str
    layer_id: Optional[int]
    qat_base_key: str
    quant_type: str = "int4block"


def expected_reverse_names(
    include_output: bool = True, layer_start: int = 0, layer_end_exclusive: int = DEFAULT_NUM_LAYERS
) -> List[str]:
    names: List[str] = ["output.conv"] if include_output else []
    for layer_id in range(layer_end_exclusive - 1, layer_start - 1, -1):
        prefix = f"layers.{layer_id}"
        for op in REVERSE_LAYER_OPS:
            names.append(f"{prefix}.{op}")
    return names




def build_qat_specs(
    include_output: bool = True, layer_start: int = 0, layer_end_exclusive: int = DEFAULT_NUM_LAYERS
) -> Dict[str, QATWeightSpec]:
    out: Dict[str, QATWeightSpec] = {}
    if include_output:
        out["output.conv"] = QATWeightSpec(
            name="output.conv",
            layer_id=None,
            qat_base_key="lm_head",
            quant_type="int8_per_channel",
        )
    for layer_id in range(layer_start, layer_end_exclusive):
        for local_op, qat_local in TARGET_MAP.items():
            name = f"layers.{layer_id}.{local_op}"
            out[name] = QATWeightSpec(
                name=name,
                layer_id=layer_id,
                qat_base_key=f"model.layers.{layer_id}.{qat_local}",
            )
    return out




def encode_weight_id(name: str) -> Tuple[int, int, int]:
    if name == "output.conv":
        return (WEIGHT_KIND_OUTPUT_CONV, INVALID_LAYER_ID, INVALID_OP_ID)
    prefix = "layers."
    if not name.startswith(prefix):
        raise ValueError(f"unsupported weight name: {name}")
    parts = name.split(".", 2)
    if len(parts) != 3:
        raise ValueError(f"unsupported weight name: {name}")
    layer_id = int(parts[1])
    op_name = parts[2]
    try:
        op_id = FORWARD_LAYER_OPS.index(op_name)
    except ValueError as exc:
        raise ValueError(f"unsupported decoder op: {name}") from exc
    return (WEIGHT_KIND_DECODER_LINEAR, layer_id, op_id)


def build_split_starts(num_layers: int, num_splits: int) -> List[int]:
    if num_splits <= 1:
        return [0]
    step = int(num_layers / num_splits)
    if step <= 0:
        raise ValueError(
            f"num_splits={num_splits} is too large for num_layers={num_layers}"
        )
    return list(range(0, num_layers, step))


def split_id_for_record(
    layer_id: Optional[int], num_layers: int, num_splits: int
) -> int:
    if num_splits <= 1:
        return 0
    if layer_id is None:
        return num_splits - 1
    split_starts = build_split_starts(num_layers, num_splits)
    split_id = 0
    for idx, start in enumerate(split_starts):
        if layer_id >= start:
            split_id = idx
        else:
            break
    return min(split_id, num_splits - 1)


def _enc_nibble_pair(high: int, low: int) -> int:
    return ((high & 0xF) << 4) | (low & 0xF)


def encode_int4_tile_32x32(tile: np.ndarray) -> bytes:
    if tile.shape != (32, 32):
        raise ValueError(f"tile must be 32x32, got {tile.shape}")

    perm = (4, 0, 5, 1, 6, 2, 7, 3)
    vals: List[int] = []
    for bc in range(4):
        for br in range(4):
            r0 = br * 8
            c0 = bc * 8
            for r in range(8):
                for c in perm:
                    v = int(tile[r0 + r, c0 + c])
                    if not (-8 <= v <= 7):
                        raise ValueError(f"int4 value out of range: {v}")
                    vals.append(v & 0xF)

    out = bytearray()
    for i in range(0, len(vals), 2):
        out.append(_enc_nibble_pair(vals[i], vals[i + 1]))
    return bytes(out)


def encode_int4_block_64xN(block64: np.ndarray) -> bytes:
    rows, cols = block64.shape
    if rows != 64:
        raise ValueError(f"int4 block rows must be 64, got {rows}")
    if cols % 32 != 0:
        raise ValueError(f"int4 block cols must be divisible by 32, got {cols}")

    out = bytearray()
    tile_cols = cols // 32
    for bc in range(tile_cols):
        c0 = bc * 32
        for br in range(2):
            r0 = br * 32
            tile = block64[r0 : r0 + 32, c0 : c0 + 32]
            out.extend(encode_int4_tile_32x32(tile))
    return bytes(out)


# -----------------------------
# 编码器 B: int8 per-channel
# -----------------------------

def encode_int8_tile_32x32(tile: np.ndarray) -> bytes:
    if tile.shape != (32, 32):
        raise ValueError(f"tile must be 32x32, got {tile.shape}")
    parts: List[bytes] = []
    for c in range(0, 32, 4):
        sub = tile[:, c : c + 4]
        parts.append(sub.reshape(-1).astype(np.int8, copy=False).tobytes())
    return b"".join(parts)


def encode_int8_block_64xN(block64: np.ndarray) -> bytes:
    rows, cols = block64.shape
    if rows != 64:
        raise ValueError(f"int8 block rows must be 64, got {rows}")
    if cols % 32 != 0:
        raise ValueError(f"int8 block cols must be divisible by 32, got {cols}")

    out: List[bytes] = []
    tile_cols = cols // 32
    for bc in range(tile_cols):
        c0 = bc * 32
        for br in range(2):
            r0 = br * 32
            tile = block64[r0 : r0 + 32, c0 : c0 + 32]
            out.append(encode_int8_tile_32x32(tile))
    return b"".join(out)


# -----------------------
# QAT checkpoint helpers
# -----------------------

def _pack_factor(bits: int) -> int:
    if bits not in {2, 3, 4, 8}:
        raise NotImplementedError(f"unsupported bits={bits}")
    return 32 // bits


def _rounded_zero_points_from_unpacked(z_unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    maxq = 2**bits - 1
    return torch.clamp(torch.round(z_unpacked), min=0, max=maxq)


def _unpack_rows(packed: torch.Tensor, bits: int, rows: int, cols: int) -> torch.Tensor:
    pack_factor = _pack_factor(bits)
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


def _unpack_cols(packed: torch.Tensor, bits: int, rows: int, cols: int) -> torch.Tensor:
    pack_factor = _pack_factor(bits)
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


def _infer_bits_from_packed_rows(in_features: int, packed_rows: int, fallback_bits: int) -> int:
    for bits in (2, 3, 4, 8):
        pack_factor = _pack_factor(bits)
        if math.ceil(in_features / pack_factor) == packed_rows:
            return bits
    return fallback_bits


def _prepare_qat_qweight(
    q_packed: torch.Tensor,
    z_packed: Optional[torch.Tensor],
    *,
    qweight_mode: str,
    bits_hint: int,
    group_size_hint: int,
) -> torch.Tensor:
    if q_packed.ndim != 2:
        raise ValueError(f"q_packed must be 2D, got {tuple(q_packed.shape)}")

    out_features = int(q_packed.shape[1])
    guessed_in_features = int(q_packed.shape[0] * _pack_factor(bits_hint))
    bits = _infer_bits_from_packed_rows(guessed_in_features, int(q_packed.shape[0]), bits_hint)
    in_features = int(q_packed.shape[0] * _pack_factor(bits))
    q_unpacked = _unpack_rows(q_packed, bits, in_features, out_features).to(torch.float16)

    if qweight_mode == "qweight_minus_qzeros" and z_packed is not None:
        num_groups = int(z_packed.shape[0])
        z_unpacked = _unpack_cols(z_packed, bits, num_groups, out_features)
        z_unpacked = _rounded_zero_points_from_unpacked(z_unpacked.to(torch.float32), bits).to(torch.float16)
        group_size = group_size_hint if group_size_hint > 0 else max(1, in_features // max(1, num_groups))
        z_expanded = z_unpacked.repeat_interleave(group_size, dim=0)
        if z_expanded.shape[0] < in_features:
            pad_rows = in_features - z_expanded.shape[0]
            z_expanded = torch.cat([z_expanded, z_expanded[-1:, :].repeat(pad_rows, 1)], dim=0)
        z_expanded = z_expanded[:in_features, :]
        q_unpacked = q_unpacked - z_expanded

    return q_unpacked.transpose(0, 1).contiguous()


def _prepare_output_qweight_int8(q_packed: torch.Tensor, bits_hint: int) -> torch.Tensor:
    del bits_hint
    if q_packed.ndim != 2:
        raise ValueError(f"output q_packed must be 2D, got {tuple(q_packed.shape)}")

    out_features = int(q_packed.shape[1])
    bits = 8
    in_features = int(q_packed.shape[0] * _pack_factor(bits))
    q_unpacked = _unpack_rows(q_packed, bits, in_features, out_features).to(torch.int16)
    return q_unpacked.transpose(0, 1).contiguous() - 128


def _to_checked_int4_matrix(q_target: torch.Tensor, spec: QATWeightSpec) -> np.ndarray:
    q = q_target.detach().cpu().to(torch.float32).contiguous()
    if q.ndim != 2:
        raise ValueError(f"{spec.name}: q_target must be 2D, got {tuple(q.shape)}")

    rounded = torch.round(q)
    diff = (q - rounded).abs().max().item() if q.numel() > 0 else 0.0
    if diff > 1e-3:
        raise ValueError(f"{spec.name}: q_target not integer-like, max_round_diff={diff}")

    rounded_i32 = rounded.to(torch.int32)
    q_min = int(rounded_i32.min().item()) if rounded_i32.numel() > 0 else 0
    q_max = int(rounded_i32.max().item()) if rounded_i32.numel() > 0 else 0
    if q_min < -8 or q_max > 7:
        raise ValueError(f"{spec.name}: int4 range violated, min={q_min}, max={q_max}")

    return rounded_i32.to(torch.int8).numpy().astype(np.int8, copy=False)


def _to_checked_int8_matrix(q_target: torch.Tensor, spec: QATWeightSpec) -> np.ndarray:
    q = q_target.detach().cpu().to(torch.int16).contiguous()
    if q.ndim != 2:
        raise ValueError(f"{spec.name}: q_target must be 2D, got {tuple(q.shape)}")

    q_min = int(q.min().item()) if q.numel() > 0 else 0
    q_max = int(q.max().item()) if q.numel() > 0 else 0
    if q_min < -128 or q_max > 127:
        raise ValueError(f"{spec.name}: int8 range violated, min={q_min}, max={q_max}")

    return q.to(torch.int8).numpy().astype(np.int8, copy=False)


def split_blocks_encoded_from_qat(
    spec: QATWeightSpec,
    qat_sd: Dict[str, torch.Tensor],
    *,
    bits_hint: int,
    group_size: int,
    qweight_mode: str,
) -> List[bytes]:
    base = spec.qat_base_key
    qkey = f"{base}.qweight"
    skey = f"{base}.scales"

    if qkey not in qat_sd:
        raise KeyError(f"missing checkpoint tensor: {qkey}")
    if skey not in qat_sd:
        raise KeyError(f"missing checkpoint tensor: {skey}")

    if spec.quant_type == "int8_per_channel":
        arr = _to_checked_int8_matrix(_prepare_output_qweight_int8(qat_sd[qkey], bits_hint), spec)
        h, w = arr.shape
        if h % 64 != 0:
            raise ValueError(f"{spec.name}: int8 tensor height must be divisible by 64, got {arr.shape}")
        if w % 32 != 0:
            raise ValueError(f"{spec.name}: int8 tensor width must be divisible by 32, got {arr.shape}")
        return [encode_int8_block_64xN(arr[r0 : r0 + 64, :]) for r0 in range(0, h, 64)]

    zkey = f"{base}.qzeros"
    q_target = _prepare_qat_qweight(
        qat_sd[qkey],
        qat_sd.get(zkey, None),
        qweight_mode=qweight_mode,
        bits_hint=bits_hint,
        group_size_hint=group_size,
    )
    arr = _to_checked_int4_matrix(q_target, spec)

    h, w = arr.shape
    if h % 64 != 0:
        raise ValueError(f"{spec.name}: int4block tensor height must be divisible by 64, got {arr.shape}")
    if w % 32 != 0:
        raise ValueError(f"{spec.name}: int4block tensor width must be divisible by 32, got {arr.shape}")
    return [encode_int4_block_64xN(arr[r0 : r0 + 64, :]) for r0 in range(0, h, 64)]


# -----------------------
# PTE strip helpers
# -----------------------

def _has_overlap(intervals: List[Tuple[int, int]], start: int, end: int) -> bool:
    for s, e in intervals:
        if end <= s:
            continue
        if start >= e:
            continue
        return True
    return False


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[int, int]] = [sorted_intervals[0]]
    for s, e in sorted_intervals[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def _invert_to_keep_intervals(total_size: int, deleted_merged: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    keep: List[Tuple[int, int]] = []
    cursor = 0
    for s, e in deleted_merged:
        if cursor < s:
            keep.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < total_size:
        keep.append((cursor, total_size))
    return keep


def _build_stripped_from_keep_ranges(source_bytes: bytes, keep_intervals: List[Tuple[int, int]]) -> bytearray:
    total_keep = sum(e - s for s, e in keep_intervals)
    out = bytearray(total_keep)
    src_view = memoryview(source_bytes)
    out_view = memoryview(out)
    dst = 0
    for s, e in keep_intervals:
        length = e - s
        out_view[dst : dst + length] = src_view[s:e]
        dst += length
    return out


def realtime_delete(
    old_pte: Path,
    specs_by_name: Dict[str, QATWeightSpec],
    qat_sd: Dict[str, torch.Tensor],
    *,
    bits_hint: int,
    group_size: int,
    qweight_mode: str,
    include_output: bool,
    strict: bool = True,
    show_progress: bool = True,
    search_direction: str = "reverse",
    num_splits: int = 1,
    num_layers: int = DEFAULT_NUM_LAYERS,
    layer_start: int = 0,
    layer_end_exclusive: int = DEFAULT_NUM_LAYERS,
    split_layer_starts: Optional[List[int]] = None,
    shard_index: Optional[int] = None,
) -> Tuple[bytearray, Dict[str, Any], List[str], bytes]:
    order = expected_reverse_names(
        include_output=include_output,
        layer_start=layer_start,
        layer_end_exclusive=layer_end_exclusive,
    )
    source_bytes = old_pte.read_bytes()
    old_size = len(source_bytes)

    records: List[Dict[str, Any]] = []
    report_lines: List[str] = []

    delete_order = 0
    total_deleted_bytes = 0
    started_at = time.time()
    available_names = [name for name in order if name in specs_by_name]
    total_layers_to_process = len(available_names)
    layer_done = 0
    taken_intervals: List[Tuple[int, int]] = []

    def record_split_id(layer_id: Optional[int]) -> int:
        if shard_index is not None:
            return shard_index
        return split_id_for_record(layer_id, num_layers, num_splits)

    for name in order:
        if name not in specs_by_name:
            msg = f"[WARN] missing weight spec for {name}, skip"
            report_lines.append(msg)
            if strict:
                raise RuntimeError(msg)
            continue

        spec = specs_by_name[name]
        blocks = split_blocks_encoded_from_qat(
            spec,
            qat_sd,
            bits_hint=bits_hint,
            group_size=group_size,
            qweight_mode=qweight_mode,
        )
        layer_done += 1

        report_lines.append(
            f"[INFO] deleting {name} quant_type={spec.quant_type} blocks={len(blocks)} from {spec.qat_base_key}"
        )
        if show_progress:
            print(
                f"\n[progress] layer {layer_done}/{total_layers_to_process}: {name} | blocks={len(blocks)} | search={search_direction}",
                flush=True,
            )

        block_items = list(enumerate(blocks))
        if search_direction == "reverse":
            block_items = list(reversed(block_items))

        for block_id, block in block_items:
            search_anchor = len(source_bytes) if search_direction == "reverse" else 0
            if search_direction == "reverse":
                hit = source_bytes.rfind(block, 0, len(source_bytes))
                while hit >= 0 and _has_overlap(taken_intervals, hit, hit + len(block)):
                    hit = source_bytes.rfind(block, 0, hit)
            else:
                hit = source_bytes.find(block, 0)
                while hit >= 0 and _has_overlap(taken_intervals, hit, hit + len(block)):
                    hit = source_bytes.find(block, hit + 1)

            if hit < 0:
                msg = (
                    f"[ERROR] block not found: name={name} block_id={block_id} "
                    f"search_start={search_anchor} block_len={len(block)}"
                )
                report_lines.append(msg)
                if strict:
                    raise RuntimeError(msg)
                records.append(
                    {
                        "name": name,
                        "weight_kind": None,
                        "layer_id": spec.layer_id,
                        "op_id": None,
                        "split_id": record_split_id(spec.layer_id),
                        "block_id": block_id,
                        "quant_type": spec.quant_type,
                        "current_offset_before_delete": None,
                        "source_offset": None,
                        "source_end_exclusive": None,
                        "length": len(block),
                        "sha256": hashlib.sha256(block).hexdigest(),
                        "delete_order": delete_order,
                        "deleted": False,
                        "error": "not_found",
                    }
                )
                delete_order += 1
                continue

            length = len(block)
            weight_kind, layer_id, op_id = encode_weight_id(name)
            records.append(
                {
                    "name": name,
                    "weight_kind": weight_kind,
                    "layer_id": layer_id,
                    "op_id": op_id,
                    "split_id": record_split_id(spec.layer_id),
                    "block_id": block_id,
                    "quant_type": spec.quant_type,
                    "current_offset_before_delete": int(hit),
                    "source_offset": int(hit),
                    "source_end_exclusive": int(hit + length),
                    "length": int(length),
                    "sha256": hashlib.sha256(block).hexdigest(),
                    "delete_order": delete_order,
                    "deleted": True,
                }
            )
            taken_intervals.append((int(hit), int(hit + length)))
            total_deleted_bytes += length
            delete_order += 1

            if show_progress:
                should_print = (
                    block_id == 0
                    or (block_id + 1) == len(blocks)
                    or ((block_id + 1) % 16 == 0)
                )
                if should_print:
                    elapsed = time.time() - started_at
                    msg = (
                        f"\r  block {block_id + 1}/{len(blocks)} "
                        f"| global_delete_order={delete_order} "
                        f"| deleted_bytes={total_deleted_bytes} "
                        f"| source_hit={hit} "
                        f"| elapsed={elapsed:.1f}s"
                    )
                    print(msg, end="", flush=True)

            report_lines.append(
                f"  - block_id={block_id} hit={hit} len={length} delete_order={delete_order - 1}"
            )

        if show_progress:
            print("", flush=True)

    deleted_merged = _merge_intervals(taken_intervals)
    keep_intervals = _invert_to_keep_intervals(old_size, deleted_merged)
    stripped_buf = _build_stripped_from_keep_ranges(source_bytes, keep_intervals)
    final_size = len(stripped_buf)

    records.sort(key=lambda rec: int(rec["source_offset"]) if rec.get("source_offset") is not None else -1)
    index_payload: Dict[str, Any] = {
        "source_pte": str(old_pte),
        "old_pte_size": old_size,
        "final_pte_size": final_size,
        "total_deleted_bytes": total_deleted_bytes,
        "search_mode": "immutable_source_search_then_rebuild_keep_ranges",
        "search_direction": search_direction,
        "delete_order_mode": "reverse_graph_order",
        "source_file_modified": False,
        "working_buffer_mode": "rebuild_from_keep_intervals",
        "deleted_intervals_merged_count": len(deleted_merged),
        "keep_intervals_count": len(keep_intervals),
        "qweight_mode": qweight_mode,
        "bits_hint": bits_hint,
        "group_size": group_size,
        "num_splits": num_splits,
        "num_layers": num_layers,
        "layer_start": layer_start,
        "layer_end_exclusive": layer_end_exclusive,
        "split_layer_starts": split_layer_starts
        if split_layer_starts is not None
        else build_split_starts(num_layers, num_splits),
        "shard_index": shard_index,
        "stripped_output_weight": include_output,
        "records": records,
    }
    return stripped_buf, index_payload, report_lines, source_bytes




def write_binary_index(index_bin: Path, index_payload: Dict[str, Any]) -> None:
    records = [rec for rec in index_payload["records"] if rec.get("deleted", False)]
    records.sort(key=lambda rec: int(rec["source_offset"]))
    header = struct.pack(
        "<IHHHHIQQ",
        BINARY_INDEX_MAGIC,
        BINARY_INDEX_VERSION,
        BINARY_INDEX_HEADER_SIZE,
        BINARY_INDEX_RECORD_SIZE,
        int(index_payload.get("num_splits", 1)),
        len(records),
        int(index_payload["old_pte_size"]),
        int(index_payload["final_pte_size"]),
    )
    payload = bytearray(header)
    for rec in records:
        if "weight_kind" in rec:
            weight_kind = int(rec["weight_kind"])
            layer_id = int(rec.get("layer_id", INVALID_LAYER_ID))
            op_id = int(rec.get("op_id", INVALID_OP_ID))
        else:
            weight_kind, layer_id, op_id = encode_weight_id(str(rec["name"]))
        payload.extend(
            struct.pack(
                "<BBBBIQI",
                weight_kind,
                layer_id,
                op_id,
                int(rec.get("split_id", 0)),
                int(rec["block_id"]),
                int(rec["source_offset"]),
                int(rec["length"]),
            )
        )
    index_bin.write_bytes(bytes(payload))


def _read_u16(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 2 > len(data):
        raise ValueError(f"flatbuffer u16 offset out of range: {offset}")
    return struct.unpack_from("<H", data, offset)[0]


def _read_u32(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise ValueError(f"flatbuffer u32 offset out of range: {offset}")
    return struct.unpack_from("<I", data, offset)[0]


def _read_i32(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise ValueError(f"flatbuffer i32 offset out of range: {offset}")
    return struct.unpack_from("<i", data, offset)[0]


def _table_field(data: bytes, table: int, field_id: int) -> Optional[int]:
    vtable = table - _read_i32(data, table)
    if vtable < 0:
        raise ValueError(f"flatbuffer vtable offset out of range: {vtable}")
    vtable_size = _read_u16(data, vtable)
    entry = vtable + 4 + 2 * field_id
    if entry + 2 > vtable + vtable_size:
        return None
    relative = _read_u16(data, entry)
    if relative == 0:
        return None
    field = table + relative
    if field >= len(data):
        raise ValueError(f"flatbuffer field offset out of range: {field}")
    return field


def _offset_target(data: bytes, offset: int) -> int:
    target = offset + _read_u32(data, offset)
    if target >= len(data):
        raise ValueError(f"flatbuffer offset target out of range: {target}")
    return target


def _table_vector(data: bytes, field: Optional[int]) -> List[int]:
    if field is None:
        return []
    vector = _offset_target(data, field)
    count = _read_u32(data, vector)
    entries = vector + 4
    if entries + 4 * count > len(data):
        raise ValueError("flatbuffer table vector exceeds PTE size")
    return [_offset_target(data, entries + 4 * index) for index in range(count)]


def _string_field(data: bytes, field: Optional[int]) -> Optional[str]:
    if field is None:
        return None
    string = _offset_target(data, field)
    count = _read_u32(data, string)
    start = string + 4
    if start + count > len(data):
        raise ValueError("flatbuffer string exceeds PTE size")
    return data[start : start + count].decode("utf-8")


def _byte_vector(data: bytes, field: Optional[int]) -> Optional[bytes]:
    if field is None:
        return None
    vector = _offset_target(data, field)
    count = _read_u32(data, vector)
    start = vector + 4
    if start + count > len(data):
        raise ValueError("flatbuffer byte vector exceeds PTE size")
    return bytes(data[start : start + count])


def extract_qnn_compile_spec_from_complete_pte(pte_bytes: bytes) -> bytes:
    # The input is the original complete PTE. Never call this for stripped.pte.
    program = _read_u32(pte_bytes, 0)
    for plan in _table_vector(pte_bytes, _table_field(pte_bytes, program, 1)):
        for delegate in _table_vector(pte_bytes, _table_field(pte_bytes, plan, 7)):
            for spec in _table_vector(pte_bytes, _table_field(pte_bytes, delegate, 2)):
                if _string_field(pte_bytes, _table_field(pte_bytes, spec, 0)) != "qnn_compile_spec":
                    continue
                value = _byte_vector(pte_bytes, _table_field(pte_bytes, spec, 1))
                if value:
                    return value
    raise ValueError("qnn_compile_spec not found in complete PTE")


def embed_qnn_compile_spec_in_manifest(
    manifest_path: Path, qnn_compile_spec: bytes
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    encoded = qnn_compile_spec.hex()
    existing = payload.get("qnn_compile_spec_hex")
    if existing is not None and existing != encoded:
        raise ValueError(
            f"qnn_compile_spec_hex in {manifest_path} does not match the complete PTE"
        )
    payload["qnn_compile_spec_hex"] = encoded
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def embed_llama_qnn_quant_profile_in_manifest(
    source_manifest_path: Path,
    destination_manifest_path: Path,
    graph_name: str,
) -> None:
    source_payload = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_graph = source_payload.get("graphs", {}).get(graph_name)
    if not isinstance(source_graph, dict):
        raise ValueError(
            f"graph {graph_name!r} is missing from source manifest: {source_manifest_path}"
        )
    profile = source_graph.get("llama_qnn_quant_profile")
    if not isinstance(profile, dict):
        raise ValueError(
            "source manifest has no embedded llama_qnn_quant_profile; re-export "
            "with --emit_llama_qnn_quant_profile"
        )
    if profile.get("schema_version") != 2 or profile.get("format") != (
        "llama-qnn-quant-profile-v2"
    ):
        raise ValueError("source manifest has an unsupported llama QNN profile schema")
    if profile.get("capabilities") != _LLAMA_QNN_QUANT_PROFILE_CAPABILITIES:
        raise ValueError(
            "source manifest has an incomplete llama QNN exact-decode profile"
        )
    gptq_source_recipe = profile.get("gptq_source_recipe")
    if not isinstance(gptq_source_recipe, dict) or (
        gptq_source_recipe.get("schema_version") != 1
        or gptq_source_recipe.get("format") != "llama-gptq-source-recipe-v1"
    ):
        raise ValueError(
            "source manifest has no complete GPTQ source recipe in "
            "llama_qnn_quant_profile; re-export with "
            "--emit_llama_qnn_quant_profile"
        )
    source_weight_bits = int(gptq_source_recipe.get("source_weight_bits", 0))
    group_size = int(gptq_source_recipe.get("group_size", 0))
    if (
        source_weight_bits != 2
        or group_size <= 0
        or group_size % 32 != 0
        or gptq_source_recipe.get("gguf_weight_type") != f"GPTQ2_{group_size}"
        or gptq_source_recipe.get("qweight_mode")
        not in {"qweight_minus_qzeros", "qweight"}
        or gptq_source_recipe.get("qnn_weight_code_contract")
        != _gptq2_qnn_weight_code_contract(group_size)
    ):
        raise ValueError("source manifest has an invalid GPTQ source recipe")
    profile_shards = profile.get("shards")
    source_shards = source_graph.get("shards")
    if not isinstance(profile_shards, list) or not isinstance(source_shards, list):
        raise ValueError("source manifest has incomplete shard/profile metadata")
    if len(profile_shards) != len(source_shards):
        raise ValueError(
            "source manifest QNN profile shard count does not match graph shard count"
        )
    for shard_index, profile_shard in enumerate(profile_shards):
        if not isinstance(profile_shard, dict):
            raise ValueError(f"QNN profile shard {shard_index} is not an object")
        if profile_shard.get("capabilities") != _LLAMA_QNN_QUANT_PROFILE_CAPABILITIES:
            raise ValueError(
                f"QNN profile shard {shard_index} has an incomplete capability set"
            )
        tensors = profile_shard.get("tensors")
        operations = profile_shard.get("operations")
        u16_tensor_index = profile_shard.get("u16_tensor_index")
        if (
            not isinstance(tensors, dict)
            or not isinstance(operations, list)
            or not isinstance(u16_tensor_index, dict)
        ):
            raise ValueError(f"QNN profile shard {shard_index} has incomplete graph data")
        expected_u16_tensors = {
            tensor_name
            for tensor_name, tensor in tensors.items()
            if isinstance(tensor, dict)
            and tensor.get("data_type") == "QNN_DATATYPE_UFIXED_POINT_16"
        }
        if set(u16_tensor_index) != expected_u16_tensors:
            raise ValueError(
                f"QNN profile shard {shard_index} has incomplete U16 tensor coverage"
            )
        for tensor_name, tensor in tensors.items():
            if not isinstance(tensor, dict):
                raise ValueError(
                    f"QNN profile shard {shard_index} tensor {tensor_name} is invalid"
                )
            if tensor.get("tensor_type") != "QNN_TENSOR_TYPE_STATIC":
                continue
            static_payload = tensor.get("static_payload")
            if not isinstance(static_payload, dict):
                raise ValueError(
                    f"QNN profile shard {shard_index} static tensor {tensor_name} "
                    "lacks exact payload metadata"
                )
            encoding = tensor.get("quantization_encoding")
            expected_storage = (
                "external_gptq_int2_source_reconstruction"
                if encoding == "QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION"
                else "embedded_exact_bytes"
            )
            if static_payload.get("storage") != expected_storage:
                raise ValueError(
                    f"QNN profile shard {shard_index} static tensor {tensor_name} "
                    "has an invalid payload contract"
                )

    destination_payload = json.loads(destination_manifest_path.read_text(encoding="utf-8"))
    destination_graphs = destination_payload.setdefault("graphs", {})
    destination_graph = destination_graphs.setdefault(graph_name, {})
    existing = destination_graph.get("llama_qnn_quant_profile")
    if existing is not None and existing != profile:
        raise ValueError(
            f"llama_qnn_quant_profile in {destination_manifest_path} does not "
            "match the source export manifest"
        )
    destination_graph["llama_qnn_quant_profile"] = profile
    destination_manifest_path.write_text(
        json.dumps(destination_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_outputs(
    out_dir: Path,
    stripped_bytes: bytearray,
    index_payload: Dict[str, Any],
    report_lines: List[str],
) -> Tuple[Path, Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    stripped_pte = out_dir / "stripped.pte"
    index_json = out_dir / "index.json"
    index_bin = out_dir / "index.bin"
    report_txt = out_dir / "report.txt"

    stripped_pte.write_bytes(bytes(stripped_bytes))
    index_json.write_text(json.dumps(index_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_binary_index(index_bin, index_payload)

    calc_deleted = index_payload["old_pte_size"] - index_payload["final_pte_size"]
    header = [
        f"source_pte={index_payload.get('source_pte')}",
        f"old_pte_size={index_payload['old_pte_size']}",
        f"final_pte_size={index_payload['final_pte_size']}",
        f"total_deleted_bytes(index)={index_payload['total_deleted_bytes']}",
        f"size_diff(old-final)={calc_deleted}",
        f"search_mode={index_payload['search_mode']}",
        f"delete_order_mode={index_payload['delete_order_mode']}",
        f"source_file_modified={index_payload.get('source_file_modified', False)}",
        f"working_buffer_mode={index_payload.get('working_buffer_mode', 'in_memory_copy')}",
        f"num_splits={index_payload.get('num_splits', 1)}",
        f"num_layers={index_payload.get('num_layers', DEFAULT_NUM_LAYERS)}",
        f"layer_range=[{index_payload.get('layer_start')}, {index_payload.get('layer_end_exclusive')})",
        f"shard_index={index_payload.get('shard_index')}",
        f"consistency_ok={calc_deleted == index_payload['total_deleted_bytes']}",
        "",
    ]
    report_txt.write_text("\n".join(header + report_lines) + "\n", encoding="utf-8")
    return stripped_pte, index_json, index_bin, report_txt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="ExecuTorch QNN .pte 基于 QAT checkpoint 重建量化权重的逆序实时删除与索引生成"
    )
    ap.add_argument(
        "--old-pte",
        default=None,
        help="Path to old.pte. Defaults to the selected manifest shard when --shard-manifest is used.",
    )
    ap.add_argument(
        "--qat-checkpoint",
        default="/root/autodl-tmp/Qwen3-1.7b-2bit/model.safetensors",
        help="Path to QAT safetensors checkpoint",
    )
    ap.add_argument("--bits-hint", type=int, default=2, help="Packed source bits hint")
    ap.add_argument("--group-size", type=int, default=32, help="QAT source group size")
    ap.add_argument(
        "--num-splits",
        type=int,
        default=None,
        help="Logical shard count used to assign stripped decoder blocks to split-level rebuild groups.",
    )
    ap.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Decoder layer count. Defaults to manifest metadata, or 28 without a manifest.",
    )
    ap.add_argument(
        "--layer-start",
        type=int,
        default=None,
        help="First decoder layer id to strip, inclusive.",
    )
    ap.add_argument(
        "--layer-end-exclusive",
        type=int,
        default=None,
        help="Last decoder layer id to strip, exclusive.",
    )
    ap.add_argument(
        "--shard-manifest",
        default=None,
        help="Exported *.shards.json. Reads the selected shard PTE and layer range from this manifest.",
    )
    ap.add_argument(
        "--shard-graph",
        default="prefill_forward",
        help="Manifest graph name to strip when --shard-manifest is used.",
    )
    ap.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Manifest shard index to strip. Required with --shard-manifest.",
    )
    ap.add_argument(
        "--qweight-mode",
        choices=["qweight", "qweight_minus_qzeros"],
        default="qweight_minus_qzeros",
        help="How to reconstruct exported qweight from checkpoint",
    )
    ap.add_argument("--out-dir", default="pte_qat_strip_out", help="Output directory")
    ap.add_argument(
        "--embed-qnn-compile-spec-manifest",
        default=None,
        help="Manifest JSON to receive qnn_compile_spec_hex from the complete PTE.",
    )
    ap.add_argument(
        "--qnn-compile-spec-only",
        action="store_true",
        help="Only extract qnn_compile_spec from --old-pte and embed it in the requested manifest.",
    )
    ap.add_argument(
        "--embed-llama-qnn-quant-profile-manifest",
        default=None,
        help=(
            "Device manifest JSON that receives the embedded QNN quantization "
            "profile from --llama-qnn-quant-profile-source-manifest or "
            "--shard-manifest."
        ),
    )
    ap.add_argument(
        "--llama-qnn-quant-profile-source-manifest",
        default=None,
        help=(
            "Exported *.shards.json containing llama_qnn_quant_profile. "
            "Defaults to --shard-manifest."
        ),
    )
    ap.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep output.conv in stripped.pte and exclude it from rebuild indexes",
    )
    ap.add_argument(
        "--no-output",
        action="store_true",
        help="The exported prefill PTE has no LM head/logits output, so output.conv is absent.",
    )
    ap.add_argument(
        "--no-strict",
        action="store_true",
        help="Do not abort when one expected block is missing",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable realtime progress display",
    )
    ap.add_argument(
        "--search-direction",
        choices=["forward", "reverse"],
        default="reverse",
        help="Search direction on immutable source buffer",
    )
    return ap.parse_args()


def read_manifest_shard(
    manifest_path: Path, graph_name: str, shard_index: int
) -> Dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    graph = payload.get("graphs", {}).get(graph_name)
    if not isinstance(graph, dict):
        raise ValueError(
            f"graph {graph_name!r} is missing from shard manifest: {manifest_path}"
        )

    shards = graph.get("shards")
    if not isinstance(shards, list):
        raise ValueError(
            "manifest has no per-shard metadata; re-export it with the current "
            "llm_wrappers.py before using --shard-manifest"
        )

    shard = next(
        (entry for entry in shards if int(entry.get("index", -1)) == shard_index),
        None,
    )
    if not isinstance(shard, dict):
        raise ValueError(
            f"shard index {shard_index} is missing from graph {graph_name!r}"
        )

    required_fields = ("pte_path", "layer_start", "layer_end_exclusive")
    missing = [field for field in required_fields if field not in shard]
    if missing:
        raise ValueError(
            f"manifest shard {shard_index} is missing required fields: {missing}"
        )

    num_layers = graph.get("num_decoder_layers", payload.get("num_decoder_layers"))
    if num_layers is None:
        raise ValueError("manifest is missing num_decoder_layers")

    pte_path = Path(str(shard["pte_path"]))
    if not pte_path.is_absolute():
        pte_path = manifest_path.parent / pte_path

    return {
        "pte_path": pte_path,
        "layer_start": int(shard["layer_start"]),
        "layer_end_exclusive": int(shard["layer_end_exclusive"]),
        "num_layers": int(num_layers),
        "num_splits": int(payload.get("num_shards", len(shards))),
        "split_layer_starts": [
            int(entry["layer_start"])
            for entry in sorted(shards, key=lambda entry: int(entry["index"]))
        ],
        "shard_index": shard_index,
        "prefill_outputs_logits": bool(payload.get("prefill_outputs_logits", True)),
    }


def main() -> None:
    args = parse_args()

    if args.qnn_compile_spec_only:
        if args.old_pte is None or args.embed_qnn_compile_spec_manifest is None:
            raise ValueError(
                "--qnn-compile-spec-only requires --old-pte and "
                "--embed-qnn-compile-spec-manifest"
            )
        old_pte = Path(args.old_pte).expanduser().resolve()
        manifest_path = Path(args.embed_qnn_compile_spec_manifest).expanduser().resolve()
        qnn_compile_spec = extract_qnn_compile_spec_from_complete_pte(
            old_pte.read_bytes()
        )
        embed_qnn_compile_spec_in_manifest(manifest_path, qnn_compile_spec)
        print(
            f"embedded qnn_compile_spec_hex ({len(qnn_compile_spec)} bytes) -> "
            f"{manifest_path}"
        )
        if args.embed_llama_qnn_quant_profile_manifest is not None:
            if args.llama_qnn_quant_profile_source_manifest is None:
                raise ValueError(
                    "--qnn-compile-spec-only profile embedding requires "
                    "--llama-qnn-quant-profile-source-manifest"
                )
            source_manifest = Path(
                args.llama_qnn_quant_profile_source_manifest
            ).expanduser().resolve()
            destination_manifest = Path(
                args.embed_llama_qnn_quant_profile_manifest
            ).expanduser().resolve()
            embed_llama_qnn_quant_profile_in_manifest(
                source_manifest, destination_manifest, args.shard_graph
            )
            print(f"embedded llama_qnn_quant_profile -> {destination_manifest}")
        return

    manifest_shard = None
    if args.shard_manifest is not None:
        if args.shard_index is None:
            raise ValueError("--shard-index is required with --shard-manifest")
        manifest_path = Path(args.shard_manifest).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"shard manifest not found: {manifest_path}")
        manifest_shard = read_manifest_shard(
            manifest_path, args.shard_graph, args.shard_index
        )

    def manifest_or_arg(name: str, fallback: int) -> int:
        arg_value = getattr(args, name)
        manifest_value = None if manifest_shard is None else manifest_shard[name]
        if arg_value is not None and manifest_value is not None and arg_value != manifest_value:
            raise ValueError(
                f"--{name.replace('_', '-')}={arg_value} conflicts with manifest value {manifest_value}"
            )
        return manifest_value if manifest_value is not None else (
            arg_value if arg_value is not None else fallback
        )

    num_layers = manifest_or_arg("num_layers", DEFAULT_NUM_LAYERS)
    layer_start = manifest_or_arg("layer_start", 0)
    layer_end_exclusive = manifest_or_arg("layer_end_exclusive", num_layers)
    num_splits = manifest_or_arg("num_splits", 1)
    old_pte_value = args.old_pte or (
        str(manifest_shard["pte_path"])
        if manifest_shard is not None
        else DEFAULT_OLD_PTE
    )
    old_pte = Path(old_pte_value).expanduser().resolve()
    if not old_pte.exists():
        raise FileNotFoundError(f"old pte not found: {old_pte}")

    qat_checkpoint = Path(args.qat_checkpoint).expanduser().resolve()
    if not qat_checkpoint.exists():
        raise FileNotFoundError(f"qat checkpoint not found: {qat_checkpoint}")

    if (
        num_layers <= 0
        or layer_start < 0
        or layer_end_exclusive > num_layers
        or layer_start >= layer_end_exclusive
    ):
        raise ValueError(
            f"invalid layer range: [{layer_start}, {layer_end_exclusive}) for num_layers={num_layers}"
        )

    include_output = not args.keep_output and not args.no_output
    if manifest_shard is not None and not manifest_shard["prefill_outputs_logits"]:
        include_output = False
        print("manifest declares prefill_outputs_logits=false; skipping output.conv strip")

    qat_sd = load_file(str(qat_checkpoint))
    specs_by_name = build_qat_specs(
        include_output=include_output,
        layer_start=layer_start,
        layer_end_exclusive=layer_end_exclusive,
    )
    strict = not args.no_strict

    stripped, index_payload, report_lines, complete_pte_bytes = realtime_delete(
        old_pte=old_pte,
        specs_by_name=specs_by_name,
        qat_sd=qat_sd,
        bits_hint=args.bits_hint,
        group_size=args.group_size,
        qweight_mode=args.qweight_mode,
        include_output=include_output,
        strict=strict,
        show_progress=not args.no_progress,
        search_direction=args.search_direction,
        num_splits=num_splits,
        num_layers=num_layers,
        layer_start=layer_start,
        layer_end_exclusive=layer_end_exclusive,
        split_layer_starts=(
            manifest_shard["split_layer_starts"] if manifest_shard is not None else None
        ),
        shard_index=(
            manifest_shard["shard_index"] if manifest_shard is not None else None
        ),
    )
    qnn_compile_spec_bytes = extract_qnn_compile_spec_from_complete_pte(
        complete_pte_bytes
    )
    if args.embed_qnn_compile_spec_manifest is not None:
        embed_qnn_compile_spec_in_manifest(
            Path(args.embed_qnn_compile_spec_manifest).expanduser().resolve(),
            qnn_compile_spec_bytes,
        )
    if args.embed_llama_qnn_quant_profile_manifest is not None:
        source_manifest_arg = (
            args.llama_qnn_quant_profile_source_manifest or args.shard_manifest
        )
        if source_manifest_arg is None:
            raise ValueError(
                "profile embedding requires --llama-qnn-quant-profile-source-manifest "
                "or --shard-manifest"
            )
        embed_llama_qnn_quant_profile_in_manifest(
            Path(source_manifest_arg).expanduser().resolve(),
            Path(args.embed_llama_qnn_quant_profile_manifest).expanduser().resolve(),
            args.shard_graph,
        )
    if manifest_shard is not None:
        index_payload["shard_manifest"] = str(manifest_path)
        index_payload["shard_graph"] = args.shard_graph

    out_dir = Path(args.out_dir).expanduser().resolve()
    stripped_pte, index_json, index_bin, report_txt = write_outputs(
        out_dir=out_dir,
        stripped_bytes=stripped,
        index_payload=index_payload,
        report_lines=report_lines,
    )

    print(f"saved stripped.pte -> {stripped_pte}")
    print(f"saved index.json  -> {index_json}")
    print(f"saved index.bin   -> {index_bin}")
    print(f"saved report.txt  -> {report_txt}")
    print(
        "summary:",
        {
            "old_pte_size": index_payload["old_pte_size"],
            "final_pte_size": index_payload["final_pte_size"],
            "total_deleted_bytes": index_payload["total_deleted_bytes"],
            "records": len(index_payload["records"]),
        },
    )


if __name__ == "__main__":
    main()
