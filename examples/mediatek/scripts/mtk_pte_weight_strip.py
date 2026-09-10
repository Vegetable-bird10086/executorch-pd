#!/usr/bin/env python3

"""Strip packed MTK weights from a PTE and rebuild it byte-for-byte.

The tool intentionally uses a small backend-neutral container.  ``stripped.pte``
contains every original byte except the selected packed weight ranges,
``weights.bin`` contains those ranges in source-offset order, and ``index.bin``
describes how to insert them again.  Android can mmap the two compact sources
and materialize only the chunk that is about to be loaded by NeuroPilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAGIC = b"MTKSTRP1"
VERSION = 1
HEADER = struct.Struct("<8sIIQQQII")
RECORD = struct.Struct("<QQQII")
GGUF_MAGIC = b"MTKGGUF2"
GGUF_VERSION = 2
GGUF_HEADER = struct.Struct("<8sIIQQQII")
GGUF_RECORD = struct.Struct("<QQIIIIII")
KIND_INT4 = 4
KIND_INT8 = 8

GGUF_OP_IDS = {
    "self_attn.q_proj": 0,
    "self_attn.k_proj": 1,
    "self_attn.v_proj": 2,
    "self_attn.o_proj": 3,
    "mlp.gate_proj": 4,
    "mlp.up_proj": 5,
    "mlp.down_proj": 6,
}
GGUF_FLAG_LLAMA_ROPE_ROWS = 1 << 31


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_mtk_int4(qweight: np.ndarray) -> bytes:
    if qweight.ndim != 2:
        raise ValueError(f"int4 qweight must be 2D, got {qweight.shape}")
    rows, cols = qweight.shape
    if rows % 16 or cols % 32:
        raise ValueError(
            f"int4 shape must satisfy rows%16=0 and cols%32=0, got {qweight.shape}"
        )
    if np.any(qweight < -8) or np.any(qweight > 7):
        raise ValueError("int4 qweight is outside [-8, 7]")

    # Recovered for mt6989 / mtk_converter 8.13.0.  A 16-row group is split
    # into four banks; each bank contains natural-row-major 4x32/64-byte tiles.
    row = np.arange(rows, dtype=np.int64)[:, None]
    col = np.arange(cols, dtype=np.int64)[None, :]
    offsets = (
        (row // 16) * (2 * cols)
        + ((row % 16) // 4) * (rows * cols // 8)
        + (col // 32) * 64
        + (row % 4) * 16
        + ((col % 32) // 2)
    )
    values = (qweight.astype(np.int16) & 0xF).astype(np.uint8)
    packed = np.zeros(rows * cols // 2, dtype=np.uint8)
    flat_col = np.broadcast_to(col, qweight.shape).ravel()
    np.bitwise_or.at(
        packed,
        offsets.ravel(),
        np.where((flat_col & 1) == 0, values.ravel(), values.ravel() << 4),
    )
    return packed.tobytes()


def encode_mtk_int8(qweight: np.ndarray) -> bytes:
    if qweight.ndim != 2:
        raise ValueError(f"int8 qweight must be 2D, got {qweight.shape}")
    rows, cols = qweight.shape
    if rows % 16 or cols % 16:
        raise ValueError(
            f"int8 shape must satisfy rows%16=0 and cols%16=0, got {qweight.shape}"
        )
    if np.any(qweight < -128) or np.any(qweight > 127):
        raise ValueError("int8 qweight is outside [-128, 127]")

    row = np.arange(rows, dtype=np.int64)[:, None]
    col = np.arange(cols, dtype=np.int64)[None, :]
    offsets = (
        (row // 16) * (4 * cols)
        + ((row % 16) // 4) * (rows * cols // 4)
        + (col // 16) * 64
        + (row % 4) * 16
        + (col % 16)
    )
    packed = np.empty(rows * cols, dtype=np.uint8)
    packed[offsets.ravel()] = qweight.astype(np.int8, copy=False).view(np.uint8).ravel()
    return packed.tobytes()


def find_all(data: bytes, needle: bytes) -> list[int]:
    hits: list[int] = []
    cursor = 0
    while True:
        hit = data.find(needle, cursor)
        if hit < 0:
            return hits
        hits.append(hit)
        cursor = hit + 1


@dataclass(frozen=True)
class WeightRecord:
    name: str
    kind: int
    source_offset: int
    length: int
    payload_offset: int
    payload_sha256: str
    encoded_mismatch_bytes: int


def locate_packed_weight(data: bytes, expected: bytes, name: str) -> tuple[int, int]:
    """Locate a packed blob using independent anchors and verify it as a whole.

    The MTK converter patches a very small number of bytes inside some otherwise
    deterministic qweight blobs.  Requiring the complete encoded qweight to be
    byte-identical therefore misses valid blobs.  Multiple exact 4 KiB anchors
    establish the start, after which a strict whole-blob mismatch bound prevents
    an unrelated region from being accepted.
    """
    anchor_size = min(4096, len(expected))
    anchor_count = min(17, max(1, len(expected) // anchor_size))
    if anchor_count == 1:
        offsets = [0]
    else:
        offsets = [
            index * (len(expected) - anchor_size) // (anchor_count - 1)
            for index in range(anchor_count)
        ]

    votes: dict[int, int] = {}
    for relative_offset in offsets:
        anchor = expected[relative_offset : relative_offset + anchor_size]
        for hit in find_all(data, anchor):
            candidate = hit - relative_offset
            if 0 <= candidate <= len(data) - len(expected):
                votes[candidate] = votes.get(candidate, 0) + 1

    minimum_votes = max(3, (anchor_count + 1) // 2)
    candidates = [start for start, count in votes.items() if count >= minimum_votes]
    if len(candidates) != 1:
        ranked = sorted(votes.items(), key=lambda item: item[1], reverse=True)[:5]
        raise ValueError(
            f"{name}: packed blob anchor vote is ambiguous; "
            f"required {minimum_votes}/{anchor_count}, best candidates={ranked}"
        )

    start = candidates[0]
    actual = memoryview(data)[start : start + len(expected)]
    mismatch_count = sum(left != right for left, right in zip(actual, expected))
    # Observed converter patches are sparse (currently <=75 bytes in 6 MiB).
    # Keep the limit tight enough that a wrong layout or shifted region fails.
    mismatch_limit = max(256, len(expected) // 20_000)
    if mismatch_count > mismatch_limit:
        raise ValueError(
            f"{name}: candidate at {start} differs in {mismatch_count} bytes "
            f"(limit {mismatch_limit})"
        )
    return start, mismatch_count


def parse_tensor_spec(value: str) -> tuple[str, int, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3 or parts[1] not in {"int4", "int8"}:
        raise argparse.ArgumentTypeError("expected NAME:int4|int8:/path/to/qweight.npy")
    return parts[0], KIND_INT4 if parts[1] == "int4" else KIND_INT8, Path(parts[2])


def load_qweight_manifest(path: Path) -> list[tuple[str, int, Path]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = []
    for record in payload.get("records", []):
        precision = record["precision"]
        if precision not in {"int4", "int8"}:
            raise ValueError(f"unsupported manifest precision: {precision}")
        consumers = record.get("consumer_module_paths") or []
        logical_name = consumers[-1] if consumers else record["attr"]
        # Preserve uniqueness for sharded lm-head tensors with the attribute.
        name = f"{logical_name}:{record['attr']}"
        specs.append(
            (
                name,
                KIND_INT4 if precision == "int4" else KIND_INT8,
                path.parent / record["path"],
            )
        )
    if not specs:
        raise ValueError(f"qweight manifest has no records: {path}")
    return specs


def load_qweight_manifest_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for item in payload.get("records", []):
        precision = item["precision"]
        if precision not in {"int4", "int8"}:
            raise ValueError(f"unsupported manifest precision: {precision}")
        consumers = item.get("consumer_module_paths") or []
        logical_name = consumers[-1] if consumers else item["attr"]
        records.append(
            {
                "name": f"{logical_name}:{item['attr']}",
                "logical_name": logical_name,
                "kind": KIND_INT4 if precision == "int4" else KIND_INT8,
                "path": path.parent / item["path"],
                "shape": list(map(int, item["shape"])),
            }
        )
    if not records:
        raise ValueError(f"qweight manifest has no records: {path}")
    return records


def write_index(path: Path, records: list[WeightRecord], sizes: tuple[int, int, int]) -> None:
    original_size, stripped_size, weights_size = sizes
    payload = bytearray(
        HEADER.pack(
            MAGIC,
            VERSION,
            RECORD.size,
            original_size,
            stripped_size,
            weights_size,
            len(records),
            0,
        )
    )
    for record in records:
        payload.extend(
            RECORD.pack(
                record.source_offset,
                record.length,
                record.payload_offset,
                record.kind,
                0,
            )
        )
    path.write_bytes(payload)


def read_index(path: Path) -> tuple[tuple[int, int, int], list[tuple[int, int, int, int]]]:
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("index is shorter than its header")
    magic, version, record_size, original, stripped, weights, count, _ = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION or record_size != RECORD.size:
        raise ValueError("unsupported MTK strip index")
    expected = HEADER.size + count * RECORD.size
    if len(data) != expected:
        raise ValueError(f"index size mismatch: got {len(data)}, expected {expected}")
    records = []
    for index in range(count):
        source, length, payload, kind, _ = RECORD.unpack_from(data, HEADER.size + index * RECORD.size)
        records.append((source, length, payload, kind))
    return (original, stripped, weights), records


def strip(args: argparse.Namespace) -> None:
    pte = args.pte.read_bytes()
    located: list[tuple[str, int, int, bytes, int]] = []
    tensor_specs = list(args.tensor or [])
    for manifest in args.qweight_manifest or []:
        tensor_specs.extend(load_qweight_manifest(manifest))
    if not tensor_specs:
        raise ValueError("provide at least one --tensor or --qweight-manifest")
    for name, kind, tensor_path in tensor_specs:
        tensor = np.load(tensor_path, mmap_mode="r")
        encoded = encode_mtk_int4(tensor) if kind == KIND_INT4 else encode_mtk_int8(tensor)
        source_offset, mismatch_count = locate_packed_weight(pte, encoded, name)
        # Store the bytes actually consumed by the compiled PTE.  The encoded
        # tensor is used only to identify and validate the physical range.
        payload = pte[source_offset : source_offset + len(encoded)]
        located.append((name, kind, source_offset, payload, mismatch_count))

    located.sort(key=lambda item: item[2])
    for previous, current in zip(located, located[1:]):
        previous_end = previous[2] + len(previous[3])
        if previous_end > current[2]:
            raise ValueError(f"overlapping weights: {previous[0]} and {current[0]}")

    stripped = bytearray()
    weights = bytearray()
    records: list[WeightRecord] = []
    source_cursor = 0
    for name, kind, source_offset, payload, mismatch_count in located:
        stripped.extend(pte[source_cursor:source_offset])
        payload_offset = len(weights)
        weights.extend(payload)
        records.append(
            WeightRecord(
                name=name,
                kind=kind,
                source_offset=source_offset,
                length=len(payload),
                payload_offset=payload_offset,
                payload_sha256=sha256(payload),
                encoded_mismatch_bytes=mismatch_count,
            )
        )
        source_cursor = source_offset + len(payload)
    stripped.extend(pte[source_cursor:])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stripped_path = args.output_dir / "stripped.pte"
    weights_path = args.output_dir / "weights.bin"
    index_path = args.output_dir / "index.bin"
    stripped_path.write_bytes(stripped)
    weights_path.write_bytes(weights)
    write_index(index_path, records, (len(pte), len(stripped), len(weights)))

    report = {
        "schema_version": VERSION,
        "source_pte": str(args.pte),
        "source_pte_size": len(pte),
        "source_pte_sha256": sha256(pte),
        "stripped_pte_size": len(stripped),
        "stripped_pte_sha256": sha256(stripped),
        "weights_size": len(weights),
        "weights_sha256": sha256(weights),
        "removed_fraction": len(weights) / len(pte),
        "records": [record.__dict__ for record in records],
    }
    (args.output_dir / "index.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def rebuild(args: argparse.Namespace) -> None:
    stripped = args.stripped_pte.read_bytes()
    weights = args.weights.read_bytes()
    (original_size, stripped_size, weights_size), records = read_index(args.index)
    if len(stripped) != stripped_size or len(weights) != weights_size:
        raise ValueError("stripped PTE or weight payload size does not match index")

    rebuilt = bytearray(original_size)
    stripped_cursor = 0
    destination_cursor = 0
    for source_offset, length, payload_offset, _kind in records:
        if source_offset < destination_cursor:
            raise ValueError("overlapping or unsorted index records")
        keep = source_offset - destination_cursor
        rebuilt[destination_cursor:source_offset] = stripped[
            stripped_cursor : stripped_cursor + keep
        ]
        rebuilt[source_offset : source_offset + length] = weights[
            payload_offset : payload_offset + length
        ]
        stripped_cursor += keep
        destination_cursor = source_offset + length
    rebuilt[destination_cursor:] = stripped[stripped_cursor:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(rebuilt)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "size": len(rebuilt),
                "sha256": sha256(rebuilt),
                "records": len(records),
            },
            indent=2,
        )
    )


def write_gguf_index(args: argparse.Namespace) -> None:
    """Upgrade a legacy byte-payload index into a semantic GGUF recipe.

    MTK subgraphs number their decoder layers locally.  ``chunk_index`` and
    ``layers_per_chunk`` turn those local ids into the global ids used by GGUF.
    No payload offsets or bytes from weights.bin are retained in this format.
    """
    legacy = json.loads(args.legacy_index_json.read_text(encoding="utf-8"))
    manifest = json.loads(args.qweight_manifest.read_text(encoding="utf-8"))
    tensors = {}
    for item in manifest.get("records", []):
        attr = item["attr"]
        if ".layers." in attr:
            attr = attr.split(".layers.", 1)[1]
            attr = "layers." + attr
        tensors[attr] = item

    records = []
    materialized_size = 0
    for record in legacy.get("records", []):
        logical = record["name"].split(":", 1)[0]
        parts = logical.split(".")
        if len(parts) < 4 or parts[0] != "layers":
            raise ValueError(f"unsupported MTK tensor name: {logical}")
        local_layer = int(parts[1])
        op_name = ".".join(parts[2:])
        if op_name not in GGUF_OP_IDS:
            raise ValueError(f"unsupported MTK GGUF op: {op_name}")
        attr = record["name"].split(":", 1)[-1]
        tensor = tensors.get(attr)
        if tensor is None:
            raise ValueError(f"missing qweight manifest entry for {attr}")
        rows, cols = map(int, tensor["shape"])
        kind = KIND_INT4 if tensor["precision"] == "int4" else KIND_INT8
        expected_length = rows * cols // 2 if kind == KIND_INT4 else rows * cols
        if expected_length != int(record["length"]):
            raise ValueError(
                f"{logical}: shape/precision length {expected_length} != "
                f"PTE record length {record['length']}"
            )
        global_layer = args.chunk_index * args.layers_per_chunk + local_layer
        records.append(
            {
                "name": logical,
                "gguf_tensor": (
                    f"blk.{global_layer}."
                    f"{('attn_q','attn_k','attn_v','attn_output','ffn_gate','ffn_up','ffn_down')[GGUF_OP_IDS[op_name]]}.weight"
                ),
                "source_offset": int(record["source_offset"]),
                "length": int(record["length"]),
                "global_layer": global_layer,
                "op_id": GGUF_OP_IDS[op_name],
                "rows": rows,
                "cols": cols,
                "kind": kind,
            }
        )
        materialized_size += int(record["length"])

    records.sort(key=lambda item: item["source_offset"])
    original_size = int(legacy["source_pte_size"])
    stripped_size = int(legacy["stripped_pte_size"])
    payload = bytearray(
        GGUF_HEADER.pack(
            GGUF_MAGIC,
            GGUF_VERSION,
            GGUF_RECORD.size,
            original_size,
            stripped_size,
            materialized_size,
            len(records),
            args.source_group_size,
        )
    )
    for record in records:
        payload.extend(
            GGUF_RECORD.pack(
                record["source_offset"],
                record["length"],
                record["global_layer"],
                record["op_id"],
                record["rows"],
                record["cols"],
                record["kind"],
                0,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    report = {
        "schema_version": GGUF_VERSION,
        "format": "mtk-gguf-rebuild-index",
        "source_group_size": args.source_group_size,
        "source_pte_size": original_size,
        "stripped_pte_size": stripped_size,
        "materialized_weight_size": materialized_size,
        "records": records,
    }
    report_path = args.report or args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def strip_gguf(args: argparse.Namespace) -> None:
    """Strip MTK weights and emit only a semantic GGUF reconstruction index."""
    pte = args.pte.read_bytes()
    manifest_records = load_qweight_manifest_records(args.qweight_manifest)
    located = []
    for item in manifest_records:
        tensor = np.load(item["path"], mmap_mode="r")
        encoded = (
            encode_mtk_int4(tensor)
            if item["kind"] == KIND_INT4
            else encode_mtk_int8(tensor)
        )
        source_offset, mismatch_count = locate_packed_weight(
            pte, encoded, item["name"]
        )
        if mismatch_count != 0:
            raise ValueError(
                f"{item['name']}: GGUF-derived export must match its MTK layout "
                f"exactly, got {mismatch_count} differing bytes"
            )
        located.append((source_offset, len(encoded), mismatch_count, item))
    located.sort(key=lambda value: value[0])
    for previous, current in zip(located, located[1:]):
        if previous[0] + previous[1] > current[0]:
            raise ValueError(
                f"overlapping weights: {previous[3]['name']} and {current[3]['name']}"
            )

    stripped = bytearray()
    source_cursor = 0
    semantic_records = []
    materialized_size = 0
    op_names = (
        "attn_q", "attn_k", "attn_v", "attn_output",
        "ffn_gate", "ffn_up", "ffn_down",
    )
    for source_offset, length, mismatch_count, item in located:
        stripped.extend(pte[source_cursor:source_offset])
        logical = item["logical_name"]
        parts = logical.split(".")
        if len(parts) < 4 or parts[0] != "layers":
            raise ValueError(f"unsupported MTK tensor name: {logical}")
        local_layer = int(parts[1])
        op_name = ".".join(parts[2:])
        if op_name not in GGUF_OP_IDS:
            raise ValueError(f"unsupported MTK GGUF op: {op_name}")
        global_layer = args.chunk_index * args.layers_per_chunk + local_layer
        rope_heads = (
            args.llama_rope_q_heads if op_name == "self_attn.q_proj" else
            args.llama_rope_kv_heads if op_name == "self_attn.k_proj" else 0
        )
        flags = GGUF_FLAG_LLAMA_ROPE_ROWS | rope_heads if rope_heads else 0
        rows, cols = item["shape"]
        semantic_records.append(
            {
                "name": logical,
                "gguf_tensor": (
                    f"blk.{global_layer}.{op_names[GGUF_OP_IDS[op_name]]}.weight"
                ),
                "source_offset": source_offset,
                "length": length,
                "global_layer": global_layer,
                "op_id": GGUF_OP_IDS[op_name],
                "rows": rows,
                "cols": cols,
                "kind": item["kind"],
                "flags": flags,
                "encoded_mismatch_bytes": mismatch_count,
            }
        )
        materialized_size += length
        source_cursor = source_offset + length
    stripped.extend(pte[source_cursor:])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stripped_path = args.output_dir / "stripped.pte"
    index_path = args.output_dir / "index.bin"
    stripped_path.write_bytes(stripped)
    payload = bytearray(
        GGUF_HEADER.pack(
            GGUF_MAGIC,
            GGUF_VERSION,
            GGUF_RECORD.size,
            len(pte),
            len(stripped),
            materialized_size,
            len(semantic_records),
            args.source_group_size,
        )
    )
    for record in semantic_records:
        payload.extend(
            GGUF_RECORD.pack(
                record["source_offset"], record["length"],
                record["global_layer"], record["op_id"],
                record["rows"], record["cols"], record["kind"], record["flags"],
            )
        )
    index_path.write_bytes(payload)
    report = {
        "schema_version": GGUF_VERSION,
        "format": "mtk-gguf-stripped-pte",
        "source_pte": str(args.pte),
        "source_pte_size": len(pte),
        "source_pte_sha256": sha256(pte),
        "stripped_pte_size": len(stripped),
        "stripped_pte_sha256": sha256(stripped),
        "materialized_weight_size": materialized_size,
        "removed_fraction": materialized_size / len(pte),
        "source_group_size": args.source_group_size,
        "weights_bin_emitted": False,
        "records": semantic_records,
    }
    (args.output_dir / "index.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    strip_parser = subparsers.add_parser("strip")
    strip_parser.add_argument("--pte", type=Path, required=True)
    strip_parser.add_argument("--output-dir", type=Path, required=True)
    strip_parser.add_argument(
        "--tensor",
        type=parse_tensor_spec,
        action="append",
        help="repeatable NAME:int4|int8:/path/to/qweight.npy",
    )
    strip_parser.add_argument(
        "--qweight-manifest",
        type=Path,
        action="append",
        help="manifest.json emitted by qwen.py --dump-qweights",
    )
    strip_parser.set_defaults(func=strip)

    rebuild_parser = subparsers.add_parser("rebuild")
    rebuild_parser.add_argument("--stripped-pte", type=Path, required=True)
    rebuild_parser.add_argument("--weights", type=Path, required=True)
    rebuild_parser.add_argument("--index", type=Path, required=True)
    rebuild_parser.add_argument("--output", type=Path, required=True)
    rebuild_parser.set_defaults(func=rebuild)

    gguf_parser = subparsers.add_parser(
        "gguf-index", help="create a semantic index with no weights.bin dependency"
    )
    gguf_parser.add_argument("--legacy-index-json", type=Path, required=True)
    gguf_parser.add_argument("--qweight-manifest", type=Path, required=True)
    gguf_parser.add_argument("--chunk-index", type=int, required=True)
    gguf_parser.add_argument("--layers-per-chunk", type=int, default=2)
    gguf_parser.add_argument("--source-group-size", type=int, default=32)
    gguf_parser.add_argument("--output", type=Path, required=True)
    gguf_parser.add_argument("--report", type=Path)
    gguf_parser.set_defaults(func=write_gguf_index)

    strip_gguf_parser = subparsers.add_parser(
        "strip-gguf", help="strip weights and emit no weights.bin"
    )
    strip_gguf_parser.add_argument("--pte", type=Path, required=True)
    strip_gguf_parser.add_argument("--qweight-manifest", type=Path, required=True)
    strip_gguf_parser.add_argument("--chunk-index", type=int, required=True)
    strip_gguf_parser.add_argument("--layers-per-chunk", type=int, default=2)
    strip_gguf_parser.add_argument("--source-group-size", type=int, default=32)
    strip_gguf_parser.add_argument("--llama-rope-q-heads", type=int, default=0)
    strip_gguf_parser.add_argument("--llama-rope-kv-heads", type=int, default=0)
    strip_gguf_parser.add_argument("--output-dir", type=Path, required=True)
    strip_gguf_parser.set_defaults(func=strip_gguf)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
