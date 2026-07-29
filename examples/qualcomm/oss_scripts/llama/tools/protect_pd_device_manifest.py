#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Copy and verify protected PD runtime fields in a device manifest.

The exported shard manifest is the source of truth.  Older exports do not
contain qnn_compile_spec_hex, so this tool can recover it once from their
complete PTE.  The destination is updated atomically to avoid exposing a
partially rewritten manifest to packaging or deployment.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from pte_qat_checkpoint_reverse_strip import (
    extract_qnn_compile_spec_from_complete_pte,
)


def _read_manifest(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest root must be an object: {path}")
    return payload


def _validate_hex(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value or len(value) % 2:
        raise ValueError(f"{field} is missing or malformed in {path}")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not valid hex in {path}") from exc
    return value.lower()


def _source_compile_spec(
    source_path: Path,
    source: Dict[str, Any],
    complete_pte_arg: Optional[str],
) -> str:
    encoded = source.get("qnn_compile_spec_hex")
    if encoded is not None:
        return _validate_hex(encoded, "qnn_compile_spec_hex", source_path)

    pte_value = complete_pte_arg or source.get("combined_pte")
    if not isinstance(pte_value, str) or not pte_value:
        raise ValueError(
            "source manifest predates protected compile specs and has no "
            "combined_pte; pass --complete-pte"
        )
    pte_path = Path(pte_value).expanduser()
    if not pte_path.is_absolute():
        pte_path = source_path.parent / pte_path
    compile_spec = extract_qnn_compile_spec_from_complete_pte(pte_path.read_bytes())
    return compile_spec.hex()


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        required=True,
        help="Primary exported *.shards.json (source of protected fields).",
    )
    parser.add_argument(
        "--device-manifest",
        required=True,
        help="Generated stripped/device manifest to protect in place.",
    )
    parser.add_argument(
        "--complete-pte",
        default=None,
        help="Complete PTE fallback for an older source manifest.",
    )
    args = parser.parse_args()

    source_path = Path(args.source_manifest).expanduser().resolve()
    device_path = Path(args.device_manifest).expanduser().resolve()
    source = _read_manifest(source_path)
    device = _read_manifest(device_path)
    expected = _source_compile_spec(source_path, source, args.complete_pte)

    existing = device.get("qnn_compile_spec_hex")
    if existing is not None:
        existing = _validate_hex(existing, "qnn_compile_spec_hex", device_path)
        if existing != expected:
            raise ValueError(
                "device qnn_compile_spec_hex does not match its exported source"
            )
    device["qnn_compile_spec_hex"] = expected
    _atomic_write(device_path, device)

    # Reopen the final inode, since this exact file is what packaging copies.
    verified = _read_manifest(device_path)
    final_value = _validate_hex(
        verified.get("qnn_compile_spec_hex"),
        "qnn_compile_spec_hex",
        device_path,
    )
    if final_value != expected:
        raise RuntimeError("protected field changed during atomic manifest update")
    print(
        f"protected qnn_compile_spec_hex: bytes={len(bytes.fromhex(expected))} "
        f"manifest={device_path}"
    )


if __name__ == "__main__":
    main()
