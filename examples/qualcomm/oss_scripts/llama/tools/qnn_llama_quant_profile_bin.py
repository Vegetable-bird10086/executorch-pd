#!/usr/bin/env python3
"""Convert a validated llama QNN runtime profile JSON to its mmap binary form."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


MAGIC = b"LQNNPRF\0"
HEADER = struct.Struct("<8sIIQQ")
VERSION = 2
ENDIAN_TAG = 0x01020304


def find_converter(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("LLAMA_QNN_PROFILE_CONVERTER"):
        candidates.append(Path(os.environ["LLAMA_QNN_PROFILE_CONVERTER"]))
    on_path = shutil.which("qnn-u16-profile-convert")
    if on_path:
        candidates.append(Path(on_path))

    workspace = Path(__file__).resolve().parents[6]
    candidates.extend(
        [
            workspace / "llama.cpp/build-x86/bin/qnn-u16-profile-convert",
            workspace / "llama.cpp/build/bin/qnn-u16-profile-convert",
        ]
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    raise SystemExit(
        "qnn-u16-profile-convert was not found; build the llama.cpp target or "
        "pass --converter /path/to/qnn-u16-profile-convert"
    )


def validate_binary(path: Path) -> None:
    size = path.stat().st_size
    if size < HEADER.size:
        raise SystemExit(f"binary profile is truncated: {path}")
    with path.open("rb") as stream:
        magic, version, endian_tag, payload_bytes, _payload_checksum = HEADER.unpack(
            stream.read(HEADER.size)
        )
    if magic != MAGIC or version != VERSION or endian_tag != ENDIAN_TAG:
        raise SystemExit(f"binary profile header is invalid: {path}")
    if payload_bytes != size - HEADER.size:
        raise SystemExit(
            f"binary profile payload size mismatch: header={payload_bytes} "
            f"actual={size - HEADER.size}"
        )


def convert_profile(
    source: Path,
    model: Path,
    output: Path,
    converter_arg: str | None = None,
) -> Path:
    source = source.resolve()
    model = model.resolve()
    output = output.resolve()
    if not source.is_file():
        raise SystemExit(f"runtime profile JSON does not exist: {source}")
    if source == output:
        raise SystemExit("profile JSON and binary output must be different files")
    if not model.is_file():
        raise SystemExit(f"GGUF model does not exist: {model}")

    converter = find_converter(converter_arg)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        subprocess.run(
            [str(converter), str(source), str(model), str(temporary)],
            check=True,
        )
        validate_binary(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, type=Path, help="runtime profile JSON")
    parser.add_argument("--gguf", required=True, type=Path, help="Decode GGUF model")
    parser.add_argument("--out", required=True, type=Path, help="output mmap binary")
    parser.add_argument("--converter", help="qnn-u16-profile-convert executable")
    args = parser.parse_args()

    source = args.json.resolve()
    output = convert_profile(source, args.gguf, args.out, args.converter)

    print(
        f"qnn-profile-bin-export: status=pass json_bytes={source.stat().st_size} "
        f"bin_bytes={output.stat().st_size} output={output}"
    )


if __name__ == "__main__":
    main()
