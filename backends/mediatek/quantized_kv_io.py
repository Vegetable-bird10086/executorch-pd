# Copyright (c) 2026. Licensed under the BSD-style ExecuTorch LICENSE.
"""Validated, KV-only Int16 boundaries for shared-observer Qwen exports."""
import json
import hashlib
import math
import re
from pathlib import Path

SPEC_KEY = "SharedKvIO"


def rewrite_kv_io(mlir, pairs):
    """pairs contains (input SSA name, delegated output index) in cache order.

    Do not infer pairing from equal scales or topological output order. The
    exporter supplies output identities, resolved against the delegated graph.
    Fail closed if a live boundary is missing or its scale is not shared.
    """
    text = mlir.decode() if isinstance(mlir, bytes) else mlir
    pattern = r'\s*(%[^ ]+) = "nir\.(QuantizeLayer|DequantizeLayer)"\((%[^)]+)\) : \((.+)\) -> (.+)'
    ops = {}
    for line in text.splitlines():
        m = re.fullmatch(pattern, line)
        if m:
            ops[m[1]] = (m[2], m[3], m[4], m[5], line)
    outputs = {int(m[2]): m[1] for m in re.finditer(
        r'"nir.OutputLayer"\((%[^)]+)\) \{link = #nir.iolink<frontend : (\d+)>', text)}
    def scale(typ):
        if 'symmetric.Int16' not in typ:
            raise ValueError(f"KV must be symmetric Int16: {typ}")
        m = re.search(r'qscales<\{([^}]+)\}', typ)
        value = float(m[1]) if m else float('nan')
        if not math.isfinite(value) or value <= 0:
            raise ValueError("KV requires one finite positive scale")
        return value
    remove, replacements, rows = set(), {}, []
    for index, (input_name, output_index) in enumerate(pairs):
        name = '%' + input_name
        qname = name + '_quantized'
        output = outputs[output_index]
        op, src, qt, ft, line = ops[output]
        if op != 'DequantizeLayer':
            raise ValueError("KV output is not a dequantization boundary")
        output_scale = scale(qt)
        if qname in ops:
            op, qsrc, oldtype, typ, qline = ops[qname]
            if op != 'QuantizeLayer' or qsrc != name or scale(typ) != output_scale:
                raise ValueError(f"KV input/output scales differ: {input_name}")
            old = f'{name}: {oldtype} {{nir.link'
            if old not in text:
                raise ValueError(f"Missing KV input declaration: {input_name}")
            text = text.replace(old, f'{name}: {typ} {{nir.link')
            remove.add(qline)
            replacements[qname] = name
        else:
            # Last-chunk attention may be dead. Its graph input still exists
            # in PTE metadata, but must not have a live unquantized NIR use.
            uses = len(re.findall(re.escape(name) + r'(?![\w.])', text))
            if uses > 1:
                raise ValueError(f"Unquantized live KV input: {input_name}")
            if uses:
                decl = re.search(re.escape(name) + r': (.+?) \{nir.link', text)
                if decl is None:
                    raise ValueError(f"Unexpected KV use: {input_name}")
                oldtype = decl[1]
                typ = oldtype.replace('!nir.Float32]>',
                    f'!nir.symmetric.Int16], !nir.qscales<{{{output_scale}}}>>')
                text = text.replace(f'{name}: {oldtype} {{nir.link', f'{name}: {typ} {{nir.link')
        old = f'"nir.OutputLayer"({output}) {{link = #nir.iolink<frontend : {output_index}>}} : ({ft})'
        if old not in text:
            raise ValueError("Missing KV output declaration")
        text = text.replace(old, f'"nir.OutputLayer"({src}) {{link = #nir.iolink<frontend : {output_index}>}} : ({qt})')
        remove.add(line)
        rows.append([index, output_scale, output_scale])
    text = '\n'.join(line for line in text.splitlines() if line not in remove) + '\n'
    for old, new in replacements.items():
        text = re.sub(re.escape(old) + r'(?![\w.])', lambda _: new, text)
    return text, rows


def prepare_kv_io(mlir, config, input_names, output_names):
    pairs = []
    for item in config['pairs']:
        if item['output'] not in output_names:
            raise ValueError(f"KV boundary left the delegate: {item}; outputs={output_names}")
        pairs.append((item['input'], output_names.index(item['output'])))
    text, rows = rewrite_kv_io(mlir, pairs)
    path = Path(config['qparams_path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'ar': config['ar'], 'rows': rows,
        'nir_ops_sha256': hashlib.sha256('\n'.join(line for line in text.splitlines()
            if '"nir.const"' not in line).encode()).hexdigest()}, indent=2))
    return text


def finalize_kv_io(buffer, method_files, cache_count, sidecar):
    """Validate all methods, write runner scales, and expose SHORT KV in PTE."""
    from executorch.exir._serialize._program import deserialize_pte_binary, serialize_pte_binary
    from executorch.exir.schema import ScalarType
    obj = deserialize_pte_binary(buffer)
    rows, by_ar = [], {}
    for file in method_files:
        data = json.loads(Path(file).read_text())
        ar = data['ar']
        scales = data['rows']
        if len(scales) != cache_count or [x[0] for x in scales] != list(range(cache_count)):
            raise ValueError("Incomplete KV scale coverage")
        if ar in by_ar:
            raise ValueError("Export one cache size per PTE; AR methods must be unique")
        if any(a != b or a <= 0 or not math.isfinite(a) for _, a, b in scales):
            raise ValueError("KV IO scales must be equal and positive")
        by_ar[ar] = scales
        rows.extend([ar, *item] for item in scales)
    if any(v != next(iter(by_ar.values())) for v in by_ar.values()):
        raise ValueError("KV scales must agree across AR methods")
    if len(obj.program.execution_plan) != len(by_ar):
        raise ValueError("KV sidecars do not cover all methods")
    for plan in obj.program.execution_plan:
        if len(plan.delegates) != 1 or plan.operators:
            raise ValueError("Quantized KV requires a single fully delegated method")
        for delegate in plan.delegates:
            delegate.compile_specs = [spec for spec in delegate.compile_specs if spec.key != SPEC_KEY]
        if len(plan.inputs) != cache_count + 3 or len(plan.outputs) not in (cache_count, cache_count + 1):
            raise ValueError("Unexpected Qwen KV interface")
        for i in plan.inputs[-cache_count:] + plan.outputs[-cache_count:]:
            tensor = plan.values[i].val
            if len(tensor.sizes) != 4 or tensor.scalar_type != ScalarType.FLOAT:
                raise ValueError("Unexpected floating KV boundary tensor")
            tensor.scalar_type = ScalarType.SHORT
    Path(sidecar).write_text(''.join(' '.join(map(str, row)) + '\n' for row in rows))
    return serialize_pte_binary(obj, extract_delegate_segments=True)


def validate_export(pte_path, sidecar_path):
    """Reject a stale FP32/unshared model when reusing an export directory."""
    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.exir.schema import ScalarType
    records = {}
    for line in Path(sidecar_path).read_text().splitlines():
        ar, index, si, so = line.split()
        ar, index, si, so = int(ar), int(index), float(si), float(so)
        if si != so or si <= 0 or not math.isfinite(si) or (ar, index) in records:
            raise ValueError("Not a shared-scale KV sidecar")
        records[ar, index] = si
    program = deserialize_pte_binary(Path(pte_path).read_bytes()).program
    expected, cache_lengths = set(), set()
    for plan in program.execution_plan:
        count = len(plan.inputs) - 3
        if count <= 0 or len(plan.delegates) != 1 or plan.operators:
            raise ValueError("Not a fully delegated Qwen KV model")
        if len(plan.outputs) not in (count, count + 1):
            raise ValueError("Unexpected KV output count")
        outputs = [plan.values[i].val for i in plan.outputs[-count:]]
        ar = outputs[0].sizes[2]
        for index, (a, b) in enumerate(zip(plan.inputs[-count:], plan.outputs[-count:])):
            x, y = plan.values[a].val, plan.values[b].val
            if x.scalar_type != ScalarType.SHORT or y.scalar_type != ScalarType.SHORT:
                raise ValueError("Stale FP32 KV model: re-export into a fresh directory")
            if len(x.sizes) != 4 or y.sizes != [x.sizes[0], x.sizes[1], ar, x.sizes[3]]:
                raise ValueError("Inconsistent KV shapes")
            cache_lengths.add(x.sizes[2])
            expected.add((ar, index))
    if set(records) != expected or len(cache_lengths) != 1:
        raise ValueError("KV sidecar/shape does not cover every method")
    for ar, index in records:
        if any(value != records[ar, index] for (other_ar, other_index), value in records.items() if index == other_index):
            raise ValueError("KV scales differ between AR methods")


if __name__ == '__main__':
    import sys
    root = Path(sys.argv[1])
    for directory in sorted(root.glob('chunk_*')):
        files = list(directory.glob('*.pte'))
        if len(files) != 1:
            raise ValueError(f"Expected one PTE in {directory}")
        validate_export(files[0], directory / 'kv_io_qparams.txt')
        print(f'Validated shared Int16 KV: {directory.name}')
