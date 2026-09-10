# Copyright (c) 2026. Licensed under the BSD-style ExecuTorch LICENSE.
"""Share per-cache calibration across history, new KV, and their concatenation."""
import torch
from torch.fx import Node
from torchao.quantization.pt2e.quantizer import QuantizationAnnotation, SharedQuantizationSpec
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

_PASSTHROUGH = {
    torch.ops.aten.to.dtype,
    torch.ops.aten.to.dtype_layout,
    torch.ops.aten._to_copy.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.alias.default,
}


def annotate_shared_kv(graph, activation_spec):
    """Opt-in KV policy; never share K with V, or different layers' caches.

    Like QNN's KV concat annotation, all concat inputs and its output share
    one observer. History and new values both update its range, so zero-filled
    history is not calibrated independently. The direct new-token graph output
    uses the same observer. Anchor at the earliest node for PT2E ordering.
    """
    matches = []
    for cat in graph.nodes:
        if cat.op != 'call_function' or cat.target != torch.ops.aten.cat.default:
            continue
        args = cat.args[0]
        dim = cat.args[1] if len(cat.args) > 1 else cat.kwargs.get("dim", 0)
        if len(args) != 2 or dim not in (2, -2):
            continue
        history, new = args
        if not isinstance(history, Node) or not isinstance(new, Node):
            continue
        path = [history]
        origin = history
        while origin.op == 'call_function' and origin.target in _PASSTHROUGH:
            origin = origin.args[0]
            if not isinstance(origin, Node):
                break
            path.append(origin)
        if not isinstance(origin, Node) or origin.op != 'placeholder':
            continue
        if not str(origin.target).startswith(('cache_', 'past_key', 'past_value')):
            continue
        hv, nv = history.meta.get('val'), new.meta.get('val')
        if hv is None or nv is None or hv.dim() != 4 or nv.dim() != 4:
            continue
        if hv.dtype != torch.float32 or nv.dtype != torch.float32:
            raise ValueError('Shared KV annotation expects floating pre-quantization tensors')
        if any(hv.shape[d] != nv.shape[d] for d in (0, 1, 3)):
            raise ValueError('History/new KV shapes disagree outside token dimension')
        shared = SharedQuantizationSpec(origin)
        ann = new.meta.setdefault(Q_ANNOTATION_KEY, QuantizationAnnotation())
        ann.output_qspec = shared
        ann._annotated = True
        for node in path:
            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                output_qspec=activation_spec if node is origin else shared, _annotated=True)
        cat.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map={history: shared, new: shared},
            output_qspec=shared, _annotated=True)
        matches.append({'history': origin.name, 'new': new.name, 'concat': cat.name})
    return matches
