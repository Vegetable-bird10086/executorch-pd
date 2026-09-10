"""Post-quantization GQA rewrite, preserving the existing Q/DQ boundaries."""
import operator
import torch


def group_gqa_matmuls(graph_module, num_heads, num_kv_heads):
    if num_heads % num_kv_heads:
        raise ValueError('Attention heads must be divisible by KV heads')
    groups = num_heads // num_kv_heads
    if groups == 1:
        return []
    graph = graph_module.graph
    changed = []
    for node in list(graph.nodes):
        if node.op != 'call_function' or node.target != torch.ops.aten.matmul.default:
            continue
        a, b = node.args
        is_qk = b.target == torch.ops.aten.transpose.int
        view = b.args[0] if is_qk else b
        if view.target != torch.ops.aten.view.default:
            continue
        repeat = view.args[0]
        if repeat.target != torch.ops.aten.repeat.default:
            continue
        if list(repeat.args[1]) != [1, 1, groups, 1]:
            continue
        if is_qk and tuple(b.args[1:]) != (2, 3):
            continue
        kv = repeat.args[0]
        with graph.inserting_before(node):
            if is_qk:
                kv = graph.call_function(torch.ops.aten.transpose.int, (kv, 2, 3))
            batch = graph.call_function(torch.ops.aten.sym_size.int, (a, 0))
            tokens = graph.call_function(torch.ops.aten.sym_size.int, (a, 2))
            last = graph.call_function(torch.ops.aten.sym_size.int, (a, 3))
            rows = graph.call_function(operator.mul, (tokens, groups))
            grouped = graph.call_function(torch.ops.aten.reshape.default, (a, [batch, num_kv_heads, rows, last]))
            product = graph.call_function(torch.ops.aten.matmul.default, (grouped, kv))
            out = graph.call_function(torch.ops.aten.reshape.default, (product, [batch, num_heads, tokens, -1]))
        node.replace_all_uses_with(out)
        graph.erase_node(node)
        changed.append({'node': node.name, 'kind': 'QK' if is_qk else 'AV'})
    graph.eliminate_dead_code()
    graph.lint()
    graph_module.recompile()
    return changed
