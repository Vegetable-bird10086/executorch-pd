import json
import tempfile
import unittest
from pathlib import Path

from executorch.backends.mediatek.quantized_kv_io import prepare_kv_io, rewrite_kv_io

F = '!nir.shape<[1x8x128x128x!nir.Float32]>'
Q = '!nir.shape<[1x8x128x128x!nir.symmetric.Int16], !nir.qscales<{0.01}>>'
V = Q.replace('0.01', '0.02')


def graph(unused=False):
    lines = ['module {', 'func.func @graph0(' + ('' if unused else f'%cache_0: {F} {{nir.link = #nir.iolink<frontend : 0>}}, ') + f'%cache_1: {F} {{nir.link = #nir.iolink<frontend : 1>}}) {{']
    if not unused:
        lines += [f' %cache_0_quantized = "nir.QuantizeLayer"(%cache_0) : ({F}) -> {Q}']
    lines += [f' %cache_1_quantized = "nir.QuantizeLayer"(%cache_1) : ({F}) -> {V}',
              f' %k = "nir.DequantizeLayer"(%new_k) : ({Q}) -> {F}',
              f' %v = "nir.DequantizeLayer"(%new_v) : ({V}) -> {F}',
              f' "nir.OutputLayer"(%v) {{link = #nir.iolink<frontend : 0>}} : ({F}) -> ()',
              f' "nir.OutputLayer"(%k) {{link = #nir.iolink<frontend : 1>}} : ({F}) -> ()', 'return', '}', '}']
    return '\n'.join(lines)


class KvBoundaryTest(unittest.TestCase):
    def test_reordered_outputs_keep_pair_identity(self):
        text, rows = rewrite_kv_io(graph().encode(), [('cache_0', 1), ('cache_1', 0)])
        self.assertEqual(rows, [[0, .01, .01], [1, .02, .02]])
        self.assertNotIn('"nir.QuantizeLayer"', text)
        self.assertNotIn('"nir.DequantizeLayer"', text)
        self.assertIn('"nir.OutputLayer"(%new_v)', text)
        self.assertIn(f'%cache_0: {Q}', text)

    def test_wrong_pair_or_scale_fails(self):
        with self.assertRaisesRegex(ValueError, 'scales differ'):
            rewrite_kv_io(graph(), [('cache_0', 0), ('cache_1', 1)])

    def test_missing_tail_input_keeps_output_scale(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = {'ar': 128, 'qparams_path': str(Path(root)/'scales.json'), 'pairs': [
                {'input': 'cache_0', 'output': 'k'}, {'input': 'cache_1', 'output': 'v'}]}
            prepare_kv_io(graph(True), cfg, ['cache_1'], ['v', 'k'])
            data = json.loads(Path(cfg['qparams_path']).read_text())
            self.assertEqual(data['rows'], [[0, .01, .01], [1, .02, .02]])

    def test_live_unquantized_input_is_rejected(self):
        source = graph().replace(f' %cache_0_quantized = "nir.QuantizeLayer"(%cache_0) : ({F}) -> {Q}', ' %use = "nir.IdentityLayer"(%cache_0)')
        with self.assertRaisesRegex(ValueError, 'live KV'):
            rewrite_kv_io(source, [('cache_0', 1), ('cache_1', 0)])

    def test_non_kv_boundary_unchanged(self):
        source = graph() + f'\n %hidden_quantized = "nir.QuantizeLayer"(%hidden) : ({F}) -> {Q}'
        text, _ = rewrite_kv_io(source, [('cache_0', 1), ('cache_1', 0)])
        self.assertIn('"nir.QuantizeLayer"(%hidden)', text)

    def test_output_outside_delegate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'left the delegate'):
            prepare_kv_io(graph(), {'pairs': [{'input': 'cache_0', 'output': 'missing'}]}, ['cache_0'], ['v', 'k'])


if __name__ == '__main__':
    unittest.main()
