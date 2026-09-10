"""Host regression for the MTK graph-output ABI and production C++ KV updater.

Run with the MTK Python environment; no model weights or phone are required.
"""
import ast
import copy
import ctypes
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.llm_models.modeling_common import Attention, ModelChunk


class IncrementalKvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        path = Path(cls.temp.name)
        (path / 'bridge.cpp').write_text('''#include "IncrementalKvCache.h"
extern "C" void update(void* h, const void* n, size_t r, size_t c, size_t t, size_t b) {
 example::UpdateSlidingKvCache(h, n, r, c, t, b);
}
extern "C" void update_valid(void* h, const void* n, size_t r, size_t c, size_t t, size_t b, size_t v) {
 example::UpdateSlidingKvCache(h, n, r, c, t, b, v);
}
''')
        subprocess.run(['c++', '-std=c++17', '-shared', '-fPIC', '-O2',
                        '-I' + str(ROOT / 'executor_runner/llama_runner'),
                        str(path / 'bridge.cpp'), '-o', str(path / 'bridge.so')], check=True)
        cls.lib = ctypes.CDLL(str(path / 'bridge.so'))
        cls.lib.update.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_size_t] * 4
        cls.lib.update.restype = None
        cls.lib.update_valid.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_size_t] * 5
        cls.lib.update_valid.restype = None

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def update(self, history, new, valid=None):
        dst, src = history.contiguous().clone(), new.contiguous()
        args = (dst.data_ptr(), src.data_ptr(), dst.shape[0] * dst.shape[1],
                dst.shape[2], src.shape[2], dst.shape[3] * dst.element_size())
        if valid is None:
            self.lib.update(*args)
        else:
            self.lib.update_valid(*args, valid)
        return dst

    @staticmethod
    def config(incremental=False):
        return SimpleNamespace(hidden_size=32, intermediate_size=48,
                               num_attention_heads=4, num_key_value_heads=2,
                               head_dim=8, combine_qkv=False, use_qk_norm=True,
                               norm_eps=1e-6, position_embedding='rope',
                               output_new_cache_only=incremental, norm='RMSNorm',
                               use_stable_embedding=False)

    def test_host_update_matches_concat_slice(self):
        for dtype in [torch.float32, torch.float16, torch.uint8, torch.int16]:
            for cache_len in [1, 8, 17]:
                history = torch.arange(2 * 3 * cache_len * 8).reshape(2, 3, cache_len, 8).to(dtype)
                for tokens in [1, 4, cache_len, cache_len + 3, 1]:
                    new = torch.randint(0, 127, (2, 3, tokens, 8)).to(dtype)
                    expected = torch.cat((history, new), dim=2)[:, :, -cache_len:, :]
                    history = self.update(history, new)
                    torch.testing.assert_close(history, expected, rtol=0, atol=0)

    def test_valid_history_matches_full_update(self):
        torch.manual_seed(42)
        for dtype in [torch.float32, torch.float16, torch.uint8, torch.int16]:
            for capacity in [1, 17, 128, 4096]:
                for valid in [0, 1, capacity // 2, capacity, capacity + 5]:
                    history = torch.zeros(2, 3, capacity, 8, dtype=dtype)
                    live = min(valid, capacity)
                    if live:
                        history[:, :, -live:] = torch.randint(0, 127, (2, 3, live, 8)).to(dtype)
                    for tokens in [0, 1, 128, capacity, capacity + 3]:
                        new = torch.randint(0, 127, (2, 3, tokens, 8)).to(dtype)
                        expected = torch.cat((history, new), dim=2)[:, :, -capacity:, :]
                        actual = self.update(history, new, valid)
                        self.assertTrue(torch.equal(actual, expected), (dtype, capacity, valid, tokens))

    def test_multiblock_attention_padding_and_decode(self):
        torch.manual_seed(9)
        old = Attention(self.config()).eval()
        new = copy.deepcopy(old)
        new.output_new_cache_only = True
        cache_len = 12
        old_cache = [torch.zeros(1, 2, cache_len, 8) for _ in range(2)]
        new_cache = [x.clone() for x in old_cache]
        seen = 0
        # Leading padding, full blocks, trailing padding, then one-token decode;
        # enough iterations to cross the full-cache boundary.
        for tokens, left, right in [(4, 2, 0), (4, 0, 0), (4, 0, 1), (1, 0, 0),
                                    (4, 0, 0), (1, 0, 0), (1, 0, 0), (4, 0, 2), (1, 0, 0)]:
            x = torch.randn(1, tokens, 32)
            pos = torch.randn(1, 2, tokens, 8)
            mask = torch.full((1, 1, tokens, cache_len + tokens), -100.)
            mask[:, :, :, cache_len - min(seen, cache_len):cache_len] = 0
            for i in range(tokens):
                if i >= left:
                    mask[:, :, i, cache_len + left:cache_len + i + 1] = 0
            a = old(x, mask, pos, *old_cache)
            b = new(x, mask, pos, *new_cache)
            torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
            old_cache = list(a[1:])
            new_cache = [self.update(h, n, seen) for h, n in zip(new_cache, b[1:])]
            for previous, updated in zip(old_cache, new_cache):
                torch.testing.assert_close(previous, updated, rtol=0, atol=0)
                # Existing runner padding postprocess is applied after updating.
                if left:
                    previous[:, :, cache_len-tokens:cache_len-tokens+left] = 0
                    updated[:, :, cache_len-tokens:cache_len-tokens+left] = 0
                if right:
                    alive = min(seen + tokens, cache_len)
                    for tensor in (previous, updated):
                        tensor[:, :, cache_len-alive+right:] = tensor[:, :, cache_len-alive:-right].clone()
                        tensor[:, :, cache_len-alive:cache_len-alive+right] = 0
            seen += tokens - left - right

    def test_calibration_reconstructs_full_history(self):
        # Execute the actual export calibration helper without importing SDKs
        # or loading checkpoint/tokenizer dependencies from the CLI module.
        tree = ast.parse((ROOT / 'model_export_scripts/qwen.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'forward_and_save')
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'qwen.py', 'exec'), namespace)
        forward = namespace['forward_and_save']
        old = ModelChunk(self.config(), 2, 0, jit_trace=True).eval()
        old.device_list = ['cpu', 'cpu']
        new = copy.deepcopy(old)
        new.config.output_new_cache_only = True
        for layer in new.layers:
            layer.self_attn.output_new_cache_only = True
        histories = [[torch.zeros(4, 2, 12, 8)] for _ in range(2)]
        for count in [4, 4, 1, 4, 1]:
            x = torch.randn(1, count, 32)
            mask = torch.zeros(1, 1, count, 12 + count)
            pos = torch.randn(1, 2, count, 8)
            a, histories[0] = forward([old], x, histories[0], mask, pos, {}, [2], 'test')
            b, histories[1] = forward([new], x, histories[1], mask, pos, {}, [2], 'test')
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(histories[0][0], histories[1][0], rtol=0, atol=0)

    def test_export_new_kv_has_no_history_dependency(self):
        model = ModelChunk(self.config(True), 2, 0, jit_trace=True).eval()
        model.device_list = ['cpu', 'cpu']
        inputs = model.get_example_inputs(4, 12)
        with torch.no_grad():
            outputs = model(*inputs)
            self.assertEqual(len(outputs), 5)
            self.assertTrue(all(x.shape == (1, 2, 4, 8) for x in outputs[1:]))
            exported = torch.export.export(model, inputs, strict=True)
            actual = exported.module()(*inputs)
            for x, y in zip(outputs, actual):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            # Final KV-only chunk preserves its ABI and drops its hidden output.
            model.prefill_no_output = True
            tail = torch.export.export(model, inputs, strict=True)
            actual = tail.module()(*inputs)
            self.assertEqual(len(actual), 4)
            for x, y in zip(outputs[1:], actual):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        # Single-layer K/V do not depend on historical KV. Verify the exported
        # output ancestry, not merely tensor shape or Python implementation.
        attention = Attention(self.config(True)).eval()
        sample = (inputs[0], inputs[1], inputs[2], inputs[3], inputs[5])
        graph = torch.export.export(attention, sample).graph
        output = next(n for n in graph.nodes if n.op == 'output')
        def ancestors(node):
            result = {node}
            for source in node.all_input_nodes:
                result |= ancestors(source)
            return result
        for node in output.args[0][1:]:
            self.assertFalse(any(n.op == 'placeholder' and n.name in ('past_key', 'past_value')
                                 for n in ancestors(node)))


if __name__ == '__main__':
    unittest.main()
