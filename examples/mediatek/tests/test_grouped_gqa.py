import unittest
import torch
from aot_utils.llm_utils.grouped_gqa import group_gqa_matmuls

class Attention(torch.nn.Module):
    def forward(self, q, k, v, mask):
        b, h, t, d = q.shape
        s = k.shape[2]
        k = k.repeat(1, 1, 2, 1).view(b, h, s, d)
        v = v.repeat(1, 1, 2, 1).view(b, h, s, d)
        scores = torch.matmul(q, k.transpose(2, 3)) / d**0.5
        return torch.matmul(torch.softmax(scores + mask, dim=-1), v)

class GroupedGQATest(unittest.TestCase):
    def test_shapes_and_values(self):
        for b, t, s in [(1, 1, 17), (1, 128, 256), (2, 7, 19)]:
            with self.subTest(batch=b, tokens=t):
                inputs = (torch.randn(b, 4, t, 8), torch.randn(b, 2, s, 8), torch.randn(b, 2, s, 8), torch.randn(b, 1, t, s))
                gm = torch.export.export(Attention(), inputs).module()
                ref = gm(*inputs)
                self.assertEqual(len(group_gqa_matmuls(gm, 4, 2)), 2)
                torch.testing.assert_close(gm(*inputs), ref, rtol=0, atol=0)
                self.assertFalse(any(n.target == torch.ops.aten.repeat.default for n in gm.graph.nodes))
                self.assertEqual(group_gqa_matmuls(gm, 4, 2), [])

if __name__ == '__main__': unittest.main()
