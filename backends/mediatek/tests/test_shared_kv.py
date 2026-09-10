import unittest
import torch
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.backends.mediatek.quantizer.quantizer import NeuropilotQuantizer
from executorch.backends.mediatek.quantizer.qconfig import Precision

class Pair(torch.nn.Module):
    def forward(self, x, cache_k, cache_v):
        k = x * 3.0
        v = x * 0.03125
        return (torch.cat([torch.ops.aten.to.dtype_layout(cache_k, dtype=torch.float32, device=x.device, layout=torch.strided), k], 2),
                torch.cat([torch.ops.aten.to.dtype_layout(cache_v, dtype=torch.float32, device=x.device, layout=torch.strided), v], 2), k, v)

class SharedKvTest(unittest.TestCase):
    def build(self, enabled):
        x=torch.linspace(-40, 30, 16).reshape(1,2,2,4)
        history=torch.zeros(1,2,8,4)
        gm=torch.export.export(Pair(),(x,history,history.clone()),strict=True).module()
        q=NeuropilotQuantizer();q.setup_precision(Precision.A16W4)
        q.set_shared_kv_quantization(enabled)
        prepared=prepare_pt2e(gm,q);prepared(x,history,history)
        return convert_pt2e(prepared,fold_quantize=False),q,x,history

    def test_zero_history_shares_new_value_range(self):
        gm,q,x,h=self.build(True)
        self.assertEqual(len(q.shared_kv_annotations),2)
        nodes=list(gm.graph.nodes);outs=next(n for n in nodes if n.op=='output').args[0]
        params=[]
        for i,name in enumerate(['cache_k','cache_v']):
            placeholder=next(n for n in nodes if n.op=='placeholder' and n.target==name)
            quant=next(iter(placeholder.users))
            self.assertIn('quantize_per_tensor',str(quant.target))
            # Direct new KV and full concat must use identical qparams.
            self.assertEqual(quant.args[1:],outs[2+i].args[1:])
            self.assertEqual(quant.args[1:],outs[i].args[1:])
            self.assertGreater(float(quant.args[1]),1e-6)
            params.append(quant.args[1:])
        self.assertNotEqual(params[0],params[1])
        first=gm(x,h,h)
        histories=[]
        for new,param in zip(first[2:],params):
            scale,zp,lo,hi,dtype=param
            code=torch.ops.quantized_decomposed.quantize_per_tensor.default(new,*param)
            restored=torch.ops.quantized_decomposed.dequantize_per_tensor.default(code,*param)
            self.assertTrue(torch.equal(restored,new))
            histories.append(torch.cat([h[:,:,2:],restored],2))
        second=gm(x*.5,*histories)
        self.assertTrue(torch.equal(second[0][:,:,:8],histories[0]))
        self.assertTrue(torch.equal(second[1][:,:,:8],histories[1]))

    def test_opt_in(self):
        _,q,_,_=self.build(False)
        self.assertEqual(q.shared_kv_annotations,[])

    def test_rotary_concat_not_matched(self):
        class Rotary(torch.nn.Module):
            def forward(self,cache_k,x):return torch.cat([cache_k,x],-1)
        x=torch.ones(1,2,2,4)
        gm=torch.export.export(Rotary(),(x,x),strict=True).module()
        q=NeuropilotQuantizer();q.set_shared_kv_quantization()
        prepare_pt2e(gm,q)
        self.assertEqual(q.shared_kv_annotations,[])

if __name__=='__main__':unittest.main()
