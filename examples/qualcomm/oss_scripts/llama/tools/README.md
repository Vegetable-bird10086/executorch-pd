# Llama Tools

## `pte_qat_checkpoint_reverse_strip.py`

This tool rebuilds the quantized weights that were actually written into an
exported ExecuTorch `.pte`, then strips those matched blocks back out of the
file and emits rebuild indexes.

It is the script you use to generate a `stripped.pte` for the PD split
inference workflow.

Canonical path:

```bash
python examples/qualcomm/oss_scripts/llama/tools/pte_qat_checkpoint_reverse_strip.py
```

### Inputs

- `--old-pte`: source `.pte` to strip
- `--qat-checkpoint`: QAT `safetensors` checkpoint used to rebuild quantized weights

### Outputs

The tool writes these files into `--out-dir`:

- `stripped.pte`
- `index.json`
- `index.bin`
- `report.txt`

### Default example

```bash
cd /root/autodl-tmp/executorch

python examples/qualcomm/oss_scripts/llama/tools/pte_qat_checkpoint_reverse_strip.py \
  --old-pte /root/autodl-tmp/executorch/llama_qnn/hybrid_llama_qnn.pte \
  --qat-checkpoint /root/autodl-tmp/Qwen3-1.7b-2bit/model.safetensors \
  --out-dir /root/autodl-tmp/executorch/llama_qnn/pte_qat_strip_out
```

### Keep `output.conv`

Use this when you want the stripped file to keep `output.conv` and exclude it
from the rebuild index:

```bash
python examples/qualcomm/oss_scripts/llama/tools/pte_qat_checkpoint_reverse_strip.py \
  --old-pte /root/autodl-tmp/executorch/llama_qnn/hybrid_llama_qnn.pte \
  --qat-checkpoint /root/autodl-tmp/Qwen3-1.7b-2bit/model.safetensors \
  --out-dir /root/autodl-tmp/executorch/llama_qnn/pte_qat_strip_out_with_output \
  --keep-output
```

### Important options

- `--bits-hint 2`: packed source bits hint
- `--group-size 32`: QAT group size hint
- `--qweight-mode qweight_minus_qzeros`: reconstruct exported packed weights using checkpoint `qweight` and `qzeros`
- `--search-direction reverse`: search blocks from the back of the immutable source buffer
- `--no-strict`: continue instead of aborting when an expected block is missing
- `--no-progress`: disable realtime progress printing

### Recommended workflow

1. Export or locate the full source `.pte`.
2. Run this tool to produce `stripped.pte` and the index files.
3. Keep `index.json` or `index.bin` together with `stripped.pte`, since the
   stripped file alone is not enough to rebuild the removed quant blocks.
4. Record the exact QAT checkpoint path used for the strip operation.

### Notes

- The current implementation handles `output.conv` plus the 28 decoder layers'
  `attention.wq/wk/wv/wo_conv` and `feed_forward.w1/w2/w3_conv`.
- `tok_embeddings` is intentionally not processed.
- The historical experimental copy still exists under
  `/root/autodl-tmp/executorch/llama_qnn/pte_qat_checkpoint_reverse_strip.py`,
  but the tools path above is the maintained location to use going forward.
