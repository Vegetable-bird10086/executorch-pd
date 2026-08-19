# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import torch
from executorch.backends.qualcomm.quantizer.custom_annotation import (
    annotate_kv_8bit,
    annotate_kv_8bit_a8,
    annotate_llm_all_token_axis_a8,
    annotate_llm_residual_token_axis_a8,
    annotate_llm_v_projection_token_axis_a8,
    annotate_llm_prefill_kv_output_per_tensor_a8,
)
from executorch.backends.qualcomm.quantizer.quant_recipe import (
    QuantGranularity,
    QuantRecipe,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from torchao.quantization.pt2e import HistogramObserver, MinMaxObserver


class PercentileHistogramObserver(HistogramObserver):
    """Linear-time percentile clipping for the experimental native-A8 path."""

    @torch.jit.export
    def reset_min_max_vals(self):
        self.min_val.fill_(float("inf"))
        self.max_val.fill_(float("-inf"))
        self.histogram.zero_()

    def calculate_qparams(self):
        if self.min_val == float("inf") or self.max_val == float("-inf"):
            return self._calculate_qparams(self.min_val, self.max_val)
        total = self.histogram.sum()
        if total <= 0 or self.max_val <= self.min_val:
            return self._calculate_qparams(self.min_val, self.max_val)

        retained = float(os.environ.get("ET_QNN_A8_RETAINED_PERCENTILE", "0.999"))
        if not 0.0 < retained <= 1.0:
            raise ValueError(
                "ET_QNN_A8_RETAINED_PERCENTILE must be in (0, 1], "
                f"got {retained}"
            )
        tail = (1.0 - retained) / 2.0
        cumulative = torch.cumsum(self.histogram, dim=0)
        lower_count = total * tail
        upper_count = total * (1.0 - tail)
        lower_bin = int(torch.searchsorted(cumulative, lower_count).item())
        upper_bin = int(torch.searchsorted(cumulative, upper_count).item())
        lower_bin = max(0, min(lower_bin, self.bins - 1))
        upper_bin = max(lower_bin, min(upper_bin, self.bins - 1))
        bin_width = (self.max_val - self.min_val) / self.bins
        new_min = self.min_val + bin_width * lower_bin
        new_max = self.min_val + bin_width * (upper_bin + 1)
        zero = torch.zeros_like(new_min)
        return self._calculate_qparams(
            torch.minimum(new_min, zero), torch.maximum(new_max, zero)
        )


class StaticLLMQuantRecipe:
    """
    Qualcomm's static LLaMA quantization recipe.
    """

    def __init__(self):
        self.recipe: Optional[QuantRecipe] = None

        # For IO bitwidth
        self.default_quant_dtype = getattr(self, "default_quant_dtype", None)
        if self.default_quant_dtype is None:
            raise ValueError("default_quant_dtype must be defined in the recipe.")

    def get_kv_io_bit_width(self) -> int:
        if self.default_quant_dtype is None:
            return 32
        elif (
            self.default_quant_dtype == QuantDtype.use_8a8w
            or annotate_kv_8bit in self.recipe.custom_quant_annotations
        ):
            return 8
        else:
            # If quantized but not 8a8w or mix_quantization, it has to be 16bit kv io.
            return 16

    def get_logits_output_bit_width(self) -> int:
        # We use 16bit logits for all quant config
        return 32 if self.default_quant_dtype is None else 16


class LlamaStories260KQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()
        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
            .add_regex(
                {r"output\.conv"},
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class LlamaStories110MQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
            .add_regex(
                {r"layers\..*\.attention\.wv.*"},
                QuantDtype.use_8a4w,
                False,
                act_observer=MinMaxObserver,
                act_symmetric=True,
                granularity=QuantGranularity.PER_CHANNEL,
            )
            .add_regex(
                {r"output\.conv"},
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class Llama3_1BQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w_block

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
                note="default with 16bit activation",
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32, 1, 1)},
                note="Annotate with 16a4w block quantization since these layers are not sensitive.",
            )
            .add_regex(
                {
                    r"output\.conv",
                    #r"layers\.[0-3]\.feed_forward\.w2_conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
                note="Down proj layer is sensitive and should be annotated with 16a8w.",
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class Llama3_3BQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w_block

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32, 1, 1)},
            )
            .add_regex(
                {
                    r"output\.conv",
                    r"layers\.2[1-7]\.feed_forward\.w2_conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)

class Llama3_8BQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w_block

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            # === Conv（如果有的话，基本照搬）===
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32, 1, 1)},
            )


            # === FFN down_proj（最敏感，必须提精度）===
            .add_regex(
                {   
                    r"output\.conv",
                    #r"layers\.\d+\.feed_forward\.w2",   # down_proj
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )

        )

        # KV cache 用 8bit（非常关键）
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)

class Llama2_7BQuantRecipe(StaticLLMQuantRecipe):
    # Default to groupwise 4-bit weights with 16-bit activations.
    # Group size defaults to 128 (matches README examples); can be adjusted
    # by constructing the recipe with a different `group_size` value.
    default_quant_dtype = QuantDtype.use_16a4w_block

    def __init__(self, verbose: bool = False, group_size: int = 128):
        super().__init__()

        block_size = (1, group_size, 1, 1)

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": block_size},
            )
            .add_regex(
                {
                    r"output\.conv",
                    r"layers\..*\.feed_forward\.w2_conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class CodegenQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        ).add_node_target(
            {
                torch.ops.aten.conv2d.default,
            },
            QuantDtype.use_16a8w,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
        )


class Gemma_2BQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 64, 1, 1)},
            )
            .add_regex(
                {
                    r"layers\..*\.attention\.wv.*",
                    r"output\.conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class Gemma2QuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32, 1, 1)},
            )
            .add_regex(
                {
                    r"layers\..*\.attention\.wv.*",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class Gemma3QuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 64, 1, 1)},
            )
            .add_regex(
                {
                    r"layers\..*\.attention\.wv.*",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class GLM_1_5B_InstructQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32, 1, 1)},
            )
            .add_regex(
                {r"output\.conv"},
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class Granite_3_3_2B_InstructQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 64, 1, 1)},
            )
            .add_regex(
                {
                    r"layers\..*\.attention\.wv.*",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class InternVL3_1B_QuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        ).add_node_target(
            {
                torch.ops.aten.conv2d.default,
            },
            QuantDtype.use_16a8w,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
        )


class Phi4MiniQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 16, 1, 1)},
            )
            .add_regex(
                {r"layers\..*\.attention\.wv.*"},
                QuantDtype.use_8a4w,
                False,
                act_observer=MinMaxObserver,
                act_symmetric=True,
                granularity=QuantGranularity.PER_CHANNEL,
            )
            .add_regex(
                {r"output\.conv"},
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class Qwen2_5_0_5BQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        ).add_node_target(
            {
                torch.ops.aten.conv2d.default,
            },
            QuantDtype.use_16a4w_block,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_BLOCK,
            extra_kwargs={"block_size": (1, 16, 1, 1)},
        )


class Qwen2_5_1_5BQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 16, 1, 1)},
            )
            .add_regex(
                {r"output\.conv"},
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )


class Qwen3_0_6BQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32, 1, 1)},
            )
            .add_regex(
                {
                    #r"layers\..*\.feed_forward\.w2_conv",
                    r"output\.conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )


class Qwen3_1_7BQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32, 1, 1)},
            )
            .add_regex(
                {
                    r"output\.conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class Qwen3_4BQuantRecipe(Qwen3_1_7BQuantRecipe):
    pass


class Qwen3_4BA8QuantRecipe(StaticLLMQuantRecipe):
    """Experimental Qwen3 recipe with native U8 activations.

    The production Qwen3 recipes above remain A16. The convolution
    placeholder weights remain INT4 block32 so the existing GPTQ2 replacement
    contract is unchanged; only activation and output domains become U8.
    """

    default_quant_dtype = QuantDtype.use_8a4w

    def __init__(self, verbose: bool = False):
        super().__init__()
        conv_weight_bits = int(os.environ.get("ET_QNN_A8_CONV_WEIGHT_BITS", "4"))
        if conv_weight_bits not in (4, 8):
            raise ValueError(
                "ET_QNN_A8_CONV_WEIGHT_BITS must be 4 or 8, "
                f"got {conv_weight_bits}"
            )
        activation_observer = (
            PercentileHistogramObserver
            if os.environ.get("ET_QNN_A8_HISTOGRAM_OBSERVER", "") == "1"
            else MinMaxObserver
        )
        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=activation_observer,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
                note="experimental native 8bit activation",
            )
            .add_node_target(
                {torch.ops.aten.conv2d.default},
                (
                    QuantDtype.use_8a8w
                    if conv_weight_bits == 8
                    else QuantDtype.use_8a4w_block
                ),
                False,
                act_observer=activation_observer,
                granularity=(
                    QuantGranularity.PER_CHANNEL
                    if conv_weight_bits == 8
                    else QuantGranularity.PER_BLOCK
                ),
                extra_kwargs=(
                    {} if conv_weight_bits == 8 else {"block_size": (1, 32, 1, 1)}
                ),
                note=f"experimental U8 activation with INT{conv_weight_bits} weight",
            )
            .add_regex(
                {r"output\.conv"},
                QuantDtype.use_8a8w,
                False,
                act_observer=activation_observer,
                granularity=QuantGranularity.PER_CHANNEL,
                note="experimental U8 output activation",
            )
        )
        selective_ops = {
            item.strip()
            for item in os.environ.get(
                "ET_QNN_A8_PERCENTILE_OPS", ""
            ).split(",")
            if item.strip()
        }
        selective_targets = set()
        if "mul" in selective_ops:
            selective_targets.add(torch.ops.aten.mul.Tensor)
        if "add" in selective_ops:
            selective_targets.add(torch.ops.aten.add.Tensor)
        if selective_targets:
            self.recipe.add_node_target(
                selective_targets,
                QuantDtype.use_8a4w,
                False,
                act_observer=PercentileHistogramObserver,
                granularity=QuantGranularity.PER_TENSOR,
                note="selective native-A8 percentile activation clipping",
            )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit_a8)
        if os.environ.get("ET_QNN_A8_TOKEN_AXIS_RESIDUAL", "") == "1":
            self.recipe.custom_quant_annotations.append(
                annotate_llm_residual_token_axis_a8
            )
        if os.environ.get("ET_QNN_A8_TOKEN_AXIS_ALL", "") == "1":
            self.recipe.custom_quant_annotations.append(
                annotate_llm_all_token_axis_a8
            )
        if os.environ.get("ET_QNN_A8_V_TOKEN_AXIS", "") == "1":
            self.recipe.custom_quant_annotations.append(
                annotate_llm_v_projection_token_axis_a8
            )
        if os.environ.get("ET_QNN_A8_KV_OUTPUT_PER_TENSOR", "") == "1":
            self.recipe.custom_quant_annotations.append(
                annotate_llm_prefill_kv_output_per_tensor_a8
            )

    def get_kv_io_bit_width(self) -> int:
        return 8

    def get_logits_output_bit_width(self) -> int:
        return 8


class Smollm2QuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        ).add_node_target(
            {
                torch.ops.aten.conv2d.default,
            },
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
        )


class Smollm3QuantRecipe(StaticLLMQuantRecipe):

    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32, 1, 1)},
            )
            .add_regex(
                {
                    r"layers\..*\.attention\.wq.*",
                    r"layers\..*\.attention\.wk.*",
                    r"layers\..*\.attention\.wv.*",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
            .add_regex(
                {
                    r"output\.conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class SmolVLMQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        ).add_node_target(
            {
                torch.ops.aten.conv2d.default,
            },
            QuantDtype.use_16a8w,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
        )
