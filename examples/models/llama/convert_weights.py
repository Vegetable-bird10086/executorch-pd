from typing import Dict

import json
import os
import torch
from executorch.examples.models.checkpoint import get_mapped_key

from safetensors.torch import load_file


# Mapping from HF/unsloth-style keys -> ExecuTorch Meta-style keys.
# Includes both base model keys (embed, lm_head, attention, mlp, norms)
# and existing LoRA unsloth keys.
_UNSLOTH_TO_META = {
    # Base model mappings
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "lm_head.weight": "output.weight",
    "model.norm.weight": "norm.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",

    # LoRA / unsloth-style mappings (existing)
    "base_model.model.model.layers.{}.mlp.down_proj.lora_A.weight": "layers.{}.feed_forward.w2.lora_a.weight",
    "base_model.model.model.layers.{}.mlp.down_proj.lora_B.weight": "layers.{}.feed_forward.w2.lora_b.weight",
    "base_model.model.model.layers.{}.mlp.gate_proj.lora_A.weight": "layers.{}.feed_forward.w1.lora_a.weight",
    "base_model.model.model.layers.{}.mlp.gate_proj.lora_B.weight": "layers.{}.feed_forward.w1.lora_b.weight",
    "base_model.model.model.layers.{}.mlp.up_proj.lora_A.weight": "layers.{}.feed_forward.w3.lora_a.weight",
    "base_model.model.model.layers.{}.mlp.up_proj.lora_B.weight": "layers.{}.feed_forward.w3.lora_b.weight",
    "base_model.model.model.layers.{}.self_attn.k_proj.lora_A.weight": "layers.{}.attention.wk.lora_a.weight",
    "base_model.model.model.layers.{}.self_attn.k_proj.lora_B.weight": "layers.{}.attention.wk.lora_b.weight",
    "base_model.model.model.layers.{}.self_attn.o_proj.lora_A.weight": "layers.{}.attention.wo.lora_a.weight",
    "base_model.model.model.layers.{}.self_attn.o_proj.lora_B.weight": "layers.{}.attention.wo.lora_b.weight",
    "base_model.model.model.layers.{}.self_attn.q_proj.lora_A.weight": "layers.{}.attention.wq.lora_a.weight",
    "base_model.model.model.layers.{}.self_attn.q_proj.lora_B.weight": "layers.{}.attention.wq.lora_b.weight",
    "base_model.model.model.layers.{}.self_attn.v_proj.lora_A.weight": "layers.{}.attention.wv.lora_a.weight",
    "base_model.model.model.layers.{}.self_attn.v_proj.lora_B.weight": "layers.{}.attention.wv.lora_b.weight",
}


def unsloth_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from unsloth format to Meta's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in unsloth format.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    converted_state_dict = {}

    for key, value in state_dict.items():
        try:
            new_key = get_mapped_key(key, _UNSLOTH_TO_META)
        except Exception as e:
            raise ValueError(f"Key {key} not found in mapping") from e

        converted_state_dict[new_key] = value

    # If lm_head (output) is not present, assume tied embeddings and copy.
    if "output.weight" not in converted_state_dict and "tok_embeddings.weight" in converted_state_dict:
        converted_state_dict["output.weight"] = converted_state_dict["tok_embeddings.weight"]

    return converted_state_dict


def load_checkpoint_from_safetensors(input_dir: str) -> Dict:
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        # Sharded checkpoint.
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        checkpoint_shards = sorted(set(weight_map.values()))

        # Load all the shards into memory
        shard_to_weights = {}
        for shard in checkpoint_shards:
            shard_to_weights[shard] = load_file(os.path.join(input_dir, shard))

        # Merge tensors into consolidated state dict.
        merged_state_dict = {}
        for weight_name, shard in weight_map.items():
            tensor = shard_to_weights[shard][weight_name]
            merged_state_dict[weight_name] = tensor
        return merged_state_dict

    # Single checkpoint.
    model_path = os.path.join(input_dir, "model.safetensors")
    if os.path.exists(model_path):
        return load_file(os.path.join(input_dir, "model.safetensors"))

    raise FileNotFoundError(f"Could not find safetensors checkpoint in {input_dir}")


def load_and_convert_unsloth_to_meta(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a checkpoint file or directory and convert it to Meta's format.

    Args:
        checkpoint_path (str): Path to the checkpoint file or directory.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    # If a directory is provided, try to load sharded safetensors or a single file.
    if os.path.isdir(checkpoint_path):
        state_dict = load_checkpoint_from_safetensors(checkpoint_path)
    else:
        # Single safetensors file
        state_dict = load_file(checkpoint_path)

    return unsloth_to_meta(state_dict)


def convert_weights(input_dir: str, output_file: str) -> None:
    """
    Compatibility wrapper used by the HF download + convert flow.

    Args:
        input_dir: Path to downloaded HF checkpoint or safetensors file (or directory).
        output_file: Path where the converted Meta-format .pth will be saved.
    """
    print("Loading checkpoint...")
    # load_and_convert_unsloth_to_meta supports a safetensors file path or
    # a single-file checkpoint. If a directory is provided, try to load the
    # index or a single safetensors file inside it.
    state_dict = None
    try:
        state_dict = load_and_convert_unsloth_to_meta(input_dir)
    except Exception:
        # Try to find a safetensors file inside the directory
        import os

        if os.path.isdir(input_dir):
            # Prefer an index file or any .safetensors shard
            candidates = [
                os.path.join(input_dir, f)
                for f in os.listdir(input_dir)
                if f.endswith(".safetensors") or f.endswith(".pt") or f.endswith(".pth")
            ]
            if not candidates:
                raise
            # Use the first candidate
            state_dict = load_and_convert_unsloth_to_meta(candidates[0])
        else:
            raise

    print("Saving converted checkpoint...")
    import torch

    torch.save(state_dict, output_file)
    print("Done.")
