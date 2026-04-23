#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/pd_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/pte_rebuilder.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

DEFINE_string(decoder_model_version, "qwen3", "The decoder model to execute.");
DEFINE_string(
    model_path,
    "kv_llama_qnn.pte",
    "Model serialized in flatbuffer format.");
DEFINE_string(
    stripped_model_path,
    "",
    "Stripped model serialized in flatbuffer format.");
DEFINE_string(
    index_bin_path,
    "",
    "Path to the binary strip index used to rebuild a stripped PTE in memory.");
DEFINE_string(
    qat_checkpoint_path,
    "",
    "Path to the QAT safetensors checkpoint used to rebuild a stripped PTE in memory.");
DEFINE_string(
    gguf_model_path,
    "",
    "Path to the T-MAC GGUF model used to rebuild stripped decoder blocks in memory.");
DEFINE_int32(
    qat_bits_hint,
    2,
    "Bit width hint for rebuilding stripped QAT blocks from the checkpoint.");
DEFINE_int32(
    qat_group_size,
    32,
    "Group size used when rebuilding stripped QAT blocks from the checkpoint.");
DEFINE_string(
    qat_qweight_mode,
    "qweight_minus_qzeros",
    "QAT qweight decoding mode used during stripped PTE rebuild.");
DEFINE_string(
    tokenizer_path,
    "tokenizer.bin",
    "Tokenizer path.");
DEFINE_string(
    prompt,
    "",
    "Prompt text to prefill and export.");
DEFINE_string(
    tokenized_prompt,
    "",
    "Optional raw uint64 token file used instead of string prompt.");
DEFINE_string(
    system_prompt,
    "",
    "Optional system prompt.");
DEFINE_string(
    prefill_export_dir,
    "",
    "Output directory for PD handoff export.");
DEFINE_string(
    kv_quant_attrs_path,
    "",
    "Required KV quant attrs JSON for 8-bit KV PD export. Use the Prefill-side file such as prefill_kv_quant_attrs.json.");
DEFINE_string(
    attention_sink_rope_path,
    "",
    "Attention sink rope PTE. Not supported in PD v1 export.");
DEFINE_int32(
    seq_len,
    1024,
    "Compiled sequence length budget to respect during prefill export.");
DEFINE_int32(
    eval_mode,
    1,
    "0: TokenGenerator(kv) / 1: HybridMode (prefill+kv) / 2: Lookahead Decoding");
DEFINE_bool(
    shared_buffer,
    false,
    "Whether to use shared RPC buffers.");

namespace {

struct ModuleBundle {
  std::unique_ptr<executorch::extension::Module> module;
  std::shared_ptr<std::vector<uint8_t>> pte_bytes;
};

struct ModuleMetaInfo {
  example::KvBitWidth kv_bitwidth{example::KvBitWidth::kWidth8};
};

std::vector<std::string> CollectPrompts(int argc, char** argv) {
  std::vector<std::string> prompts;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--prompt" && i + 1 < argc) {
      prompts.push_back(argv[i + 1]);
      i++;
    }
  }
  return prompts;
}

std::vector<uint8_t> read_binary_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  ET_CHECK_MSG(input.is_open(), "Unable to read file: %s", path.c_str());
  return std::vector<uint8_t>(
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

bool should_rebuild_from_stripped() {
  const bool has_stripped = !FLAGS_stripped_model_path.empty();
  const bool has_index = !FLAGS_index_bin_path.empty();
  const bool has_checkpoint = !FLAGS_qat_checkpoint_path.empty();
  const bool has_gguf = !FLAGS_gguf_model_path.empty();
  ET_CHECK_MSG(
      !(has_checkpoint && has_gguf),
      "Provide either qat_checkpoint_path or gguf_model_path, but not both");
  const bool has_rebuild_source = has_checkpoint || has_gguf;
  ET_CHECK_MSG(
      has_stripped == has_index && has_index == has_rebuild_source,
      "Provide stripped_model_path, index_bin_path, and one rebuild source together");
  return has_stripped;
}

ModuleBundle load_module_from_file_or_rebuild() {
  ModuleBundle bundle;
  if (!should_rebuild_from_stripped()) {
    bundle.pte_bytes =
        std::make_shared<std::vector<uint8_t>>(read_binary_file(FLAGS_model_path));
    auto data_loader = std::make_unique<executorch::extension::BufferDataLoader>(
        bundle.pte_bytes->data(), bundle.pte_bytes->size());
    bundle.module = std::make_unique<executorch::extension::Module>(std::move(data_loader));
    return bundle;
  }

  const std::vector<uint8_t> stripped_pte = read_binary_file(FLAGS_stripped_model_path);
  const std::vector<uint8_t> index_bytes = read_binary_file(FLAGS_index_bin_path);
  example::PteRebuildResult rebuild_result;
  if (!FLAGS_gguf_model_path.empty()) {
    const std::vector<uint8_t> gguf_bytes = read_binary_file(FLAGS_gguf_model_path);
    rebuild_result = example::rebuild_pte_from_stripped_gguf(
        stripped_pte, index_bytes, gguf_bytes);
  } else {
    const std::vector<uint8_t> checkpoint_bytes =
        read_binary_file(FLAGS_qat_checkpoint_path);
    rebuild_result = example::rebuild_pte_from_stripped_checkpoint(
        stripped_pte,
        index_bytes,
        checkpoint_bytes,
        FLAGS_qat_bits_hint,
        FLAGS_qat_group_size,
        FLAGS_qat_qweight_mode);
  }
  bundle.pte_bytes = rebuild_result.rebuilt_pte;
  auto data_loader = std::make_unique<executorch::extension::BufferDataLoader>(
      bundle.pte_bytes->data(), bundle.pte_bytes->size());
  bundle.module = std::make_unique<executorch::extension::Module>(std::move(data_loader));
  return bundle;
}

ModuleMetaInfo read_module_meta(executorch::extension::Module* module) {
  ModuleMetaInfo meta;
  auto method_names = module->method_names();
  ET_CHECK_MSG(method_names.ok(), "Failed to read module method names");
  if (method_names->count("get_kv_io_bit_width") > 0) {
    meta.kv_bitwidth = static_cast<example::KvBitWidth>(
        module->get("get_kv_io_bit_width").get().toScalar().to<int64_t>());
  }
  return meta;
}

std::string get_model_path_for_runner() {
  if (!FLAGS_stripped_model_path.empty()) {
    return FLAGS_stripped_model_path;
  }
  return FLAGS_model_path;
}

std::string get_formatted_prompt(
    const std::string& prompt,
    const std::string& system_prompt,
    example::DecoderModelVersion decoder_model_version) {
  std::string formatted_prompt;
  switch (decoder_model_version) {
    case example::DecoderModelVersion::kLlama2:
    case example::DecoderModelVersion::kQwen2_5:
    case example::DecoderModelVersion::kCodegen:
      formatted_prompt.append(prompt);
      break;
    case example::DecoderModelVersion::kLlama3:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|start_header_id|>system<|end_header_id|>\n\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|eot_id|>");
      }
      formatted_prompt.append("<|start_header_id|>user<|end_header_id|>\n\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append(
          "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
      break;
    case example::DecoderModelVersion::kGemma:
    case example::DecoderModelVersion::kGemma3:
      formatted_prompt.append("<start_of_turn>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<end_of_turn>\n");
      formatted_prompt.append("<start_of_turn>model\n");
      if (!system_prompt.empty()) {
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<end_of_turn>\n");
      }
      break;
    case example::DecoderModelVersion::kGemma2:
      formatted_prompt.append("<start_of_turn>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<end_of_turn>\n");
      formatted_prompt.append("<start_of_turn>model\n");
      break;
    case example::DecoderModelVersion::kGranite:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|start_of_role|>system<|end_of_role|>");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|end_of_text|>\n");
      }
      formatted_prompt.append("<|start_of_role|>user<|end_of_role|>");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|end_of_text|>\n");
      formatted_prompt.append("<|start_of_role|>assistant<|end_of_role|>");
      break;
    case example::DecoderModelVersion::kPhi4:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|system|>");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|end|>");
      }
      formatted_prompt.append("<|user|>");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|end|><|assistant|>");
      break;
    case example::DecoderModelVersion::kQwen3:
      formatted_prompt.append("<|im_start|>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|im_end|>\n");
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|im_end|>\n");
      }
      formatted_prompt.append("<|im_start|>assistant");
      break;
    case example::DecoderModelVersion::kSmollm2_135m:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|im_end|>\n");
      }
      formatted_prompt.append("<|im_start|>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|im_end|>\n");
      formatted_prompt.append("<|im_start|>assistant\n\n");
      break;
    case example::DecoderModelVersion::kSmollm3:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("\n\n");
      }
      formatted_prompt.append("<|im_start|>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|im_end|>\n");
      formatted_prompt.append("<|im_start|>assistant\n");
      break;
    case example::DecoderModelVersion::kGlm:
      formatted_prompt.append("<|user|>\n");
      formatted_prompt.append(prompt);
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|system|>\n");
        formatted_prompt.append(system_prompt);
      }
      formatted_prompt.append("<|assistant|>\n");
      break;
    default:
      ET_CHECK_MSG(false, "unsupported decoder version");
      break;
  }
  return formatted_prompt;
}

template <typename T>
void run_pd_export(
    ModuleBundle module_bundle,
    const std::string& prompt_input,
    bool tokenized_prompt,
    std::unique_ptr<executorch::extension::Module> attention_sink_rope_module) {
  example::PDPrefillRunner<T> runner(
      std::move(module_bundle.module),
      FLAGS_decoder_model_version.c_str(),
      get_model_path_for_runner(),
      FLAGS_tokenizer_path.c_str(),
      module_bundle.pte_bytes,
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      nullptr,
      std::move(attention_sink_rope_module));

  const auto decoder_version = runner.get_decoder_model_version().get();
  const std::string formatted_prompt = tokenized_prompt
      ? prompt_input
      : get_formatted_prompt(prompt_input, FLAGS_system_prompt, decoder_version);
  ET_CHECK_MSG(
      runner.export_prefill_handoff(
          formatted_prompt,
          tokenized_prompt,
          FLAGS_seq_len,
          FLAGS_prefill_export_dir,
          FLAGS_kv_quant_attrs_path) == executorch::runtime::Error::Ok,
      "PD prefill export failed");
}

} // namespace

int main(int argc, char** argv) {
  std::vector<std::string> prompts = CollectPrompts(argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ET_CHECK_MSG(
      !FLAGS_prefill_export_dir.empty(),
      "--prefill_export_dir is required for qnn_llama_pd_runner");
  ET_CHECK_MSG(
      FLAGS_attention_sink_rope_path.empty(),
      "PD prefill export v1 does not support attention sink");
  ET_CHECK_MSG(
      FLAGS_eval_mode != 2,
      "PD prefill export v1 does not support lookahead decoding");
  ET_CHECK_MSG(
      gflags::GetCommandLineFlagInfoOrDie("prompt").is_default ||
          gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default,
      "Only provide prompt or tokenized_prompt, not both");

  const bool use_tokenized_prompt =
      !gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default;

  ET_CHECK_MSG(
      use_tokenized_prompt || prompts.size() == 1,
      "PD prefill export v1 only supports a single prompt");
  ET_CHECK_MSG(
      use_tokenized_prompt || !prompts.empty(),
      "Provide --prompt or --tokenized_prompt");
  ET_CHECK_MSG(
      !use_tokenized_prompt || FLAGS_system_prompt.empty(),
      "tokenized_prompt mode does not support system_prompt reformatting");

  ModuleBundle module_bundle = load_module_from_file_or_rebuild();
  ModuleMetaInfo module_meta = read_module_meta(module_bundle.module.get());
  std::unique_ptr<executorch::extension::Module> attention_sink_rope_module;

  const std::string prompt_input =
      use_tokenized_prompt ? FLAGS_tokenized_prompt : prompts.front();
  if (module_meta.kv_bitwidth == example::KvBitWidth::kWidth8) {
    run_pd_export<uint8_t>(
        std::move(module_bundle),
        prompt_input,
        use_tokenized_prompt,
        std::move(attention_sink_rope_module));
  } else if (module_meta.kv_bitwidth == example::KvBitWidth::kWidth16) {
    run_pd_export<uint16_t>(
        std::move(module_bundle),
        prompt_input,
        use_tokenized_prompt,
        std::move(attention_sink_rope_module));
  } else {
    ET_CHECK_MSG(
        false,
        "Unsupported kv bitwidth: %ld",
        static_cast<int64_t>(module_meta.kv_bitwidth));
  }
  return 0;
}
