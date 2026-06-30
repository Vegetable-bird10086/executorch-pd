/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * This tool can run Llama2 110M, Llama3.2 1B / 3B, Gemma 2B, Gemma2 2B, Gemma3
 * 1B, Granite3.3 2B, phi4-mini-instruct, Qwen2.5 0.5B / 1.5B, Qwen3 0.6B
 * / 1.7B, SmolLM2 135M, SmolLM3 3B with Qualcomm AI Engine Direct.
 *
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/pte_rebuilder.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

DEFINE_string(decoder_model_version, "llama2", "The decoder model to execute.");
DEFINE_string(
    model_path,
    "kv_llama_qnn.pte",
    "Model serialized in flatbuffer format.");
DEFINE_string(
    stripped_model_path,
    "",
    "Stripped model serialized in flatbuffer format. When provided with index_bin_path and a rebuild source, the original PTE is rebuilt in memory.");
DEFINE_string(
    index_bin_path,
    "",
    "Path to the binary strip index used to rebuild a stripped PTE in memory.");

DEFINE_string(
    qat_checkpoint_path,
    "",
    "Path to the QAT safetensors checkpoint used to rebuild a stripped PTE in memory.");
DEFINE_string(
    tmac_model_path,
    "",
    "Path to the T-MAC GGUF model used to rebuild stripped decoder blocks in memory.");
DEFINE_string(
    gguf_model_path,
    "",
    "Path to the llama.cpp GPTQ2_32 GGUF model used to rebuild stripped decoder blocks in memory.");
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
    attention_sink_rope_path,
    "",
    "[Attention Sink] The Attention Sink Rope Model is serialized using the flatbuffer format. If specified, seq_len can exceed the context length defined in the model.");
DEFINE_string(
    output_path,
    "outputs.txt",
    "Executorch inference data output path.");
DEFINE_string(
    performance_output_path,
    "inference_speed.txt",
    "Records inference speed. For CI purpose.");
DEFINE_string(
    dump_logits_path,
    "",
    "If path is provided, program will dump all logits generated. This option is for analysis purpose. It is not recommended for general usage as it will cause token rate drop and increase in memory usage.");
DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer stuff.");
DEFINE_string(
    prompt,
    "The answer to the ultimate question is",
    "User prompts for Llama. When multiple prompts are entered, a multi-turn conversation will be initiated. Note that this feature is currently for testing purposes only.");
DEFINE_string(
    tokenized_prompt,
    "",
    "This is an alternative of passing prompts. Users could provide this in a raw file, with tokens saved in uint64 format.");
DEFINE_string(
    system_prompt,
    "",
    "Tells the model what kind of assistant it should be. For example, You are a helpful AI assistant for travel tips and recommendations. Default is None");
DEFINE_string(
    wikitext_path,
    "",
    "Path to a local WikiText text file. When provided, the runner computes WikiText perplexity instead of generating text.");
DEFINE_int32(
    wikitext_max_tokens,
    0,
    "Maximum number of WikiText target tokens to score. Non-positive values mean score all available tokens.");
DEFINE_double(
    temperature,
    0.0f,
    "Temperature; Default is 0.0f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");
DEFINE_int32(
    seq_len,
    128,
    "Total number of tokens to generate (prompt + output).");
DEFINE_int32(
    eval_mode,
    1,
    "0: TokenGenerator(kv) / 1: HybridMode (prefill+kv) / 2: Lookahead Decoding");
DEFINE_bool(
    shared_buffer,
    false,
    "Specifies to use shared buffers for zero-copy use case between the application and device/co-processor associated with the backend.");
DEFINE_int32(num_iters, 1, "total num of iterations to run.");
DEFINE_int32(
    ngram,
    0,
    "[Lookahead Decoding] Represents the size of the n-grams used in the lookahead process.");
DEFINE_int32(
    window,
    0,
    "[Lookahead Decoding] Determines how many future tokens the algorithm attempts to predict in each step.");
DEFINE_int32(
    gcap,
    0,
    "[Lookahead Decoding] Represents the maximum number of speculations or candidate n-grams that the algorithm considers in each step for verification. It balances the trade-off between computation efficiency and exploring more possibilities.");

namespace {

struct ModuleBundle {
  std::unique_ptr<executorch::extension::Module> module;
  std::shared_ptr<std::vector<uint8_t>> rebuilt_pte;
  double rebuild_time_ms{0.0};
  size_t rebuilt_records{0};
  bool rebuilt_from_stripped{false};
  bool specialized_fast_path_used{false};
};

struct ModuleMetaInfo {
  example::KvBitWidth kv_bitwidth{example::KvBitWidth::kWidth8};
  float logits_scale{1.0f};
  int32_t logits_zero_point{0};
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
  const bool has_tmac_gguf = !FLAGS_tmac_model_path.empty();
  const bool has_gguf = !FLAGS_gguf_model_path.empty();
  const int rebuild_source_count =
      static_cast<int>(has_checkpoint) +
      static_cast<int>(has_tmac_gguf) +
      static_cast<int>(has_gguf);
  ET_CHECK_MSG(
      rebuild_source_count <= 1,
      "Provide only one of qat_checkpoint_path, tmac_model_path, or gguf_model_path");
  const bool has_rebuild_source =
      has_checkpoint || has_tmac_gguf || has_gguf;
  ET_CHECK_MSG(
      has_stripped == has_index && has_index == has_rebuild_source,
      "Provide stripped_model_path, index_bin_path, and one rebuild source together");
  return has_stripped;
}

ModuleBundle load_module_from_file_or_rebuild() {
  ModuleBundle bundle;
  if (!should_rebuild_from_stripped()) {
    bundle.module = std::make_unique<executorch::extension::Module>(
        FLAGS_model_path.c_str(),
        executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
    return bundle;
  }

  const std::vector<uint8_t> stripped_pte = read_binary_file(FLAGS_stripped_model_path);
  const std::vector<uint8_t> index_bytes = read_binary_file(FLAGS_index_bin_path);
  example::PteRebuildResult rebuild_result;
  if (!FLAGS_gguf_model_path.empty()) {
    const std::vector<uint8_t> gguf_bytes = read_binary_file(FLAGS_gguf_model_path);
    rebuild_result = example::rebuild_pte_from_stripped_gguf(
        stripped_pte, index_bytes, gguf_bytes);
  } else if (!FLAGS_tmac_model_path.empty()) {
    const std::vector<uint8_t> gguf_bytes = read_binary_file(FLAGS_tmac_model_path);
    rebuild_result = example::rebuild_pte_from_stripped_tmac_gguf(
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

  bundle.rebuilt_pte = rebuild_result.rebuilt_pte;
  bundle.rebuild_time_ms = rebuild_result.rebuild_time_ms;
  bundle.rebuilt_records = rebuild_result.rebuilt_records;
  bundle.specialized_fast_path_used = rebuild_result.specialized_fast_path_used;
  bundle.rebuilt_from_stripped = true;

  auto data_loader = std::make_unique<executorch::extension::BufferDataLoader>(
      bundle.rebuilt_pte->data(), bundle.rebuilt_pte->size());
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

  if (method_names->count("get_logits_scale") > 0) {
    meta.logits_scale =
        static_cast<float>(module->get("get_logits_scale").get().toDouble());
    ET_CHECK_MSG(
        method_names->count("get_logits_zero_point") > 0,
        "Quantized logits require get_logits_zero_point metadata");
    meta.logits_zero_point =
        module->get("get_logits_zero_point").get().toScalar().to<int64_t>();
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
        formatted_prompt.append(
            "<|start_header_id|>system<|end_header_id|>\n\n");
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
      ET_CHECK_MSG(false, "unsupported llama version");
      break;
  }
  return formatted_prompt;
}

} // namespace

template <typename T>
void start_runner(
    ModuleBundle module_bundle,
    ModuleMetaInfo module_meta,
    std::vector<std::string>& prompts,
    std::unique_ptr<executorch::extension::Module> attention_sink_rope_module) {
  bool use_tokenized_prompt =
      gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default ? false
                                                                         : true;
  example::Runner<T> runner(
      std::move(module_bundle.module),
      FLAGS_decoder_model_version.c_str(),
      get_model_path_for_runner(),
      FLAGS_tokenizer_path.c_str(),
      FLAGS_performance_output_path.c_str(),
      FLAGS_dump_logits_path.c_str(),
      FLAGS_temperature,
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      FLAGS_ngram,
      FLAGS_window,
      FLAGS_gcap,
      nullptr,
      std::move(attention_sink_rope_module));

  if (module_bundle.rebuilt_from_stripped) {
    ET_LOG(
        Info,
        "pte_rebuild_time_ms=%.3f rebuilt_records=%zu specialized_fast_path=%s",
        module_bundle.rebuild_time_ms,
        module_bundle.rebuilt_records,
        module_bundle.specialized_fast_path_used ? "true" : "false");
  }

  if (!FLAGS_wikitext_path.empty()) {
    double wiki_ppl = 0.0;
    const auto ppl_error = runner.evaluate_wikitext_ppl(
        FLAGS_wikitext_path,
        FLAGS_wikitext_max_tokens,
        module_meta.logits_scale,
        module_meta.logits_zero_point,
        &wiki_ppl);
    ET_CHECK_MSG(
        ppl_error == executorch::runtime::Error::Ok,
        "Failed to evaluate WikiText perplexity");
    std::ofstream fout(FLAGS_output_path.c_str());
    fout << "wiki_ppl=" << wiki_ppl << "\n";
    fout.close();
    ET_LOG(Info, "wiki_ppl=%f", wiki_ppl);
    return;
  }

  auto decoder_model_version = runner.get_decoder_model_version();
  std::vector<char> buf;
  buf.reserve(5 * FLAGS_seq_len);
  std::ofstream fout(FLAGS_output_path.c_str());
  auto callback = [&](const std::string& piece) {
    for (const char c : piece) {
      buf.push_back(c);
    }
  };
  executorch::extension::llm::GenerationConfig config{
      true,
      false,
      -1,
      false,
      FLAGS_seq_len,
      static_cast<float>(FLAGS_temperature),
      0,
      0};
  if (use_tokenized_prompt) {
    runner.generate_from_prompt_or_file(
        FLAGS_tokenized_prompt.c_str(), use_tokenized_prompt, config, callback);
  } else {
    for (int i = 0; i < FLAGS_num_iters; i++) {
      for (const auto& prompt : prompts) {
        std::string formatted_prompt =
            get_formatted_prompt(prompt, FLAGS_system_prompt, decoder_model_version.get());
        runner.generate_from_prompt_or_file(
            formatted_prompt.c_str(), use_tokenized_prompt, config, callback);
      }
    }
  }

  fout.write(buf.data(), buf.size());
  fout.close();
}

int main(int argc, char** argv) {
  std::vector<std::string> prompts = CollectPrompts(argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (!gflags::GetCommandLineFlagInfoOrDie("prompt").is_default &&
      !gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default) {
    ET_CHECK_MSG(false, "Only provide prompt or tokenized_input but not both.");
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("dump_logits_path").is_default &&
      FLAGS_eval_mode != 0) {
    ET_CHECK_MSG(
        false, "Only TokenGenerator(kv) mode is supported to dump all logits.");
  }
  if (!FLAGS_wikitext_path.empty()) {
    ET_CHECK_MSG(
        gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default,
        "tokenized_prompt is not supported in wikitext PPL mode");
  }

  ModuleBundle module_bundle = load_module_from_file_or_rebuild();
  std::unique_ptr<executorch::extension::Module> attention_sink_rope_module;
  if (!FLAGS_attention_sink_rope_path.empty()) {
    attention_sink_rope_module =
        std::make_unique<executorch::extension::Module>(
            FLAGS_attention_sink_rope_path.c_str(),
            executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
  }
  ModuleMetaInfo module_meta = read_module_meta(module_bundle.module.get());

  if (module_meta.kv_bitwidth == example::KvBitWidth::kWidth8) {
    start_runner<uint8_t>(
        std::move(module_bundle),
        module_meta,
        prompts,
        std::move(attention_sink_rope_module));
  } else if (module_meta.kv_bitwidth == example::KvBitWidth::kWidth16) {
    start_runner<uint16_t>(
        std::move(module_bundle),
        module_meta,
        prompts,
        std::move(attention_sink_rope_module));
  } else {
    ET_CHECK_MSG(
        false,
        "Unsupported kv bitwidth: %ld",
        static_cast<int64_t>(module_meta.kv_bitwidth));
  }

  return 0;
}
