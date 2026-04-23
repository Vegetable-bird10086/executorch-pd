#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/pd_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/pte_rebuilder.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
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
DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer path.");
DEFINE_string(
    output_path,
    "outputs.txt",
    "Output path. In WikiText PPL mode, writes wiki_ppl=<value>.");
DEFINE_string(
    performance_output_path,
    "inference_speed.txt",
    "Performance report output path used by WikiText PPL mode.");
DEFINE_string(prompt, "", "Prompt text to prefill and decode.");
DEFINE_string(
    tokenized_prompt,
    "",
    "Optional raw uint64 token file used instead of string prompt.");
DEFINE_string(system_prompt, "", "Optional system prompt.");
DEFINE_string(
    wikitext_path,
    "",
    "Path to a local WikiText text file. When provided, the runner computes WikiText perplexity and skips PD handoff/decode.");
DEFINE_int32(
    wikitext_max_tokens,
    0,
    "Maximum number of WikiText target tokens to score. Non-positive values mean score all available tokens.");
DEFINE_string(
    prefill_export_dir,
    "",
    "Optional output directory for PD handoff export. When omitted, a temporary directory is created.");
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

DEFINE_bool(
    prefill_only,
    false,
    "Only export the PD handoff and skip llama.cpp decode.");
DEFINE_string(
    llama_pd_cli_path,
    "",
    "Path to the llama-pd-cli executable used for decode. Required unless --prefill_only=true.");
DEFINE_string(
    decode_gguf_path,
    "",
    "Path to the decode-side GGUF model. When omitted, --gguf_model_path is reused.");
DEFINE_int32(
    decode_n_predict,
    128,
    "Number of tokens to generate after importing the PD handoff.");
DEFINE_int32(
    decode_threads,
    4,
    "Decode-side thread count passed to llama-pd-cli.");
DEFINE_int32(
    decode_ctx,
    2048,
    "Decode-side llama.cpp context size.");
DEFINE_double(
    decode_temp,
    0.0f,
    "Decode-side sampling temperature.");
DEFINE_int32(
    decode_ngl,
    0,
    "Decode-side number of offloaded layers (-ngl).");
DEFINE_bool(
    decode_import_ro,
    false,
    "Only validate/import the PD handoff in llama.cpp without continuing decode.");
DEFINE_bool(
    decode_roundtrip_check,
    false,
    "Ask llama-pd-cli to re-serialize the imported KV sequence and compare it byte-for-byte with the imported PD blob.");
DEFINE_bool(
    decode_native_compare,
    false,
    "Ask llama-pd-cli to compare imported-KV resume logits against a native GGUF prefill resume on the same prompt tokens.");

namespace fs = std::filesystem;

namespace {

struct ModuleBundle {
  std::unique_ptr<executorch::extension::Module> module;
  std::shared_ptr<std::vector<uint8_t>> pte_bytes;
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
    bundle.module =
        std::make_unique<executorch::extension::Module>(std::move(data_loader));
    return bundle;
  }

  const std::vector<uint8_t> stripped_pte =
      read_binary_file(FLAGS_stripped_model_path);
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
  bundle.module =
      std::make_unique<executorch::extension::Module>(std::move(data_loader));
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

std::string create_temp_handoff_dir() {
  std::string pattern = "/tmp/qnn_pd_handoff_XXXXXX";
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back('\0');
  char* created = mkdtemp(writable.data());
  ET_CHECK_MSG(created != nullptr, "mkdtemp failed: %s", std::strerror(errno));
  return std::string(created);
}

std::string resolve_decode_gguf_path() {
  if (!FLAGS_decode_gguf_path.empty()) {
    return FLAGS_decode_gguf_path;
  }
  return FLAGS_gguf_model_path;
}

int run_decode_process(const std::string& handoff_dir) {
  ET_CHECK_MSG(
      !FLAGS_llama_pd_cli_path.empty(),
      "--llama_pd_cli_path is required unless --prefill_only=true");
  const std::string decode_gguf_path = resolve_decode_gguf_path();
  ET_CHECK_MSG(
      !decode_gguf_path.empty(),
      "Provide --decode_gguf_path or --gguf_model_path for decode");

  std::vector<std::string> args = {
      FLAGS_llama_pd_cli_path,
      "--pd-import",
      handoff_dir,
      "-m",
      decode_gguf_path,
      "-n",
      std::to_string(FLAGS_decode_n_predict),
      "-c",
      std::to_string(FLAGS_decode_ctx),
      "-ngl",
      std::to_string(FLAGS_decode_ngl),
      "--temp",
      std::to_string(FLAGS_decode_temp),
  };
  if (FLAGS_decode_threads > 0) {
    args.push_back("-t");
    args.push_back(std::to_string(FLAGS_decode_threads));
  }
  if (FLAGS_decode_import_ro) {
    args.push_back("--pd-import-ro");
  }
  if (FLAGS_decode_roundtrip_check) {
    args.push_back("--pd-roundtrip-check");
  }
  if (FLAGS_decode_native_compare) {
    args.push_back("--pd-native-compare");
  }

  ET_LOG(Info, "Launching decode via llama-pd-cli");
  for (const auto& arg : args) {
    ET_LOG(Info, "  arg: %s", arg.c_str());
  }

  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);

  const pid_t pid = fork();
  ET_CHECK_MSG(pid >= 0, "fork failed: %s", std::strerror(errno));
  if (pid == 0) {
    execvp(argv[0], argv.data());
    std::fprintf(
        stderr,
        "execvp failed for %s: %s\n",
        FLAGS_llama_pd_cli_path.c_str(),
        std::strerror(errno));
    _exit(127);
  }

  int status = 0;
  ET_CHECK_MSG(waitpid(pid, &status, 0) == pid, "waitpid failed: %s", std::strerror(errno));
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    return 128 + WTERMSIG(status);
  }
  return 1;
}

template <typename T>
void run_wikitext_ppl(
    ModuleBundle module_bundle,
    const ModuleMetaInfo& module_meta,
    std::unique_ptr<executorch::extension::Module> attention_sink_rope_module) {
  example::Runner<T> runner(
      std::move(module_bundle.module),
      FLAGS_decoder_model_version.c_str(),
      get_model_path_for_runner(),
      FLAGS_tokenizer_path.c_str(),
      FLAGS_performance_output_path.c_str(),
      "",
      0.0f,
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      0,
      0,
      0,
      nullptr,
      std::move(attention_sink_rope_module));

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
  ET_LOG(
      Info,
      "wiki_ppl=%f (ExecuTorch-side prompt-logit evaluation; PD handoff/decode skipped)",
      wiki_ppl);
}

template <typename T>
void run_pd_e2e(
    ModuleBundle module_bundle,
    const std::string& prompt_input,
    bool tokenized_prompt,
    const std::string& handoff_dir,
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
          handoff_dir,
          FLAGS_kv_quant_attrs_path) == executorch::runtime::Error::Ok,
      "PD prefill export failed");
}

} // namespace

int main(int argc, char** argv) {
  std::vector<std::string> prompts = CollectPrompts(argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ET_CHECK_MSG(
      FLAGS_attention_sink_rope_path.empty(),
      "PD prefill export does not support attention sink in v1");
  ET_CHECK_MSG(
      FLAGS_eval_mode != 2,
      "PD prefill export does not support lookahead decoding in v1");
  ET_CHECK_MSG(
      gflags::GetCommandLineFlagInfoOrDie("prompt").is_default ||
          gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default,
      "Only provide prompt or tokenized_prompt, not both");

  const bool use_tokenized_prompt =
      !gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default;
  if (!FLAGS_wikitext_path.empty()) {
    ET_CHECK_MSG(
        gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default,
        "tokenized_prompt is not supported in wikitext PPL mode");
  } else {
    ET_CHECK_MSG(
        use_tokenized_prompt || prompts.size() == 1,
        "PD flow only supports a single prompt");
    ET_CHECK_MSG(
        use_tokenized_prompt || !prompts.empty(),
        "Provide --prompt or --tokenized_prompt");
    ET_CHECK_MSG(
        !use_tokenized_prompt || FLAGS_system_prompt.empty(),
        "tokenized_prompt mode does not support system_prompt reformatting");
  }

  ModuleBundle module_bundle = load_module_from_file_or_rebuild();
  ModuleMetaInfo module_meta = read_module_meta(module_bundle.module.get());
  std::unique_ptr<executorch::extension::Module> attention_sink_rope_module;

  if (!FLAGS_wikitext_path.empty()) {
    if (module_meta.kv_bitwidth == example::KvBitWidth::kWidth8) {
      run_wikitext_ppl<uint8_t>(
          std::move(module_bundle),
          module_meta,
          std::move(attention_sink_rope_module));
    } else if (module_meta.kv_bitwidth == example::KvBitWidth::kWidth16) {
      run_wikitext_ppl<uint16_t>(
          std::move(module_bundle),
          module_meta,
          std::move(attention_sink_rope_module));
    } else {
      ET_CHECK_MSG(
          false,
          "Unsupported kv bitwidth: %ld",
          static_cast<int64_t>(module_meta.kv_bitwidth));
    }
    return 0;
  }

  const std::string handoff_dir = FLAGS_prefill_export_dir.empty()
      ? create_temp_handoff_dir()
      : FLAGS_prefill_export_dir;
  fs::create_directories(handoff_dir);
  ET_LOG(Info, "Using PD handoff directory: %s", handoff_dir.c_str());

  const std::string prompt_input =
      use_tokenized_prompt ? FLAGS_tokenized_prompt : prompts.front();
  if (module_meta.kv_bitwidth == example::KvBitWidth::kWidth8) {
    run_pd_e2e<uint8_t>(
        std::move(module_bundle),
        prompt_input,
        use_tokenized_prompt,
        handoff_dir,
        std::move(attention_sink_rope_module));
  } else if (module_meta.kv_bitwidth == example::KvBitWidth::kWidth16) {
    run_pd_e2e<uint16_t>(
        std::move(module_bundle),
        prompt_input,
        use_tokenized_prompt,
        handoff_dir,
        std::move(attention_sink_rope_module));
  } else {
    ET_CHECK_MSG(
        false,
        "Unsupported kv bitwidth: %ld",
        static_cast<int64_t>(module_meta.kv_bitwidth));
  }

  if (FLAGS_prefill_only) {
    ET_LOG(Info, "Prefill completed; handoff is ready at %s", handoff_dir.c_str());
    return 0;
  }

  const int decode_rc = run_decode_process(handoff_dir);
  ET_CHECK_MSG(decode_rc == 0, "llama-pd-cli exited with code %d", decode_rc);
  return 0;
}
