#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/pd_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/pte_rebuilder.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
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
    tmac_model_path,
    "",
    "Path to the T-MAC GGUF model used to rebuild stripped decoder blocks in memory.");
DEFINE_string(
    gguf_model_path,
    "",
    "Path to the llama.cpp GPTQ2_32 GGUF model used to rebuild stripped decoder blocks in memory.");
DEFINE_string(
    prefill_shard_manifest_path,
    "",
    "Path to the shard manifest JSON. When provided, prefill_forward is executed by loading the listed prefill shard PTEs sequentially while decode still uses llama-pd-cli.");
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
DEFINE_bool(
    prefill_shard_pipeline,
    false,
    "Preload stripped shard inputs and rebuild one shard ahead on a CPU worker while QNN executes the current shard.");
DEFINE_bool(
    prefill_shard_stage_major,
    false,
    "For static Qwen3 shards, execute every AR block through one shard before advancing to the next shard.");
DEFINE_string(
    llama_pd_cli_path,
    "",
    "Path to the llama-pd-cli executable used for decode. Required unless --prefill_only=true.");
DEFINE_string(
    decode_gguf_path,
    "",
    "Path to the decode-side GGUF model. When omitted, --gguf_model_path is reused, then --tmac_model_path.");
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
DEFINE_bool(
    decode_native_first_token,
    false,
    "Ask llama-pd-cli to choose the first continuation token from a native GGUF prompt prefill instead of the QNN prefill logits.");

namespace fs = std::filesystem;

namespace {

using SteadyClock = std::chrono::steady_clock;

double elapsed_ms(
    SteadyClock::time_point start,
    SteadyClock::time_point end = SteadyClock::now()) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

struct ProcessMemorySnapshot {
  uint64_t rss_bytes{0};
  uint64_t hwm_bytes{0};
};

uint64_t read_proc_status_bytes(const char* field) {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind(field, 0) != 0) {
      continue;
    }
    std::istringstream value(line.substr(std::strlen(field)));
    uint64_t kib = 0;
    value >> kib;
    return kib * 1024;
  }
  return 0;
}

ProcessMemorySnapshot process_memory_snapshot() {
  return {
      read_proc_status_bytes("VmRSS:"),
      read_proc_status_bytes("VmHWM:"),
  };
}

double bytes_to_mib(uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

struct PdE2ERuntimeStats {
  int32_t prompt_tokens{0};
  double runner_setup_ms{0.0};
  double qnn_export_total_ms{0.0};
  example::PDPrefillRunner<uint16_t>::RuntimeStats prefill{};
  std::vector<example::DecoderRunner::PrefillShardRuntimeStats> shard_stats;
  ProcessMemorySnapshot before_runner;
  ProcessMemorySnapshot after_runner;
  ProcessMemorySnapshot after_export;
};

struct DecodeProcessResult {
  int exit_code{1};
  double wall_ms{0.0};
  ProcessMemorySnapshot before;
  ProcessMemorySnapshot after;
};

void log_pd_e2e_runtime_summary(
    const PdE2ERuntimeStats& prefill,
    const DecodeProcessResult& decode,
    double total_ms) {
  ET_LOG(
      Info,
      "PD E2E runtime summary: prompt_tokens=%d shards=%zu decode_n_predict=%d",
      prefill.prompt_tokens,
      prefill.shard_stats.size(),
      FLAGS_decode_n_predict);
  double shard_preload_total_ms = 0.0;
  double pipeline_wait_total_ms = 0.0;
  for (const auto& shard : prefill.shard_stats) {
    shard_preload_total_ms += shard.preload_ms;
    pipeline_wait_total_ms += shard.pipeline_wait_ms;
  }
  ET_LOG(
      Info,
      "PD E2E shard setup: preload_ms=%.3f pipeline_wait_ms=%.3f pipeline_enabled=%d stage_major_enabled=%d",
      shard_preload_total_ms,
      pipeline_wait_total_ms,
      static_cast<int>(FLAGS_prefill_shard_pipeline),
      static_cast<int>(FLAGS_prefill_shard_stage_major));
  ET_LOG(
      Info,
      "PD E2E timing: runner_setup_ms=%.3f tokenize_ms=%.3f qnn_prefill_ms=%.3f "
      "handoff_ms=%.3f qnn_export_total_ms=%.3f decode_process_ms=%.3f e2e_total_ms=%.3f",
      prefill.runner_setup_ms,
      prefill.prefill.tokenize_ms,
      prefill.prefill.prefill_ms,
      prefill.prefill.handoff_total_ms,
      prefill.qnn_export_total_ms,
      decode.wall_ms,
      total_ms);
  ET_LOG(
      Info,
      "PD E2E handoff detail: kv_layout_ms=%.3f kv_write_ms=%.3f fingerprint_ms=%.3f",
      prefill.prefill.kv_layout_ms,
      prefill.prefill.kv_write_ms,
      prefill.prefill.fingerprint_ms);
  ET_LOG(
      Info,
      "PD E2E parent memory MiB: before_runner_rss=%.2f after_runner_rss=%.2f "
      "after_export_rss=%.2f after_decode_rss=%.2f hwm=%.2f",
      bytes_to_mib(prefill.before_runner.rss_bytes),
      bytes_to_mib(prefill.after_runner.rss_bytes),
      bytes_to_mib(prefill.after_export.rss_bytes),
      bytes_to_mib(decode.after.rss_bytes),
      bytes_to_mib(std::max(prefill.after_export.hwm_bytes, decode.after.hwm_bytes)));
  for (size_t i = 0; i < prefill.shard_stats.size(); ++i) {
    const auto& shard = prefill.shard_stats[i];
    ET_LOG(
        Info,
        "PD E2E shard summary: index=%zu layers=[%zu,%zu) runs=%zu "
        "preload_ms=%.3f materialize_ms=%.3f rebuild_ms=%.3f "
        "pipeline_wait_ms=%.3f execute_ms=%.3f total_ms=%.3f",
        i,
        shard.layer_offset,
        shard.layer_offset + shard.layer_count,
        shard.execution_count,
        shard.preload_ms,
        shard.materialize_ms,
        shard.rebuild_ms,
        shard.pipeline_wait_ms,
        shard.execute_ms,
        shard.total_ms);

    const double lifecycle_known_ms =
        shard.materialize_ms + shard.pipeline_wait_ms +
        shard.qnn_load_method_ms + shard.input_binding_ms +
        shard.output_binding_ms + shard.execute_ms +
        shard.output_copy_ms + shard.release_ms;
    const double other_stage_ms =
        std::max(0.0, shard.total_ms - lifecycle_known_ms);
    ET_LOG(
        Info,
        "PD E2E shard lifecycle: index=%zu qnn_load_method_ms=%.3f "
        "input_binding_ms=%.3f output_binding_ms=%.3f "
        "output_copy_ms=%.3f release_ms=%.3f other_ms=%.3f",
        i,
        shard.qnn_load_method_ms,
        shard.input_binding_ms,
        shard.output_binding_ms,
        shard.output_copy_ms,
        shard.release_ms,
        other_stage_ms);

    const uint64_t peak_sample_rss = std::max(
        std::max(shard.rss_after_materialize_bytes, shard.rss_after_load_bytes),
        std::max(shard.rss_after_execute_bytes, shard.rss_after_release_bytes));
    const uint64_t peak_hwm = std::max(
        std::max(shard.hwm_after_materialize_bytes, shard.hwm_after_load_bytes),
        std::max(shard.hwm_after_execute_bytes, shard.hwm_after_release_bytes));
    const uint64_t baseline_rss = prefill.after_runner.rss_bytes;
    const uint64_t baseline_hwm = prefill.after_runner.hwm_bytes;
    ET_LOG(
        Info,
        "PD E2E shard memory MiB: index=%zu baseline_rss=%.2f "
        "peak_sample_rss=%.2f peak_sample_delta=%.2f "
        "hwm_before=%.2f hwm_peak=%.2f hwm_delta=%.2f",
        i,
        bytes_to_mib(baseline_rss),
        bytes_to_mib(peak_sample_rss),
        bytes_to_mib(
            peak_sample_rss > baseline_rss ? peak_sample_rss - baseline_rss : 0),
        bytes_to_mib(shard.hwm_before_bytes),
        bytes_to_mib(peak_hwm),
        bytes_to_mib(
            peak_hwm > baseline_hwm ? peak_hwm - baseline_hwm : 0));
    ET_LOG(
        Info,
        "PD E2E shard RSS MiB: index=%zu before=%.2f materialize=%.2f "
        "load=%.2f execute=%.2f release=%.2f",
        i,
        bytes_to_mib(shard.rss_before_bytes),
        bytes_to_mib(shard.rss_after_materialize_bytes),
        bytes_to_mib(shard.rss_after_load_bytes),
        bytes_to_mib(shard.rss_after_execute_bytes),
        bytes_to_mib(shard.rss_after_release_bytes));
  }
}

struct ModuleBundle {
  std::unique_ptr<executorch::extension::Module> module;
  std::shared_ptr<std::vector<uint8_t>> pte_bytes;
  size_t materialized_weight_bytes{0};
  example::PteSplitMaterializationStats split_stats;
};

struct ModuleMetaInfo {
  example::KvBitWidth kv_bitwidth{example::KvBitWidth::kWidth8};
  float logits_scale{1.0f};
  int32_t logits_zero_point{0};
};

struct PrefillShardFiles {
  std::vector<std::string> pte_paths;
  std::vector<std::string> index_bin_paths;
  bool qwen3_static_plan{false};
  int32_t static_aux_size{64};
  int32_t static_hidden_size{2048};
  int32_t context_len{0};
  int32_t prefill_ar_len{0};
  int32_t token_generator_ar_len{1};
  int32_t vocab_size{0};
  int32_t kv_bitwidth{8};
  int64_t num_layers{0};
  int64_t num_heads{0};
  int64_t head_dim{0};
  bool use_int64_token{false};
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
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  ET_CHECK_MSG(input.is_open(), "Unable to read file: %s", path.c_str());
  const std::streampos end = input.tellg();
  ET_CHECK_MSG(end >= 0, "Unable to determine file size: %s", path.c_str());
  const size_t size_bytes = static_cast<size_t>(end);
  input.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(size_bytes);
  if (size_bytes != 0) {
    input.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(size_bytes));
    ET_CHECK_MSG(
        input.good() || input.eof(),
        "Unable to read file: %s",
        path.c_str());
    ET_CHECK_MSG(
        static_cast<size_t>(input.gcount()) == size_bytes,
        "Short read from file: %s",
        path.c_str());
  }
  return bytes;
}

std::string read_text_file(const std::string& path) {
  std::ifstream input(path);
  ET_CHECK_MSG(input.is_open(), "Unable to read file: %s", path.c_str());
  std::stringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::string resolve_manifest_path(
    const fs::path& manifest_dir,
    const std::string& path) {
  if (path.empty()) {
    return path;
  }
  fs::path p(path);
  if (p.is_absolute()) {
    return p.string();
  }
  return (manifest_dir / p).lexically_normal().string();
}

std::vector<std::string> resolve_manifest_paths(
    const fs::path& manifest_dir,
    std::vector<std::string> paths) {
  for (auto& path : paths) {
    path = resolve_manifest_path(manifest_dir, path);
  }
  return paths;
}

std::vector<std::string> read_string_array_field(
    const std::string& manifest,
    size_t graph_pos,
    const std::string& field_name) {
  const size_t field_pos = manifest.find("\"" + field_name + "\"", graph_pos);
  if (field_pos == std::string::npos) {
    return {};
  }
  const size_t array_start = manifest.find('[', field_pos);
  const size_t array_end = manifest.find(']', array_start);
  ET_CHECK_MSG(
      array_start != std::string::npos && array_end != std::string::npos,
      "Invalid %s array in shard manifest",
      field_name.c_str());
  const std::string array_body =
      manifest.substr(array_start, array_end - array_start + 1);
  std::regex path_regex("\"([^\"]+)\"");
  std::sregex_iterator begin(array_body.begin(), array_body.end(), path_regex);
  std::sregex_iterator end;
  std::vector<std::string> paths;
  for (auto it = begin; it != end; ++it) {
    paths.push_back((*it)[1].str());
  }
  return paths;
}

std::string read_string_field(
    const std::string& manifest,
    size_t start_pos,
    const std::string& field_name) {
  const size_t field_pos = manifest.find("\"" + field_name + "\"", start_pos);
  if (field_pos == std::string::npos) {
    return "";
  }
  const size_t colon_pos = manifest.find(':', field_pos);
  const size_t value_start = manifest.find('"', colon_pos);
  const size_t value_end = manifest.find('"', value_start + 1);
  ET_CHECK_MSG(
      colon_pos != std::string::npos && value_start != std::string::npos &&
          value_end != std::string::npos,
      "Invalid %s string in shard manifest",
      field_name.c_str());
  return manifest.substr(value_start + 1, value_end - value_start - 1);
}

int32_t read_int_field_or(
    const std::string& manifest,
    size_t start_pos,
    const std::string& field_name,
    int32_t default_value) {
  const size_t field_pos = manifest.find("\"" + field_name + "\"", start_pos);
  if (field_pos == std::string::npos) {
    return default_value;
  }
  const size_t colon_pos = manifest.find(':', field_pos);
  ET_CHECK_MSG(colon_pos != std::string::npos, "Invalid %s integer in shard manifest", field_name.c_str());
  size_t value_start = manifest.find_first_of("-0123456789", colon_pos + 1);
  ET_CHECK_MSG(value_start != std::string::npos, "Invalid %s integer in shard manifest", field_name.c_str());
  size_t value_end = value_start;
  while (value_end < manifest.size() &&
         (manifest[value_end] == '-' ||
          (manifest[value_end] >= '0' && manifest[value_end] <= '9'))) {
    ++value_end;
  }
  return static_cast<int32_t>(std::stol(manifest.substr(value_start, value_end - value_start)));
}

bool read_bool_field_or(
    const std::string& manifest,
    size_t start_pos,
    const std::string& field_name,
    bool default_value) {
  const size_t field_pos = manifest.find("\"" + field_name + "\"", start_pos);
  if (field_pos == std::string::npos) {
    return default_value;
  }
  const size_t colon_pos = manifest.find(':', field_pos);
  ET_CHECK_MSG(colon_pos != std::string::npos, "Invalid %s bool in shard manifest", field_name.c_str());
  const size_t value_start = manifest.find_first_not_of(" \t\r\n", colon_pos + 1);
  ET_CHECK_MSG(value_start != std::string::npos, "Invalid %s bool in shard manifest", field_name.c_str());
  if (manifest.compare(value_start, 4, "true") == 0) {
    return true;
  }
  if (manifest.compare(value_start, 5, "false") == 0) {
    return false;
  }
  ET_CHECK_MSG(false, "Invalid %s bool in shard manifest", field_name.c_str());
  return default_value;
}

PrefillShardFiles read_prefill_shard_files(const std::string& manifest_path) {
  PrefillShardFiles files;
  if (manifest_path.empty()) {
    return files;
  }
  const std::string manifest = read_text_file(manifest_path);
  const fs::path manifest_dir = fs::absolute(fs::path(manifest_path)).parent_path();
  const size_t graph_pos = manifest.find("\"prefill_forward\"");
  ET_CHECK_MSG(
      graph_pos != std::string::npos,
      "prefill_forward graph is missing from shard manifest: %s",
      manifest_path.c_str());

  files.pte_paths = resolve_manifest_paths(
      manifest_dir,
      read_string_array_field(manifest, graph_pos, "stripped_pte_paths"));
  files.index_bin_paths = resolve_manifest_paths(
      manifest_dir,
      read_string_array_field(manifest, graph_pos, "index_bin_paths"));
  const std::string plan_type = read_string_field(manifest, graph_pos, "prefill_plan_type");
  files.qwen3_static_plan = plan_type == "qwen3_4x7_static";
  const size_t metadata_pos = manifest.find("\"prefill_metadata\"");
  if (metadata_pos != std::string::npos) {
    files.static_aux_size = read_int_field_or(manifest, metadata_pos, "aux_size", files.static_aux_size);
    files.static_hidden_size = read_int_field_or(manifest, metadata_pos, "hidden_size", files.static_hidden_size);
    files.context_len = read_int_field_or(manifest, metadata_pos, "context_len", files.context_len);
    files.prefill_ar_len = read_int_field_or(manifest, metadata_pos, "prefill_ar_len", files.prefill_ar_len);
    files.token_generator_ar_len = read_int_field_or(
        manifest, metadata_pos, "token_generator_ar_len", files.token_generator_ar_len);
    files.vocab_size = read_int_field_or(manifest, metadata_pos, "vocab_size", files.vocab_size);
    files.kv_bitwidth = read_int_field_or(manifest, metadata_pos, "kv_bitwidth", files.kv_bitwidth);
    files.num_layers = read_int_field_or(manifest, metadata_pos, "num_layers", files.num_layers);
    files.num_heads = read_int_field_or(manifest, metadata_pos, "num_heads", files.num_heads);
    files.head_dim = read_int_field_or(manifest, metadata_pos, "head_dim", files.head_dim);
    files.use_int64_token = read_bool_field_or(
        manifest, metadata_pos, "use_int64_token", files.use_int64_token);
  }
  if (files.qwen3_static_plan) {
    ET_LOG(
        Info,
        "prefill shard static plan: %s layers=%lld ctx=%d ar=%d heads=%lld head_dim=%lld vocab=%d aux_size=%d hidden_size=%d",
        plan_type.c_str(),
        static_cast<long long>(files.num_layers),
        files.context_len,
        files.prefill_ar_len,
        static_cast<long long>(files.num_heads),
        static_cast<long long>(files.head_dim),
        files.vocab_size,
        files.static_aux_size,
        files.static_hidden_size);
  }
  if (files.pte_paths.empty()) {
    files.pte_paths = resolve_manifest_paths(
        manifest_dir, read_string_array_field(manifest, graph_pos, "pte_paths"));
  }
  ET_CHECK_MSG(
      !files.pte_paths.empty(),
      "No prefill shard paths found in shard manifest: %s",
      manifest_path.c_str());
  if (!files.index_bin_paths.empty()) {
    ET_CHECK_MSG(
        files.index_bin_paths.size() == files.pte_paths.size(),
        "prefill shard stripped_pte_paths/index_bin_paths size mismatch: %zu vs %zu",
        files.pte_paths.size(),
        files.index_bin_paths.size());
  }
  for (size_t i = 0; i < files.pte_paths.size(); ++i) {
    ET_LOG(Info, "prefill shard manifest path: %s", files.pte_paths[i].c_str());
    if (!files.index_bin_paths.empty()) {
      ET_LOG(Info, "prefill shard index path: %s", files.index_bin_paths[i].c_str());
    }
  }
  return files;
}

std::vector<std::string> read_prefill_shard_paths(const std::string& manifest_path) {
  return read_prefill_shard_files(manifest_path).pte_paths;
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
  bundle.split_stats = example::analyze_split_materialization(index_bytes);
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
  bundle.pte_bytes = rebuild_result.rebuilt_pte;
  bundle.materialized_weight_bytes = rebuild_result.materialized_weight_bytes;
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
  if (!FLAGS_gguf_model_path.empty()) {
    return FLAGS_gguf_model_path;
  }
  return FLAGS_tmac_model_path;
}

example::DecoderRunner::PrefillShardRebuildConfig make_prefill_shard_rebuild_config(
    const PrefillShardFiles& files) {
  example::DecoderRunner::PrefillShardRebuildConfig config;
  if (files.index_bin_paths.empty()) {
    return config;
  }

  const bool has_checkpoint = !FLAGS_qat_checkpoint_path.empty();
  const bool has_tmac_gguf = !FLAGS_tmac_model_path.empty();
  const bool has_gguf = !FLAGS_gguf_model_path.empty();
  const int rebuild_source_count = static_cast<int>(has_checkpoint) +
      static_cast<int>(has_tmac_gguf) + static_cast<int>(has_gguf);
  ET_CHECK_MSG(
      rebuild_source_count == 1,
      "Prefill stripped shards require exactly one rebuild source: qat_checkpoint_path, tmac_model_path, or gguf_model_path");

  if (has_gguf) {
    config.source_kind =
        example::DecoderRunner::PrefillShardRebuildConfig::SourceKind::Gguf;
    config.source_bytes =
        std::make_shared<std::vector<uint8_t>>(read_binary_file(FLAGS_gguf_model_path));
  } else if (has_tmac_gguf) {
    config.source_kind =
        example::DecoderRunner::PrefillShardRebuildConfig::SourceKind::TmacGguf;
    config.source_bytes =
        std::make_shared<std::vector<uint8_t>>(read_binary_file(FLAGS_tmac_model_path));
  } else {
    config.source_kind =
        example::DecoderRunner::PrefillShardRebuildConfig::SourceKind::QatCheckpoint;
    config.source_bytes = std::make_shared<std::vector<uint8_t>>(
        read_binary_file(FLAGS_qat_checkpoint_path));
  }
  config.bits_hint = FLAGS_qat_bits_hint;
  config.group_size = FLAGS_qat_group_size;
  config.qweight_mode = FLAGS_qat_qweight_mode;
  config.stage_major_execution = FLAGS_prefill_shard_stage_major;
  config.pipeline_rebuild = FLAGS_prefill_shard_pipeline;
  ET_LOG(
      Info,
      "PD shard rebuild source: kind=%d size_bytes=%zu capacity_bytes=%zu",
      static_cast<int>(config.source_kind),
      config.source_bytes->size(),
      config.source_bytes->capacity());
  return config;
}

DecodeProcessResult run_decode_process(const std::string& handoff_dir) {
  ET_CHECK_MSG(
      !FLAGS_llama_pd_cli_path.empty(),
      "--llama_pd_cli_path is required unless --prefill_only=true");
  const std::string decode_gguf_path = resolve_decode_gguf_path();
  ET_CHECK_MSG(
      !decode_gguf_path.empty(),
      "Provide --decode_gguf_path, --gguf_model_path, or --tmac_model_path for decode");

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
  if (FLAGS_decode_native_first_token) {
    args.push_back("--pd-native-first-token");
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

  DecodeProcessResult result;
  result.before = process_memory_snapshot();
  const auto decode_start = SteadyClock::now();
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
  result.wall_ms = elapsed_ms(decode_start);
  result.after = process_memory_snapshot();
  if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    result.exit_code = 128 + WTERMSIG(status);
  }
  return result;
}

template <typename T>
void run_wikitext_ppl(
    ModuleBundle module_bundle,
    const ModuleMetaInfo& module_meta,
    std::unique_ptr<executorch::extension::Module> attention_sink_rope_module) {
  example::Runner<T> runner(
      std::move(module_bundle.module),
      read_prefill_shard_paths(FLAGS_prefill_shard_manifest_path),
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
PdE2ERuntimeStats run_pd_e2e(
    ModuleBundle module_bundle,
    const std::string& prompt_input,
    bool tokenized_prompt,
    const std::string& handoff_dir,
    PrefillShardFiles prefill_shard_files,
    example::DecoderRunner::PrefillShardRebuildConfig prefill_shard_rebuild,
    std::unique_ptr<executorch::extension::Module> attention_sink_rope_module) {
  typename example::PDPrefillRunner<T>::StaticMetadata static_metadata;
  if (prefill_shard_files.qwen3_static_plan) {
    static_metadata.enabled = true;
    static_metadata.context_len = prefill_shard_files.context_len;
    static_metadata.prompt_ar_len = prefill_shard_files.prefill_ar_len;
    static_metadata.token_generator_ar_len = prefill_shard_files.token_generator_ar_len;
    static_metadata.vocab_size = prefill_shard_files.vocab_size;
    static_metadata.sliding_window = prefill_shard_files.context_len;
    static_metadata.num_layers = prefill_shard_files.num_layers;
    static_metadata.num_heads = prefill_shard_files.num_heads;
    static_metadata.head_dim = prefill_shard_files.head_dim;
    static_metadata.use_int64_token = prefill_shard_files.use_int64_token;
    static_metadata.cache_mode = CacheMode::StaticCahce;
  }

  PdE2ERuntimeStats stats;
  stats.before_runner = process_memory_snapshot();
  const auto runner_setup_start = SteadyClock::now();
  example::PDPrefillRunner<T> runner(
      std::move(module_bundle.module),
      std::move(prefill_shard_files.pte_paths),
      std::move(prefill_shard_files.index_bin_paths),
      std::move(prefill_shard_rebuild),
      prefill_shard_files.qwen3_static_plan,
      prefill_shard_files.static_aux_size,
      prefill_shard_files.static_hidden_size,
      static_metadata,
      FLAGS_decoder_model_version.c_str(),
      get_model_path_for_runner(),
      FLAGS_tokenizer_path.c_str(),
      module_bundle.pte_bytes,
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      nullptr,
      std::move(attention_sink_rope_module));
  stats.runner_setup_ms = elapsed_ms(runner_setup_start);
  stats.after_runner = process_memory_snapshot();

  const auto decoder_version = runner.get_decoder_model_version().get();
  const std::string formatted_prompt = tokenized_prompt
      ? prompt_input
      : get_formatted_prompt(prompt_input, FLAGS_system_prompt, decoder_version);
  const auto qnn_export_start = SteadyClock::now();
  ET_CHECK_MSG(
      runner.export_prefill_handoff(
          formatted_prompt,
          tokenized_prompt,
          FLAGS_seq_len,
          handoff_dir,
          FLAGS_kv_quant_attrs_path) == executorch::runtime::Error::Ok,
      "PD prefill export failed");
  stats.qnn_export_total_ms = elapsed_ms(qnn_export_start);
  const auto runner_stats = runner.last_runtime_stats();
  stats.prompt_tokens = runner_stats.prompt_tokens;
  stats.prefill.prompt_tokens = runner_stats.prompt_tokens;
  stats.prefill.tokenize_ms = runner_stats.tokenize_ms;
  stats.prefill.prefill_ms = runner_stats.prefill_ms;
  stats.prefill.handoff_total_ms = runner_stats.handoff_total_ms;
  stats.prefill.kv_layout_ms = runner_stats.kv_layout_ms;
  stats.prefill.kv_write_ms = runner_stats.kv_write_ms;
  stats.prefill.fingerprint_ms = runner_stats.fingerprint_ms;
  stats.shard_stats = runner.prefill_shard_runtime_stats();
  stats.after_export = process_memory_snapshot();
  return stats;
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

  PrefillShardFiles prefill_shard_files =
      read_prefill_shard_files(FLAGS_prefill_shard_manifest_path);
  const bool manifest_only_prefill = prefill_shard_files.qwen3_static_plan &&
      !FLAGS_prefill_shard_manifest_path.empty();
  ET_CHECK_MSG(
      !manifest_only_prefill || FLAGS_wikitext_path.empty(),
      "WikiText PPL mode still requires a main PTE; manifest-only is PD handoff only");

  ModuleBundle module_bundle;
  ModuleMetaInfo module_meta;
  if (manifest_only_prefill) {
    ET_CHECK_MSG(
        prefill_shard_files.kv_bitwidth == 8 ||
            prefill_shard_files.kv_bitwidth == 16,
        "Unsupported static prefill KV bitwidth in manifest: %d",
        prefill_shard_files.kv_bitwidth);
    // The Float MethodMeta on static QNN shards is an ABI carrier type. The
    // underlying KV storage width is specified by the prefill manifest and
    // must match the normal QNN runner's KVManager instantiation.
    module_meta.kv_bitwidth = static_cast<example::KvBitWidth>(
        prefill_shard_files.kv_bitwidth);
    ET_LOG(
        Info,
        "skipping main PTE load; using manifest-only prefill metadata with %d-bit KV storage",
        prefill_shard_files.kv_bitwidth);
  } else {
    module_bundle = load_module_from_file_or_rebuild();
    if (should_rebuild_from_stripped()) {
      ET_LOG(
          Info,
          "pte_materialized_weight_bytes=%zu split_peak_weight_bytes=%zu num_splits=%zu",
          module_bundle.materialized_weight_bytes,
          module_bundle.split_stats.peak_split_materialized_weight_bytes,
          module_bundle.split_stats.num_splits);
    }
    module_meta = read_module_meta(module_bundle.module.get());
  }
  auto prefill_shard_rebuild =
      make_prefill_shard_rebuild_config(prefill_shard_files);
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

  const auto e2e_start = SteadyClock::now();
  const std::string handoff_dir = FLAGS_prefill_export_dir.empty()
      ? create_temp_handoff_dir()
      : FLAGS_prefill_export_dir;
  fs::create_directories(handoff_dir);
  ET_LOG(Info, "Using PD handoff directory: %s", handoff_dir.c_str());

  const std::string prompt_input =
      use_tokenized_prompt ? FLAGS_tokenized_prompt : prompts.front();
  PdE2ERuntimeStats prefill_runtime;
  if (module_meta.kv_bitwidth == example::KvBitWidth::kWidth8) {
    prefill_runtime = run_pd_e2e<uint8_t>(
        std::move(module_bundle),
        prompt_input,
        use_tokenized_prompt,
        handoff_dir,
        prefill_shard_files,
        prefill_shard_rebuild,
        std::move(attention_sink_rope_module));
  } else if (module_meta.kv_bitwidth == example::KvBitWidth::kWidth16) {
    prefill_runtime = run_pd_e2e<uint16_t>(
        std::move(module_bundle),
        prompt_input,
        use_tokenized_prompt,
        handoff_dir,
        prefill_shard_files,
        prefill_shard_rebuild,
        std::move(attention_sink_rope_module));
  } else {
    ET_CHECK_MSG(
        false,
        "Unsupported kv bitwidth: %ld",
        static_cast<int64_t>(module_meta.kv_bitwidth));
  }

  if (FLAGS_prefill_only) {
    DecodeProcessResult no_decode;
    no_decode.before = process_memory_snapshot();
    no_decode.after = no_decode.before;
    log_pd_e2e_runtime_summary(
        prefill_runtime,
        no_decode,
        elapsed_ms(e2e_start));
    ET_LOG(Info, "Prefill completed; handoff is ready at %s", handoff_dir.c_str());
    return 0;
  }

  const DecodeProcessResult decode = run_decode_process(handoff_dir);
  ET_CHECK_MSG(
      decode.exit_code == 0,
      "llama-pd-cli exited with code %d",
      decode.exit_code);
  log_pd_e2e_runtime_summary(
      prefill_runtime,
      decode,
      elapsed_ms(e2e_start));
  return 0;
}
