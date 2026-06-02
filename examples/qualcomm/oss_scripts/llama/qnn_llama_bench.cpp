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
 * Benchmark tool for Qualcomm Llama runner. It sweeps prefill length and
 * decode length combinations using dummy prompt tokens and reports throughput.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/pte_rebuilder.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>
#include <pytorch/tokenizers/tokenizer.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
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
    attention_sink_rope_path,
    "",
    "[Attention Sink] The Attention Sink Rope Model serialized in flatbuffer format.");
DEFINE_string(
    tokenizer_path,
    "tokenizer.bin",
    "Tokenizer path. Kept for compatibility with the runner interface.");
DEFINE_string(
    performance_output_path,
    "inference_speed.txt",
    "Path used by runner internals to store speed for each run.");
DEFINE_double(
    temperature,
    0.0f,
    "Temperature; 0 = greedy argmax sampling (deterministic).");
DEFINE_int32(
    eval_mode,
    1,
    "0: TokenGenerator(kv) / 1: HybridMode (prefill+kv) / 2: Lookahead Decoding");
DEFINE_bool(
    shared_buffer,
    false,
    "Use shared buffers for zero-copy use case between app and device.");
DEFINE_int32(
    ngram,
    0,
    "[Lookahead Decoding] n-gram size used in the lookahead process.");
DEFINE_int32(
    window,
    0,
    "[Lookahead Decoding] Number of future tokens attempted per step.");
DEFINE_int32(
    gcap,
    0,
    "[Lookahead Decoding] Max speculation candidates considered per step.");
DEFINE_string(
    prefill_lengths,
    "32,128,512",
    "Comma-separated prefill token lengths to benchmark.");
DEFINE_string(
    generation_lengths,
    "0,32,128",
    "Comma-separated decode token counts to benchmark.");
DEFINE_int32(num_iters, 3, "Measured iterations per (prefill, generation) pair.");
DEFINE_int32(warmup_iters, 1, "Warmup iterations per (prefill, generation) pair.");
DEFINE_uint64(
    dummy_token_id,
    1,
    "Token id used to build dummy prompt tokens for prefill.");
DEFINE_string(
    output_path,
    "qnn_llama_bench.csv",
    "CSV output path for benchmark results.");

namespace {

struct ModuleBundle {
  std::unique_ptr<executorch::extension::Module> module;
  std::shared_ptr<std::vector<uint8_t>> rebuilt_pte;
  double rebuild_time_ms{0.0};
  size_t rebuilt_records{0};
  bool rebuilt_from_stripped{false};
  bool specialized_fast_path_used{false};
};

double safe_rate(double token_count, double time_ms) {
  return time_ms > 0.0 ? token_count * 1000.0 / time_ms : 0.0;
}

std::string trim(std::string value) {
  value.erase(
      value.begin(),
      std::find_if(
          value.begin(), value.end(), [](unsigned char c) { return !std::isspace(c); }));
  value.erase(
      std::find_if(
          value.rbegin(), value.rend(), [](unsigned char c) { return !std::isspace(c); })
          .base(),
      value.end());
  return value;
}

std::vector<int32_t> parse_int_list(
    const std::string& csv,
    const char* flag_name,
    bool allow_zero) {
  std::vector<int32_t> values;
  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = trim(token);
    ET_CHECK_MSG(
        !token.empty(), "Empty value found in --%s: %s", flag_name, csv.c_str());

    char* end_ptr = nullptr;
    const long parsed = std::strtol(token.c_str(), &end_ptr, 10);
    ET_CHECK_MSG(
        end_ptr != token.c_str() && *end_ptr == '\0',
        "Invalid integer value '%s' in --%s",
        token.c_str(),
        flag_name);
    ET_CHECK_MSG(
        parsed <= std::numeric_limits<int32_t>::max(),
        "Value '%s' in --%s is too large",
        token.c_str(),
        flag_name);
    ET_CHECK_MSG(
        allow_zero ? (parsed >= 0) : (parsed > 0),
        "Value '%s' in --%s must be %s",
        token.c_str(),
        flag_name,
        allow_zero ? ">= 0" : "> 0");
    values.push_back(static_cast<int32_t>(parsed));
  }
  ET_CHECK_MSG(!values.empty(), "--%s cannot be empty", flag_name);
  return values;
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
    bundle.module = std::make_unique<executorch::extension::Module>(
        FLAGS_model_path.c_str(),
        executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
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

std::unique_ptr<executorch::extension::Module> maybe_load_attention_sink_module() {
  if (FLAGS_attention_sink_rope_path.empty()) {
    return nullptr;
  }
  return std::make_unique<executorch::extension::Module>(
      FLAGS_attention_sink_rope_path.c_str(),
      executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
}

example::KvBitWidth read_kv_bitwidth(executorch::extension::Module* module) {
  auto method_names = module->method_names();
  ET_CHECK_MSG(method_names.ok(), "Failed to read module method names");
  if (method_names->count("get_kv_io_bit_width") > 0) {
    return static_cast<example::KvBitWidth>(
        module->get("get_kv_io_bit_width").get().toScalar().to<int64_t>());
  }
  return example::KvBitWidth::kWidth8;
}

std::string get_model_path_for_runner() {
  if (!FLAGS_stripped_model_path.empty()) {
    return FLAGS_stripped_model_path;
  }
  return FLAGS_model_path;
}

class BenchTokenizer final : public tokenizers::Tokenizer {
 public:
  static constexpr const char* kBenchPromptMagic = "__qnn_llama_bench_prompt__";
  static constexpr uint64_t kEosSentinel =
      std::numeric_limits<uint64_t>::max() - 1;

  explicit BenchTokenizer(uint64_t dummy_token_id)
      : prefill_len_(1), dummy_token_id_(dummy_token_id) {
    initialized_ = true;
    vocab_size_ = std::numeric_limits<int32_t>::max();
    bos_tok_ = dummy_token_id_;
    eos_tok_ = kEosSentinel;
  }

  void set_prefill_len(int32_t prefill_len) {
    ET_CHECK_MSG(prefill_len > 0, "prefill_len must be > 0");
    prefill_len_ = prefill_len;
  }

  tokenizers::Error load(const std::string&) override {
    initialized_ = true;
    return tokenizers::Error::Ok;
  }

  tokenizers::Result<std::string> id_to_piece(uint64_t token) const override {
    return std::to_string(token);
  }

  tokenizers::Result<uint64_t> piece_to_id(const std::string& text) const override {
    if (text == kBenchPromptMagic) {
      return dummy_token_id_;
    }
    return kEosSentinel;
  }

  tokenizers::Result<std::vector<uint64_t>> encode(
      const std::string& input,
      int8_t /*bos*/,
      int8_t /*eos*/) const override {
    if (input != kBenchPromptMagic) {
      return std::vector<uint64_t>{kEosSentinel};
    }
    return std::vector<uint64_t>(
        static_cast<size_t>(prefill_len_), dummy_token_id_);
  }

  tokenizers::Result<std::string> decode(
      uint64_t,
      uint64_t,
      bool /*skip_special_tokens*/ = false) const override {
    return std::string();
  }

 private:
  int32_t prefill_len_;
  uint64_t dummy_token_id_;
};


struct Sample {
  double prefill_ms;
  double decode_ms;
  double total_ms;
  double prefill_tok_per_s;
  double decode_tok_per_s;
  double total_tok_per_s;
  double ttfb_ms;
};

struct Aggregate {
  int32_t count{0};
  double prefill_ms{0.0};
  double decode_ms{0.0};
  double total_ms{0.0};
  double prefill_tok_per_s{0.0};
  double decode_tok_per_s{0.0};
  double total_tok_per_s{0.0};
  double ttfb_ms{0.0};

  void add(const Sample& sample) {
    ++count;
    prefill_ms += sample.prefill_ms;
    decode_ms += sample.decode_ms;
    total_ms += sample.total_ms;
    prefill_tok_per_s += sample.prefill_tok_per_s;
    decode_tok_per_s += sample.decode_tok_per_s;
    total_tok_per_s += sample.total_tok_per_s;
    ttfb_ms += sample.ttfb_ms;
  }

  Sample average() const {
    ET_CHECK_MSG(count > 0, "Attempted to average empty benchmark samples");
    return Sample{
        prefill_ms / count,
        decode_ms / count,
        total_ms / count,
        prefill_tok_per_s / count,
        decode_tok_per_s / count,
        total_tok_per_s / count,
        ttfb_ms / count};
  }
};

template <typename T>
Sample run_single(
    example::Runner<T>* runner,
    BenchTokenizer* bench_tokenizer,
    int32_t prefill_len,
    int32_t generation_len) {
  runner->reset();
  bench_tokenizer->set_prefill_len(prefill_len);

  ET_CHECK_MSG(
      static_cast<int64_t>(prefill_len) + generation_len + 1 <=
          std::numeric_limits<int32_t>::max(),
      "prefill_len + generation_len is too large");
  const executorch::extension::llm::GenerationConfig config{
      false,
      true,
      -1,
      false,
      prefill_len + generation_len + 1,
      static_cast<float>(FLAGS_temperature),
      0,
      0};

  long inference_start_ms{0};
  long prompt_eval_end_ms{0};
  long inference_end_ms{0};
  long first_token_ms{0};
  int64_t num_prompt_tokens{0};
  int64_t num_generated_tokens{0};
  bool got_stats = false;
  const auto token_callback = [](const std::string&) {};
  const auto stats_callback = [&](const executorch::llm::Stats& run_stats) {
    inference_start_ms = run_stats.inference_start_ms;
    prompt_eval_end_ms = run_stats.prompt_eval_end_ms;
    inference_end_ms = run_stats.inference_end_ms;
    first_token_ms = run_stats.first_token_ms;
    num_prompt_tokens = run_stats.num_prompt_tokens;
    num_generated_tokens = run_stats.num_generated_tokens;
    got_stats = true;
  };

  const auto err = runner->generate_from_prompt_or_file(
      BenchTokenizer::kBenchPromptMagic,
      false,
      config,
      token_callback,
      stats_callback);
  ET_CHECK_MSG(
      err == executorch::runtime::Error::Ok,
      "Runner invocation failed with error code %d",
      static_cast<int>(err));
  ET_CHECK_MSG(got_stats, "Failed to collect benchmark stats");
  ET_CHECK_MSG(
      num_prompt_tokens == prefill_len,
      "Prefill length mismatch (requested %d, got %ld)",
      prefill_len,
      num_prompt_tokens);
  ET_CHECK_MSG(
      num_generated_tokens == generation_len,
      "Decode token mismatch (requested %d, got %ld). Increase --seq_len at export time or use attention sink.",
      generation_len,
      num_generated_tokens);

  const double prefill_ms = prompt_eval_end_ms - inference_start_ms;
  const double decode_ms = inference_end_ms - prompt_eval_end_ms;
  const double total_ms = inference_end_ms - inference_start_ms;
  const double ttfb_ms = first_token_ms - inference_start_ms;

  return Sample{
      prefill_ms,
      decode_ms,
      total_ms,
      safe_rate(prefill_len, prefill_ms),
      safe_rate(generation_len, decode_ms),
      safe_rate(generation_len, total_ms),
      ttfb_ms};
}

template <typename T>
void run_benchmarks(
    ModuleBundle module_bundle,
    std::unique_ptr<executorch::extension::Module> attention_sink_rope_module,
    const std::vector<int32_t>& prefill_lengths,
    const std::vector<int32_t>& generation_lengths) {
  auto tokenizer = std::make_unique<BenchTokenizer>(FLAGS_dummy_token_id);
  BenchTokenizer* bench_tokenizer = tokenizer.get();

  example::Runner<T> runner(
      std::move(module_bundle.module),
      FLAGS_decoder_model_version.c_str(),
      get_model_path_for_runner(),
      FLAGS_tokenizer_path.c_str(),
      FLAGS_performance_output_path.c_str(),
      "",
      static_cast<float>(FLAGS_temperature),
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      FLAGS_ngram,
      FLAGS_window,
      FLAGS_gcap,
      std::move(tokenizer),
      std::move(attention_sink_rope_module));

  if (module_bundle.rebuilt_from_stripped) {
    ET_LOG(
        Info,
        "pte_rebuild_time_ms=%.3f rebuilt_records=%zu specialized_fast_path=%s",
        module_bundle.rebuild_time_ms,
        module_bundle.rebuilt_records,
        module_bundle.specialized_fast_path_used ? "true" : "false");
  }

  std::ofstream out(FLAGS_output_path.c_str());
  ET_CHECK_MSG(
      out.is_open(),
      "Failed to open output csv path: %s",
      FLAGS_output_path.c_str());
  out << "prefill_len,generation_len,avg_prefill_ms,avg_decode_ms,avg_total_ms,"
         "avg_ttfb_ms,avg_prefill_tok_per_s,avg_decode_tok_per_s,"
         "avg_end_to_end_decode_tok_per_s\n";

  std::printf(
      "%10s %14s %14s %14s %14s %14s %18s %18s %18s\n",
      "prefill",
      "generation",
      "prefill_ms",
      "decode_ms",
      "total_ms",
      "ttfb_ms",
      "prefill_tok/s",
      "decode_tok/s",
      "e2e_tok/s");

  for (int32_t prefill_len : prefill_lengths) {
    for (int32_t generation_len : generation_lengths) {
      for (int i = 0; i < FLAGS_warmup_iters; ++i) {
        (void)run_single(&runner, bench_tokenizer, prefill_len, generation_len);
      }

      Aggregate aggregate;
      for (int i = 0; i < FLAGS_num_iters; ++i) {
        aggregate.add(run_single(&runner, bench_tokenizer, prefill_len, generation_len));
      }
      const Sample avg = aggregate.average();

      out << prefill_len << "," << generation_len << "," << avg.prefill_ms
          << "," << avg.decode_ms << "," << avg.total_ms << "," << avg.ttfb_ms
          << "," << avg.prefill_tok_per_s << "," << avg.decode_tok_per_s << ","
          << avg.total_tok_per_s << "\n";
      out.flush();

      std::printf(
          "%10d %14d %14.2f %14.2f %14.2f %14.2f %18.2f %18.2f %18.2f\n",
          prefill_len,
          generation_len,
          avg.prefill_ms,
          avg.decode_ms,
          avg.total_ms,
          avg.ttfb_ms,
          avg.prefill_tok_per_s,
          avg.decode_tok_per_s,
          avg.total_tok_per_s);
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ET_CHECK_MSG(FLAGS_num_iters > 0, "--num_iters must be > 0");
  ET_CHECK_MSG(FLAGS_warmup_iters >= 0, "--warmup_iters must be >= 0");
  ET_CHECK_MSG(
      FLAGS_dummy_token_id <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
      "--dummy_token_id must fit int32 for models with int32 token input");

  const std::vector<int32_t> prefill_lengths =
      parse_int_list(FLAGS_prefill_lengths, "prefill_lengths", false);
  const std::vector<int32_t> generation_lengths =
      parse_int_list(FLAGS_generation_lengths, "generation_lengths", true);

  ModuleBundle module_bundle = load_module_from_file_or_rebuild();
  auto attention_sink_rope_module = maybe_load_attention_sink_module();
  const example::KvBitWidth kv_bitwidth =
      read_kv_bitwidth(module_bundle.module.get());

  if (kv_bitwidth == example::KvBitWidth::kWidth8) {
    run_benchmarks<uint8_t>(
        std::move(module_bundle),
        std::move(attention_sink_rope_module),
        prefill_lengths,
        generation_lengths);
  } else if (kv_bitwidth == example::KvBitWidth::kWidth16) {
    run_benchmarks<uint16_t>(
        std::move(module_bundle),
        std::move(attention_sink_rope_module),
        prefill_lengths,
        generation_lengths);
  } else {
    ET_CHECK_MSG(
        false,
        "Unsupported kv bitwidth: %ld",
        static_cast<int64_t>(kv_bitwidth));
  }

  return 0;
}
