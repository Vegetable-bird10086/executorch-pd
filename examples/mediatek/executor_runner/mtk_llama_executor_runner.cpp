/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#include "executorch/backends/mediatek/runtime/include/NeuronBufferAllocator.h"

#include <ctime>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

#ifdef MTK_LLAMA_PD_JOINT
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "pd_cli_inprocess.h"
#endif

#include <gflags/gflags.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>

#include "llama_runner/LlamaConfig.h"
#include "llama_runner/LlamaRuntime.h"
#include "llama_runner/MtkStageMajorPrefill.h"
#include "llama_runner/ModelChunk.h"
#include "llama_runner/QnnKvAbi.h"
#include "llama_runner/Utils.h"
#include "llama_runner/llm_helper/include/llm_types.h"

#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <pytorch/tokenizers/tiktoken.h>

// Llama model options
DEFINE_uint64(
    prompt_token_batch_size,
    128,
    "Token batch size for prompt model.");
DEFINE_uint64(cache_size, 1024, "Model cache size.");
DEFINE_uint64(hidden_size, 4096, "Model hidden size.");
DEFINE_uint64(num_head, 32, "Number of attention heads in each layer.");
DEFINE_uint64(num_layer, 32, "Number of layers in the model.");
DEFINE_uint64(head_dim, 0, "Head dimension of the model.");
DEFINE_uint64(window_size, 0, "Window size of Sliding Window Attention.");
DEFINE_uint64(
    max_token_length,
    2048,
    "Maximum token length that the model supports.");
DEFINE_uint64(
    layer_debug_output_count,
    0,
    "Number of diagnostic hidden-state outputs appended by each model chunk.");
DEFINE_int32(
    layer_debug_chunk_index,
    -1,
    "Only this model chunk has diagnostic outputs. -1 means every chunk.");
DEFINE_string(
    layer_debug_dump_dir,
    "",
    "Directory for first-prefill layer dumps. Disabled when empty.");
DEFINE_string(
    chunk_debug_dump_dir,
    "",
    "Directory for first-prefill chunk output dumps. Disabled when empty.");
DEFINE_bool(
    copy_chunk_io,
    false,
    "Copy the primary output between model chunks instead of sharing one "
    "backend buffer. Useful when adjacent chunks use different precision.");
DEFINE_string(
    first_chunk_input_path,
    "",
    "Replace the first chunk input on the first prefill invocation with an "
    "exact-size raw tensor file. Disabled when empty.");
DEFINE_string(
    combined_logits_dump_dir,
    "",
    "Dump each CPU-combined last-token logits buffer. Disabled when empty.");
DEFINE_double(partial_rotary_factor, 1, "Partial rotary factor of the model.");
DEFINE_double(
    rot_emb_base,
    10000,
    "Rotary embedding base value, aka 'rope_theta'.");

// Model IO Types
DEFINE_string(input_type, "int16", "Model input type. Default to 'int16'");
DEFINE_string(output_type, "int16", "Model output type. Default to 'int16'");
DEFINE_string(cache_type, "int16", "Model cache type. Default to 'int16'");
DEFINE_string(mask_type, "int16", "Model mask type. Default to 'int16'");
DEFINE_string(
    rot_emb_type,
    "int16",
    "Model rotary embedding type. Default to 'int16'");

// Model Paths
DEFINE_string(
    token_embedding_path,
    "embedding.bin",
    "Input token embedding lookup table path.");
DEFINE_string(prompt_model_paths, "", "Comma-separated prompt model paths.");
DEFINE_string(gen_model_paths, "", "Comma-separated generative model paths.");
DEFINE_string(
    model_package_paths,
    "",
    "Comma-separated weight-shared model package paths.");

// Tokenizer
DEFINE_string(tokenizer_path, "tokenizer.model", "tokenizer.model vocab path.");
DEFINE_string(
    tokenizer_type,
    "tiktoken",
    "Tokenizer type. One of ['bpe', 'tiktoken'].");
DEFINE_uint64(vocab_size, 128000, "Tokenizer vocab size.");
DEFINE_uint64(
    logit_shard_count,
    1,
    "Number of independent final logits outputs. Defaults to one.");
DEFINE_uint64(bos_token, 128000, "BOS token id.");
DEFINE_uint64(eos_token, 128001, "EOS token id.");

// Inference
DEFINE_uint64(max_response, 50, "Maximum number of tokens to generate.");
DEFINE_string(prompt_file, "", "File containing the prompt text.");
DEFINE_bool(
    prefill_only,
    false,
    "Run one prefill pass and exit without argmax or token generation. "
    "Useful for diagnostic chunks whose output is a hidden state.");
DEFINE_string(
    pd_export_dir,
    "",
    "Export prompt tokens, first token, and canonical FP16 KV for llama.cpp "
    "PD decode. Empty disables PD export.");
DEFINE_string(
    pd_qnn_kv_abi_path,
    "",
    "PD-only QNN U8 KV ABI generated from the matching Decode profile. "
    "When set, MTK Prefill additionally emits byte-exact QNN-domain KV; "
    "normal MTK generation is unchanged.");
DEFINE_string(
    pd_prompt_tokens_path,
    "",
    "Raw little-endian uint64 prompt tokens for PD prefill. When set, bypass "
    "text tokenization and require --pd_export_dir.");
DEFINE_bool(
    pd_prefill_pipeline,
    true,
    "Pack each completed MTK chunk's KV on a worker while later chunks run.");
DEFINE_string(
    mtk_ppl_prompt_tokens_path,
    "",
    "Raw little-endian uint64 context tokens for direct MTK teacher-forcing PPL.");
DEFINE_string(
    mtk_ppl_tokens_path,
    "",
    "Raw little-endian uint64 predictor-plus-target tokens for direct MTK PPL. "
    "As in QNN WikiText PPL, token 0 is fed as the first AR-1 predictor and "
    "token 1 is the first scored target.");
DEFINE_string(
    mtk_ppl_output,
    "",
    "Write direct MTK PPL metrics as JSON. Empty disables MTK PPL mode.");
DEFINE_uint64(
    mtk_ppl_max_tokens,
    0,
    "Maximum targets to score after the first predictor token; zero scores "
    "all remaining tokens in the file.");
DEFINE_string(
    pd_stage_major_stripped_pte_paths,
    "",
    "Comma-separated stripped.pte paths for low-memory stage-major Prefill.");
DEFINE_string(
    pd_stage_major_index_paths,
    "",
    "Comma-separated MTK weight index.bin paths in chunk order.");
DEFINE_string(
    pd_stage_major_weight_paths,
    "",
    "Legacy comma-separated weights.bin paths. Omit in the joint runner when "
    "semantic MTKGGUF2 indexes and --pd_joint_gguf_path are used.");
DEFINE_bool(
    pd_stage_major_three_stage,
    true,
    "Overlap MTK chunk i execution with chunk i+1 loading and chunk i+2 "
    "PTE reconstruction. Disable for a serial performance baseline.");
DEFINE_bool(
    pd_stage_major_async_release,
    false,
    "With the three-stage pipeline, overlap release of chunk i with execution "
    "of chunk i+1. At most one release job may be outstanding.");
DEFINE_bool(
    pd_stage_major_persistent_chunk0,
    true,
    "Prebuild and keep chunk 0 alive for the Prefill session. Preparation "
    "runs in parallel with tokenizer/prompt initialization; later requests "
    "reset and reuse the same Neuron context.");
DEFINE_bool(
    pd_stage_major_detach_pte_after_load,
    true,
    "Recycle each reconstructed PTE backing immediately after Neuron load, "
    "including persistent chunk 0. Disable explicitly only for backends that "
    "require the original PTE bytes to remain resident.");
DEFINE_uint64(
    pd_stage_major_session_repeats,
    1,
    "Run the same stage-major request repeatedly in one process to validate "
    "persistent chunk-0 reuse. For values above one, each handoff is written "
    "under request_<index> in --pd_export_dir.");
#ifdef MTK_LLAMA_PD_JOINT
DEFINE_string(
    pd_joint_gguf_path,
    "",
    "Memory-mapped llama.cpp Decode GGUF. Its persistent initialization runs "
    "in parallel with MTK chunk-0 preparation.");
DEFINE_string(
    pd_joint_disk_embedding_path,
    "",
    "Optional external embedding matrix passed to llama.cpp Decode.");
DEFINE_int32(pd_joint_decode_n_predict, 32, "Decode tokens to generate.");
DEFINE_int32(pd_joint_decode_ctx, 2048, "llama.cpp Decode context size.");
DEFINE_int32(pd_joint_decode_threads, 6, "llama.cpp Decode CPU threads.");
DEFINE_double(pd_joint_decode_temp, 0.0, "llama.cpp Decode temperature.");
DEFINE_string(
    decode_ppl_tokens_path,
    "",
    "Raw uint64 continuation-token file for in-process teacher-forced PD WikiPPL.");
DEFINE_string(
    decode_ppl_output_path,
    "",
    "Output file written by the in-process Decode WikiPPL path.");
DEFINE_int32(
    decode_ppl_max_tokens,
    0,
    "Maximum in-process PD WikiPPL targets; zero scores the full continuation.");
DEFINE_bool(
    pd_joint_import_only,
    false,
    "Import and validate the MTK KV handoff without generating tokens.");
#endif

// Global BOS and EOS option for tokenization (encoding)
static constexpr int8_t kAddBos = 1;
static constexpr int8_t kAddEos = 0;

using namespace example::llm_helper;
using example::LlamaModelOptions;
using example::LlamaModelPaths;
using example::LlamaModelChunk;
using example::LlamaRuntime;
using example::MtkStrippedChunkPaths;
using example::MtkStageMajorPrefillSession;
using example::MtkGgufWeightSource;
using example::LoadMtkGgufWeightSourceIntoRam;
using example::MtkGgufWeightSourceData;
using example::MtkGgufWeightSourceSize;
using example::QnnKvAbi;
using example::QnnKvAbiStats;
using example::utils::argmax;
using example::utils::read_file;
using example::utils::split;
using example::utils::Timer;
using example::utils::to_string;
using executorch::runtime::Error;
using executorch::runtime::Result;
using tokenizers::HFTokenizer;
using tokenizers::Llama2cTokenizer;
using tokenizers::Tokenizer;

#ifdef MTK_LLAMA_PD_JOINT
namespace {

struct JointDecodeRuntime {
  std::shared_ptr<MtkGgufWeightSource> model;
  llama_pd_inprocess_runtime* runtime{nullptr};
  double initializationMs{0.0};

  ~JointDecodeRuntime() {
    if (runtime != nullptr) {
      llama_pd_inprocess_runtime_destroy(runtime);
    }
  }
};

std::shared_ptr<JointDecodeRuntime> prepare_joint_decode(
    std::shared_ptr<MtkGgufWeightSource> modelSource) {
  ET_CHECK_MSG(modelSource != nullptr, "joint Decode requires GGUF RAM store");
  if (std::getenv("GGML_QNN_U16_ACTIVATIONS") == nullptr) {
    setenv("GGML_QNN_U16_ACTIVATIONS", "1", 0);
  }
  if (std::getenv("GGML_QNN_U16_BLOCKWISE_REQUANT") == nullptr) {
    setenv("GGML_QNN_U16_BLOCKWISE_REQUANT", "1", 0);
  }
  auto out = std::make_shared<JointDecodeRuntime>();
  out->model = std::move(modelSource);
  std::vector<std::string> args = {
      "mtk_llama_pd_joint_runner", "-m", "in-process.gguf", "-n",
      std::to_string(FLAGS_pd_joint_decode_n_predict), "-c",
      std::to_string(FLAGS_pd_joint_decode_ctx), "-b", "1", "-ub", "1",
      "-ngl", "0", "--temp", std::to_string(FLAGS_pd_joint_decode_temp),
      "-fit", "off", "-t", std::to_string(FLAGS_pd_joint_decode_threads)};
  if (!FLAGS_pd_joint_disk_embedding_path.empty()) {
    args.push_back("--pd-disk-embedding");
    args.push_back(FLAGS_pd_joint_disk_embedding_path);
  }
  // Match the QNN joint runner: only the continuation tokens are file-backed.
  // Prefill KV remains in the request's in-process handoff_data buffer.
  if (!FLAGS_decode_ppl_tokens_path.empty()) {
    args.push_back("--pd-ppl-tokens");
    args.push_back(FLAGS_decode_ppl_tokens_path);
    if (!FLAGS_decode_ppl_output_path.empty()) {
      args.push_back("--pd-ppl-output");
      args.push_back(FLAGS_decode_ppl_output_path);
    }
    if (FLAGS_decode_ppl_max_tokens > 0) {
      args.push_back("--pd-ppl-max-tokens");
      args.push_back(std::to_string(FLAGS_decode_ppl_max_tokens));
    }
  }
  if (FLAGS_pd_joint_import_only) {
    args.push_back("--pd-import-ro");
  }
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }
  out->runtime = llama_pd_inprocess_runtime_create(
      static_cast<int>(argv.size()),
      argv.data(),
      MtkGgufWeightSourceData(out->model),
      MtkGgufWeightSourceSize(out->model),
      &out->initializationMs);
  ET_CHECK_MSG(out->runtime != nullptr, "Unable to initialize joint llama.cpp Decode");
  return out;
}

} // namespace
#endif

class PdKvExportPipeline {
 public:
  PdKvExportPipeline(
      const size_t promptLength,
      const size_t totalLayers,
      const size_t numKVHeads,
      const size_t headDim,
      const bool threaded)
      : promptLength_(promptLength),
        totalLayers_(totalLayers),
        threaded_(threaded),
        canonicalKv_(
            2 * totalLayers * numKVHeads * promptLength * headDim) {
    if (threaded_) {
      worker_ = std::thread([this]() { WorkerLoop(); });
    }
  }

  ~PdKvExportPipeline() {
    Finish();
  }

  void Enqueue(const size_t chunkIndex, LlamaModelChunk& chunk) {
    Job job{
        chunkIndex,
        nextLayerOffset_,
        &chunk,
    };
    nextLayerOffset_ += chunk.GetCacheLayerCount();
    if (!threaded_) {
      Copy(job);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      jobs_.push_back(job);
    }
    condition_.notify_one();
  }

  void Finish() {
    if (finished_) {
      return;
    }
    if (threaded_) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
      }
      condition_.notify_one();
      if (worker_.joinable()) {
        worker_.join();
      }
    }
    ET_CHECK_MSG(
        nextLayerOffset_ == totalLayers_,
        "PD KV pipeline copied %zu layers, expected %zu",
        nextLayerOffset_,
        totalLayers_);
    finished_ = true;
  }

  const std::vector<uint16_t>& GetCanonicalKv() const {
    ET_CHECK_MSG(finished_, "Finish PD KV pipeline before reading its output");
    return canonicalKv_;
  }

  double GetCopyMs() const {
    return copyMs_;
  }

 private:
  struct Job {
    size_t chunkIndex;
    size_t layerOffset;
    LlamaModelChunk* chunk;
  };

  void Copy(const Job& job) {
    const auto start = std::chrono::steady_clock::now();
    job.chunk->CopyCacheToCanonicalFp16(
        promptLength_, job.layerOffset, totalLayers_, canonicalKv_);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start);
    copyMs_ += elapsed.count();
    ET_LOG(
        Info,
        "MTK PD KV packed: chunk=%zu layers=[%zu,%zu) ms=%.3f",
        job.chunkIndex,
        job.layerOffset,
        job.layerOffset + job.chunk->GetCacheLayerCount(),
        elapsed.count());
  }

  void WorkerLoop() {
    while (true) {
      Job job{};
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this]() { return stop_ || !jobs_.empty(); });
        if (jobs_.empty()) {
          if (stop_) {
            return;
          }
          continue;
        }
        job = jobs_.front();
        jobs_.pop_front();
      }
      Copy(job);
    }
  }

  size_t promptLength_;
  size_t totalLayers_;
  bool threaded_;
  std::vector<uint16_t> canonicalKv_;
  size_t nextLayerOffset_{0};
  double copyMs_{0.0};
  bool stop_{false};
  bool finished_{false};
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<Job> jobs_;
  std::thread worker_;
};

template <typename T>
void write_pd_binary(const std::filesystem::path& path, const T* data, size_t count) {
  std::ofstream stream(path, std::ios::binary);
  ET_CHECK_MSG(stream.good(), "Unable to open PD output %s", path.c_str());
  stream.write(
      reinterpret_cast<const char*>(data),
      static_cast<std::streamsize>(count * sizeof(T)));
  ET_CHECK_MSG(stream.good(), "Unable to write PD output %s", path.c_str());
}

void write_pd_handoff(
    const std::string& exportDir,
    const std::vector<uint64_t>& promptTokens,
    const uint64_t firstToken,
    const LlamaRuntime& runtime,
    const PdKvExportPipeline& pipeline) {
  const std::filesystem::path root(exportDir);
  std::error_code error;
  std::filesystem::create_directories(root, error);
  ET_CHECK_MSG(
      !error,
      "Unable to create PD export directory %s: %s",
      exportDir.c_str(),
      error.message().c_str());

  write_pd_binary(root / "prompt_tokens.bin", promptTokens.data(), promptTokens.size());
  write_pd_binary(root / "first_token.bin", &firstToken, size_t{1});
  const auto& kv = pipeline.GetCanonicalKv();
  write_pd_binary(root / "kv.bin", kv.data(), kv.size());
  const size_t sourceKvBitWidth =
      getLLMTypeSize(getLLMTypeFromName(FLAGS_cache_type.c_str())) * 8;

  std::ofstream manifest(root / "manifest.json");
  ET_CHECK_MSG(manifest.good(), "Unable to open PD manifest in %s", exportDir.c_str());
  manifest << "{\n"
           << "  \"format_version\": \"pd-handoff-v1\",\n"
           << "  \"decoder_model_version\": \"qwen3\",\n"
           << "  \"context_length\": " << FLAGS_max_token_length << ",\n"
           << "  \"prompt_length\": " << promptTokens.size() << ",\n"
           << "  \"original_prompt_length\": " << promptTokens.size() << ",\n"
           << "  \"first_token_is_prompt_tail\": false,\n"
           << "  \"first_token_owner\": \"executorch_mtk\",\n"
           << "  \"first_token_id\": " << firstToken << ",\n"
           << "  \"num_layers\": " << FLAGS_num_layer << ",\n"
           << "  \"num_kv_heads\": " << runtime.GetNumKVHeads() << ",\n"
           << "  \"head_dim\": " << runtime.GetCacheHeadDim() << ",\n"
           << "  \"cache_mode\": \"static\",\n"
           << "  \"source_backend\": \"executorch_mtk\",\n"
           << "  \"source_kv_bit_width\": " << sourceKvBitWidth << ",\n"
           << "  \"canonical_kv_dtype\": \"fp16\",\n"
           << "  \"canonical_kv_layout\": {\n"
           << "    \"order\": \"K_then_V\",\n"
           << "    \"shape\": \"[layer,kv_head,seq,head_dim]\",\n"
           << "    \"endianness\": \"little\"\n"
           << "  },\n"
           << "  \"canonical_k_export_transform\": \"identity\",\n"
           << "  \"prompt_tokens_file\": \"prompt_tokens.bin\",\n"
           << "  \"prompt_tokens_dtype\": \"uint64\",\n"
           << "  \"first_token_file\": \"first_token.bin\",\n"
           << "  \"kv_file\": \"kv.bin\",\n"
           << "  \"kv_file_size_bytes\": " << kv.size() * sizeof(uint16_t) << ",\n"
           << "  \"rope\": {\"freq_base\": " << FLAGS_rot_emb_base
           << ", \"freq_scale\": 1.0}\n"
           << "}\n";
  manifest.flush();
  ET_CHECK_MSG(manifest.good(), "Unable to write PD manifest in %s", exportDir.c_str());
  ET_LOG(
      Info,
      "MTK PD handoff ready: dir=%s prompt=%zu kv_bytes=%zu first_token=%llu "
      "pipeline=%d kv_pack_ms=%.3f",
      exportDir.c_str(),
      promptTokens.size(),
      kv.size() * sizeof(uint16_t),
      static_cast<unsigned long long>(firstToken),
      static_cast<int>(FLAGS_pd_prefill_pipeline),
      pipeline.GetCopyMs());
}

void write_stage_major_pd_handoff(
    const std::string& exportDir,
    const std::vector<uint64_t>& cachedPromptTokens,
    const uint64_t promptTail,
    const example::MtkStageMajorPrefillResult& result,
    const std::vector<uint8_t>* qnnU8Kv) {
  const std::filesystem::path root(exportDir);
  std::error_code error;
  std::filesystem::create_directories(root, error);
  ET_CHECK_MSG(!error, "Unable to create PD export directory %s", exportDir.c_str());
  write_pd_binary(
      root / "prompt_tokens.bin",
      cachedPromptTokens.data(),
      cachedPromptTokens.size());
  write_pd_binary(root / "first_token.bin", &promptTail, size_t{1});
  write_pd_binary(
      root / "kv.bin", result.canonicalKv.data(), result.canonicalKv.size());
  if (qnnU8Kv != nullptr) {
    write_pd_binary(root / "kv_qnn_u8.bin", qnnU8Kv->data(), qnnU8Kv->size());
  }
  const size_t sourceKvBitWidth =
      getLLMTypeSize(getLLMTypeFromName(FLAGS_cache_type.c_str())) * 8;
  std::ofstream manifest(root / "manifest.json");
  ET_CHECK_MSG(manifest.good(), "Unable to open PD manifest in %s", exportDir.c_str());
  manifest << "{\n"
           << "  \"format_version\": \"pd-handoff-v1\",\n"
           << "  \"decoder_model_version\": \"qwen3\",\n"
           << "  \"context_length\": " << FLAGS_max_token_length << ",\n"
           << "  \"prompt_length\": " << cachedPromptTokens.size() << ",\n"
           << "  \"original_prompt_length\": " << cachedPromptTokens.size() + 1 << ",\n"
           << "  \"first_token_is_prompt_tail\": true,\n"
           << "  \"first_token_owner\": \"prompt_tail_bridge\",\n"
           << "  \"first_token_id\": " << promptTail << ",\n"
           << "  \"num_layers\": " << FLAGS_num_layer << ",\n"
           << "  \"num_kv_heads\": " << result.numKvHeads << ",\n"
           << "  \"head_dim\": " << result.headDim << ",\n"
           << "  \"cache_mode\": \"static\",\n"
           << "  \"source_backend\": \"executorch_mtk_stage_major\",\n"
           << "  \"source_kv_bit_width\": " << sourceKvBitWidth << ",\n"
           << "  \"canonical_kv_dtype\": \"fp16\",\n"
           << "  \"canonical_kv_layout\": {\n"
           << "    \"order\": \"K_then_V\",\n"
           << "    \"shape\": \"[layer,kv_head,seq,head_dim]\",\n"
           << "    \"endianness\": \"little\"\n"
           << "  },\n"
           << "  \"canonical_k_export_transform\": \"identity\",\n"
           << "  \"prompt_tokens_file\": \"prompt_tokens.bin\",\n"
           << "  \"prompt_tokens_dtype\": \"uint64\",\n"
           << "  \"first_token_file\": \"first_token.bin\",\n"
           << "  \"kv_file\": \"kv.bin\",\n"
           << "  \"kv_file_size_bytes\": "
           << result.canonicalKv.size() * sizeof(uint16_t) << ",\n";
  if (qnnU8Kv != nullptr) {
    manifest << "  \"qnn_u8_kv_file\": \"kv_qnn_u8.bin\",\n"
             << "  \"qnn_u8_kv_file_size_bytes\": " << qnnU8Kv->size() << ",\n"
             << "  \"qnn_u8_kv_dtype\": \"uint8\",\n"
             << "  \"qnn_u8_kv_layout\": {\"order\": \"K_then_V\", "
                "\"shape\": \"[layer,kv_head,seq,head_dim]\"},\n"
             << "  \"qnn_k_export_transform\": \"sylvester_hadamard_128\",\n";
  }
  manifest
           << "  \"rope\": {\"freq_base\": " << FLAGS_rot_emb_base
           << ", \"freq_scale\": 1.0}\n"
           << "}\n";
  manifest.flush();
  ET_CHECK_MSG(manifest.good(), "Unable to write stage-major PD manifest");
}

LlamaModelOptions get_model_options() {
  LlamaModelOptions options = {
      // Sizes
      .prompt_token_batch_size = FLAGS_prompt_token_batch_size,
      .cache_size = FLAGS_cache_size,
      .hidden_size = FLAGS_hidden_size,
      .num_head = FLAGS_num_head,
      .num_layer = FLAGS_num_layer,
      .head_dim = FLAGS_head_dim,
      .window_size = FLAGS_window_size,
      .max_token_length = FLAGS_max_token_length,
      .layer_debug_output_count = FLAGS_layer_debug_output_count,
      .logit_shard_count = FLAGS_logit_shard_count,
      .vocab_size = FLAGS_vocab_size,
      .layer_debug_chunk_index = FLAGS_layer_debug_chunk_index,
      .partial_rotary_factor = FLAGS_partial_rotary_factor,
      .rot_emb_base = FLAGS_rot_emb_base,
      .layer_debug_dump_dir = FLAGS_layer_debug_dump_dir,
      .chunk_debug_dump_dir = FLAGS_chunk_debug_dump_dir,
      .first_chunk_input_path = FLAGS_first_chunk_input_path,
      .combined_logits_dump_dir = FLAGS_combined_logits_dump_dir,
      .copy_chunk_io = FLAGS_copy_chunk_io,

      // Types
      .model_input_type = getLLMTypeFromName(FLAGS_input_type.c_str()),
      .model_output_type = getLLMTypeFromName(FLAGS_output_type.c_str()),
      .cache_type = getLLMTypeFromName(FLAGS_cache_type.c_str()),
      .mask_type = getLLMTypeFromName(FLAGS_mask_type.c_str()),
      .rot_emb_type = getLLMTypeFromName(FLAGS_rot_emb_type.c_str())};
  return options;
}

LlamaModelPaths get_model_paths() {
  LlamaModelPaths model_paths = {
      .tokenizer_path = FLAGS_tokenizer_path,
      .token_embedding_path = FLAGS_token_embedding_path,
      .prompt_model_paths = split(FLAGS_prompt_model_paths, ','),
      .gen_model_paths = split(FLAGS_gen_model_paths, ','),
      .model_package_paths = split(FLAGS_model_package_paths, ','),
  };
  return model_paths;
}

Result<uint64_t> digest_prompt(
    LlamaRuntime& llama_runtime,
    const std::unique_ptr<Tokenizer>& tokenizer,
    const std::vector<uint64_t> input_tokens,
    PdKvExportPipeline* pdPipeline = nullptr) {
  const auto input_token_count = input_tokens.size();
  const auto prompt_token_batch_size = llama_runtime.GetTokenBatchSize();
  size_t cur_token_index = 0;

  Timer timer_digest_prompt([=](const auto elapsed_sec) {
    // Ideal prompt size is a multiple of prompt batch size
    const size_t ideal_prompt_size =
        std::ceil(float(input_token_count) / prompt_token_batch_size) *
        prompt_token_batch_size;
    ET_LOG(
        Info,
        "Done analyzing prompt in %f sec (%f tok/s)",
        elapsed_sec,
        (float)ideal_prompt_size / elapsed_sec);
  });

  auto getNextTokens = [&]() {
    const size_t num_tok_remain = input_token_count - cur_token_index;
    const size_t remainder = num_tok_remain % prompt_token_batch_size;
    const size_t num_new_tokens =
        remainder ? remainder : prompt_token_batch_size;
    const auto start = cur_token_index;
    const auto end = start + num_new_tokens;
    return std::vector(
        input_tokens.begin() + start, input_tokens.begin() + end);
  };

  void* logits;
  timer_digest_prompt.Start();
  while (cur_token_index < input_token_count) {
    const auto next_tokens = getNextTokens();
    ET_LOG(
        Debug,
        "Digest next tokens (size=%zu), 1st tok=%lu",
        next_tokens.size(),
        next_tokens[0]);
    const bool finalPrefillBlock =
        cur_token_index + next_tokens.size() == input_token_count;
    LlamaRuntime::ChunkCompleteCallback chunkCompleteCallback;
    if (pdPipeline != nullptr && finalPrefillBlock) {
      chunkCompleteCallback = [pdPipeline](
                                  const size_t chunkIndex,
                                  LlamaModelChunk& chunk) {
        pdPipeline->Enqueue(chunkIndex, chunk);
      };
    }
    logits = llama_runtime.Run(next_tokens, true, chunkCompleteCallback);
    cur_token_index += next_tokens.size();
  }
  timer_digest_prompt.End();

  const auto vocab_size = tokenizer->vocab_size();
  const auto logits_type = llama_runtime.GetModelOptions().model_output_type;
  const auto first_output_token = argmax(logits_type, logits, vocab_size);
  return first_output_token;
}

Error gen_response(
    LlamaRuntime& llama_runtime,
    const std::unique_ptr<Tokenizer>& tokenizer,
    const uint64_t input_token) {
  Timer timer_model_swap(
      [](const auto elapsed_sec) { ET_LOG(Info, "Model swapped."); });

  // Swap to gen mode
  timer_model_swap.Start();
  llama_runtime.SwapModel(1);
  timer_model_swap.End();

  size_t gen_tok_count = 0;
  uint64_t prev_token = input_token;
  uint64_t output_token = input_token;

  auto decode_res = tokenizer->decode(prev_token, output_token);
  ET_CHECK_OR_RETURN_ERROR(
      decode_res.ok(),
      InvalidState,
      "Tokenizer failed to decode first generated token: %lu",
      output_token);
  std::string full_response = std::move(decode_res.get());
  std::vector<uint64_t> full_response_tokens = {input_token};

  const auto vocab_size = tokenizer->vocab_size();
  const auto logits_type = llama_runtime.GetModelOptions().model_output_type;

  double gen_total_time_sec = 0;
  Timer timer_gen_token(
      [&](const auto elapsed_sec) { gen_total_time_sec += elapsed_sec; });

  // Print first output token
  std::cout << "\n[Real-time Response]" << std::endl;
  std::cout << full_response << std::flush;

  while (gen_tok_count++ < FLAGS_max_response &&
         llama_runtime.GetTokenIndex() < FLAGS_max_token_length) {
    timer_gen_token.Start();
    void* logits = llama_runtime.Run({output_token});
    timer_gen_token.End();

    prev_token = output_token;
    output_token = argmax(logits_type, logits, vocab_size);
    full_response_tokens.push_back(output_token);

    // Stop when output is EOS
    if (output_token == tokenizer->eos_tok()) {
      std::cout << "</eos>" << std::flush;
      break;
    }
    auto decode_res = tokenizer->decode(prev_token, output_token);
    ET_CHECK_OR_RETURN_ERROR(
        decode_res.ok(),
        InvalidState,
        "Tokenizer failed to decode generated token %lu",
        output_token);
    const std::string tok_str = std::move(decode_res.get());
    full_response += tok_str;
    std::cout << tok_str << std::flush;
  }

  std::cout << "\n\n[Generated Tokens]\n"
            << to_string(full_response_tokens) << std::endl;

  ET_LOG(
      Info,
      "Token generation speed: %f tok/s",
      gen_tok_count / gen_total_time_sec);

  return Error::Ok;
}

Error inference(
    LlamaRuntime& llama_runtime,
    const std::unique_ptr<Tokenizer>& tokenizer,
    const std::string& prompt) {
  // Tokenize input prompt
  auto encode_res = tokenizer->encode(prompt, kAddBos, kAddEos);
  ET_CHECK_OR_RETURN_ERROR(
      encode_res.ok(), InvalidState, "Tokenizer failed to encode prompt");
  const auto input_tokens = std::move(encode_res.get());

  std::cout << "\n[Input Prompt]\n" << prompt << std::endl;
  std::cout << "\n[Input Prompt Tokens]\n"
            << to_string(input_tokens) << std::endl;

  // Run prompt mode (pre-fill)
  auto prefill_res = digest_prompt(llama_runtime, tokenizer, input_tokens);
  ET_CHECK_OR_RETURN_ERROR(
      prefill_res.ok(), InvalidState, "Failed to digest prompt");
  const auto first_output_token = prefill_res.get();

  // run generation mode (decoding)
  return gen_response(llama_runtime, tokenizer, first_output_token);
}

Error export_pd_prefill(
    LlamaRuntime& llama_runtime,
    const std::unique_ptr<Tokenizer>& tokenizer,
    const std::string& prompt,
    const std::string& exportDir,
    std::vector<uint64_t> inputTokens) {
  if (inputTokens.empty()) {
    auto encode_res = tokenizer->encode(prompt, kAddBos, kAddEos);
    ET_CHECK_OR_RETURN_ERROR(
        encode_res.ok(), InvalidState, "Tokenizer failed to encode prompt");
    inputTokens = std::move(encode_res.get());
  }
  ET_CHECK_MSG(!inputTokens.empty(), "PD prefill prompt cannot be empty");
  ET_CHECK_MSG(
      inputTokens.size() <= llama_runtime.GetCacheLength(),
      "PD prompt has %zu tokens but MTK cache holds only %zu",
      inputTokens.size(),
      llama_runtime.GetCacheLength());

  if (!prompt.empty()) {
    std::cout << "\n[Input Prompt]\n" << prompt << std::endl;
  }
  std::cout << "\n[Input Prompt Tokens]\n"
            << to_string(inputTokens) << std::endl;

  PdKvExportPipeline pipeline(
      inputTokens.size(),
      FLAGS_num_layer,
      llama_runtime.GetNumKVHeads(),
      llama_runtime.GetCacheHeadDim(),
      FLAGS_pd_prefill_pipeline);
  auto prefill = digest_prompt(llama_runtime, tokenizer, inputTokens, &pipeline);
  ET_CHECK_OR_RETURN_ERROR(
      prefill.ok(), InvalidState, "Failed to run MTK PD prefill");
  pipeline.Finish();
  write_pd_handoff(
      exportDir, inputTokens, prefill.get(), llama_runtime, pipeline);
  return Error::Ok;
}

std::vector<uint64_t> load_pd_prompt_tokens(const std::string& path) {
  if (path.empty()) {
    return {};
  }
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  ET_CHECK_MSG(stream.good(), "Unable to open PD prompt tokens %s", path.c_str());
  const auto byteCount = static_cast<std::streamoff>(stream.tellg());
  ET_CHECK_MSG(
      byteCount > 0 && byteCount % static_cast<std::streamoff>(sizeof(uint64_t)) == 0,
      "PD prompt token file must contain non-empty uint64 values: %s",
      path.c_str());
  std::vector<uint64_t> tokens(
      static_cast<size_t>(byteCount) / sizeof(uint64_t));
  stream.seekg(0);
  stream.read(
      reinterpret_cast<char*>(tokens.data()),
      static_cast<std::streamsize>(tokens.size() * sizeof(uint64_t)));
  ET_CHECK_MSG(stream.good(), "Unable to read PD prompt tokens %s", path.c_str());
  return tokens;
}

Error run_mtk_ppl(
    LlamaRuntime& llama_runtime,
    const std::vector<uint64_t>& promptTokens,
    std::vector<uint64_t> scoreTokens,
    const std::string& outputPath) {
  ET_CHECK_MSG(!promptTokens.empty(), "MTK PPL prompt cannot be empty");
  ET_CHECK_MSG(!scoreTokens.empty(), "MTK PPL continuation cannot be empty");
  ET_CHECK_MSG(
      promptTokens.size() <= llama_runtime.GetCacheLength(),
      "MTK PPL prompt has %zu tokens but cache holds only %zu",
      promptTokens.size(),
      llama_runtime.GetCacheLength());
  ET_CHECK_MSG(
      scoreTokens.size() >= 2,
      "MTK PPL continuation requires a predictor and at least one target");
  const size_t availableScoredTokens = scoreTokens.size() - 1;
  const size_t scoredTokenCount = FLAGS_mtk_ppl_max_tokens > 0
      ? std::min(availableScoredTokens, FLAGS_mtk_ppl_max_tokens)
      : availableScoredTokens;
  ET_CHECK_MSG(
      promptTokens.size() + scoredTokenCount <= llama_runtime.GetCacheLength(),
      "MTK PPL range exceeds context: prompt=%zu scored=%zu cache=%zu",
      promptTokens.size(),
      scoredTokenCount,
      llama_runtime.GetCacheLength());

  const size_t batchSize = llama_runtime.GetTokenBatchSize();
  size_t cursor = 0;
  const auto prefillStart = std::chrono::steady_clock::now();
  while (cursor < promptTokens.size()) {
    const size_t remaining = promptTokens.size() - cursor;
    size_t count = remaining % batchSize;
    if (count == 0) {
      count = std::min(batchSize, remaining);
    }
    llama_runtime.Run(
        std::vector<uint64_t>(
            promptTokens.begin() + cursor,
            promptTokens.begin() + cursor + count),
        true);
    cursor += count;
  }
  const double prefillSeconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - prefillStart)
          .count();

  const auto logitsType = llama_runtime.GetModelOptions().model_output_type;
  ET_CHECK_MSG(
      logitsType == example::llm_helper::LLMType::FP32,
      "Direct MTK PPL currently requires FP32 output logits, got %s",
      getLLMTypeName(logitsType));
  const size_t vocabSize = llama_runtime.GetModelOptions().vocab_size;
  const size_t logitShardCount =
      llama_runtime.GetModelOptions().logit_shard_count;

  size_t sanitizedNonfiniteLogits = 0;
  auto tokenNll = [vocabSize, &sanitizedNonfiniteLogits](
                      const float* values, const uint64_t target) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t index = 0; index < vocabSize; ++index) {
      if (std::isfinite(values[index])) {
        maximum = std::max(maximum, values[index]);
      } else {
        ++sanitizedNonfiniteLogits;
      }
    }
    if (!std::isfinite(values[target]) || !std::isfinite(maximum)) {
      return std::numeric_limits<double>::infinity();
    }
    double sum = 0.0;
    for (size_t index = 0; index < vocabSize; ++index) {
      if (std::isfinite(values[index])) {
        sum += std::exp(static_cast<double>(values[index] - maximum));
      }
    }
    return std::log(sum) + static_cast<double>(maximum) - values[target];
  };

  double totalNll = 0.0;
  const auto scoringStart = std::chrono::steady_clock::now();
  bool finiteNll = true;
  const auto scoreOne = [&](
                            const float* values,
                            const size_t scoreIndex,
                            const size_t targetIndex) {
    const uint64_t target = scoreTokens[targetIndex];
    ET_CHECK_MSG(
        target < vocabSize,
        "MTK PPL target token %llu exceeds vocab size %zu",
        static_cast<unsigned long long>(target),
        vocabSize);
    const double nll = tokenNll(values, target);
    if (!std::isfinite(nll)) {
      ET_LOG(Error, "Non-finite MTK PPL NLL at score %zu", scoreIndex);
      finiteNll = false;
      return;
    }
    totalNll += nll;
    if ((scoreIndex + 1) % 64 == 0 || scoreIndex + 1 == scoredTokenCount) {
      ET_LOG(
          Info,
          "MTK PPL progress: scored=%zu/%zu running_ppl=%.6f",
          scoreIndex + 1,
          scoredTokenCount,
          std::exp(totalNll / static_cast<double>(scoreIndex + 1)));
    }
  };

  if (logitShardCount == 1) {
    // Match QNN's target convention: continuation token 0 is the first
    // predictor and continuation token 1 is the first scored target. Keep
    // using the causal prefill method for legacy single-output models.
    size_t feedCursor = 0;
    while (finiteNll && feedCursor < scoredTokenCount) {
      // Feed real look-ahead tokens when available so the exported 128-token
      // method always executes a full batch. Causality means those extra rows
      // do not alter the earlier logits included in the PPL calculation.
      const size_t count = std::min(batchSize, scoreTokens.size() - feedCursor);
      const size_t rowsToScore =
          std::min(count, scoredTokenCount - feedCursor);
      const std::vector<uint64_t> input(
          scoreTokens.begin() + feedCursor,
          scoreTokens.begin() + feedCursor + count);
      const float* batchLogits =
          static_cast<const float*>(llama_runtime.Run(input, false));
      for (size_t row = 0; row < rowsToScore; ++row) {
        scoreOne(
            batchLogits + row * vocabSize,
            feedCursor + row,
            feedCursor + row + 1);
        if (!finiteNll) {
          break;
        }
      }
      feedCursor += rowsToScore;
    }
  } else {
    // Use the same AR-1 graph as normal generation. The separately quantized
    // 128-token prefill graph is causal but is not numerically identical to
    // the decode graph, so it must not be substituted in an accuracy metric.
    for (size_t scoreIndex = 0;
         finiteNll && scoreIndex < scoredTokenCount;
         ++scoreIndex) {
      const float* stepLogits = static_cast<const float*>(llama_runtime.Run(
          std::vector<uint64_t>{scoreTokens[scoreIndex]}, true));
      scoreOne(stepLogits, scoreIndex, scoreIndex + 1);
    }
  }
  if (!finiteNll) {
    return Error::InvalidState;
  }
  const double scoringSeconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - scoringStart)
          .count();
  const double meanNll = totalNll / static_cast<double>(scoredTokenCount);
  const double perplexity = std::exp(meanNll);
  const double tokensPerSecond = scoredTokenCount / scoringSeconds;

  std::ofstream output(outputPath);
  ET_CHECK_MSG(output.good(), "Unable to open MTK PPL output %s", outputPath.c_str());
  output << std::setprecision(12)
         << "{\n"
         << "  \"backend\": \"executorch_mtk\",\n"
         << "  \"mode\": \""
         << (logitShardCount == 1
                 ? "batched_prefill_teacher_forcing"
                 : "sharded_decode_teacher_forcing")
         << "\",\n"
         << "  \"prompt_tokens\": " << promptTokens.size() << ",\n"
         << "  \"first_predictor_index\": 0,\n"
         << "  \"first_target_index\": 1,\n"
         << "  \"scored_tokens\": " << scoredTokenCount << ",\n"
         << "  \"total_nll\": " << totalNll << ",\n"
         << "  \"mean_nll\": " << meanNll << ",\n"
         << "  \"perplexity\": " << perplexity << ",\n"
         << "  \"sanitized_nonfinite_logits\": "
         << sanitizedNonfiniteLogits << ",\n"
         << "  \"prefill_seconds\": " << prefillSeconds << ",\n"
         << "  \"scoring_seconds\": " << scoringSeconds << ",\n"
         << "  \"tokens_per_second\": " << tokensPerSecond << "\n"
         << "}\n";
  output.flush();
  ET_CHECK_MSG(output.good(), "Unable to write MTK PPL output %s", outputPath.c_str());
  ET_LOG(
      Info,
      "Direct MTK PPL complete: prompt=%zu scored=%zu total_nll=%.6f "
      "ppl=%.6f prefill_s=%.6f scoring_s=%.6f tok/s=%.3f",
      promptTokens.size(),
      scoredTokenCount,
      totalNll,
      perplexity,
      prefillSeconds,
      scoringSeconds,
      tokensPerSecond);
  if (sanitizedNonfiniteLogits > 0) {
    ET_LOG(
        Error,
        "MTK PPL sanitized %zu non-target non-finite logits as negative infinity",
        sanitizedNonfiniteLogits);
  }
  return Error::Ok;
}

std::unique_ptr<Tokenizer> load_tokenizer() {
#ifdef MTK_LLAMA_PD_JOINT
  ET_CHECK_MSG(
      false,
      "Joint runner supports only stage-major mode with pre-tokenized input");
  return nullptr;
#else
  std::unique_ptr<Tokenizer> tokenizer;
  if (FLAGS_tokenizer_type == "bpe") {
    tokenizer = std::make_unique<Llama2cTokenizer>();
  } else if (FLAGS_tokenizer_type == "tiktoken") {
    tokenizer = example::get_tiktoken_for_llama();
  } else if (FLAGS_tokenizer_type == "hf") {
    tokenizer = std::make_unique<HFTokenizer>();
  }
  ET_CHECK_MSG(
      tokenizer, "Invalid tokenizer type: %s", FLAGS_tokenizer_type.c_str());
  tokenizer->load(FLAGS_tokenizer_path);
  return tokenizer;
#endif
}

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 1) {
    std::string msg = "Extra commandline args:";
    for (int i = 1 /* skip argv[0] (program name) */; i < argc; i++) {
      msg += std::string(" ") + argv[i];
    }
    ET_LOG(Error, "%s", msg.c_str());
    return 1;
  }

  LlamaModelOptions model_options = get_model_options();
  LlamaModelPaths model_paths = get_model_paths();

  const bool stageMajor =
      !FLAGS_pd_stage_major_stripped_pte_paths.empty() ||
      !FLAGS_pd_stage_major_index_paths.empty() ||
      !FLAGS_pd_stage_major_weight_paths.empty();
  if (stageMajor) {
#ifdef MTK_LLAMA_PD_JOINT
    const bool ggufBacked = !FLAGS_pd_joint_gguf_path.empty();
#else
    const bool ggufBacked = false;
#endif
    ET_CHECK_MSG(
        !FLAGS_pd_stage_major_stripped_pte_paths.empty() &&
            !FLAGS_pd_stage_major_index_paths.empty() &&
            (ggufBacked || !FLAGS_pd_stage_major_weight_paths.empty()),
        "Stage-major requires stripped/index lists and either a shared GGUF "
        "or the legacy weights list");
#ifndef MTK_LLAMA_PD_JOINT
    ET_CHECK_MSG(
        !FLAGS_pd_export_dir.empty(),
        "Stage-major Prefill requires --pd_export_dir");
#else
    ET_CHECK_MSG(
        !FLAGS_pd_export_dir.empty() || !FLAGS_pd_joint_gguf_path.empty(),
        "Stage-major Prefill requires --pd_export_dir or --pd_joint_gguf_path");
#endif
    ET_CHECK_MSG(
        FLAGS_logit_shard_count == 1,
        "No-lm-head stage-major Prefill requires --logit_shard_count=1");
    const auto stripped = split(FLAGS_pd_stage_major_stripped_pte_paths, ',');
    const auto indexes = split(FLAGS_pd_stage_major_index_paths, ',');
    const auto weights = FLAGS_pd_stage_major_weight_paths.empty()
        ? std::vector<std::string>{}
        : split(FLAGS_pd_stage_major_weight_paths, ',');
    ET_CHECK_MSG(
        !stripped.empty() && stripped.size() == indexes.size() &&
            (ggufBacked || stripped.size() == weights.size()),
        "Stage-major path-list sizes differ");
    std::vector<MtkStrippedChunkPaths> chunks;
    chunks.reserve(stripped.size());
    for (size_t index = 0; index < stripped.size(); ++index) {
      chunks.push_back({
          stripped[index],
          indexes[index],
          ggufBacked ? std::string{} : weights[index]});
    }

#ifdef MTK_LLAMA_PD_JOINT
    const std::string stageGgufPath = ggufBacked ? FLAGS_pd_joint_gguf_path : "";
#else
    const std::string stageGgufPath;
#endif
    std::shared_ptr<MtkGgufWeightSource> sharedGgufSource;
#ifdef MTK_LLAMA_PD_JOINT
    if (ggufBacked) {
      const auto ramStoreStart = std::chrono::steady_clock::now();
      sharedGgufSource = LoadMtkGgufWeightSourceIntoRam(stageGgufPath);
      ET_LOG(
          Info,
          "MTK PD GGUF RAM store ready: bytes=%zu load_ms=%.3f "
          "shared_prefill_decode_pointer=%p timing_scope=bootstrap",
          MtkGgufWeightSourceSize(sharedGgufSource),
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - ramStoreStart)
              .count(),
          MtkGgufWeightSourceData(sharedGgufSource));
    }
#endif


    std::unique_ptr<QnnKvAbi> qnnKvAbi;
    if (!FLAGS_pd_qnn_kv_abi_path.empty()) {
      qnnKvAbi = std::make_unique<QnnKvAbi>(FLAGS_pd_qnn_kv_abi_path);
      ET_CHECK_MSG(
          qnnKvAbi->NumLayers() == FLAGS_num_layer &&
              qnnKvAbi->HeadDim() == FLAGS_head_dim,
          "QNN KV ABI dimensions do not match the MTK model");
      ET_LOG(
          Info,
          "MTK PD will emit QNN U8 KV: abi=%s layers=%zu heads=%zu dim=%zu",
          FLAGS_pd_qnn_kv_abi_path.c_str(),
          qnnKvAbi->NumLayers(),
          qnnKvAbi->NumHeads(),
          qnnKvAbi->HeadDim());
    }
    std::string stageEmbeddingPath = FLAGS_token_embedding_path;
#ifdef MTK_LLAMA_PD_JOINT
    if (!FLAGS_pd_joint_disk_embedding_path.empty()) {
      stageEmbeddingPath = FLAGS_pd_joint_disk_embedding_path;
      ET_LOG(
          Info,
          "MTK PD prefill and QNN decode share FP16 embedding: %s",
          stageEmbeddingPath.c_str());
    }
#endif
    MtkStageMajorPrefillSession stageSession(
        model_options,
        stageEmbeddingPath,
        chunks,
        stageGgufPath,
        sharedGgufSource,
        false,
        FLAGS_pd_stage_major_three_stage,
        FLAGS_pd_stage_major_async_release,
        FLAGS_pd_stage_major_persistent_chunk0,
        FLAGS_pd_stage_major_detach_pte_after_load,
        qnnKvAbi.get());
    const auto overlapStart = std::chrono::steady_clock::now();
    auto prewarmFuture = std::async(
        std::launch::async, [&stageSession]() { stageSession.Prepare(); });
#ifdef MTK_LLAMA_PD_JOINT
    auto decodeInitFuture = std::async(
        std::launch::async,
        [sharedGgufSource]() { return prepare_joint_decode(sharedGgufSource); });
#endif
    const auto requestInitStart = std::chrono::steady_clock::now();
#ifndef MTK_LLAMA_PD_JOINT
    const auto tokenizer = load_tokenizer();
#endif
    std::vector<uint64_t> inputTokens =
        load_pd_prompt_tokens(FLAGS_pd_prompt_tokens_path);
    std::string prompt;
    if (inputTokens.empty()) {
#ifdef MTK_LLAMA_PD_JOINT
      ET_CHECK_MSG(
          false,
          "Joint runner requires --pd_prompt_tokens_path to avoid linking a "
          "second tokenizer/Unicode runtime");
#else
      ET_CHECK_MSG(!FLAGS_prompt_file.empty(), "No stage-major prompt provided");
      prompt = read_file(FLAGS_prompt_file);
      auto encoded = tokenizer->encode(prompt, kAddBos, kAddEos);
      ET_CHECK_MSG(encoded.ok(), "Tokenizer failed to encode stage-major prompt");
      inputTokens = std::move(encoded.get());
#endif
    }
    const double requestInitMs =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - requestInitStart)
            .count();
    prewarmFuture.get();
#ifdef MTK_LLAMA_PD_JOINT
    auto jointDecode = decodeInitFuture.get();
#endif
    const double overlapWallMs =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - overlapStart)
            .count();
    ET_LOG(
        Info,
        "MTK chunk-0/request initialization overlap: request_init_ms=%.3f "
        "wall_ms=%.3f persistent_chunk0=%d"
#ifdef MTK_LLAMA_PD_JOINT
        " joint_decode_init_ms=%.3f"
#endif
        ,
        requestInitMs,
        overlapWallMs,
        static_cast<int>(FLAGS_pd_stage_major_persistent_chunk0)
#ifdef MTK_LLAMA_PD_JOINT
        , jointDecode->initializationMs
#endif
        );
    ET_CHECK_MSG(
        inputTokens.size() >= 2,
        "No-output stage-major Prefill requires at least two prompt tokens");
    const uint64_t promptTail = inputTokens.back();
    std::vector<uint64_t> cachedPromptTokens(
        inputTokens.begin(), inputTokens.end() - 1);
    ET_LOG(
        Info,
        "Begin MTK low-memory stage-major Prefill: chunks=%zu cached_tokens=%zu "
        "prompt_tail=%llu original_tokens=%zu",
        chunks.size(),
        cachedPromptTokens.size(),
        static_cast<unsigned long long>(promptTail),
        inputTokens.size());
    ET_CHECK_MSG(
        FLAGS_pd_stage_major_session_repeats > 0,
        "--pd_stage_major_session_repeats must be positive");
    for (size_t requestIndex = 0;
         requestIndex < FLAGS_pd_stage_major_session_repeats;
         ++requestIndex) {
      ET_LOG(
          Info,
          "MTK stage-major session request begin: index=%zu total=%llu",
          requestIndex,
          static_cast<unsigned long long>(
              FLAGS_pd_stage_major_session_repeats));
#ifdef MTK_LLAMA_PD_JOINT
      ET_LOG(
          Info,
          "MTK joint GGUF RAM store retained for Prefill: request=%zu "
          "model_bytes=%zu shared_pointer=%p",
          requestIndex,
          MtkGgufWeightSourceSize(jointDecode->model),
          MtkGgufWeightSourceData(jointDecode->model));
#endif
      const auto result = stageSession.Run(cachedPromptTokens);
      if (qnnKvAbi) {
        ET_LOG(
            Info,
            "MTK PD QNN U8 KV ready: values=%zu worker_ms=%.3f "
            "boundary_finalize_ms=%.3f non_finite=%zu code0=%zu code255=%zu",
            result.qnnKvStats.values,
            result.kvHandoffWorkerMs,
            result.kvHandoffBoundaryMs,
            result.qnnKvStats.nonFinite,
            result.qnnKvStats.codeZero,
            result.qnnKvStats.code255);
      }
      if (!FLAGS_pd_export_dir.empty()) {
        const std::string requestExportDir =
            FLAGS_pd_stage_major_session_repeats == 1
            ? FLAGS_pd_export_dir
            : FLAGS_pd_export_dir + "/request_" +
                std::to_string(requestIndex);
        std::filesystem::create_directories(requestExportDir);
        std::vector<uint8_t> exportQnnKv;
        if (qnnKvAbi) {
          exportQnnKv.assign(
              result.qnnDirectHandoff.begin() +
                  result.qnnDirectKvOffsetBytes,
              result.qnnDirectHandoff.end());
        }
        write_stage_major_pd_handoff(
            requestExportDir,
            cachedPromptTokens,
            promptTail,
            result,
            qnnKvAbi ? &exportQnnKv : nullptr);
      }
#ifdef MTK_LLAMA_PD_JOINT
      llama_pd_inprocess_result decodeResult{};
      llama_pd_inprocess_request request{};
      request.model_data = MtkGgufWeightSourceData(jointDecode->model);
      request.model_size = MtkGgufWeightSourceSize(jointDecode->model);
      request.prompt_length = static_cast<int32_t>(cachedPromptTokens.size());
      request.num_layers = static_cast<int32_t>(FLAGS_num_layer);
      request.num_kv_heads = static_cast<int32_t>(result.numKvHeads);
      request.head_dim = static_cast<int32_t>(result.headDim);
      request.first_token = static_cast<int32_t>(promptTail);
      request.first_token_is_prompt_tail = true;
      request.result = &decodeResult;
      if (qnnKvAbi) {
        request.handoff_data = result.qnnDirectHandoff.data();
        request.handoff_size = result.qnnDirectHandoff.size();
      } else {
        request.prompt_tokens = cachedPromptTokens.data();
        request.kv_fp16 = result.canonicalKv.data();
        request.kv_fp16_values = result.canonicalKv.size();
      }
      const int decodeStatus = llama_pd_inprocess_runtime_run(
          jointDecode->runtime, &request);
      ET_CHECK_MSG(decodeStatus == 0, "Joint MTK/llama.cpp Decode failed");
      ET_LOG(
          Info,
          "MTK joint Decode request=%zu initialization_once_ms=%.3f "
          "boundary_ms=%.3f generation_ms=%.3f generated_tokens=%d",
          requestIndex,
          decodeResult.initialization_ms,
          decodeResult.boundary_ms,
          decodeResult.generation_ms,
          decodeResult.generated_tokens);
#endif
    }
    std::cout << "\n[PD Stage-Major Prefill Export Complete]" << std::endl;
    return 0;
  }

  if (model_paths.prompt_model_paths.empty() &&
      model_paths.model_package_paths.empty()) {
    model_options.prompt_token_batch_size = 1;
    ET_LOG(
        Info,
        "No prompt model paths provided, overriding prompt_token_batch_size to 1");
  }

  // Prepare timers
  Timer timer_init(
      [](const auto elapsed_sec) { ET_LOG(Info, "Model initialized."); });
  Timer timer_release(
      [](const auto elapsed_sec) { ET_LOG(Info, "Model released."); });

  LlamaRuntime llama_runtime;

  // Initialize model
  ET_LOG(Info, "Begin model loading.");
  timer_init.Start();
  const auto tokenizer = load_tokenizer();
  llama_runtime.Initialize(model_options, model_paths);
  timer_init.End();

  // Run model
  const bool mtkPplMode = !FLAGS_mtk_ppl_output.empty();
  ET_CHECK_MSG(
      !FLAGS_prompt_file.empty() || !FLAGS_pd_prompt_tokens_path.empty() ||
          mtkPplMode,
      "No prompt file, PD prompt tokens, or MTK PPL request provided.");
  ET_CHECK_MSG(
      FLAGS_pd_prompt_tokens_path.empty() || !FLAGS_pd_export_dir.empty(),
      "--pd_prompt_tokens_path requires --pd_export_dir");
  ET_CHECK_MSG(
      FLAGS_prompt_file.empty() || FLAGS_pd_prompt_tokens_path.empty(),
      "--prompt_file and --pd_prompt_tokens_path are mutually exclusive");
  std::string prompt;
  if (!FLAGS_prompt_file.empty()) {
    prompt = read_file(FLAGS_prompt_file);
  }
  ET_CHECK_MSG(
      FLAGS_pd_export_dir.empty() || !FLAGS_prefill_only,
      "--pd_export_dir and --prefill_only are mutually exclusive");
  if (mtkPplMode) {
    ET_CHECK_MSG(
        !FLAGS_mtk_ppl_prompt_tokens_path.empty() &&
            !FLAGS_mtk_ppl_tokens_path.empty(),
        "MTK PPL requires --mtk_ppl_prompt_tokens_path and --mtk_ppl_tokens_path");
    ET_CHECK_MSG(
        FLAGS_prompt_file.empty() && FLAGS_pd_prompt_tokens_path.empty() &&
            FLAGS_pd_export_dir.empty(),
        "MTK PPL mode cannot be combined with prompt or PD export modes");
    const auto status = run_mtk_ppl(
        llama_runtime,
        load_pd_prompt_tokens(FLAGS_mtk_ppl_prompt_tokens_path),
        load_pd_prompt_tokens(FLAGS_mtk_ppl_tokens_path),
        FLAGS_mtk_ppl_output);
    if (status != Error::Ok) {
      ET_LOG(
          Error,
          "Direct MTK PPL failed with status 0x%x",
          static_cast<unsigned int>(status));
      llama_runtime.Release();
      return 1;
    }
  } else if (!FLAGS_pd_export_dir.empty()) {
    const auto status = export_pd_prefill(
        llama_runtime,
        tokenizer,
        prompt,
        FLAGS_pd_export_dir,
        load_pd_prompt_tokens(FLAGS_pd_prompt_tokens_path));
    ET_CHECK_MSG(
        status == Error::Ok,
        "MTK PD prefill export failed with status 0x%x",
        static_cast<unsigned int>(status));
    std::cout << "\n[PD Prefill Export Complete]" << std::endl;
  } else if (FLAGS_prefill_only) {
    auto encode_res = tokenizer->encode(prompt, kAddBos, kAddEos);
    ET_CHECK_MSG(encode_res.ok(), "Tokenizer failed to encode prompt");
    const auto input_tokens = std::move(encode_res.get());
    std::cout << "\n[Input Prompt Tokens]\n"
              << to_string(input_tokens) << std::endl;
    llama_runtime.Run(input_tokens);
    std::cout << "\n[Prefill Only Complete]" << std::endl;
  } else {
    inference(llama_runtime, tokenizer, prompt);
  }

  // Release model
  timer_release.Start();
  llama_runtime.Release();
  timer_release.End();

  return 0;
}
