#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorchBackend.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/pd_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/pte_rebuilder.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>
#ifdef QNN_LLAMA_PD_JOINT
#include "llama.h"
#include "pd_cli_inprocess.h"
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
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
DEFINE_bool(
    model_ram_store,
    true,
    "Load the shared Prefill/Decode GGUF into a sealed memfd before E2E timing.");
DEFINE_bool(
    model_anonymous_buffer,
    false,
    "Load the shared Prefill/Decode GGUF into one read-only anonymous memory "
    "buffer consumed directly by both PTE rebuild and joint Decode.");
DEFINE_bool(
    model_residency_probe,
    false,
    "Diagnostic: report mincore residency of the shared GGUF buffer at PD stages.");
DEFINE_string(
    model_residency_profile_path,
    "",
    "Write periodic shared-model residency and memory-pressure samples to CSV.");
DEFINE_int32(
    model_residency_profile_interval_ms,
    250,
    "Sampling interval for --model_residency_profile_path (minimum 50 ms).");
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
DEFINE_string(
    session_prompts_path,
    "",
    "Optional UTF-8 text file containing one prompt per non-empty line. "
    "All prompts run in one process and reuse Prefill/Decode runtime state.");
DEFINE_string(system_prompt, "", "Optional system prompt.");
DEFINE_string(
    wikitext_path,
    "",
    "Path to a local WikiText text file. When provided, the runner computes WikiText perplexity and skips PD handoff/decode.");
DEFINE_int32(
    wikitext_max_tokens,
    0,
    "Maximum number of WikiText target tokens to score. Non-positive values mean score all available tokens.");
DEFINE_int32(
    wikitext_start_token,
    0,
    "Zero-based predictor-token row at which WikiText scoring starts.");
DEFINE_double(
    wikitext_logits_scale,
    0.0,
    "Explicit QNN logits scale for manifest-only WikiText PPL.");
DEFINE_int32(
    wikitext_logits_zero_point,
    0,
    "Explicit QNN logits zero point for manifest-only WikiText PPL.");
DEFINE_string(
    attention_sink_rope_path,
    "",
    "Attention sink rope PTE. Not supported in PD v1 export.");
DEFINE_int32(
    seq_len,
    4096,
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
    "Run QNN Prefill only, release its in-memory handoff, and skip Decode.");
DEFINE_bool(
    prefill_shard_pipeline,
    false,
    "Preload stripped shard inputs and rebuild one shard ahead on a CPU worker while QNN executes the current shard.");
DEFINE_bool(
    prefill_shard_pipeline_3stage,
    true,
    "Stage-major pipeline: rebuild(i+2), QNN load(i+1), and execute(i) on separate stages.");
DEFINE_bool(
    prefill_qnn_backend_prewarm,
    true,
    "Prewarm the process-lifetime QNN backend/device from qnn_compile_spec_hex in the shard manifest without creating a QNN context or graph.");
DEFINE_bool(
    prefill_persistent_shard0,
    true,
    "Rebuild and QNN-load prefill shard 0 during runner preparation, retain "
    "its context through the request, and start the three-stage pipeline at "
    "shard 1.");
DEFINE_bool(
    prefill_release_pte_backing_after_load,
    false,
    "Experimental: after each rebuilt Prefill shard finishes QNN method load "
    "and output binding, return its PTE backing to the rebuild pool before "
    "Execute. Later rebuilds may overwrite it; use only for ownership and "
    "correctness validation.");
DEFINE_bool(
    prefill_detach_shard0_qnn_after_load,
    false,
    "Experimental: after persistent shard0 QNN Load, retain a detached QNN "
    "execution shell and destroy its ExecuTorch Module plus complete PTE "
    "backing before Execute.");
DEFINE_bool(
    prefill_detach_all_qnn_after_load,
    true,
    "At the end of every Prefill shard Load, retain only a "
    "detached QNN execution shell, destroy Module/Method, return the complete "
    "PTE backing, and limit the three-stage rebuild pool to two buffers.");
DEFINE_bool(
    prefill_release_stripped_pte_after_rebuild,
    true,
    "Single-request default: release each stripped PTE immediately "
    "after its rebuilt PTE has been produced.");
DEFINE_bool(
    decode_stream_sidecar_during_prefill,
    true,
    "Joint-PD default: after Decode warmup discard the physical pages of each "
    "stable per-shard sidecar malloc buffer and read that shard from the V5 "
    "payload asynchronously after its Prefill QNN Execute.");
DEFINE_bool(
    prefill_release_shard0_after_execute,
    true,
    "Release an early-prepared shard0 immediately after its only QNN execute "
    "so unused rebuilt-PTE pages do not remain resident during CPU Decode.");
DEFINE_bool(
    prefill_unload_shard0_method_after_execute,
    false,
    "Diagnostic: unload shard0's QNN method/graph after execute while retaining "
    "its Module and rebuilt PTE backing.");
DEFINE_bool(
    prefill_destroy_shard0_module_keep_pte_after_execute,
    false,
    "Diagnostic: destroy shard0's complete Module after execute while retaining "
    "its rebuilt PTE backing allocation.");
DEFINE_bool(
    prefill_discard_shard0_pte_pages_after_execute,
    false,
    "Diagnostic: after destroying shard0 Module, MADV_DONTNEED the retained "
    "PTE backing pages while keeping the allocation and capacity.");
DEFINE_bool(
    prefill_release_htp_vote_before_decode,
    false,
    "Diagnostic: down-vote HTP before CPU Decode while retaining backend/device.");
DEFINE_bool(
    prefill_release_all_before_decode,
    false,
    "Legacy diagnostic: tear down all Prefill and process-global QNN resources "
    "before Decode. This is not the seamless joint-PD production lifecycle.");
DEFINE_int32(
    decode_cooldown_ms,
    0,
    "Diagnostic idle interval after Prefill resource handling and before Decode handoff.");
DEFINE_bool(
    decode_pretouch_model,
    false,
    "Diagnostic: read one byte from every shared GGUF page before Decode.");
DEFINE_string(
    decode_sidecar_reread_path,
    "",
    "Diagnostic: after Prefill and immediately before the resident Decode "
    "handoff, sequentially read the complete Decode sidecar through a small "
    "temporary buffer. This warms file cache without retaining a second copy.");
DEFINE_bool(
    decode_sidecar_pretouch_mapping,
    false,
    "Diagnostic: after the optional boundary reread, touch every page of the "
    "existing Decode sidecar mmap so its VMA page tables are populated before "
    "the first post-Prefill Decode call.");
DEFINE_string(
    prefill_etdump_dir,
    "",
    "Write one selected prefill shard's ETDump and intermediate tensor buffer here. "
    "The shard PTE must be exported with --dump_intermediate_outputs.");
DEFINE_int32(
    prefill_etdump_shard,
    0,
    "Zero-based prefill shard index to capture when --prefill_etdump_dir is set.");
DEFINE_int64(
    prefill_etdump_debug_buffer_bytes,
    536870912,
    "Bytes reserved for the selected shard's ETDump intermediate tensor buffer.");
DEFINE_bool(
    prefill_shard_stage_major,
    true,
    "For static Qwen3 shards, execute every AR block through one shard before advancing to the next shard.");
DEFINE_bool(
    prefill_gguf_relayout_blocks,
    false,
    "Before prefill rebuild, copy raw GPTQ2_32 64-row blocks into contiguous "
    "PTE-record order. Layout-only benchmark; increases resident memory.");
DEFINE_bool(
    prefill_gguf_relayout_gs32_source,
    false,
    "Before prefill rebuild, copy raw GPTQ2_32 bytes into the GS32 source-read "
    "order. Preserves raw qbytes and scale/zero_bias fields; increases resident memory.");
DEFINE_bool(
    prefill_no_output,
    false,
    "The prefill PTE has no logits output; export the last prompt token as the llama.cpp bridge token.");
DEFINE_bool(
    prefill_force_logits,
    false,
    "Override the shard manifest and use prefill logits to select the first decode token. "
    "Use only with shard PTEs that retain the logits output.");
DEFINE_bool(
    prefill_separate_embed,
    false,
    "The prefill PTE takes hidden_states from a separate embedding matrix.");
DEFINE_string(
    prefill_embedding_matrix_path,
    "",
    "Path to separate_embed_matrix.bin when --prefill_separate_embed=true.");
DEFINE_bool(
    prefill_embedding_resident,
    false,
    "Keep separate embedding matrix resident in memory; false uses row-on-demand file reads.");
DEFINE_string(
    llama_pd_cli_path,
    "",
    "Path to the llama-pd-cli executable used for decode. Required unless --prefill_only=true.");
DEFINE_string(
    decode_gguf_path,
    "",
    "Path to the decode-side GGUF model. When omitted, --gguf_model_path is reused, then --tmac_model_path.");
DEFINE_bool(
    decode_use_prefill_embedding,
    true,
    "Pass the separate Prefill embedding matrix to Decode as --pd-disk-embedding. "
    "Disable when the Decode GGUF contains its own embedding tensor.");
DEFINE_bool(
    decode_native_compare,
    false,
    "Compare imported handoff logits against native Decode prompt prefill.");
DEFINE_bool(
    decode_defer_runtime_until_after_prefill,
    false,
    "Legacy diagnostic: create and warm Decode after Prefill. Production joint "
    "PD initializes and warms Decode before Prefill for a seamless handoff.");
DEFINE_bool(
    decode_stage_model_only_before_prefill,
    false,
    "Experimental: load model-only Decode state before Prefill while deferring "
    "QNN metadata, context, KV, TG reserve, and warmup to the handoff boundary.");
DEFINE_int32(
    decode_n_predict,
    128,
    "Number of tokens to generate after importing the PD handoff.");
DEFINE_int32(
    decode_threads,
    6,
    "Decode-side thread count; six threads is the validated Meizu 21 default.");
DEFINE_int32(
    decode_ctx,
    4096,
    "Decode-side llama.cpp context size matching the production Prefill export.");
DEFINE_double(
    decode_temp,
    0.0f,
    "Decode-side sampling temperature.");
DEFINE_int32(
    decode_ngl,
    0,
    "Decode-side number of offloaded layers (-ngl).");
DEFINE_string(
    decode_ppl_tokens_path,
    "",
    "Raw uint64 continuation-token file for teacher-forced PD WikiPPL.");
DEFINE_string(
    decode_ppl_output_path,
    "",
    "Output file written by llama-pd-cli in teacher-forced PD WikiPPL mode.");
DEFINE_int32(
    decode_ppl_max_tokens,
    0,
    "Maximum number of teacher-forced PD target tokens; zero scores the whole continuation file.");
DEFINE_bool(
    decode_import_ro,
    false,
    "Only validate/import the PD handoff in llama.cpp without continuing decode.");

namespace fs = std::filesystem;

namespace {

using SteadyClock = std::chrono::steady_clock;

std::string g_model_ram_fd_spec;

double elapsed_ms(
    SteadyClock::time_point start,
    SteadyClock::time_point end = SteadyClock::now()) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

struct ProcessMemorySnapshot {
  uint64_t rss_bytes{0};
  uint64_t hwm_bytes{0};
  uint64_t pss_bytes{0};
  uint64_t minor_faults{0};
  uint64_t major_faults{0};
};

uint64_t read_proc_status_bytes(const std::string& path, const char* field) {
  std::ifstream status(path);
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

std::pair<uint64_t, uint64_t> read_proc_faults(const std::string& path) {
  std::ifstream stat(path);
  std::string line;
  std::getline(stat, line);
  const size_t command_end = line.rfind(')');
  if (command_end == std::string::npos || command_end + 2 >= line.size()) {
    return {};
  }
  std::istringstream fields(line.substr(command_end + 2));
  char state = 0;
  uint64_t ignored = 0;
  uint64_t minor_faults = 0;
  uint64_t major_faults = 0;
  fields >> state;
  for (int field = 4; field <= 9; ++field) {
    fields >> ignored;
  }
  fields >> minor_faults >> ignored >> major_faults;
  return fields ? std::make_pair(minor_faults, major_faults)
                : std::pair<uint64_t, uint64_t>{};
}

ProcessMemorySnapshot process_memory_snapshot() {
  const auto faults = read_proc_faults("/proc/self/stat");
  return {
      read_proc_status_bytes("/proc/self/status", "VmRSS:"),
      read_proc_status_bytes("/proc/self/status", "VmHWM:"),
      read_proc_status_bytes("/proc/self/smaps_rollup", "Pss:"),
      faults.first,
      faults.second,
  };
}

ProcessMemorySnapshot process_memory_snapshot(pid_t pid) {
  const std::string proc = "/proc/" + std::to_string(pid);
  const auto faults = read_proc_faults(proc + "/stat");
  return {
      read_proc_status_bytes(proc + "/status", "VmRSS:"),
      read_proc_status_bytes(proc + "/status", "VmHWM:"),
      read_proc_status_bytes(proc + "/smaps_rollup", "Pss:"),
      faults.first,
      faults.second,
  };
}

ProcessMemorySnapshot process_memory_status_snapshot() {
  const auto faults = read_proc_faults("/proc/self/stat");
  return {
      read_proc_status_bytes("/proc/self/status", "VmRSS:"),
      read_proc_status_bytes("/proc/self/status", "VmHWM:"),
      0,
      faults.first,
      faults.second,
  };
}

double bytes_to_mib(uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void reread_decode_sidecar_before_handoff(const std::string& path) {
  if (path.empty()) {
    return;
  }
  constexpr size_t kReadBufferBytes = 4 * 1024 * 1024;
  const auto start = SteadyClock::now();
  const auto before = process_memory_status_snapshot();
  std::ifstream input(path, std::ios::binary);
  ET_CHECK_MSG(
      input.is_open(),
      "Failed to open Decode sidecar for boundary reread: %s",
      path.c_str());
  std::vector<char> buffer(kReadBufferBytes);
  uint64_t bytes = 0;
  uint64_t checksum = 0;
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count <= 0) {
      break;
    }
    bytes += static_cast<uint64_t>(count);
    checksum += static_cast<uint8_t>(buffer.front());
    checksum += static_cast<uint8_t>(buffer[static_cast<size_t>(count) - 1]);
  }
  ET_CHECK_MSG(
      input.eof(),
      "Decode sidecar boundary reread failed before EOF: path=%s bytes=%" PRIu64,
      path.c_str(),
      bytes);
  buffer.clear();
  buffer.shrink_to_fit();
  const auto after = process_memory_status_snapshot();
  const double reread_ms = elapsed_ms(start);
  ET_LOG(
      Info,
      "PD Decode sidecar boundary reread: path=%s bytes=%" PRIu64 " "
      "reread_ms=%.3f throughput_mib_s=%.3f checksum=%" PRIu64 " "
      "major_faults_before=%" PRIu64 " major_faults_after=%" PRIu64 " "
      "minor_faults_before=%" PRIu64 " minor_faults_after=%" PRIu64 " "
      "rss_before_mib=%.3f rss_after_mib=%.3f",
      path.c_str(),
      bytes,
      reread_ms,
      reread_ms > 0.0 ? bytes_to_mib(bytes) * 1000.0 / reread_ms : 0.0,
      checksum,
      before.major_faults,
      after.major_faults,
      before.minor_faults,
      after.minor_faults,
      bytes_to_mib(before.rss_bytes),
      bytes_to_mib(after.rss_bytes));
}

void pretouch_decode_sidecar_mapping_before_handoff(
    const std::string& path) {
  if (!FLAGS_decode_sidecar_pretouch_mapping) {
    return;
  }
  ET_CHECK_MSG(
      !path.empty(),
      "--decode_sidecar_pretouch_mapping requires "
      "--decode_sidecar_reread_path");
  const auto start = SteadyClock::now();
  const auto before = process_memory_status_snapshot();
  std::ifstream maps("/proc/self/maps");
  ET_CHECK_MSG(maps.is_open(), "Failed to open /proc/self/maps");
  const size_t page_size =
      static_cast<size_t>(std::max<long>(sysconf(_SC_PAGESIZE), 1));
  uint64_t touched_pages = 0;
  uint64_t mapped_bytes = 0;
  uint64_t checksum = 0;
  std::string line;
  while (std::getline(maps, line)) {
    std::istringstream fields(line);
    std::string address;
    std::string permissions;
    std::string offset;
    std::string device;
    std::string inode;
    fields >> address >> permissions >> offset >> device >> inode;
    std::string mapped_path;
    std::getline(fields, mapped_path);
    const size_t first = mapped_path.find_first_not_of(" \t");
    mapped_path = first == std::string::npos
        ? std::string()
        : mapped_path.substr(first);
    if (mapped_path != path || permissions.empty() || permissions[0] != 'r') {
      continue;
    }
    const size_t dash = address.find('-');
    ET_CHECK_MSG(
        dash != std::string::npos,
        "Malformed /proc/self/maps address: %s",
        address.c_str());
    const uintptr_t begin =
        static_cast<uintptr_t>(std::stoull(address.substr(0, dash), nullptr, 16));
    const uintptr_t end =
        static_cast<uintptr_t>(std::stoull(address.substr(dash + 1), nullptr, 16));
    const volatile uint8_t* bytes =
        reinterpret_cast<const volatile uint8_t*>(begin);
    for (uintptr_t current = begin; current < end; current += page_size) {
      checksum += bytes[current - begin];
      ++touched_pages;
    }
    if (end > begin) {
      checksum += bytes[end - begin - 1];
      mapped_bytes += static_cast<uint64_t>(end - begin);
    }
  }
  ET_CHECK_MSG(
      mapped_bytes != 0,
      "Decode sidecar mapping not found for boundary pretouch: %s",
      path.c_str());
  const auto after = process_memory_status_snapshot();
  const double pretouch_ms = elapsed_ms(start);
  ET_LOG(
      Info,
      "PD Decode sidecar mapping pretouch: path=%s mapped_bytes=%" PRIu64 " "
      "touched_pages=%" PRIu64 " pretouch_ms=%.3f checksum=%" PRIu64 " "
      "major_faults_before=%" PRIu64 " major_faults_after=%" PRIu64 " "
      "minor_faults_before=%" PRIu64 " minor_faults_after=%" PRIu64 " "
      "rss_before_mib=%.3f rss_after_mib=%.3f",
      path.c_str(),
      mapped_bytes,
      touched_pages,
      pretouch_ms,
      checksum,
      before.major_faults,
      after.major_faults,
      before.minor_faults,
      after.minor_faults,
      bytes_to_mib(before.rss_bytes),
      bytes_to_mib(after.rss_bytes));
}

struct PdE2ERuntimeStats {
  int32_t prompt_tokens{0};
  double runner_setup_ms{0.0};
  double prefill_prepare_ms{0.0};
  double bootstrap_tokenize_ms{0.0};
  double prepare_overlap_wall_ms{0.0};
  double qnn_export_total_ms{0.0};
  double qnn_backend_prewarm_ms{0.0};
  bool qnn_backend_prewarmed{false};
  double persistent_shard0_prepare_ms{0.0};
  bool persistent_shard0_prepared{false};
  example::PDPrefillRunner<uint16_t>::RuntimeStats prefill{};
  std::vector<example::DecoderRunner::PrefillShardRuntimeStats> shard_stats;
  ProcessMemorySnapshot before_runner;
  ProcessMemorySnapshot after_runner;
  ProcessMemorySnapshot after_export;
  std::string shared_embedding_matrix_path;
  example::PDPrefillRunner<uint8_t>::MemoryHandoff memory_handoff;
};

struct DecodeProcessResult {
  int exit_code{1};
  double startup_ms{0.0};
  double process_wall_ms{0.0};
  double wall_ms{0.0};
  double boundary_ms{0.0};
  double generation_ms{0.0};
  int32_t generated_tokens{0};
  ProcessMemorySnapshot before;
  ProcessMemorySnapshot after;
  ProcessMemorySnapshot child_peak;
  uint64_t process_tree_peak_rss_bytes{0};
  uint64_t process_tree_peak_pss_bytes{0};
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
  double qat_checkpoint_context_total_ms = 0.0;
  double qat_recipe_total_ms = 0.0;
  double gguf_checkpoint_context_total_ms = 0.0;
  double gguf_recipe_total_ms = 0.0;
  double gguf_relayout_total_ms = 0.0;
  size_t gguf_relayout_total_bytes = 0;
  double rebuild_allocation_total_ms = 0.0;
  double rebuild_static_copy_total_ms = 0.0;
  double rebuild_weight_materialization_total_ms = 0.0;
  double pipeline_wait_total_ms = 0.0;
  for (const auto& shard : prefill.shard_stats) {
    shard_preload_total_ms += shard.preload_ms;
    qat_checkpoint_context_total_ms += shard.qat_checkpoint_context_ms;
    qat_recipe_total_ms += shard.qat_recipe_ms;
    gguf_checkpoint_context_total_ms += shard.gguf_checkpoint_context_ms;
    gguf_recipe_total_ms += shard.gguf_recipe_ms;
    gguf_relayout_total_ms += shard.gguf_relayout_ms;
    gguf_relayout_total_bytes += shard.gguf_relayout_bytes;
    rebuild_allocation_total_ms += shard.rebuild_allocation_ms;
    rebuild_static_copy_total_ms += shard.rebuild_static_copy_ms;
    rebuild_weight_materialization_total_ms +=
        shard.rebuild_weight_materialization_ms;
    pipeline_wait_total_ms += shard.pipeline_wait_ms;
  }
  ET_LOG(
      Info,
      "PD E2E shard setup: preload_ms=%.3f qat_checkpoint_context_ms=%.3f "
      "qat_recipe_ms=%.3f gguf_checkpoint_context_ms=%.3f gguf_recipe_ms=%.3f "
      "qnn_backend_prewarm_ms=%.3f qnn_backend_prewarmed=%d pipeline_wait_ms=%.3f "
      "pipeline_enabled=%d stage_major_enabled=%d three_stage_enabled=%d "
      "persistent_shard0_prepare_ms=%.3f persistent_shard0_prepared=%d",
      shard_preload_total_ms,
      qat_checkpoint_context_total_ms,
      qat_recipe_total_ms,
      gguf_checkpoint_context_total_ms,
      gguf_recipe_total_ms,
      prefill.qnn_backend_prewarm_ms,
      static_cast<int>(prefill.qnn_backend_prewarmed),
      pipeline_wait_total_ms,
      static_cast<int>(
          FLAGS_prefill_shard_pipeline || FLAGS_prefill_shard_pipeline_3stage),
      static_cast<int>(FLAGS_prefill_shard_stage_major),
      static_cast<int>(FLAGS_prefill_shard_pipeline_3stage),
      prefill.persistent_shard0_prepare_ms,
      static_cast<int>(prefill.persistent_shard0_prepared));
  ET_LOG(
      Info,
      "PD E2E rebuild breakdown: allocation_ms=%.3f static_copy_ms=%.3f "
      "weight_materialization_ms=%.3f",
      rebuild_allocation_total_ms,
      rebuild_static_copy_total_ms,
      rebuild_weight_materialization_total_ms);
  ET_LOG(
      Info,
      "PD E2E GGUF source-block relayout: enabled=%d relayout_ms=%.3f relayout_bytes=%zu",
      static_cast<int>(gguf_relayout_total_bytes > 0),
      gguf_relayout_total_ms,
      gguf_relayout_total_bytes);
  ET_LOG(
      Info,
      "PD E2E timing: runner_setup_ms=%.3f prefill_prepare_ms=%.3f "
      "bootstrap_tokenize_ms=%.3f prepare_overlap_wall_ms=%.3f "
      "tokenize_ms=%.3f embedding_prepare_ms=%.3f qnn_prefill_ms=%.3f "
      "handoff_ms=%.3f qnn_export_total_ms=%.3f decode_startup_ms=%.3f "
      "decode_process_ms=%.3f e2e_total_ms=%.3f",
      prefill.runner_setup_ms,
      prefill.prefill_prepare_ms,
      prefill.bootstrap_tokenize_ms,
      prefill.prepare_overlap_wall_ms,
      prefill.prefill.tokenize_ms,
      prefill.prefill.embedding_prepare_ms,
      prefill.prefill.prefill_ms,
      prefill.prefill.handoff_total_ms,
      prefill.qnn_export_total_ms,
      decode.startup_ms,
      decode.wall_ms,
      total_ms);
  const double prefill_round_ms =
      prefill.prefill.tokenize_ms + prefill.prefill.prefill_ms;
  const double pd_boundary_ms =
      prefill.prefill.handoff_total_ms + decode.boundary_ms;
  const double decode_round_ms = pd_boundary_ms + decode.generation_ms;
  const double decode_round_ms_per_token = decode.generated_tokens > 0
      ? decode_round_ms / static_cast<double>(decode.generated_tokens)
      : 0.0;
  ET_LOG(
      Info,
      "PD steady-state timing: initialization_once_ms=%.3f "
      "prefill_round_ms=%.3f prefill_handoff_ms=%.3f "
      "decode_boundary_ms=%.3f pd_boundary_ms=%.3f "
      "decode_generation_ms=%.3f decode_round_ms=%.3f "
      "decode_generated_tokens=%d decode_round_ms_per_token=%.3f "
      "qnn_backend_prewarmed=%d",
      prefill.runner_setup_ms + prefill.prepare_overlap_wall_ms +
          decode.startup_ms,
      prefill_round_ms,
      prefill.prefill.handoff_total_ms,
      decode.boundary_ms,
      pd_boundary_ms,
      decode.generation_ms,
      decode_round_ms,
      decode.generated_tokens,
      decode_round_ms_per_token,
      static_cast<int>(prefill.qnn_backend_prewarmed));
  if (!prefill.qnn_backend_prewarmed) {
    ET_LOG(
        Error,
        "PD steady-state Prefill timing is contaminated by first-use QNN "
        "backend initialization; export qnn_compile_spec_hex and keep "
        "--prefill_qnn_backend_prewarm enabled");
  }
  ET_LOG(
      Info,
      "PD E2E memory handoff: kv_pack_ms=%.3f size_bytes=%zu",
      prefill.prefill.kv_layout_ms,
      prefill.memory_handoff.size_bytes -
          static_cast<size_t>(prefill.memory_handoff.prompt_length) *
              sizeof(uint64_t));
  ET_LOG(
      Info,
      "PD E2E parent memory MiB: before_runner_rss=%.2f after_runner_rss=%.2f "
      "after_export_rss=%.2f after_decode_rss=%.2f hwm=%.2f",
      bytes_to_mib(prefill.before_runner.rss_bytes),
      bytes_to_mib(prefill.after_runner.rss_bytes),
      bytes_to_mib(prefill.after_export.rss_bytes),
      bytes_to_mib(decode.after.rss_bytes),
      bytes_to_mib(std::max(prefill.after_export.hwm_bytes, decode.after.hwm_bytes)));
  ET_LOG(
      Info,
      "PD E2E parent faults: before_runner_minflt=%" PRIu64
      " after_runner_minflt=%" PRIu64 " after_export_minflt=%" PRIu64
      " after_decode_minflt=%" PRIu64 " before_runner_majflt=%" PRIu64
      " after_runner_majflt=%" PRIu64 " after_export_majflt=%" PRIu64
      " after_decode_majflt=%" PRIu64,
      prefill.before_runner.minor_faults,
      prefill.after_runner.minor_faults,
      prefill.after_export.minor_faults,
      decode.after.minor_faults,
      prefill.before_runner.major_faults,
      prefill.after_runner.major_faults,
      prefill.after_export.major_faults,
      decode.after.major_faults);
  ET_LOG(
      Info,
      "PD E2E decode memory MiB: child_peak_rss=%.2f child_hwm=%.2f "
      "process_tree_peak_rss=%.2f process_tree_peak_pss=%.2f poll_interval_ms=5",
      bytes_to_mib(decode.child_peak.rss_bytes),
      bytes_to_mib(decode.child_peak.hwm_bytes),
      bytes_to_mib(decode.process_tree_peak_rss_bytes),
      bytes_to_mib(decode.process_tree_peak_pss_bytes));
  for (size_t i = 0; i < prefill.shard_stats.size(); ++i) {
    const auto& shard = prefill.shard_stats[i];
    ET_LOG(
        Info,
        "PD E2E shard config: index=%zu layers=[%zu,%zu) runs=%zu "
        "preload_ms=%.3f qat_checkpoint_context_ms=%.3f qat_recipe_ms=%.3f "
        "gguf_checkpoint_context_ms=%.3f gguf_recipe_ms=%.3f",
        i,
        shard.layer_offset,
        shard.layer_offset + shard.layer_count,
        shard.execution_count,
        shard.preload_ms,
        shard.qat_checkpoint_context_ms,
        shard.qat_recipe_ms,
        shard.gguf_checkpoint_context_ms,
        shard.gguf_recipe_ms);
    if (shard.gguf_relayout_bytes > 0) {
      ET_LOG(
          Info,
          "PD E2E shard GGUF source relayout: index=%zu relayout_ms=%.3f "
          "relayout_bytes=%zu",
          i,
          shard.gguf_relayout_ms,
          shard.gguf_relayout_bytes);
    }
    ET_LOG(
        Info,
        "PD E2E shard rebuild: index=%zu materialize_ms=%.3f rebuild_ms=%.3f "
        "allocation_ms=%.3f static_copy_ms=%.3f weight_materialization_ms=%.3f",
        i,
        shard.materialize_ms,
        shard.rebuild_ms,
        shard.rebuild_allocation_ms,
        shard.rebuild_static_copy_ms,
        shard.rebuild_weight_materialization_ms);
    ET_LOG(
        Info,
        "PD E2E shard execution: index=%zu pipeline_wait_ms=%.3f "
        "execute_ms=%.3f total_ms=%.3f",
        i,
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
  bool outputs_logits{true};
  bool use_separate_embed{false};
  std::string embedding_matrix_path;
  bool embedding_qnn_u16_input{false};
  float embedding_qnn_u16_scale{0.0f};
  int32_t embedding_qnn_u16_zero_point{0};
  std::shared_ptr<std::vector<uint8_t>> qnn_compile_spec_bytes;
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

void append_session_prompts(
    const std::string& path,
    std::vector<std::string>* prompts) {
  if (path.empty()) {
    return;
  }
  std::ifstream input(path);
  ET_CHECK_MSG(input.is_open(), "Unable to read session prompts: %s", path.c_str());
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty()) {
      prompts->push_back(std::move(line));
    }
  }
  ET_CHECK_MSG(
      !prompts->empty(),
      "Session prompt file contains no non-empty prompts: %s",
      path.c_str());
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
  std::regex path_regex("\"([^\"]*)\"");
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

float read_float_field_or(
    const std::string& manifest,
    size_t start_pos,
    const std::string& field_name,
    float default_value) {
  const size_t field_pos = manifest.find("\"" + field_name + "\"", start_pos);
  if (field_pos == std::string::npos) return default_value;
  const size_t colon_pos = manifest.find(":", field_pos);
  ET_CHECK_MSG(colon_pos != std::string::npos, "Invalid %s float in shard manifest", field_name.c_str());
  const size_t value_start = manifest.find_first_of("-+.0123456789", colon_pos + 1);
  ET_CHECK_MSG(value_start != std::string::npos, "Invalid %s float in shard manifest", field_name.c_str());
  size_t consumed = 0;
  const float value = std::stof(manifest.substr(value_start), &consumed);
  ET_CHECK_MSG(consumed > 0, "Invalid %s float in shard manifest", field_name.c_str());
  return value;
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

uint8_t decode_manifest_hex_nibble(char character, const char* field_name) {
  if (character >= '0' && character <= '9') {
    return static_cast<uint8_t>(character - '0');
  }
  if (character >= 'a' && character <= 'f') {
    return static_cast<uint8_t>(character - 'a' + 10);
  }
  if (character >= 'A' && character <= 'F') {
    return static_cast<uint8_t>(character - 'A' + 10);
  }
  ET_CHECK_MSG(false, "%s contains a non-hex character", field_name);
  return 0;
}

std::vector<uint8_t> decode_hex_manifest_bytes(
    const std::string& value,
    const char* field_name) {
  ET_CHECK_MSG(
      value.size() % 2 == 0,
      "%s must contain an even number of hex characters",
      field_name);
  std::vector<uint8_t> bytes;
  bytes.reserve(value.size() / 2);
  for (size_t index = 0; index < value.size(); index += 2) {
    bytes.push_back(static_cast<uint8_t>(
        (decode_manifest_hex_nibble(value[index], field_name) << 4) |
        decode_manifest_hex_nibble(value[index + 1], field_name)));
  }
  return bytes;
}

PrefillShardFiles read_prefill_shard_files(const std::string& manifest_path) {
  PrefillShardFiles files;
  if (manifest_path.empty()) {
    return files;
  }
  const std::string manifest = read_text_file(manifest_path);
  const fs::path manifest_dir = fs::absolute(fs::path(manifest_path)).parent_path();
  const std::string qnn_compile_spec_hex =
      read_string_field(manifest, 0, "qnn_compile_spec_hex");
  if (!qnn_compile_spec_hex.empty()) {
    files.qnn_compile_spec_bytes =
        std::make_shared<std::vector<uint8_t>>(decode_hex_manifest_bytes(
            qnn_compile_spec_hex, "qnn_compile_spec_hex"));
  }
  ET_LOG(
      Info,
      "loaded qnn_compile_spec from shard manifest: bytes=%zu",
      files.qnn_compile_spec_bytes == nullptr
          ? 0
          : files.qnn_compile_spec_bytes->size());
  files.outputs_logits =
      read_bool_field_or(manifest, 0, "prefill_outputs_logits", files.outputs_logits);
  files.use_separate_embed =
      read_bool_field_or(manifest, 0, "separate_embed", files.use_separate_embed);
  if (files.use_separate_embed) {
    const std::string matrix_path =
        read_string_field(manifest, 0, "separate_embed_matrix");
    ET_CHECK_MSG(
        !matrix_path.empty(),
        "separate_embed manifest is missing separate_embed_matrix: %s",
        manifest_path.c_str());
    files.embedding_matrix_path =
        resolve_manifest_path(manifest_dir, matrix_path);
  }
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
  // The static Qwen3 ABI is independent of the number of evenly split
  // decoder shards. Keep accepting the legacy 4x7 name and support names
  // such as qwen3_14x2_static emitted by a 14-shard export.
  const std::string static_suffix = "_static";
  files.qwen3_static_plan =
      plan_type == "qwen3_static" ||
      (plan_type.rfind("qwen3_", 0) == 0 &&
       plan_type.size() > std::strlen("qwen3_") + static_suffix.size() &&
       plan_type.compare(
           plan_type.size() - static_suffix.size(),
           static_suffix.size(),
           static_suffix) == 0);
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
    files.embedding_qnn_u16_input = read_bool_field_or(
        manifest, metadata_pos, "embedding_qnn_u16_input", false);
    files.embedding_qnn_u16_scale = read_float_field_or(
        manifest, metadata_pos, "embedding_qnn_u16_scale", 0.0f);
    files.embedding_qnn_u16_zero_point = read_int_field_or(
        manifest, metadata_pos, "embedding_qnn_u16_zero_point", 0);
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
      formatted_prompt.append("<|im_start|>assistant\n");
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

std::string resolve_decode_gguf_path() {
  if (!g_model_ram_fd_spec.empty()) {
    return g_model_ram_fd_spec;
  }
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
    if (FLAGS_model_anonymous_buffer) {
      ET_CHECK_MSG(
          FLAGS_decode_gguf_path.empty() ||
              FLAGS_decode_gguf_path == FLAGS_gguf_model_path,
          "--model_anonymous_buffer requires Prefill and Decode to use the same GGUF");
      const auto load_start = SteadyClock::now();
      config.mapped_source_bytes =
          example::ReadOnlyMappedFile::load_into_anonymous_buffer(
              FLAGS_gguf_model_path);
      g_model_ram_fd_spec.clear();
      ET_LOG(
          Info,
          "PD anonymous model buffer ready: bytes=%zu load_ms=%.3f "
          "timing_scope=bootstrap",
          config.mapped_source_bytes->size(),
          elapsed_ms(load_start));
    } else if (FLAGS_model_ram_store) {
      ET_CHECK_MSG(
          FLAGS_decode_gguf_path.empty() ||
              FLAGS_decode_gguf_path == FLAGS_gguf_model_path,
          "--model_ram_store requires Prefill and Decode to use the same GGUF");
      const auto load_start = SteadyClock::now();
      config.mapped_source_bytes =
          example::ReadOnlyMappedFile::load_into_shared_memory(
              FLAGS_gguf_model_path);
      g_model_ram_fd_spec =
          config.mapped_source_bytes->inherited_fd_spec();
      ET_LOG(
          Info,
          "PD model RAM store ready: bytes=%zu load_ms=%.3f fd_spec=%s "
          "timing_scope=bootstrap",
          config.mapped_source_bytes->size(),
          elapsed_ms(load_start),
          g_model_ram_fd_spec.c_str());
    } else {
      config.mapped_source_bytes =
          example::ReadOnlyMappedFile::open(FLAGS_gguf_model_path);
    }
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
  ET_CHECK_MSG(
      !(FLAGS_prefill_gguf_relayout_blocks && FLAGS_prefill_gguf_relayout_gs32_source),
      "Use at most one GGUF relayout benchmark flag");
  config.gguf_relayout_kind = FLAGS_prefill_gguf_relayout_gs32_source
      ? example::PteGgufRecipeRelayoutKind::Gs32Source
      : (FLAGS_prefill_gguf_relayout_blocks
             ? example::PteGgufRecipeRelayoutKind::RawBlocks
             : example::PteGgufRecipeRelayoutKind::None);
  ET_CHECK_MSG(
      !FLAGS_prefill_shard_pipeline_3stage || FLAGS_prefill_shard_stage_major,
      "--prefill_shard_pipeline_3stage requires --prefill_shard_stage_major");
  config.stage_major_execution = FLAGS_prefill_shard_stage_major;
  config.pipeline_qnn_load = FLAGS_prefill_shard_pipeline_3stage;
  config.pipeline_rebuild =
      FLAGS_prefill_shard_pipeline || FLAGS_prefill_shard_pipeline_3stage;
  config.prewarm_qnn_backend = FLAGS_prefill_qnn_backend_prewarm;
  config.release_stripped_pte_after_rebuild =
      FLAGS_prefill_release_stripped_pte_after_rebuild;
  ET_CHECK_MSG(
      !FLAGS_prefill_persistent_shard0 ||
          FLAGS_prefill_shard_pipeline_3stage,
      "--prefill_persistent_shard0 requires "
      "--prefill_shard_pipeline_3stage");
  config.persistent_shard0_context = FLAGS_prefill_persistent_shard0;
  ET_CHECK_MSG(
      !FLAGS_prefill_detach_shard0_qnn_after_load ||
          FLAGS_prefill_persistent_shard0,
      "--prefill_detach_shard0_qnn_after_load requires "
      "--prefill_persistent_shard0");
  ET_CHECK_MSG(
      !FLAGS_prefill_detach_all_qnn_after_load ||
          FLAGS_prefill_shard_pipeline_3stage,
      "--prefill_detach_all_qnn_after_load requires "
      "--prefill_shard_pipeline_3stage");
  ET_CHECK_MSG(
      !FLAGS_prefill_release_pte_backing_after_load ||
          FLAGS_prefill_etdump_dir.empty(),
      "--prefill_release_pte_backing_after_load is incompatible with ETDump "
      "method unload/reload");
  config.release_rebuilt_pte_backing_after_load =
      FLAGS_prefill_release_pte_backing_after_load;
  ET_CHECK_MSG(
      !(FLAGS_prefill_detach_shard0_qnn_after_load ||
        FLAGS_prefill_detach_all_qnn_after_load) ||
          (!FLAGS_prefill_release_pte_backing_after_load &&
           FLAGS_prefill_etdump_dir.empty() &&
           !FLAGS_prefill_unload_shard0_method_after_execute &&
           !FLAGS_prefill_destroy_shard0_module_keep_pte_after_execute),
      "detached QNN execution is incompatible with whole-PTE release, "
      "ETDump, and post-execute Module lifetime diagnostics");
  config.detach_shard0_qnn_after_load =
      FLAGS_prefill_detach_shard0_qnn_after_load;
  config.detach_all_qnn_after_load =
      FLAGS_prefill_detach_all_qnn_after_load;
  config.release_prepared_shard0_after_execute =
      FLAGS_prefill_release_shard0_after_execute;
  config.unload_prepared_shard0_method_after_execute =
      FLAGS_prefill_unload_shard0_method_after_execute;
  config.destroy_prepared_shard0_module_keep_pte_after_execute =
      FLAGS_prefill_destroy_shard0_module_keep_pte_after_execute;
  config.discard_prepared_shard0_pte_pages_after_execute =
      FLAGS_prefill_discard_shard0_pte_pages_after_execute;
  config.qnn_compile_spec_bytes = files.qnn_compile_spec_bytes;
  ET_LOG(
      Info,
      "PD shard rebuild source: kind=%d size_bytes=%zu capacity_bytes=%zu",
      static_cast<int>(config.source_kind),
      config.mapped_source_bytes ? config.mapped_source_bytes->size()
                                 : config.source_bytes->size(),
      config.mapped_source_bytes ? size_t{0} : config.source_bytes->capacity());
  return config;
}

constexpr uint32_t PD_RESIDENT_READY_MAGIC = 0x50445259U; // "PDRY"
constexpr uint32_t PD_RESIDENT_PREPARE_MAGIC = 0x50445052U; // "PDPR"
constexpr uint32_t PD_RESIDENT_REQUEST_MAGIC = 0x50444b56U; // "PDKV"
// v5 stores both K and V as [layer, head, token, dim].
constexpr uint32_t PD_RESIDENT_PROTOCOL_VERSION = 5;

struct PdResidentReady {
  uint32_t magic{PD_RESIDENT_READY_MAGIC};
  uint32_t version{PD_RESIDENT_PROTOCOL_VERSION};
};

struct PdResidentPrepare {
  uint32_t magic{PD_RESIDENT_PREPARE_MAGIC};
  uint32_t version{PD_RESIDENT_PROTOCOL_VERSION};
};

struct PdResidentRequest {
  uint32_t magic{PD_RESIDENT_REQUEST_MAGIC};
  uint32_t version{PD_RESIDENT_PROTOCOL_VERSION};
  uint64_t memory_size{0};
  int32_t prompt_length{0};
  int32_t num_layers{0};
  int32_t num_kv_heads{0};
  int32_t head_dim{0};
  int32_t first_token{-1};
  uint32_t first_token_is_prompt_tail{0};
};

#ifndef QNN_LLAMA_PD_JOINT
struct ResidentDecodeProcess {
  pid_t pid{-1};
  int control_fd{-1};
  bool runtime_prepare_sent{false};
  SteadyClock::time_point start{};
  DecodeProcessResult result{};
};

ResidentDecodeProcess start_resident_decode_process(
    const std::string& shared_embedding_matrix_path) {
  ET_CHECK_MSG(
      !FLAGS_llama_pd_cli_path.empty(),
      "--llama_pd_cli_path is required unless --prefill_only=true");
  const std::string decode_gguf_path = resolve_decode_gguf_path();
  ET_CHECK_MSG(
      !decode_gguf_path.empty(),
      "Provide --decode_gguf_path, --gguf_model_path, or --tmac_model_path for decode");

  int sockets[2] = {-1, -1};
  ET_CHECK_MSG(
      socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sockets) == 0,
      "resident Decode socketpair failed: %s",
      std::strerror(errno));

  std::vector<std::string> args = {
      FLAGS_llama_pd_cli_path,
      "--pd-control-fd",
      std::to_string(sockets[1]),
      "-m",
      decode_gguf_path,
      "-n",
      std::to_string(FLAGS_decode_n_predict),
      "-c",
      std::to_string(FLAGS_decode_ctx),
      "-b",
      "1",
      "-ub",
      "1",
      "-ngl",
      std::to_string(FLAGS_decode_ngl),
      "--temp",
      std::to_string(FLAGS_decode_temp),
      "-fit",
      "off",
  };
  if (FLAGS_decode_threads > 0) {
    args.push_back("-t");
    args.push_back(std::to_string(FLAGS_decode_threads));
  }
  if (!shared_embedding_matrix_path.empty()) {
    args.push_back("--pd-disk-embedding");
    args.push_back(shared_embedding_matrix_path);
  }
  if (FLAGS_decode_native_compare) {
    args.push_back("--pd-native-compare");
  }
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
  if (FLAGS_decode_import_ro) {
    args.push_back("--pd-import-ro");
  }

  ET_LOG(Info, "Starting resident Decode before Prefill");
  for (const auto& arg : args) {
    ET_LOG(Info, "  arg: %s", arg.c_str());
  }

  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);

  ResidentDecodeProcess resident;
  resident.result.before = process_memory_snapshot();
  resident.start = SteadyClock::now();
  resident.pid = fork();
  ET_CHECK_MSG(resident.pid >= 0, "fork failed: %s", std::strerror(errno));
  if (resident.pid == 0) {
    close(sockets[0]);
    execvp(argv[0], argv.data());
    std::fprintf(
        stderr,
        "execvp failed for %s: %s\n",
        FLAGS_llama_pd_cli_path.c_str(),
        std::strerror(errno));
    _exit(127);
  }
  close(sockets[1]);
  resident.control_fd = sockets[0];

  PdResidentReady ready;
  ssize_t received;
  do {
    received = recv(resident.control_fd, &ready, sizeof(ready), 0);
  } while (received < 0 && errno == EINTR);
  ET_CHECK_MSG(
      received == static_cast<ssize_t>(sizeof(ready)) &&
          ready.magic == PD_RESIDENT_READY_MAGIC &&
          ready.version == PD_RESIDENT_PROTOCOL_VERSION,
      "resident Decode failed before ready (recv=%zd errno=%s)",
      received,
      std::strerror(errno));
  resident.result.child_peak = process_memory_snapshot(resident.pid);
  resident.result.startup_ms = elapsed_ms(resident.start);
  resident.result.process_tree_peak_rss_bytes =
      resident.result.before.rss_bytes + resident.result.child_peak.rss_bytes;
  resident.result.process_tree_peak_pss_bytes =
      resident.result.before.pss_bytes + resident.result.child_peak.pss_bytes;
  ET_LOG(
      Info,
      "Resident Decode ready before Prefill: pid=%d startup_ms=%.3f child_rss_mib=%.2f",
      static_cast<int>(resident.pid),
      elapsed_ms(resident.start),
      bytes_to_mib(resident.result.child_peak.rss_bytes));
  return resident;
}

void start_resident_decode_runtime_prepare(int control_fd) {
  const PdResidentPrepare request;
  ssize_t sent;
  do {
    sent = send(control_fd, &request, sizeof(request), MSG_NOSIGNAL);
  } while (sent < 0 && errno == EINTR);
  ET_CHECK_MSG(
      sent == static_cast<ssize_t>(sizeof(request)),
      "failed to start resident Decode runtime preparation: %s",
      std::strerror(errno));
  ET_LOG(
      Info,
      "Prefill released rebuild inputs; Decode metadata/context/KV preparation started asynchronously");
}

void send_resident_handoff(
    int control_fd,
    const example::PDPrefillRunner<uint8_t>::MemoryHandoff& memory_handoff) {
  ET_CHECK_MSG(
      memory_handoff.fd >= 0,
      "resident Decode requires an in-memory UINT8 KV handoff");
  PdResidentRequest request;
  request.memory_size = memory_handoff.size_bytes;
  request.prompt_length = memory_handoff.prompt_length;
  request.num_layers = memory_handoff.num_layers;
  request.num_kv_heads = memory_handoff.num_kv_heads;
  request.head_dim = memory_handoff.head_dim;
  request.first_token = memory_handoff.first_token;
  request.first_token_is_prompt_tail =
      memory_handoff.first_token_is_prompt_tail ? 1U : 0U;

  iovec iov {};
  iov.iov_base = &request;
  iov.iov_len = sizeof(request);
  alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))] = {};
  msghdr message {};
  message.msg_iov = &iov;
  message.msg_iovlen = 1;
  message.msg_control = control;
  message.msg_controllen = sizeof(control);
  cmsghdr* cmsg = CMSG_FIRSTHDR(&message);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(cmsg), &memory_handoff.fd, sizeof(int));

  ssize_t sent;
  do {
    sent = sendmsg(control_fd, &message, MSG_NOSIGNAL);
  } while (sent < 0 && errno == EINTR);
  ET_CHECK_MSG(
      sent == static_cast<ssize_t>(sizeof(request)),
      "failed to send resident Decode handoff: %s",
      std::strerror(errno));
}

DecodeProcessResult finish_resident_decode_process(
    ResidentDecodeProcess resident) {
  const auto decode_start = SteadyClock::now();
  ET_CHECK_MSG(
      resident.control_fd < 0,
      "resident Decode handoff must be sent before waiting for completion");

  int status = 0;
  while (true) {
    const pid_t wait_result = waitpid(resident.pid, &status, WNOHANG);
    ET_CHECK_MSG(wait_result >= 0, "waitpid failed: %s", std::strerror(errno));
    if (wait_result == resident.pid) {
      break;
    }
    const auto child_memory = process_memory_snapshot(resident.pid);
    const auto parent_memory = process_memory_snapshot();
    if (child_memory.rss_bytes > resident.result.child_peak.rss_bytes) {
      resident.result.child_peak.rss_bytes = child_memory.rss_bytes;
    }
    resident.result.child_peak.hwm_bytes = std::max(
        resident.result.child_peak.hwm_bytes, child_memory.hwm_bytes);
    resident.result.process_tree_peak_rss_bytes = std::max(
        resident.result.process_tree_peak_rss_bytes,
        parent_memory.rss_bytes + child_memory.rss_bytes);
    resident.result.process_tree_peak_pss_bytes = std::max(
        resident.result.process_tree_peak_pss_bytes,
        parent_memory.pss_bytes + child_memory.pss_bytes);
    usleep(5000);
  }
  resident.result.wall_ms = elapsed_ms(decode_start);
  resident.result.after = process_memory_snapshot();
  if (WIFEXITED(status)) {
    resident.result.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    resident.result.exit_code = 128 + WTERMSIG(status);
  }
  return resident.result;
}

void begin_resident_decode_handoff(
    ResidentDecodeProcess& resident,
    const example::PDPrefillRunner<uint8_t>::MemoryHandoff& memory_handoff) {
  if (!resident.runtime_prepare_sent) {
    start_resident_decode_runtime_prepare(resident.control_fd);
    resident.runtime_prepare_sent = true;
  }
  send_resident_handoff(resident.control_fd, memory_handoff);
  close(resident.control_fd);
  resident.control_fd = -1;
  ET_LOG(Info, "Prefill handed KV memfd to already-resident Decode pid=%d", resident.pid);
}
#else
std::shared_ptr<example::ReadOnlyMappedFile> g_joint_model_source;

void log_joint_model_residency(const char* stage) {
  if (!FLAGS_model_residency_probe || !g_joint_model_source) {
    return;
  }
  const auto start = SteadyClock::now();
  const auto residency = g_joint_model_source->residency();
  const double percent = residency.total_pages == 0
      ? 0.0
      : 100.0 * static_cast<double>(residency.resident_pages) /
          static_cast<double>(residency.total_pages);
  ET_LOG(
      Info,
      "PD model residency: stage=%s resident_pages=%zu total_pages=%zu "
      "resident_percent=%.3f error=%d probe_ms=%.3f",
      stage,
      residency.resident_pages,
      residency.total_pages,
      percent,
      residency.error,
      elapsed_ms(start));
}

uint64_t g_joint_prefill_complete_major_faults{0};

struct RawMemoryResidency {
  size_t resident_pages{0};
  size_t total_pages{0};
  int error{0};
};

RawMemoryResidency raw_memory_residency(const void* data, size_t size) {
  RawMemoryResidency result;
  if (data == nullptr || size == 0) {
    return result;
  }
  const long page_size_long = sysconf(_SC_PAGESIZE);
  if (page_size_long <= 0) {
    result.error = errno != 0 ? errno : EINVAL;
    return result;
  }
  const uintptr_t page_size = static_cast<uintptr_t>(page_size_long);
  const uintptr_t address = reinterpret_cast<uintptr_t>(data);
  const uintptr_t begin = address - address % page_size;
  const uintptr_t end =
      (address + size + page_size - 1) / page_size * page_size;
  result.total_pages = static_cast<size_t>((end - begin) / page_size);
  std::vector<unsigned char> state(result.total_pages);
  if (mincore(
          reinterpret_cast<void*>(begin),
          static_cast<size_t>(end - begin),
          state.data()) != 0) {
    result.error = errno;
    return result;
  }
  result.resident_pages = static_cast<size_t>(std::count_if(
      state.begin(), state.end(), [](unsigned char value) {
        return (value & 1U) != 0;
      }));
  return result;
}

const void* g_joint_sidecar_data{nullptr};
size_t g_joint_sidecar_size{0};
bool g_joint_sidecar_anonymous{false};

void log_joint_pd_boundary_probe(
    const char* event,
    int32_t decode_call_index,
    int32_t token,
    bool establish_prefill_baseline) {
  if (!FLAGS_model_residency_probe || !g_joint_model_source) {
    return;
  }
  const auto start = SteadyClock::now();
  const auto residency = g_joint_model_source->residency();
  const auto sidecar_residency =
      raw_memory_residency(g_joint_sidecar_data, g_joint_sidecar_size);
  const auto memory = process_memory_status_snapshot();
  if (establish_prefill_baseline) {
    g_joint_prefill_complete_major_faults = memory.major_faults;
  }
  const uint64_t major_faults_since_prefill =
      memory.major_faults >= g_joint_prefill_complete_major_faults
      ? memory.major_faults - g_joint_prefill_complete_major_faults
      : 0;
  const double percent = residency.total_pages == 0
      ? 0.0
      : 100.0 * static_cast<double>(residency.resident_pages) /
            static_cast<double>(residency.total_pages);
  const double sidecar_percent = sidecar_residency.total_pages == 0
      ? 0.0
      : 100.0 * static_cast<double>(sidecar_residency.resident_pages) /
            static_cast<double>(sidecar_residency.total_pages);
  ET_LOG(
      Info,
      "PD boundary probe: event=%s decode_call_index=%d token=%d "
      "resident_pages=%zu total_pages=%zu resident_percent=%.6f "
      "sidecar_resident_pages=%zu sidecar_total_pages=%zu "
      "sidecar_resident_percent=%.6f sidecar_error=%d sidecar_anonymous=%d "
      "major_faults=%" PRIu64 " major_faults_since_prefill=%" PRIu64 " "
      "minor_faults=%" PRIu64 " rss_bytes=%" PRIu64 " hwm_bytes=%" PRIu64 " "
      "probe_ms=%.3f",
      event,
      decode_call_index,
      token,
      residency.resident_pages,
      residency.total_pages,
      percent,
      sidecar_residency.resident_pages,
      sidecar_residency.total_pages,
      sidecar_percent,
      sidecar_residency.error,
      static_cast<int>(g_joint_sidecar_anonymous),
      memory.major_faults,
      major_faults_since_prefill,
      memory.minor_faults,
      memory.rss_bytes,
      memory.hwm_bytes,
      elapsed_ms(start));
}

void joint_pd_decode_event_probe(
    void*,
    const char* event,
    int32_t decode_call_index,
    int32_t token) {
  char label[64];
  std::snprintf(
      label,
      sizeof(label),
      "decode_call_%d_%s",
      decode_call_index,
      event);
  log_joint_pd_boundary_probe(label, decode_call_index, token, false);
}

enum class JointResidencyStage : int {
  ModelReady,
  DecodeModelInit,
  PrefillSetup,
  Prefill,
  PrefillComplete,
  PrefillRelease,
  DecodeContext,
  DecodeWarmup,
  Handoff,
  Decode,
  DecodeComplete,
  Complete,
};

const char* joint_residency_stage_name(JointResidencyStage stage) {
  switch (stage) {
    case JointResidencyStage::ModelReady: return "model_ready";
    case JointResidencyStage::DecodeModelInit: return "decode_model_init";
    case JointResidencyStage::PrefillSetup: return "prefill_setup";
    case JointResidencyStage::Prefill: return "prefill";
    case JointResidencyStage::PrefillComplete: return "prefill_complete";
    case JointResidencyStage::PrefillRelease: return "prefill_release";
    case JointResidencyStage::DecodeContext: return "decode_context";
    case JointResidencyStage::DecodeWarmup: return "decode_warmup";
    case JointResidencyStage::Handoff: return "handoff";
    case JointResidencyStage::Decode: return "decode";
    case JointResidencyStage::DecodeComplete: return "decode_complete";
    case JointResidencyStage::Complete: return "complete";
  }
  return "unknown";
}

class JointModelResidencyProfiler {
 public:
  JointModelResidencyProfiler(
      std::shared_ptr<example::ReadOnlyMappedFile> source,
      const std::string& path,
      int interval_ms)
      : source_(std::move(source)),
        interval_ms_(std::max(interval_ms, 50)),
        start_(SteadyClock::now()) {
    output_ = std::fopen(path.c_str(), "w");
    ET_CHECK_MSG(output_ != nullptr, "Failed to open model residency profile: %s", path.c_str());
    std::fprintf(
        output_,
        "elapsed_ms,stage,resident_pages,total_pages,resident_percent,"
        "sidecar_resident_pages,sidecar_total_pages,sidecar_resident_percent,"
        "sidecar_error,sidecar_anonymous,"
        "rss_bytes,hwm_bytes,process_swap_bytes,mem_available_bytes,"
        "swap_free_bytes,swap_cached_bytes,minor_faults,major_faults,sample_ms,"
        "program_swap_bytes,rss_plus_program_swap_bytes\n");
    sample();
    worker_ = std::thread([this]() {
      while (!stop_.load(std::memory_order_relaxed)) {
        usleep(static_cast<useconds_t>(interval_ms_) * 1000U);
        if (!stop_.load(std::memory_order_relaxed)) {
          sample();
        }
      }
    });
  }

  ~JointModelResidencyProfiler() {
    stop();
  }

  void set_stage(JointResidencyStage stage) {
    stage_.store(stage, std::memory_order_relaxed);
    sample();
  }

  void set_sidecar(const void* data, size_t size, bool anonymous) {
    {
      std::lock_guard<std::mutex> lock(sample_mutex_);
      sidecar_data_ = data;
      sidecar_size_ = size;
      sidecar_anonymous_ = anonymous;
    }
    sample();
  }

  void stop() {
    if (stopped_.exchange(true, std::memory_order_relaxed)) {
      return;
    }
    stop_.store(true, std::memory_order_relaxed);
    if (worker_.joinable()) {
      worker_.join();
    }
    sample();
    if (output_ != nullptr) {
      std::fclose(output_);
      output_ = nullptr;
    }
  }

 private:
  void sample() {
    std::lock_guard<std::mutex> lock(sample_mutex_);
    if (output_ == nullptr) {
      return;
    }
    const auto sample_start = SteadyClock::now();
    const auto residency = source_->residency();
    const auto sidecar_residency =
        raw_memory_residency(sidecar_data_, sidecar_size_);
    const auto memory = process_memory_status_snapshot();
    const uint64_t process_swap_bytes =
        read_proc_status_bytes("/proc/self/status", "VmSwap:");
    const uint64_t page_size =
        static_cast<uint64_t>(std::max<long>(sysconf(_SC_PAGESIZE), 1));
    const uint64_t nonresident_model_bytes =
        static_cast<uint64_t>(residency.total_pages - residency.resident_pages) *
        page_size;
    const uint64_t program_swap_bytes =
        process_swap_bytes + nonresident_model_bytes;
    const double percent = residency.total_pages == 0
        ? 0.0
        : 100.0 * static_cast<double>(residency.resident_pages) /
            static_cast<double>(residency.total_pages);
    const double sidecar_percent = sidecar_residency.total_pages == 0
        ? 0.0
        : 100.0 * static_cast<double>(sidecar_residency.resident_pages) /
              static_cast<double>(sidecar_residency.total_pages);
    const auto stage = stage_.load(std::memory_order_relaxed);
    std::fprintf(
        output_,
        "%.3f,%s,%zu,%zu,%.6f,%zu,%zu,%.6f,%d,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64
        ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
        ",%.3f,%" PRIu64 ",%" PRIu64 "\n",
        elapsed_ms(start_),
        joint_residency_stage_name(stage),
        residency.resident_pages,
        residency.total_pages,
        percent,
        sidecar_residency.resident_pages,
        sidecar_residency.total_pages,
        sidecar_percent,
        sidecar_residency.error,
        static_cast<int>(sidecar_anonymous_),
        memory.rss_bytes,
        memory.hwm_bytes,
        process_swap_bytes,
        read_proc_status_bytes("/proc/meminfo", "MemAvailable:"),
        read_proc_status_bytes("/proc/meminfo", "SwapFree:"),
        read_proc_status_bytes("/proc/meminfo", "SwapCached:"),
        memory.minor_faults,
        memory.major_faults,
        elapsed_ms(sample_start),
        program_swap_bytes,
        memory.rss_bytes + program_swap_bytes);
    std::fflush(output_);
  }

  std::shared_ptr<example::ReadOnlyMappedFile> source_;
  const void* sidecar_data_{nullptr};
  size_t sidecar_size_{0};
  bool sidecar_anonymous_{false};
  int interval_ms_{250};
  SteadyClock::time_point start_;
  std::atomic<JointResidencyStage> stage_{JointResidencyStage::ModelReady};
  std::atomic<bool> stop_{false};
  std::atomic<bool> stopped_{false};
  FILE* output_{nullptr};
  std::mutex sample_mutex_;
  std::thread worker_;
};

std::unique_ptr<JointModelResidencyProfiler> g_joint_residency_profiler;

void start_joint_model_residency_profiler() {
  if (FLAGS_model_residency_profile_path.empty()) {
    return;
  }
  g_joint_residency_profiler = std::make_unique<JointModelResidencyProfiler>(
      g_joint_model_source,
      FLAGS_model_residency_profile_path,
      FLAGS_model_residency_profile_interval_ms);
}

void attach_joint_sidecar_residency_source(
    llama_pd_inprocess_runtime* runtime) {
  if (runtime == nullptr) {
    return;
  }
  const void* data = nullptr;
  size_t size = 0;
  bool anonymous = false;
  if (!llama_pd_inprocess_runtime_profile_backing(
          runtime, &data, &size, &anonymous)) {
    g_joint_sidecar_data = nullptr;
    g_joint_sidecar_size = 0;
    g_joint_sidecar_anonymous = true;
    if (g_joint_residency_profiler) {
      g_joint_residency_profiler->set_sidecar(nullptr, 0, true);
    }
    ET_LOG(
        Info,
        "Decode sidecar uses discontiguous per-shard buffers; contiguous "
        "residency source is unavailable");
    return;
  }
  g_joint_sidecar_data = data;
  g_joint_sidecar_size = size;
  g_joint_sidecar_anonymous = anonymous;
  if (g_joint_residency_profiler) {
    g_joint_residency_profiler->set_sidecar(data, size, anonymous);
  }
  const auto residency = raw_memory_residency(data, size);
  ET_LOG(
      Info,
      "Decode sidecar residency source attached: ptr=%p bytes=%zu "
      "anonymous=%d resident_pages=%zu total_pages=%zu error=%d",
      data,
      size,
      static_cast<int>(anonymous),
      residency.resident_pages,
      residency.total_pages,
      residency.error);
}

class JointSidecarShardStreamer {
 public:
  JointSidecarShardStreamer(
      llama_pd_inprocess_runtime* runtime,
      size_t shard_count)
      : runtime_(runtime),
        shard_count_(shard_count),
        enqueued_(shard_count, false) {}

  ~JointSidecarShardStreamer() {
    stop_worker();
  }

  bool prepare() {
    const auto start = SteadyClock::now();
    if (!llama_pd_inprocess_runtime_profile_stream_prepare(
            runtime_, shard_count_)) {
      return false;
    }
    worker_ = std::thread([this]() { worker_loop(); });
    ET_LOG(
        Info,
        "Decode per-shard sidecar buffers released for streaming: "
        "shards=%zu ms=%.3f",
        shard_count_,
        elapsed_ms(start));
    return true;
  }

  bool enqueue(size_t shard_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closing_ || failed_ || shard_index >= shard_count_ ||
        enqueued_[shard_index]) {
      return false;
    }
    enqueued_[shard_index] = true;
    queue_.push_back(shard_index);
    cv_.notify_one();
    return true;
  }

  bool finish(double* boundary_wait_ms, size_t* bytes_loaded) {
    const auto wait_start = SteadyClock::now();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closing_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
    const double wait_ms = elapsed_ms(wait_start);
    if (boundary_wait_ms != nullptr) {
      *boundary_wait_ms = wait_ms;
    }
    bool valid = false;
    size_t loaded = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      valid = !failed_ && completed_ == shard_count_;
      loaded = total_bytes_loaded_;
    }
    if (bytes_loaded != nullptr) {
      *bytes_loaded = loaded;
    }
    if (!valid ||
        !llama_pd_inprocess_runtime_profile_stream_finish(runtime_)) {
      return false;
    }
    finished_ = true;
    ET_LOG(
        Info,
        "Decode sidecar streaming complete: shards=%zu bytes=%zu "
        "boundary_wait_ms=%.3f io_total_ms=%.3f",
        shard_count_,
        loaded,
        wait_ms,
        io_total_ms_);
    return true;
  }

 private:
  void worker_loop() {
    while (true) {
      size_t shard_index = 0;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return closing_ || !queue_.empty(); });
        if (queue_.empty()) {
          return;
        }
        shard_index = queue_.front();
        queue_.pop_front();
      }
      const auto io_start = SteadyClock::now();
      size_t loaded = 0;
      const bool ok = llama_pd_inprocess_runtime_profile_stream_fill(
          runtime_, shard_index, &loaded);
      const double io_ms = elapsed_ms(io_start);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ok) {
          failed_ = true;
          return;
        }
        ++completed_;
        total_bytes_loaded_ += loaded;
        io_total_ms_ += io_ms;
      }
      const auto memory = process_memory_status_snapshot();
      ET_LOG(
          Info,
          "Decode sidecar shard chunk loaded: shard=%zu bytes=%zu ms=%.3f "
          "rss_mib=%.2f hwm_mib=%.2f vmswap_mib=%.2f",
          shard_index,
          loaded,
          io_ms,
          memory.rss_bytes / (1024.0 * 1024.0),
          memory.hwm_bytes / (1024.0 * 1024.0),
          read_proc_status_bytes("/proc/self/status", "VmSwap:") /
              (1024.0 * 1024.0));
    }
  }

  void stop_worker() {
    if (finished_ || !worker_.joinable()) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closing_ = true;
    }
    cv_.notify_all();
    worker_.join();
  }

  llama_pd_inprocess_runtime* runtime_{nullptr};
  size_t shard_count_{0};
  std::vector<bool> enqueued_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<size_t> queue_;
  std::thread worker_;
  bool closing_{false};
  bool failed_{false};
  bool finished_{false};
  size_t completed_{0};
  size_t total_bytes_loaded_{0};
  double io_total_ms_{0.0};
};

void set_joint_model_residency_stage(JointResidencyStage stage) {
  if (g_joint_residency_profiler) {
    g_joint_residency_profiler->set_stage(stage);
  }
}

void stop_joint_model_residency_profiler() {
  if (g_joint_residency_profiler) {
    g_joint_residency_profiler->stop();
    g_joint_residency_profiler.reset();
  }
}

std::vector<uint64_t> joint_tokenize_prompt(
    const std::string& formatted_prompt,
    bool tokenized_prompt) {
  if (tokenized_prompt) {
    const auto bytes = read_binary_file(formatted_prompt);
    ET_CHECK_MSG(
        bytes.size() % sizeof(uint64_t) == 0,
        "tokenized prompt size is not uint64 aligned");
    std::vector<uint64_t> tokens(bytes.size() / sizeof(uint64_t));
    std::memcpy(tokens.data(), bytes.data(), bytes.size());
    return tokens;
  }

  llama_model_params model_params = llama_model_default_params();
  model_params.vocab_only = true;
  std::optional<std::string> deferred_profile_path;
  if (const char* path = std::getenv("LLAMA_QNN_U16_QPARAMS_MANIFEST");
      path != nullptr && path[0] != '\0') {
    deferred_profile_path = path;
    unsetenv("LLAMA_QNN_U16_QPARAMS_MANIFEST");
  }
  std::unique_ptr<llama_model, decltype(&llama_model_free)> tokenizer_model(
      llama_model_load_from_buffer(
          g_joint_model_source->data(),
          g_joint_model_source->size(),
          model_params),
      llama_model_free);
  if (deferred_profile_path.has_value()) {
    setenv(
        "LLAMA_QNN_U16_QPARAMS_MANIFEST",
        deferred_profile_path->c_str(),
        1);
  }
  ET_CHECK_MSG(
      tokenizer_model != nullptr,
      "failed to create shared llama.cpp tokenizer");
  const llama_vocab* vocab = llama_model_get_vocab(tokenizer_model.get());
  const int32_t required = -llama_tokenize(
      vocab,
      formatted_prompt.data(),
      static_cast<int32_t>(formatted_prompt.size()),
      nullptr,
      0,
      true,
      true);
  ET_CHECK_MSG(required > 0, "shared llama.cpp tokenizer returned no tokens");
  std::vector<llama_token> llama_tokens(static_cast<size_t>(required));
  const int32_t count = llama_tokenize(
      vocab,
      formatted_prompt.data(),
      static_cast<int32_t>(formatted_prompt.size()),
      llama_tokens.data(),
      required,
      true,
      true);
  ET_CHECK_MSG(count == required, "shared llama.cpp tokenizer size changed");
  std::vector<uint64_t> tokens;
  tokens.reserve(llama_tokens.size());
  for (llama_token token : llama_tokens) {
    tokens.push_back(static_cast<uint64_t>(token));
  }
  ET_LOG(
      Info,
      "Joint tokenizer produced %zu tokens from model_ptr=%p",
      tokens.size(),
      g_joint_model_source->data());
  return tokens;
}

struct ResidentDecodeProcess {
  pid_t pid{getpid()};
  int control_fd{-1};
  bool runtime_prepare_sent{false};
  std::string shared_embedding_matrix_path;
  std::vector<std::string> args;
  llama_pd_inprocess_runtime* runtime{nullptr};
  DecodeProcessResult result{};
};

std::vector<std::string> make_joint_decode_args(
    const std::string& shared_embedding_matrix_path) {
  std::vector<std::string> args = {
      "qnn_llama_pd_joint_runner",
      "-m",
      "in-process.gguf",
      "-n",
      std::to_string(FLAGS_decode_n_predict),
      "-c",
      std::to_string(FLAGS_decode_ctx),
      "-b",
      "1",
      "-ub",
      "1",
      "-ngl",
      std::to_string(FLAGS_decode_ngl),
      "--temp",
      std::to_string(FLAGS_decode_temp),
      "-fit",
      "off",
  };
  if (FLAGS_decode_threads > 0) {
    args.push_back("-t");
    args.push_back(std::to_string(FLAGS_decode_threads));
  }
  if (!shared_embedding_matrix_path.empty()) {
    args.push_back("--pd-disk-embedding");
    args.push_back(shared_embedding_matrix_path);
  }
  if (FLAGS_decode_native_compare) {
    args.push_back("--pd-native-compare");
  }
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
  if (FLAGS_decode_import_ro) {
    args.push_back("--pd-import-ro");
  }
  return args;
}

ResidentDecodeProcess start_resident_decode_process(
    const std::string& shared_embedding_matrix_path) {
  ET_CHECK_MSG(
      g_joint_model_source && !g_joint_model_source->empty(),
      "joint PD runner requires the Prefill GGUF mapping to remain alive");
  ResidentDecodeProcess resident;
  resident.shared_embedding_matrix_path = shared_embedding_matrix_path;
  resident.args = make_joint_decode_args(shared_embedding_matrix_path);
  resident.result.before = process_memory_snapshot();
  resident.result.exit_code = 0;
  if (FLAGS_decode_defer_runtime_until_after_prefill &&
      FLAGS_decode_stage_model_only_before_prefill) {
    std::vector<char*> argv;
    argv.reserve(resident.args.size() + 1);
    for (auto& arg : resident.args) {
      argv.push_back(arg.data());
    }
    argv.push_back(nullptr);
    resident.runtime = llama_pd_inprocess_runtime_create_model_only(
        static_cast<int>(resident.args.size()),
        argv.data(),
        g_joint_model_source->data(),
        g_joint_model_source->size(),
        &resident.result.startup_ms);
    ET_CHECK_MSG(
        resident.runtime != nullptr,
        "failed to create model-only joint Decode runtime");
    ET_LOG(
        Info,
        "Joint Decode model-only initialization complete before Prefill: "
        "model_ptr=%p model_bytes=%zu initialization_ms=%.3f",
        g_joint_model_source->data(),
        g_joint_model_source->size(),
        resident.result.startup_ms);
    return resident;
  }
  if (FLAGS_decode_defer_runtime_until_after_prefill) {
    ET_LOG(
        Info,
        "Joint Decode runtime creation deferred until after Prefill: "
        "model_ptr=%p model_bytes=%zu",
        g_joint_model_source->data(),
        g_joint_model_source->size());
    return resident;
  }
  std::vector<char*> argv;
  argv.reserve(resident.args.size() + 1);
  for (auto& arg : resident.args) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);
  resident.runtime = llama_pd_inprocess_runtime_create(
      static_cast<int>(resident.args.size()),
      argv.data(),
      g_joint_model_source->data(),
      g_joint_model_source->size(),
      &resident.result.startup_ms);
  ET_CHECK_MSG(
      resident.runtime != nullptr,
      "failed to create persistent joint Decode runtime");
  ET_LOG(
      Info,
      "Joint Decode initialized in-process before Prefill: model_ptr=%p "
      "model_bytes=%zu initialization_ms=%.3f",
      g_joint_model_source->data(),
      g_joint_model_source->size(),
      resident.result.startup_ms);
  return resident;
}

void start_resident_decode_runtime_prepare(int) {
  if (FLAGS_decode_defer_runtime_until_after_prefill) {
    ET_LOG(
        Info,
        "Prefill rebuild buffers released; deferred joint Decode will create "
        "its runtime at the direct-pointer boundary");
  } else {
    ET_LOG(
        Info,
        "Prefill rebuild buffers released; resident joint Decode runtime is ready");
  }
}

void begin_resident_decode_handoff(
    ResidentDecodeProcess& resident,
    const example::PDPrefillRunner<uint8_t>::MemoryHandoff& memory_handoff) {
  ET_CHECK_MSG(
      memory_handoff.direct_pointer && memory_handoff.bytes &&
          !memory_handoff.bytes->empty(),
      "joint Decode requires a direct-pointer KV handoff");

  if (FLAGS_decode_defer_runtime_until_after_prefill &&
      FLAGS_decode_stage_model_only_before_prefill &&
      resident.runtime != nullptr) {
    set_joint_model_residency_stage(JointResidencyStage::DecodeContext);
    double context_ms = 0.0;
    double warmup_ms = 0.0;
    ET_CHECK_MSG(
        llama_pd_inprocess_runtime_prepare_context(
            resident.runtime, &context_ms),
        "failed to prepare deferred joint Decode context");
    log_joint_model_residency("before_warmup");
    set_joint_model_residency_stage(JointResidencyStage::DecodeWarmup);
    ET_CHECK_MSG(
        llama_pd_inprocess_runtime_warmup(
            resident.runtime, &warmup_ms),
        "failed to warm deferred joint Decode runtime");
    log_joint_model_residency("after_warmup");
    set_joint_model_residency_stage(JointResidencyStage::Handoff);
    resident.result.startup_ms += context_ms + warmup_ms;
    ET_LOG(
        Info,
        "Joint Decode deferred context ready after Prefill: "
        "context_ms=%.3f warmup_ms=%.3f initialization_ms=%.3f",
        context_ms,
        warmup_ms,
        resident.result.startup_ms);
  }

  if (resident.runtime == nullptr) {
    set_joint_model_residency_stage(JointResidencyStage::DecodeContext);
    const auto deferred_init_start = SteadyClock::now();
    std::vector<char*> argv;
    argv.reserve(resident.args.size() + 1);
    for (auto& arg : resident.args) {
      argv.push_back(arg.data());
    }
    argv.push_back(nullptr);
    resident.runtime = llama_pd_inprocess_runtime_create(
        static_cast<int>(resident.args.size()),
        argv.data(),
        g_joint_model_source->data(),
        g_joint_model_source->size(),
        &resident.result.startup_ms);
    ET_CHECK_MSG(
        resident.runtime != nullptr,
        "failed to create deferred joint Decode runtime");
    attach_joint_sidecar_residency_source(resident.runtime);
    ET_LOG(
        Info,
        "Joint Decode initialized in-process after Prefill: model_ptr=%p "
        "model_bytes=%zu initialization_ms=%.3f wall_ms=%.3f",
        g_joint_model_source->data(),
        g_joint_model_source->size(),
        resident.result.startup_ms,
        elapsed_ms(deferred_init_start));
  }

  llama_pd_inprocess_request request{
      g_joint_model_source->data(),
      g_joint_model_source->size(),
      memory_handoff.bytes->data(),
      memory_handoff.bytes->size(),
      memory_handoff.prompt_length,
      memory_handoff.num_layers,
      memory_handoff.num_kv_heads,
      memory_handoff.head_dim,
      static_cast<int32_t>(memory_handoff.first_token),
      memory_handoff.first_token_is_prompt_tail,
      nullptr,
  };
  llama_pd_inprocess_result inprocess_result{};
  request.result = &inprocess_result;
  if (FLAGS_model_residency_probe) {
    request.decode_event_callback = joint_pd_decode_event_probe;
    request.decode_event_call_limit = 2;
  }
  set_joint_model_residency_stage(JointResidencyStage::Decode);
  const auto decode_start = SteadyClock::now();
  resident.result.exit_code =
      llama_pd_inprocess_runtime_run(resident.runtime, &request);
  resident.result.process_wall_ms = elapsed_ms(decode_start);
  set_joint_model_residency_stage(JointResidencyStage::DecodeComplete);
  resident.result.startup_ms = inprocess_result.initialization_ms;
  resident.result.boundary_ms = inprocess_result.boundary_ms;
  resident.result.generation_ms = inprocess_result.generation_ms;
  resident.result.generated_tokens = inprocess_result.generated_tokens;
  resident.result.wall_ms =
      resident.result.boundary_ms + resident.result.generation_ms;
  resident.result.after = process_memory_snapshot();
  resident.result.process_tree_peak_rss_bytes =
      resident.result.after.hwm_bytes;
  resident.result.process_tree_peak_pss_bytes =
      resident.result.after.pss_bytes;
  resident.runtime_prepare_sent = true;
  ET_LOG(
      Info,
      "Joint Decode completed in-process: model_ptr=%p kv_ptr=%p "
      "initialization_once_ms=%.3f boundary_ms=%.3f generation_ms=%.3f "
      "decode_round_ms=%.3f process_wall_ms=%.3f",
      request.model_data,
      request.handoff_data,
      resident.result.startup_ms,
      resident.result.boundary_ms,
      resident.result.generation_ms,
      resident.result.wall_ms,
      resident.result.process_wall_ms);
}

DecodeProcessResult finish_resident_decode_process(
    ResidentDecodeProcess resident) {
  if (resident.runtime != nullptr) {
    llama_pd_inprocess_runtime_destroy(resident.runtime);
    resident.runtime = nullptr;
  }
  return resident.result;
}
#endif

template <typename T>
void run_wikitext_ppl(
    ModuleBundle module_bundle,
    const ModuleMetaInfo& module_meta,
    PrefillShardFiles prefill_shard_files,
    example::DecoderRunner::PrefillShardRebuildConfig prefill_shard_rebuild,
    std::unique_ptr<executorch::extension::Module> attention_sink_rope_module) {
  const bool effective_outputs_logits = FLAGS_prefill_no_output
      ? false
      : (FLAGS_prefill_force_logits || prefill_shard_files.outputs_logits);
  const bool effective_separate_embed =
      FLAGS_prefill_separate_embed || prefill_shard_files.use_separate_embed;
  const std::string effective_embedding_matrix_path =
      !FLAGS_prefill_embedding_matrix_path.empty()
      ? FLAGS_prefill_embedding_matrix_path
      : prefill_shard_files.embedding_matrix_path;
  ET_CHECK_MSG(effective_outputs_logits, "QNN WikiPPL requires prefill logits");

  typename example::PDPrefillRunner<T>::StaticMetadata static_metadata;
  if (prefill_shard_files.qwen3_static_plan) {
    static_metadata.enabled = true;
    static_metadata.context_len = prefill_shard_files.context_len;
    static_metadata.prompt_ar_len = prefill_shard_files.prefill_ar_len;
    static_metadata.token_generator_ar_len =
        prefill_shard_files.token_generator_ar_len;
    static_metadata.vocab_size = prefill_shard_files.vocab_size;
    static_metadata.sliding_window = prefill_shard_files.context_len;
    static_metadata.num_layers = prefill_shard_files.num_layers;
    static_metadata.num_heads = prefill_shard_files.num_heads;
    static_metadata.head_dim = prefill_shard_files.head_dim;
    static_metadata.use_int64_token = prefill_shard_files.use_int64_token;
    static_metadata.cache_mode = CacheMode::StaticCahce;
    static_metadata.outputs_logits = true;
    static_metadata.use_separate_embed = effective_separate_embed;
    static_metadata.embedding_matrix_path = effective_embedding_matrix_path;
    static_metadata.resident_embedding = FLAGS_prefill_embedding_resident;
    static_metadata.embedding_qnn_u16_input = prefill_shard_files.embedding_qnn_u16_input;
    static_metadata.embedding_qnn_u16_scale = prefill_shard_files.embedding_qnn_u16_scale;
    static_metadata.embedding_qnn_u16_zero_point = prefill_shard_files.embedding_qnn_u16_zero_point;
  }

  example::PDPrefillRunner<T> runner(
      std::move(module_bundle.module),
      std::move(prefill_shard_files.pte_paths),
      std::move(prefill_shard_files.index_bin_paths),
      std::move(prefill_shard_rebuild),
      prefill_shard_files.qwen3_static_plan,
      prefill_shard_files.static_aux_size,
      prefill_shard_files.static_hidden_size,
      true,
      effective_separate_embed,
      effective_embedding_matrix_path,
      FLAGS_prefill_embedding_resident,
      static_metadata,
      FLAGS_decoder_model_version.c_str(),
      get_model_path_for_runner(),
      FLAGS_tokenizer_path.c_str(),
      module_bundle.pte_bytes,
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      nullptr,
      std::move(attention_sink_rope_module));

  double wiki_ppl = 0.0;
  int64_t scored_tokens = 0;
  const auto ppl_error = runner.evaluate_wikitext_ppl(
      FLAGS_wikitext_path,
      FLAGS_wikitext_start_token,
      FLAGS_wikitext_max_tokens,
      module_meta.logits_scale,
      module_meta.logits_zero_point,
      &wiki_ppl,
      &scored_tokens);
  ET_CHECK_MSG(
      ppl_error == executorch::runtime::Error::Ok,
      "Failed to evaluate WikiText perplexity");

  std::ofstream fout(FLAGS_output_path.c_str());
  fout << std::setprecision(15);
  fout << "wiki_ppl=" << wiki_ppl << "\n";
  fout << "scored_tokens=" << scored_tokens << "\n";
  fout << "logits_scale=" << module_meta.logits_scale << "\n";
  fout << "logits_zero_point=" << module_meta.logits_zero_point << "\n";
  fout.close();
  ET_LOG(
      Info,
      "wiki_ppl=%f scored_tokens=%ld (pure QNN sharded prefill)",
      wiki_ppl,
      scored_tokens);
}

template <typename T>
struct PdPrefillSession {
  std::unique_ptr<example::PDPrefillRunner<T>> runner;
  example::DecoderModelVersion decoder_version;
  ProcessMemorySnapshot before_runner;
  ProcessMemorySnapshot after_runner;
  std::string shared_embedding_matrix_path;
  double runner_setup_ms{0.0};
  size_t request_count{0};
};

template <typename T>
PdPrefillSession<T> create_pd_prefill_session(
    ModuleBundle module_bundle,
    PrefillShardFiles prefill_shard_files,
    example::DecoderRunner::PrefillShardRebuildConfig prefill_shard_rebuild,
    std::unique_ptr<executorch::extension::Module> attention_sink_rope_module) {
  ET_CHECK_MSG(
      !(FLAGS_prefill_no_output && FLAGS_prefill_force_logits),
      "--prefill_no_output and --prefill_force_logits cannot both be true");
  const bool effective_outputs_logits = FLAGS_prefill_no_output
      ? false
      : (FLAGS_prefill_force_logits || prefill_shard_files.outputs_logits);
  const bool effective_separate_embed =
      FLAGS_prefill_separate_embed || prefill_shard_files.use_separate_embed;
  const std::string effective_embedding_matrix_path =
      !FLAGS_prefill_embedding_matrix_path.empty()
      ? FLAGS_prefill_embedding_matrix_path
      : prefill_shard_files.embedding_matrix_path;
  ET_CHECK_MSG(
      !effective_separate_embed || !effective_embedding_matrix_path.empty(),
      "Separate prefill embedding requires --prefill_embedding_matrix_path or separate_embed_matrix in the shard manifest");
  ET_CHECK_MSG(
      !effective_separate_embed || !FLAGS_prefill_embedding_resident,
      "Separate embedding must remain non-resident in the PD pipeline; "
      "--prefill_embedding_resident=true is not supported");

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
    static_metadata.outputs_logits = effective_outputs_logits;
    static_metadata.use_separate_embed = effective_separate_embed;
    static_metadata.embedding_matrix_path = effective_embedding_matrix_path;
    static_metadata.resident_embedding = false;
    static_metadata.embedding_qnn_u16_input = prefill_shard_files.embedding_qnn_u16_input;
    static_metadata.embedding_qnn_u16_scale = prefill_shard_files.embedding_qnn_u16_scale;
    static_metadata.embedding_qnn_u16_zero_point = prefill_shard_files.embedding_qnn_u16_zero_point;
  }

  PdPrefillSession<T> session;
  session.before_runner = process_memory_snapshot();
  const auto runner_setup_start = SteadyClock::now();
  session.runner = std::make_unique<example::PDPrefillRunner<T>>(
      std::move(module_bundle.module),
      std::move(prefill_shard_files.pte_paths),
      std::move(prefill_shard_files.index_bin_paths),
      std::move(prefill_shard_rebuild),
      prefill_shard_files.qwen3_static_plan,
      prefill_shard_files.static_aux_size,
      prefill_shard_files.static_hidden_size,
      effective_outputs_logits,
      effective_separate_embed,
      effective_embedding_matrix_path,
      FLAGS_prefill_embedding_resident,
      static_metadata,
      FLAGS_decoder_model_version.c_str(),
      get_model_path_for_runner(),
      FLAGS_tokenizer_path.c_str(),
      module_bundle.pte_bytes,
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      nullptr,
      std::move(attention_sink_rope_module));
  if (!FLAGS_prefill_etdump_dir.empty()) {
    ET_CHECK_MSG(
        FLAGS_prefill_etdump_shard >= 0,
        "--prefill_etdump_shard must be non-negative");
    ET_CHECK_MSG(
        FLAGS_prefill_etdump_debug_buffer_bytes > 0,
        "--prefill_etdump_debug_buffer_bytes must be positive");
    session.runner->set_prefill_etdump_config({
        FLAGS_prefill_etdump_dir,
        FLAGS_prefill_etdump_shard,
        static_cast<size_t>(FLAGS_prefill_etdump_debug_buffer_bytes),
    });
  }
  session.runner_setup_ms = elapsed_ms(runner_setup_start);
  session.after_runner = process_memory_snapshot();
  session.shared_embedding_matrix_path = effective_separate_embed
      ? effective_embedding_matrix_path
      : std::string{};
  session.decoder_version = session.runner->get_decoder_model_version().get();
  return session;
}

template <typename T>
PdE2ERuntimeStats run_pd_e2e_request(
    PdPrefillSession<T>& session,
    const std::string& prompt_input,
    bool tokenized_prompt) {
  ET_CHECK_MSG(session.runner != nullptr, "PD Prefill session is not initialized");
  PdE2ERuntimeStats stats;
#ifdef QNN_LLAMA_PD_JOINT
  stats.memory_handoff.direct_pointer = true;
#endif
  stats.shared_embedding_matrix_path = session.shared_embedding_matrix_path;
  stats.before_runner = session.request_count == 0
      ? session.before_runner
      : process_memory_status_snapshot();
  stats.after_runner = session.request_count == 0
      ? session.after_runner
      : stats.before_runner;
  stats.runner_setup_ms =
      session.request_count == 0 ? session.runner_setup_ms : 0.0;
  session.runner->begin_request();

  const std::string formatted_prompt = tokenized_prompt
      ? prompt_input
      : get_formatted_prompt(
            prompt_input, FLAGS_system_prompt, session.decoder_version);
#ifdef QNN_LLAMA_PD_JOINT
  const auto prepare_overlap_start = SteadyClock::now();
  auto tokenize_future = std::async(
      std::launch::async,
      [formatted_prompt, tokenized_prompt]() {
        const auto tokenize_start = SteadyClock::now();
        auto tokens =
            joint_tokenize_prompt(formatted_prompt, tokenized_prompt);
        return std::make_pair(
            std::move(tokens), elapsed_ms(tokenize_start));
      });
  const auto prepare_start = SteadyClock::now();
  ET_CHECK_MSG(
      session.runner->load() == executorch::runtime::Error::Ok,
      "Failed to prepare joint QNN Prefill runner");
  stats.prefill_prepare_ms = elapsed_ms(prepare_start);
  auto tokenization = tokenize_future.get();
  stats.bootstrap_tokenize_ms = tokenization.second;
  stats.prepare_overlap_wall_ms = elapsed_ms(prepare_overlap_start);
  session.runner->set_prefill_tokens(std::move(tokenization.first));
  ET_LOG(
      Info,
      "Joint Prefill initialization overlap: prepare_ms=%.3f "
      "tokenize_ms=%.3f wall_ms=%.3f hidden_ms=%.3f",
      stats.prefill_prepare_ms,
      stats.bootstrap_tokenize_ms,
      stats.prepare_overlap_wall_ms,
      stats.prefill_prepare_ms + stats.bootstrap_tokenize_ms -
          stats.prepare_overlap_wall_ms);
#endif
  const auto qnn_export_start = SteadyClock::now();
  ET_CHECK_MSG(
      session.runner->export_prefill_memory_handoff(
          formatted_prompt,
          tokenized_prompt,
          FLAGS_seq_len,
          &stats.memory_handoff) == executorch::runtime::Error::Ok,
      "PD prefill export failed");
  stats.qnn_export_total_ms = elapsed_ms(qnn_export_start);
  const auto runner_stats = session.runner->last_runtime_stats();
  stats.prompt_tokens = runner_stats.prompt_tokens;
  stats.prefill.prompt_tokens = runner_stats.prompt_tokens;
  stats.prefill.tokenize_ms = runner_stats.tokenize_ms;
  stats.prefill.embedding_prepare_ms = runner_stats.embedding_prepare_ms;
  stats.prefill.prefill_ms = runner_stats.prefill_ms;
  stats.prefill.handoff_total_ms = runner_stats.handoff_total_ms;
  stats.prefill.kv_layout_ms = runner_stats.kv_layout_ms;
  stats.prefill.kv_write_ms = runner_stats.kv_write_ms;
  stats.prefill.fingerprint_ms = runner_stats.fingerprint_ms;
  stats.shard_stats = session.runner->prefill_shard_runtime_stats();
  stats.qnn_backend_prewarm_ms =
      session.runner->prefill_qnn_backend_prewarm_ms();
  stats.qnn_backend_prewarmed =
      session.runner->prefill_qnn_backend_prewarmed();
  stats.persistent_shard0_prepare_ms =
      session.runner->prefill_persistent_shard0_prepare_ms();
  stats.persistent_shard0_prepared =
      session.runner->prefill_persistent_shard0_prepared();
  // The handoff must be sent immediately after Prefill. Reading
  // smaps_rollup here can stall for tens of milliseconds; the concurrent
  // dense monitor already records PSS for the full process tree.
  stats.after_export = process_memory_status_snapshot();
  ++session.request_count;
  return stats;
}

} // namespace

int qnn_llama_pd_e2e_main(int argc, char** argv) {
  std::vector<std::string> prompts = CollectPrompts(argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  const bool use_session_prompt_file = !FLAGS_session_prompts_path.empty();
  ET_CHECK_MSG(
      !use_session_prompt_file || prompts.empty(),
      "--session_prompts_path cannot be combined with --prompt");
  append_session_prompts(FLAGS_session_prompts_path, &prompts);

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
  ET_CHECK_MSG(
      !use_tokenized_prompt || !use_session_prompt_file,
      "--session_prompts_path cannot be combined with --tokenized_prompt");
  if (!FLAGS_wikitext_path.empty()) {
    ET_CHECK_MSG(
        gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default,
        "tokenized_prompt is not supported in wikitext PPL mode");
  } else {
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
      !(FLAGS_prefill_no_output && FLAGS_prefill_force_logits),
      "--prefill_no_output and --prefill_force_logits cannot both be true");
  const bool effective_prefill_no_output = FLAGS_prefill_no_output ||
      (!FLAGS_prefill_force_logits && !prefill_shard_files.outputs_logits);
  const bool effective_prefill_separate_embed =
      FLAGS_prefill_separate_embed || prefill_shard_files.use_separate_embed;
  ET_CHECK_MSG(
      !effective_prefill_no_output || manifest_only_prefill,
      "--prefill_no_output requires a manifest-only static prefill shard export");
  ET_CHECK_MSG(
      !effective_prefill_separate_embed || manifest_only_prefill ||
          !FLAGS_wikitext_path.empty(),
      "--prefill_separate_embed requires static prefill shards");
  ET_CHECK_MSG(
      !effective_prefill_separate_embed ||
          !FLAGS_prefill_embedding_matrix_path.empty() ||
          !prefill_shard_files.embedding_matrix_path.empty(),
      "--prefill_embedding_matrix_path is required when --prefill_separate_embed=true");

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
    if (!FLAGS_wikitext_path.empty()) {
      ET_CHECK_MSG(
          FLAGS_wikitext_logits_scale > 0.0,
          "Manifest-only WikiPPL requires --wikitext_logits_scale");
      module_meta.logits_scale =
          static_cast<float>(FLAGS_wikitext_logits_scale);
      module_meta.logits_zero_point = FLAGS_wikitext_logits_zero_point;
    }
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
  const size_t request_count = use_tokenized_prompt ? 1 : prompts.size();
  const bool use_decode_sidecar_streaming =
      FLAGS_decode_stream_sidecar_during_prefill && !FLAGS_prefill_only &&
      request_count == 1;
  if (FLAGS_decode_stream_sidecar_during_prefill &&
      !use_decode_sidecar_streaming) {
    ET_LOG(
        Info,
        "Decode sidecar streaming disabled for this run: prefill_only=%d "
        "request_count=%zu",
        FLAGS_prefill_only,
        request_count);
  }
  if (prefill_shard_rebuild.release_stripped_pte_after_rebuild &&
      (FLAGS_prefill_only || request_count > 1)) {
    prefill_shard_rebuild.release_stripped_pte_after_rebuild = false;
    ET_LOG(
        Info,
        "retaining stripped Prefill PTE for reusable session: "
        "prefill_only=%d request_count=%zu",
        FLAGS_prefill_only,
        request_count);
  }
  if (request_count > 1 &&
      prefill_shard_rebuild.release_prepared_shard0_after_execute) {
    prefill_shard_rebuild.release_prepared_shard0_after_execute = false;
    ET_LOG(
        Info,
        "retaining persistent shard0 across a multi-request session: "
        "request_count=%zu",
        request_count);
  }
  ET_CHECK_MSG(
      !FLAGS_prefill_unload_shard0_method_after_execute ||
          request_count == 1,
      "--prefill_unload_shard0_method_after_execute is single-request only");
  ET_CHECK_MSG(
      !FLAGS_prefill_destroy_shard0_module_keep_pte_after_execute ||
          request_count == 1,
      "--prefill_destroy_shard0_module_keep_pte_after_execute is "
      "single-request only");
  ET_CHECK_MSG(
      !FLAGS_prefill_discard_shard0_pte_pages_after_execute ||
          FLAGS_prefill_destroy_shard0_module_keep_pte_after_execute,
      "--prefill_discard_shard0_pte_pages_after_execute requires "
      "--prefill_destroy_shard0_module_keep_pte_after_execute");
  ET_CHECK_MSG(
      !(FLAGS_prefill_unload_shard0_method_after_execute &&
        FLAGS_prefill_destroy_shard0_module_keep_pte_after_execute),
      "shard0 method-only and Module-only diagnostics are mutually exclusive");
#ifdef QNN_LLAMA_PD_JOINT
  ET_CHECK_MSG(
      FLAGS_model_ram_store || FLAGS_model_anonymous_buffer,
      "joint PD runner requires --model_ram_store=true or "
      "--model_anonymous_buffer=true so the one-time GGUF read completes "
      "before Prefill/Decode timing");
  ET_CHECK_MSG(
      prefill_shard_rebuild.mapped_source_bytes &&
          !prefill_shard_rebuild.mapped_source_bytes->empty(),
      "joint PD runner requires --gguf_model_path as the shared Prefill/Decode model");
  g_joint_model_source = prefill_shard_rebuild.mapped_source_bytes;
  log_joint_model_residency("after_model_load");
  ET_CHECK_MSG(
      !use_decode_sidecar_streaming ||
          (FLAGS_prefill_release_stripped_pte_after_rebuild &&
           FLAGS_decode_sidecar_reread_path.empty() &&
           !FLAGS_decode_sidecar_pretouch_mapping),
      "Decode sidecar streaming requires "
      "--prefill_release_stripped_pte_after_rebuild and no legacy "
      "sidecar reread/pretouch diagnostics");
#endif
  std::unique_ptr<executorch::extension::Module> attention_sink_rope_module;

  if (!FLAGS_wikitext_path.empty()) {
    if (module_meta.kv_bitwidth == example::KvBitWidth::kWidth8) {
      run_wikitext_ppl<uint8_t>(
          std::move(module_bundle),
          module_meta,
          std::move(prefill_shard_files),
          std::move(prefill_shard_rebuild),
          std::move(attention_sink_rope_module));
    } else if (module_meta.kv_bitwidth == example::KvBitWidth::kWidth16) {
      run_wikitext_ppl<uint16_t>(
          std::move(module_bundle),
          module_meta,
          std::move(prefill_shard_files),
          std::move(prefill_shard_rebuild),
          std::move(attention_sink_rope_module));
    } else {
      ET_CHECK_MSG(
          false,
          "Unsupported kv bitwidth: %ld",
          static_cast<int64_t>(module_meta.kv_bitwidth));
    }
    return 0;
  }

#ifdef QNN_LLAMA_PD_JOINT
  start_joint_model_residency_profiler();
#endif
  const auto e2e_start = SteadyClock::now();
  ET_CHECK_MSG(
      module_meta.kv_bitwidth == example::KvBitWidth::kWidth8,
      "PD E2E requires UINT8 KV");
  ET_LOG(Info, "Using in-memory PD handoff");

  const std::string resident_embedding_matrix_path =
      effective_prefill_separate_embed && FLAGS_decode_use_prefill_embedding
      ? (!FLAGS_prefill_embedding_matrix_path.empty()
             ? FLAGS_prefill_embedding_matrix_path
             : prefill_shard_files.embedding_matrix_path)
      : std::string{};
  ResidentDecodeProcess resident_decode;
#ifdef QNN_LLAMA_PD_JOINT
  std::unique_ptr<JointSidecarShardStreamer> sidecar_streamer;
#endif
  if (!FLAGS_prefill_only) {
#ifdef QNN_LLAMA_PD_JOINT
    set_joint_model_residency_stage(JointResidencyStage::DecodeModelInit);
#endif
    resident_decode =
        start_resident_decode_process(resident_embedding_matrix_path);
    attach_joint_sidecar_residency_source(resident_decode.runtime);
#ifdef QNN_LLAMA_PD_JOINT
    if (use_decode_sidecar_streaming) {
      const size_t sidecar_shard_count = prefill_shard_files.pte_paths.size();
      ET_CHECK_MSG(
          sidecar_shard_count != 0,
          "sidecar streaming requires Prefill shards");
      sidecar_streamer = std::make_unique<JointSidecarShardStreamer>(
          resident_decode.runtime, sidecar_shard_count);
      ET_CHECK_MSG(
          sidecar_streamer->prepare(),
          "failed to prepare Decode sidecar streaming arena");
      attach_joint_sidecar_residency_source(resident_decode.runtime);
      prefill_shard_rebuild.shard_execute_begin_callback =
          [streamer = sidecar_streamer.get()](size_t shard_index) {
            ET_CHECK_MSG(
                streamer->enqueue(shard_index),
                "failed to enqueue Decode sidecar chunk for shard %zu",
                shard_index);
          };
    }
#endif
    prefill_shard_rebuild.final_shard_overlap_callback =
        [&resident_decode]() {
          start_resident_decode_runtime_prepare(resident_decode.control_fd);
          resident_decode.runtime_prepare_sent = true;
        };
  }

#ifdef QNN_LLAMA_PD_JOINT
  set_joint_model_residency_stage(JointResidencyStage::PrefillSetup);
#endif
  PdPrefillSession<uint8_t> prefill_session =
      create_pd_prefill_session<uint8_t>(
          std::move(module_bundle),
          std::move(prefill_shard_files),
          std::move(prefill_shard_rebuild),
          std::move(attention_sink_rope_module));
  ET_LOG(
      Info,
      "PD session ready: requests=%zu persistent_shard0=%d decode_resident=%d",
      request_count,
      static_cast<int>(FLAGS_prefill_persistent_shard0),
      static_cast<int>(!FLAGS_prefill_only));

  for (size_t request_index = 0; request_index < request_count;
       ++request_index) {
    const auto request_start = SteadyClock::now();
    ET_LOG(
        Info,
        "PD session request begin: index=%zu total=%zu",
        request_index,
        request_count);

    std::atomic<bool> stop_memory_monitor{false};
    std::thread memory_monitor;
    if (!FLAGS_prefill_only) {
      memory_monitor = std::thread([&]() {
        while (!stop_memory_monitor.load(std::memory_order_relaxed)) {
          const auto parent_memory = process_memory_snapshot();
#ifdef QNN_LLAMA_PD_JOINT
          resident_decode.result.process_tree_peak_rss_bytes = std::max(
              resident_decode.result.process_tree_peak_rss_bytes,
              parent_memory.rss_bytes);
          resident_decode.result.process_tree_peak_pss_bytes = std::max(
              resident_decode.result.process_tree_peak_pss_bytes,
              parent_memory.pss_bytes);
#else
          const auto child_memory = process_memory_snapshot(resident_decode.pid);
          resident_decode.result.child_peak.rss_bytes = std::max(
              resident_decode.result.child_peak.rss_bytes,
              child_memory.rss_bytes);
          resident_decode.result.child_peak.hwm_bytes = std::max(
              resident_decode.result.child_peak.hwm_bytes,
              child_memory.hwm_bytes);
          resident_decode.result.child_peak.pss_bytes = std::max(
              resident_decode.result.child_peak.pss_bytes,
              child_memory.pss_bytes);
          resident_decode.result.process_tree_peak_rss_bytes = std::max(
              resident_decode.result.process_tree_peak_rss_bytes,
              parent_memory.rss_bytes + child_memory.rss_bytes);
          resident_decode.result.process_tree_peak_pss_bytes = std::max(
              resident_decode.result.process_tree_peak_pss_bytes,
              parent_memory.pss_bytes + child_memory.pss_bytes);
#endif
          usleep(5000);
        }
      });
    }

    const std::string prompt_input = use_tokenized_prompt
        ? FLAGS_tokenized_prompt
        : prompts[request_index];
#ifdef QNN_LLAMA_PD_JOINT
    set_joint_model_residency_stage(JointResidencyStage::Prefill);
#endif
    PdE2ERuntimeStats prefill_runtime = run_pd_e2e_request<uint8_t>(
        prefill_session, prompt_input, use_tokenized_prompt);
#ifdef QNN_LLAMA_PD_JOINT
    set_joint_model_residency_stage(JointResidencyStage::PrefillComplete);
    log_joint_pd_boundary_probe("prefill_complete", -1, -1, true);
#endif

    if (FLAGS_prefill_only) {
      if (prefill_runtime.memory_handoff.fd >= 0) {
        close(prefill_runtime.memory_handoff.fd);
        prefill_runtime.memory_handoff.fd = -1;
      }
      DecodeProcessResult no_decode;
      no_decode.before = process_memory_snapshot();
      no_decode.after = no_decode.before;
      log_pd_e2e_runtime_summary(
          prefill_runtime,
          no_decode,
          elapsed_ms(request_start));
      ET_LOG(
          Info,
          "PD session request complete: index=%zu prefill_only=1 wall_ms=%.3f",
          request_index,
          elapsed_ms(request_start));
      continue;
    }

#ifdef QNN_LLAMA_PD_JOINT
    if (sidecar_streamer) {
      double sidecar_boundary_wait_ms = 0.0;
      size_t sidecar_bytes_loaded = 0;
      ET_CHECK_MSG(
          sidecar_streamer->finish(
              &sidecar_boundary_wait_ms, &sidecar_bytes_loaded),
          "Decode sidecar streaming did not complete before handoff");
      attach_joint_sidecar_residency_source(resident_decode.runtime);
    }
#endif

    reread_decode_sidecar_before_handoff(
        FLAGS_decode_sidecar_reread_path);
    pretouch_decode_sidecar_mapping_before_handoff(
        FLAGS_decode_sidecar_reread_path);

#ifdef QNN_LLAMA_PD_JOINT
    set_joint_model_residency_stage(JointResidencyStage::PrefillRelease);
#endif
    // Production joint PD reaches this boundary with Decode already warm and
    // all request-scoped Prefill resources retired by their normal shard
    // lifetimes. Whole-runtime teardown remains a legacy diagnostic only.
    if (FLAGS_prefill_release_all_before_decode) {
      prefill_session.runner->release_prefill_resources_before_decode();
      executorch::backends::qnn::ReleaseQnnBackendBundles();
      ET_LOG(Info, "All QNN Prefill resources released before Decode handoff");
    } else if (FLAGS_prefill_release_htp_vote_before_decode) {
      executorch::backends::qnn::ReleaseQnnPerformanceVotes();
      ET_LOG(Info, "HTP performance vote released before Decode handoff");
    }
    if (FLAGS_decode_cooldown_ms > 0) {
      ET_LOG(
          Info,
          "Decode cooldown after Prefill: ms=%d",
          FLAGS_decode_cooldown_ms);
      std::this_thread::sleep_for(
          std::chrono::milliseconds(FLAGS_decode_cooldown_ms));
    }
#ifdef QNN_LLAMA_PD_JOINT
    set_joint_model_residency_stage(JointResidencyStage::Handoff);
    log_joint_model_residency("after_prefill_release");
#endif
#ifdef QNN_LLAMA_PD_JOINT
    if (FLAGS_decode_pretouch_model) {
      const auto pretouch_start = SteadyClock::now();
      const volatile uint8_t* bytes = g_joint_model_source->data();
      uint64_t checksum = 0;
      for (size_t offset = 0; offset < g_joint_model_source->size();
           offset += 4096) {
        checksum += bytes[offset];
      }
      checksum += bytes[g_joint_model_source->size() - 1];
      ET_LOG(
          Info,
          "Decode model page pretouch complete: bytes=%zu checksum=%" PRIu64
          " ms=%.3f",
          g_joint_model_source->size(),
          checksum,
          elapsed_ms(pretouch_start));
    }

    // Send the ready KV
#endif

    // Send the ready KV immediately. Joining the dense PSS monitor can take
    // tens of milliseconds on Android and must not sit on the PD boundary.
    begin_resident_decode_handoff(
        resident_decode,
        prefill_runtime.memory_handoff);
    stop_memory_monitor.store(true, std::memory_order_relaxed);
    memory_monitor.join();
#ifndef QNN_LLAMA_PD_JOINT
    // The forked Decode path completes asynchronously after the handoff. Wait
    // for the child before reading its result; the joint path is synchronous
    // and has already populated resident_decode.result at this point.
    resident_decode.result =
        finish_resident_decode_process(std::move(resident_decode));
#endif

    const DecodeProcessResult decode = resident_decode.result;
    if (prefill_runtime.memory_handoff.fd >= 0) {
      close(prefill_runtime.memory_handoff.fd);
      prefill_runtime.memory_handoff.fd = -1;
    }
    ET_CHECK_MSG(
        decode.exit_code == 0,
        "llama-pd-cli exited with code %d on session request %zu",
        decode.exit_code,
        request_index);
    log_pd_e2e_runtime_summary(
        prefill_runtime,
        decode,
        elapsed_ms(request_start));
    ET_LOG(
        Info,
        "PD session request complete: index=%zu generated_tokens=%d wall_ms=%.3f",
        request_index,
        decode.generated_tokens,
        elapsed_ms(request_start));
  }

  if (!FLAGS_prefill_only) {
#ifdef QNN_LLAMA_PD_JOINT
    finish_resident_decode_process(std::move(resident_decode));
#endif
  }
#ifdef QNN_LLAMA_PD_JOINT
  set_joint_model_residency_stage(JointResidencyStage::Complete);
  stop_joint_model_residency_profiler();
#endif
  ET_LOG(
      Info,
      "PD session complete: requests=%zu wall_ms=%.3f",
      request_count,
      elapsed_ms(e2e_start));
#ifdef QNN_LLAMA_PD_JOINT
  if (request_count > 1) {
    // Release live QNN contexts explicitly, then avoid the Android QNN SDK's
    // process-global static teardown path. The OS reclaims the process-lifetime
    // backend/device bundle.
    prefill_session.runner.reset();
    std::fflush(nullptr);
    _Exit(0);
  }
#endif
  return 0;
}

#ifndef QNN_LLAMA_PD_E2E_NO_MAIN
int main(int argc, char** argv) {
  return qnn_llama_pd_e2e_main(argc, argv);
}
#endif
