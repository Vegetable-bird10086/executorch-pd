/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Standalone tool to verify that PTE rebuild (stripped.pte + index.bin + GGUF)
// produces binary-identical output to a known-good original PTE.
//
// Usage:
//   pte_rebuild_compare \
//       --original_shard_dir /path/to/original/shards \
//       --stripped_manifest_path /path/to/shard/manifest.json \
//       --gguf_model_path /path/to/model.gguf

// Minimal stubs so we can compile without linking the full executorch runtime.
#ifndef ET_INLINE
#define ET_INLINE __attribute__((always_inline)) inline
#endif

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

// Stub out executorch logging.
namespace executorch {
namespace runtime {
enum LogLevel { Debug = 0, Info = 1, Error = 2, Fatal = 3 };
namespace internal {
inline void stub_log(LogLevel level, const char* format, ...) {
  const char* prefix = "";
  switch (level) {
    case LogLevel::Error:   prefix = "E "; break;
    case LogLevel::Info:    prefix = "I "; break;
    case LogLevel::Debug:   prefix = "D "; break;
    case LogLevel::Fatal:   prefix = "F "; break;
  }
  std::fprintf(stderr, "%s", prefix);
  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);
  std::fprintf(stderr, "\n");
}
}  // namespace internal
}  // namespace runtime
}  // namespace executorch

// Stub the log macros used by pte_rebuilder.  It does not actually log, so
// empty macros are fine.
#define ET_LOG(level, ...) \
  executorch::runtime::internal::stub_log( \
      executorch::runtime::LogLevel::level, __VA_ARGS__)
#define ET_CHECK_MSG(cond, ...) \
  do { if (!(cond)) { std::fprintf(stderr, "FATAL: " __VA_ARGS__); std::fprintf(stderr, "\n"); std::abort(); } } while (0)

// Now include the real pte_rebuilder header.
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/pte_rebuilder.h>

namespace {

std::vector<uint8_t> read_binary_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    std::fprintf(stderr, "E Unable to read file: %s\n", path.c_str());
    std::abort();
  }
  return std::vector<uint8_t>(
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string read_text_file(const std::string& path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    std::fprintf(stderr, "E Unable to read file: %s\n", path.c_str());
    std::abort();
  }
  std::stringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::string hex_dump(const uint8_t* data, size_t len, size_t max_bytes) {
  std::ostringstream oss;
  const size_t n = std::min(len, max_bytes);
  for (size_t i = 0; i < n; ++i) {
    oss << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<int>(data[i]);
    if ((i + 1) % 16 == 0 && i + 1 < n) {
      oss << "\n  ";
    } else if (i + 1 < n) {
      oss << " ";
    }
  }
  if (len > max_bytes) {
    oss << "... (truncated, total=" << len << ")";
  }
  return oss.str();
}

struct MismatchInfo {
  size_t offset{0};
  size_t count{0};
  std::vector<std::pair<size_t, std::pair<uint8_t, uint8_t>>> samples;
};

MismatchInfo find_mismatches(
    const uint8_t* rebuilt,
    const uint8_t* original,
    size_t size,
    size_t max_samples) {
  MismatchInfo info;
  for (size_t i = 0; i < size; ++i) {
    if (rebuilt[i] != original[i]) {
      if (info.count == 0) {
        info.offset = i;
      }
      ++info.count;
      if (info.samples.size() < max_samples) {
        info.samples.push_back({i, {rebuilt[i], original[i]}});
      }
    }
  }
  return info;
}

// Minimal JSON helpers to read shard paths from the manifest without
// dragging in a JSON library.
std::vector<std::string> extract_string_array(
    const std::string& text,
    size_t pos,
    const char* key) {
  std::vector<std::string> result;
  std::string search = std::string("\"") + key + "\"";
  size_t key_pos = text.find(search, pos);
  if (key_pos == std::string::npos) {
    return result;
  }
  size_t bracket = text.find('[', key_pos + search.size());
  if (bracket == std::string::npos) {
    return result;
  }
  size_t close_bracket = text.find(']', bracket + 1);
  if (close_bracket == std::string::npos) {
    return result;
  }
  size_t cursor = bracket + 1;
  while (cursor < close_bracket) {
    size_t open = text.find('"', cursor);
    if (open == std::string::npos || open >= close_bracket) {
      break;
    }
    size_t close = text.find('"', open + 1);
    if (close == std::string::npos || close >= close_bracket) {
      break;
    }
    result.push_back(text.substr(open + 1, close - open - 1));
    cursor = close + 1;
  }
  return result;
}

int run_compare(int argc, char** argv) {
  std::string original_shard_dir;
  std::string stripped_manifest_path;
  std::string gguf_model_path;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--original_shard_dir" && i + 1 < argc) {
      original_shard_dir = argv[++i];
    } else if (arg == "--stripped_manifest_path" && i + 1 < argc) {
      stripped_manifest_path = argv[++i];
    } else if (arg == "--gguf_model_path" && i + 1 < argc) {
      gguf_model_path = argv[++i];
    }
  }

  if (original_shard_dir.empty() || stripped_manifest_path.empty() ||
      gguf_model_path.empty()) {
    std::fprintf(stderr,
        "Usage: pte_rebuild_compare \\\n"
        "    --original_shard_dir /path/to/original/shards \\\n"
        "    --stripped_manifest_path /path/to/shard/manifest.json \\\n"
        "    --gguf_model_path /path/to/model.gguf\n");
    return 1;
  }

  while (!original_shard_dir.empty() && original_shard_dir.back() == '/') {
    original_shard_dir.pop_back();
  }

  ET_LOG(Info, "Loading GGUF model: %s", gguf_model_path.c_str());
  auto gguf_bytes = example::ReadOnlyMappedFile::open(gguf_model_path);
  auto gguf_context = example::create_pte_gguf_rebuild_context(gguf_bytes);
  ET_LOG(Info, "GGUF model size: %zu bytes (read-only mmap)", gguf_bytes->size());

  const std::string manifest = read_text_file(stripped_manifest_path);
  const std::string manifest_dir = [&]() {
    std::string dir = stripped_manifest_path;
    size_t slash = dir.rfind('/');
    return slash != std::string::npos ? dir.substr(0, slash) : ".";
  }();

  size_t graph_pos = manifest.find("\"prefill_forward\"");
  if (graph_pos == std::string::npos) {
    ET_LOG(Error, "prefill_forward graph not found in manifest");
    return 1;
  }

  std::vector<std::string> stripped_paths =
      extract_string_array(manifest, graph_pos, "stripped_pte_paths");
  std::vector<std::string> index_paths =
      extract_string_array(manifest, graph_pos, "index_bin_paths");

  if (stripped_paths.empty()) {
    stripped_paths =
        extract_string_array(manifest, graph_pos, "pte_paths");
  }

  ET_LOG(Info, "Found %zu shard paths in manifest", stripped_paths.size());

  if (index_paths.size() != stripped_paths.size()) {
    ET_LOG(Error,
        "Index paths (%zu) != stripped paths (%zu)",
        index_paths.size(), stripped_paths.size());
    return 1;
  }

  auto resolve = [&](const std::string& p) -> std::string {
    if (p.empty()) return p;
    if (p[0] == '/') return p;
    return manifest_dir + "/" + p;
  };

  size_t total_mismatched_bytes = 0;
  int mismatched_shards = 0;
  constexpr size_t kMaxSamples = 8;

  for (size_t shard_idx = 0; shard_idx < stripped_paths.size(); ++shard_idx) {
    const std::string stripped_path = resolve(stripped_paths[shard_idx]);
    const std::string index_path = resolve(index_paths[shard_idx]);

    std::ostringstream orig_name;
    orig_name << original_shard_dir << "/hybrid_llama_qnn.prefill_forward.shard"
              << shard_idx << ".pte";

    ET_LOG(Info, "--- Shard %zu ---", shard_idx);
    ET_LOG(Info, "  Original:  %s", orig_name.str().c_str());
    ET_LOG(Info, "  Stripped:  %s", stripped_path.c_str());
    ET_LOG(Info, "  Index:     %s", index_path.c_str());

    const std::vector<uint8_t> original_bytes =
        read_binary_file(orig_name.str());
    const std::vector<uint8_t> stripped_bytes =
        read_binary_file(stripped_path);
    auto index_bytes = std::make_shared<std::vector<uint8_t>>(
        read_binary_file(index_path));

    ET_LOG(Info, "  Sizes: original=%zu stripped=%zu index=%zu",
        original_bytes.size(), stripped_bytes.size(), index_bytes->size());

    auto recipe = example::prepare_pte_gguf_shard_recipe(
        gguf_context, index_bytes, 32);
    example::PteRebuildResult rebuild_result =
        example::rebuild_pte_from_stripped_gguf_recipe(
            stripped_bytes, *recipe, nullptr);
    example::discard_pte_gguf_rebuild_source_pages(gguf_context);

    if (rebuild_result.rebuilt_pte_buffer == nullptr) {
      ET_LOG(Error, "  REBUILD FAILED");
      ++mismatched_shards;
      continue;
    }

    const example::PteRebuildBuffer& rebuilt =
        *rebuild_result.rebuilt_pte_buffer;

    ET_LOG(Info,
        "  Rebuilt:   %zu bytes  records=%zu  weight_bytes=%zu  time=%.2f ms",
        rebuilt.size(), rebuild_result.rebuilt_records,
        rebuild_result.materialized_weight_bytes,
        rebuild_result.rebuild_time_ms);

    if (rebuilt.size() != original_bytes.size()) {
      ET_LOG(Error,
          "  SIZE MISMATCH: rebuilt=%zu original=%zu (diff=%+zd)",
          rebuilt.size(), original_bytes.size(),
          static_cast<ssize_t>(rebuilt.size()) -
              static_cast<ssize_t>(original_bytes.size()));
      ++mismatched_shards;
      continue;
    }

    MismatchInfo mismatches = find_mismatches(
        rebuilt.data(), original_bytes.data(), rebuilt.size(), kMaxSamples);

    if (mismatches.count == 0) {
      ET_LOG(Info, "  IDENTICAL - rebuild matches original");
    } else {
      ++mismatched_shards;
      total_mismatched_bytes += mismatches.count;
      ET_LOG(Error,
          "  MISMATCH: %zu differing bytes (first at offset %zu, %.4f%% of %zu bytes)",
          mismatches.count, mismatches.offset,
          100.0 * static_cast<double>(mismatches.count) /
              static_cast<double>(rebuilt.size()),
          rebuilt.size());

      for (const auto& sample : mismatches.samples) {
        ET_LOG(Info,
            "    offset=%-10zu rebuilt=0x%02x original=0x%02x",
            sample.first,
            static_cast<unsigned>(sample.second.first),
            static_cast<unsigned>(sample.second.second));
      }

      constexpr size_t kDumpRadius = 32;
      const size_t dump_start =
          mismatches.offset > kDumpRadius ? mismatches.offset - kDumpRadius : 0;
      const size_t dump_len =
          std::min<size_t>(128, rebuilt.size() - dump_start);
      ET_LOG(Info,
          "  Rebuilt  @ %zu:\n  %s",
          dump_start,
          hex_dump(rebuilt.data() + dump_start, dump_len, 128).c_str());
      ET_LOG(Info,
          "  Original @ %zu:\n  %s",
          dump_start,
          hex_dump(original_bytes.data() + dump_start, dump_len, 128).c_str());
    }
  }

  if (mismatched_shards == 0) {
    ET_LOG(Info, "ALL %zu SHARDS IDENTICAL - rebuild is consistent",
        stripped_paths.size());
    return 0;
  } else {
    ET_LOG(Error,
        "%d/%zu SHARDS HAVE MISMATCHES (%zu total differing bytes)",
        mismatched_shards, stripped_paths.size(), total_mismatched_bytes);
    return 1;
  }
}

} // namespace

int main(int argc, char** argv) {
  return run_compare(argc, argv);
}
