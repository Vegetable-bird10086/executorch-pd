#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>

namespace example {

// Preserve the existing MTK right-aligned sliding-window ABI. This performs
// exactly cat(history, new_tokens)[:, :, -cache_length:, :] on host buffers.
// Source and destination are distinct, contiguous [row, token, feature] arrays.
// History before the right-aligned valid suffix must be zero. Its length may
// conservatively overestimate live tokens (e.g. after rollback). The default
// retains the full-history behavior for callers without valid-length tracking.
inline void UpdateSlidingKvCache(
    void* history, const void* newTokens, size_t rows, size_t cacheLength,
    size_t tokenCount, size_t tokenBytes,
    size_t validHistoryTokens = std::numeric_limits<size_t>::max()) {
  if (tokenCount == 0 || cacheLength == 0 || rows == 0 || tokenBytes == 0) {
    return;
  }
  const size_t retained = std::min(cacheLength, tokenCount);
  const size_t oldCount = std::min(validHistoryTokens, cacheLength - retained);
  const size_t appendOffset = (cacheLength - retained) * tokenBytes;
  auto* dst = static_cast<unsigned char*>(history);
  const auto* src = static_cast<const unsigned char*>(newTokens);
  for (size_t row = 0; row < rows; ++row) {
    auto* output = dst + row * cacheLength * tokenBytes;
    const auto* input = src + (row * tokenCount + tokenCount - retained) * tokenBytes;
    if (oldCount != 0) {
      std::memmove(
          output + appendOffset - oldCount * tokenBytes,
          output + (cacheLength - oldCount) * tokenBytes,
          oldCount * tokenBytes);
    }
    std::memcpy(output + appendOffset, input, retained * tokenBytes);
  }
}

} // namespace example
