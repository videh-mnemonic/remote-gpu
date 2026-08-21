#pragma once

#include <cstdint>

// CUDA tensor maps currently support at most five dimensions. Keep this wire
// format independent of CUDA's enum ABI and zero-fill unused array entries so
// client and server can safely evolve together.
static constexpr std::uint32_t LUPINE_TENSOR_MAP_MAX_RANK = 5;

struct lupine_tensormap_tiled_request {
  std::uint32_t data_type;
  std::uint32_t rank;
  std::uint64_t global_address;
  std::uint64_t global_dim[LUPINE_TENSOR_MAP_MAX_RANK];
  std::uint64_t global_strides[LUPINE_TENSOR_MAP_MAX_RANK];
  std::uint32_t box_dim[LUPINE_TENSOR_MAP_MAX_RANK];
  std::uint32_t element_strides[LUPINE_TENSOR_MAP_MAX_RANK];
  std::uint32_t interleave;
  std::uint32_t swizzle;
  std::uint32_t l2_promotion;
  std::uint32_t oob_fill;
};
