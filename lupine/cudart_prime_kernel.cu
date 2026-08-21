#include <cuda_runtime_api.h>

// A real Runtime-registered kernel is needed to initialize libcudart's private
// per-context launch state. Driver-only work (and even cudaMemset) does not
// establish all of the state PyTorch later expects when a process transitions
// between local and remotely routed primary contexts.
__global__ void lupine_runtime_anchor_kernel() {}

extern "C" int lupine_launch_runtime_anchor_kernel() {
  lupine_runtime_anchor_kernel<<<1, 1>>>();
  return static_cast<int>(cudaGetLastError());
}
