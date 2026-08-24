#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <vector>

#define CUDA_OK(call)                                                          \
  do {                                                                         \
    cudaError_t status = (call);                                                \
    if (status != cudaSuccess) {                                                \
      std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(status)); \
      return 1;                                                                \
    }                                                                          \
  } while (false)

int main(int argc, char** argv) {
  const int selected_device = argc > 1 ? std::atoi(argv[1]) : 1;
  int devices = 0;
  CUDA_OK(cudaGetDeviceCount(&devices));
  if (selected_device < 0 || selected_device >= devices) {
    std::fprintf(stderr, "selected CUDA device is unavailable: %d\n", selected_device);
    return 2;
  }
  CUDA_OK(cudaSetDevice(selected_device));
  constexpr std::size_t width = 37;
  constexpr std::size_t height = 19;
  constexpr std::size_t host_pitch = 48;
  std::vector<unsigned char> input(host_pitch * height, 0);
  std::vector<unsigned char> output(host_pitch * height, 0);
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t column = 0; column < width; ++column) {
      input[row * host_pitch + column] =
          static_cast<unsigned char>((row * 17 + column * 29) & 0xff);
    }
  }

  void* source = nullptr;
  void* destination = nullptr;
  std::size_t source_pitch = 0;
  std::size_t destination_pitch = 0;
  CUDA_OK(cudaMallocPitch(&source, &source_pitch, width, height));
  CUDA_OK(cudaMallocPitch(&destination, &destination_pitch, width, height));
  CUDA_OK(cudaMemcpy2D(source, source_pitch, input.data(), host_pitch, width, height,
                       cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy2DAsync(destination, destination_pitch, source, source_pitch, width, height,
                            cudaMemcpyDeviceToDevice));
  CUDA_OK(cudaMemcpy2D(output.data(), host_pitch, destination, destination_pitch, width, height,
                       cudaMemcpyDeviceToHost));

  CUDA_OK(cudaFree(destination));
  CUDA_OK(cudaFree(source));
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t column = 0; column < width; ++column) {
      if (input[row * host_pitch + column] != output[row * host_pitch + column]) {
        std::fprintf(stderr, "2D copy mismatch at row=%zu column=%zu\n", row, column);
        return 3;
      }
    }
  }
  std::puts("mixed remote cudaMemcpy2D passed");
  return 0;
}
