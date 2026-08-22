#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error = (call);                                                  \
    if (error != cudaSuccess) {                                                  \
      std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(error)); \
      return 1;                                                                 \
    }                                                                           \
  } while (0)

__global__ void increment(const int* input, int* output) {
  *output = *input + 1;
}

int main() {
  int* host_input = nullptr;
  int* host_output = nullptr;
  int* device_input = nullptr;
  int* device_output = nullptr;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;

  CUDA_CHECK(cudaMallocHost(&host_input, sizeof(*host_input)));
  CUDA_CHECK(cudaMallocHost(&host_output, sizeof(*host_output)));
  CUDA_CHECK(cudaMalloc(&device_input, sizeof(*device_input)));
  CUDA_CHECK(cudaMalloc(&device_output, sizeof(*device_output)));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  *host_input = 10;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(cudaMemcpyAsync(device_input, host_input, sizeof(*host_input),
                             cudaMemcpyHostToDevice, stream));
  increment<<<1, 1, 0, stream>>>(device_input, device_output);
  CUDA_CHECK(cudaMemcpyAsync(host_output, device_output, sizeof(*host_output),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));

  for (int value : {21, 37, 91}) {
    *host_input = value;
    *host_output = -1;
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (*host_output != value + 1) {
      std::fprintf(stderr, "dynamic graph input %d produced %d, expected %d\n",
                   value, *host_output, value + 1);
      return 2;
    }
  }

  // Captures on one stream must not share D2H metadata, and updating an
  // executable must switch that metadata to the newly installed graph.
  int* alternate_input = nullptr;
  int* alternate_output = nullptr;
  cudaGraph_t alternate_graph = nullptr;
  CUDA_CHECK(cudaMallocHost(&alternate_input, sizeof(*alternate_input)));
  CUDA_CHECK(cudaMallocHost(&alternate_output, sizeof(*alternate_output)));
  *alternate_input = 50;
  *alternate_output = -1;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(cudaMemcpyAsync(device_input, alternate_input,
                             sizeof(*alternate_input), cudaMemcpyHostToDevice,
                             stream));
  increment<<<1, 1, 0, stream>>>(device_input, device_output);
  CUDA_CHECK(cudaMemcpyAsync(alternate_output, device_output,
                             sizeof(*alternate_output), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &alternate_graph));

  cudaGraphExecUpdateResultInfo update_info{};
  CUDA_CHECK(cudaGraphExecUpdate(executable, alternate_graph, &update_info));
  if (update_info.result != cudaGraphExecUpdateSuccess) {
    std::fprintf(stderr, "graph executable update result %d\n",
                 static_cast<int>(update_info.result));
    return 3;
  }
  *host_output = -7;
  *alternate_input = 63;
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (*alternate_output != 64 || *host_output != -7) {
    std::fprintf(stderr,
                 "updated graph produced primary=%d alternate=%d, expected "
                 "primary=-7 alternate=64\n",
                 *host_output, *alternate_output);
    return 4;
  }

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(alternate_graph));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFreeHost(host_output));
  CUDA_CHECK(cudaFreeHost(host_input));
  CUDA_CHECK(cudaFreeHost(alternate_output));
  CUDA_CHECK(cudaFreeHost(alternate_input));
  std::puts("dynamic CUDA Graph host inputs pass");
  return 0;
}
