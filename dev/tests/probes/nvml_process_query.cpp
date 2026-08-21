#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <vector>

struct process_info {
  unsigned int pid;
  unsigned long long used_gpu_memory;
  unsigned int gpu_instance_id;
  unsigned int compute_instance_id;
};

int main(int argc, char **argv) {
  void *library = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) return 1;
  auto symbol = [library](const char *name) { return dlsym(library, name); };
  auto init = reinterpret_cast<int (*)()>(symbol("nvmlInit_v2"));
  auto count = reinterpret_cast<int (*)(unsigned int *)>(
      symbol("nvmlDeviceGetCount_v2"));
  auto handle = reinterpret_cast<int (*)(unsigned int, void **)>(
      symbol("nvmlDeviceGetHandleByIndex_v2"));
  auto processes = reinterpret_cast<int (*)(void *, unsigned int *,
                                             process_info *)>(
      symbol(argc > 1 ? argv[1] : "nvmlDeviceGetComputeRunningProcesses_v2"));
  if (init == nullptr || count == nullptr || handle == nullptr ||
      processes == nullptr || init() != 0) return 2;
  unsigned int devices = 0;
  if (count(&devices) != 0) return 3;
  for (unsigned int index = 0; index < devices; ++index) {
    void *device = nullptr;
    if (handle(index, &device) != 0) return 4;
    unsigned int entries = 0;
    int status = processes(device, &entries, nullptr);
    std::vector<process_info> values(entries == 0 ? 1 : entries);
    unsigned int capacity = static_cast<unsigned int>(values.size());
    int data_status = processes(device, &capacity, values.data());
    std::printf("device=%u size_status=%d count=%u data_status=%d", index,
                status, capacity, data_status);
    for (unsigned int item = 0; item < capacity; ++item) {
      std::printf(" pid=%u memory=%llu", values[item].pid,
                  values[item].used_gpu_memory);
    }
    std::printf("\n");
  }
  return 0;
}
