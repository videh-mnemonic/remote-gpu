#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

using get_export_table_fn = int (*)(const void **, const void *);

int main(int argc, char **argv) {
  const char *library = argc > 1 ? argv[1] : "libnvidia-ml.so.1";
  void *handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    std::fprintf(stderr, "dlopen: %s\n", dlerror());
    return 1;
  }
  auto get_table = reinterpret_cast<get_export_table_fn>(
      dlsym(handle, "nvmlInternalGetExportTable"));
  if (get_table == nullptr) {
    std::fprintf(stderr, "dlsym: %s\n", dlerror());
    return 1;
  }
  const unsigned char id[16] = {0xc4, 0xfe, 0x3e, 0x6c, 0xc9, 0x8f,
                                0x6c, 0x4e, 0xa3, 0x27, 0xee, 0x69,
                                0x6e, 0x12, 0xf7, 0xc4};
  const void *table = nullptr;
  int result = get_table(&table, id);
  std::printf("result=%d table=%p\n", result, table);
  if (result != 0 || table == nullptr) {
    return 2;
  }
  const auto *words = static_cast<const std::uintptr_t *>(table);
  for (unsigned int i = 0; i < 0x948 / sizeof(std::uintptr_t); ++i) {
    if (argc > 2 && i != static_cast<unsigned int>(std::strtoul(argv[2], nullptr, 0)))
      continue;
    Dl_info info{};
    const char *symbol = "";
    const char *object = "";
    if (words[i] != 0 && dladdr(reinterpret_cast<const void *>(words[i]),
                                &info) != 0) {
      symbol = info.dli_sname == nullptr ? "" : info.dli_sname;
      object = info.dli_fname == nullptr ? "" : info.dli_fname;
    }
    auto offset = info.dli_fbase == nullptr
                      ? 0ul
                      : static_cast<unsigned long>(words[i] -
                          reinterpret_cast<std::uintptr_t>(info.dli_fbase));
    std::printf("%03u 0x%016lx +0x%lx %s %s\n", i,
                static_cast<unsigned long>(words[i]), offset, symbol, object);
  }
}
