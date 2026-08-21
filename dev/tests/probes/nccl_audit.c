#define _GNU_SOURCE
#include <link.h>
#include <stdio.h>
#include <string.h>

unsigned int la_version(unsigned int version) { return version; }

unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie) {
  (void)map;
  (void)lmid;
  (void)cookie;
  return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}

uintptr_t la_symbind64(Elf64_Sym *symbol, unsigned int index,
                       uintptr_t *reference_cookie,
                       uintptr_t *definition_cookie, unsigned int *flags,
                       const char *name) {
  (void)index;
  (void)reference_cookie;
  (void)definition_cookie;
  (void)flags;
  if (strncmp(name, "nccl", 4) == 0) {
    fprintf(stderr, "NCCL_AUDIT %s\n", name);
  }
  return symbol->st_value;
}

Elf64_Addr la_x86_64_gnu_pltenter(
    Elf64_Sym *symbol, unsigned int index, uintptr_t *reference_cookie,
    uintptr_t *definition_cookie, La_x86_64_regs *registers,
    unsigned int *flags, const char *name, long int *framesize) {
  (void)index;
  (void)reference_cookie;
  (void)definition_cookie;
  (void)registers;
  (void)flags;
  (void)framesize;
  if (strncmp(name, "nccl", 4) == 0 || strncmp(name, "pnccl", 5) == 0) {
    fprintf(stderr, "NCCL_ENTER %s\n", name);
  }
  return symbol->st_value;
}
