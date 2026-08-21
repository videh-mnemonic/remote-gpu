#include "config.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <string>

namespace {

constexpr const char *kDefaultServerFile = "/etc/rgpu/endpoints";
constexpr size_t kMaxServerConfigurationBytes = 4096;

void trim(std::string *value) {
  auto whitespace = [](unsigned char byte) { return std::isspace(byte) != 0; };
  value->erase(value->begin(),
               std::find_if_not(value->begin(), value->end(), whitespace));
  value->erase(
      std::find_if_not(value->rbegin(), value->rend(), whitespace).base(),
      value->end());
}

} // namespace

std::string lupine_server_configuration() {
  const char *environment = std::getenv("LUPINE_SERVER");
  if (environment != nullptr && environment[0] != '\0') {
    return environment;
  }

  const char *override_path = std::getenv("LUPINE_SERVER_FILE");
  const char *path =
      override_path != nullptr && override_path[0] != '\0'
          ? override_path
          : kDefaultServerFile;
  std::ifstream stream(path);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.size() > kMaxServerConfigurationBytes) {
      return {};
    }
    trim(&line);
    if (!line.empty() && line[0] != '#') {
      return line;
    }
  }
  return {};
}
