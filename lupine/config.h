#ifndef LUPINE_CONFIG_H
#define LUPINE_CONFIG_H

#include <string>

// Resolve the remote endpoint list without requiring per-process environment
// setup. LUPINE_SERVER remains the explicit override; otherwise the first
// non-empty line of LUPINE_SERVER_FILE (default /etc/rgpu/endpoints) is used.
std::string lupine_server_configuration();

#endif
