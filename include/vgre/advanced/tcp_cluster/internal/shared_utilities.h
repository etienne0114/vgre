#pragma once

/**
 * VGRE Shared Utilities - Facade Header
 * 
 * This header aggregates the modular utility components of the TCP cluster.
 * To improve compilation times and maintainability, prefer including specific
 * headers from the internal/ directory when possible.
 */

#include "vgre/advanced/tcp_cluster/internal/shared_types.h"
#include "vgre/advanced/tcp_cluster/internal/logging_patterns.h"
#include "vgre/advanced/tcp_cluster/internal/error_handling_patterns.h"
#include "vgre/advanced/tcp_cluster/internal/network_utilities.h"
#include "vgre/advanced/tcp_cluster/internal/shared_crypto_utilities.h"
#include "vgre/advanced/tcp_cluster/internal/configuration_manager.h"