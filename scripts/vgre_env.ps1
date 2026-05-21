# VGRE Environment Setup Script for PowerShell
# Usage: . ./scripts/vgre_env.ps1

Write-Host "Configuring VGRE environment..." -ForegroundColor Cyan

# Define a helper function to mimic 'export' behavior for users coming from bash
function Global:set-vgre-env($Name, $Value) {
    if ($null -ne $Name -and $Name -ne "") {
        Set-Item -Path "Env:$Name" -Value $Value
        Write-Host "Set $Name to '$Value'" -ForegroundColor Gray
    }
}

# ── Authentication ───────────────────────────────────────────────────────────
# Set Authentication Token only if explicitly provided by user.
# No hardcoded default secret is applied.
# Prefer VGRE_TCP_AUTH_TOKEN_FILE over VGRE_TCP_AUTH_TOKEN to keep the token
# out of the process listing.
if ($null -eq $env:VGRE_TCP_AUTH_TOKEN -or $env:VGRE_TCP_AUTH_TOKEN -eq "") {
    if ($null -eq $env:VGRE_TCP_AUTH_TOKEN_FILE -or $env:VGRE_TCP_AUTH_TOKEN_FILE -eq "") {
        Write-Host "VGRE_TCP_AUTH_TOKEN / VGRE_TCP_AUTH_TOKEN_FILE not set (recommended for secure cluster mode)." -ForegroundColor Yellow
        Write-Host "  Example: set-vgre-env VGRE_TCP_AUTH_TOKEN_FILE `"$env:USERPROFILE\.vgre\token`"" -ForegroundColor DarkGray
    }
}

# ── Cluster Discovery ────────────────────────────────────────────────────────
# LAN mode (default): master broadcasts on UDP 7778; workers auto-discover.
# WAN mode: set VGRE_CLUSTER_MASTER_ADDRESS on each worker (no broadcast needed).
#
# NEVER set a hardcoded default for VGRE_CLUSTER_NODES — the old loopback
# default (127.0.0.1:7777) silently overwrote any value the user already had.

if ($null -eq $env:VGRE_CLUSTER_NODES -or $env:VGRE_CLUSTER_NODES -eq "") {
    Write-Host "VGRE_CLUSTER_NODES not set." -ForegroundColor DarkGray
    Write-Host "  LAN  (auto-discover) : leave unset — UDP broadcast handles it." -ForegroundColor DarkGray
    Write-Host "  WAN  (direct connect): set-vgre-env VGRE_CLUSTER_MASTER_ADDRESS '<PUBLIC_IP>:7777'" -ForegroundColor DarkGray
    Write-Host "  Mesh (explicit peers): set-vgre-env VGRE_MESH_PEERS '<IP1>:7777,<IP2>:7777'" -ForegroundColor DarkGray
} else {
    Write-Host "VGRE_CLUSTER_NODES = $env:VGRE_CLUSTER_NODES" -ForegroundColor DarkGray
}

# -- WAN / NAT connectivity ---------------------------------------------------
# VGRE_CLUSTER_MASTER_ADDRESS  (worker side)
#   The worker connects directly to this address instead of waiting for a
#   UDP broadcast. Supports IPv4, IPv6 ([::1]:port), and hostnames.
#   The proactive-reconnect loop will re-establish the connection automatically
#   if the master restarts (exponential backoff, default cap 120 s).
#   Example: set-vgre-env "VGRE_CLUSTER_MASTER_ADDRESS" "203.0.113.5:7777"
# set-vgre-env "VGRE_CLUSTER_MASTER_ADDRESS" ""

# VGRE_CLUSTER_ADVERTISED_ADDRESS  (master side)
#   The master includes this address in its UDP ping message so workers that
#   receive the LAN broadcast still connect to the correct public IP.
#   Useful when the master is behind NAT or a VPN gateway.
#   Example: set-vgre-env "VGRE_CLUSTER_ADVERTISED_ADDRESS" "203.0.113.5:7777"
# set-vgre-env "VGRE_CLUSTER_ADVERTISED_ADDRESS" ""

# VGRE_CLUSTER_CONNECT_TIMEOUT_SEC  (default 10, range 1-120)
#   TCP connect timeout for both direct dial-out and proactive reconnection.
#   Increase for very high-latency WAN links.
# set-vgre-env "VGRE_CLUSTER_CONNECT_TIMEOUT_SEC" "10"

# VGRE_CLUSTER_MAX_BACKOFF_SEC  (default 120, range 10-3600)
#   Maximum retry interval for the proactive reconnection loop.
# set-vgre-env "VGRE_CLUSTER_MAX_BACKOFF_SEC" "120"

# VGRE_MESH_PEERS  — full mesh: comma-separated host:port list.
#   Each node connects to every other node.  All nodes must list all peers.
#   Example: set-vgre-env "VGRE_MESH_PEERS" "10.0.0.2:7777,10.0.0.3:7777"
# set-vgre-env "VGRE_MESH_PEERS" ""

# UDP announce port — master broadcasts on this port (workers listen).
# Default: 7778. Must match on master and all workers.
# set-vgre-env "VGRE_CLUSTER_UDP_ANNOUNCE_PORT" "7778"

# UDP worker port — workers broadcast on this port (master listens).
# Default: 7779. Must match on master and all workers.
# set-vgre-env "VGRE_CLUSTER_UDP_WORKER_PORT" "7779"

# VGRE_CLUSTER_DISCOVERY=OFF — disable UDP broadcast entirely (pure WAN mode).
# set-vgre-env "VGRE_CLUSTER_DISCOVERY" "OFF"

# ── Cluster Security ─────────────────────────────────────────────────────────
# Strict auth mode — set to 1 to reject connections with mismatched tokens.
# Default: 0 (fallback to default key on mismatch — development-friendly).
# set-vgre-env "VGRE_CLUSTER_STRICT_AUTH" "0"

# Auth fallback — set to 1 to allow fallback to default encrypted key on mismatch.
# Only meaningful when VGRE_CLUSTER_STRICT_AUTH=0 (default).
# set-vgre-env "VGRE_ALLOW_AUTH_FALLBACK" "0"

# PBKDF2 iteration count for session key derivation. Default: 600000 (NIST 2025).
# Increase for higher security on slower hardware. Must match on both ends.
# set-vgre-env "VGRE_PBKDF2_ITERATIONS" "600000"

# ── Cluster Performance ──────────────────────────────────────────────────────
# Bandwidth re-probe interval in seconds. Default: 300 (5 min).
# Range: 30–86400. Lower values mean more accurate bandwidth estimates.
# set-vgre-env "VGRE_CLUSTER_BANDWIDTH_REPROBE_SEC" "300"

# ── Runtime ──────────────────────────────────────────────────────────────────
# Log level: DEBUG, INFO, WARN, ERROR
# set-vgre-env "VGRE_LOG_LEVEL" "INFO"

# JIT kernel cache directory
# set-vgre-env "VGRE_CACHE_DIR" "$env:USERPROFILE\.vgre\cache"

# Override auto-detected device count
# set-vgre-env "VGRE_DEVICE_COUNT" "4"

# SIMD level: SSE4, AVX, AVX2, AVX512, native
# set-vgre-env "VGRE_SIMD_LEVEL" "AVX2"

# Worker thread count (0 = auto-detect)
# set-vgre-env "VGRE_WORKER_THREADS" "0"

# Adaptive engine exponential moving-average alpha (0.0–1.0, default 0.3)
# set-vgre-env "VGRE_ADAPTIVE_ALPHA" "0.3"

# ── Add VGRE bin directory to PATH ────────────────────────────────────────────
# Use $env:LOCALAPPDATA (always set by Windows) rather than constructing
# the path from $HOME, which may be absent in non-interactive sessions.
$vgrePath = $env:LOCALAPPDATA + "\VGRE"
if ($env:PATH -notlike "*$vgrePath*") {
    $env:PATH = "$vgrePath;$env:PATH"
    Write-Host "Added $vgrePath to PATH"
}

Write-Host "`nVGRE environment configured successfully." -ForegroundColor Green
Write-Host "You can now run 'vgre-worker' or 'vgre-dashboard' directly."
Write-Host "To set cluster variables, use: set-vgre-env VGRE_CLUSTER_NODES '192.168.1.50:7777'"
Write-Host "For secure clusters: set-vgre-env VGRE_TCP_AUTH_TOKEN_FILE `"$env:USERPROFILE\.vgre\token`"`n"
