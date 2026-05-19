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
# Cluster Nodes — manual cross-subnet configuration.
# Only set the loopback default when the variable is not already in the
# environment.  Workers connecting to a remote master MUST NOT have this
# overridden; the user should set VGRE_CLUSTER_NODES to the master IP before
# sourcing this file, or pass --master-ip to vgre-start.
if ($null -eq $env:VGRE_CLUSTER_NODES -or $env:VGRE_CLUSTER_NODES -eq "") {
    Write-Host "VGRE_CLUSTER_NODES not set — defaulting to 127.0.0.1:7777 (local test only)." -ForegroundColor DarkGray
    Write-Host "  For remote workers set: set-vgre-env VGRE_CLUSTER_NODES '<MASTER_IP>:7777'" -ForegroundColor DarkGray
    set-vgre-env "VGRE_CLUSTER_NODES" "127.0.0.1:7777"
}

# UDP announce port — master broadcasts presence on this port (workers listen)
# Default: 7778. Must match on master and all workers.
# set-vgre-env "VGRE_CLUSTER_UDP_ANNOUNCE_PORT" "7778"

# UDP worker port — workers broadcast presence on this port (master listens)
# Default: 7779. Must match on master and all workers.
# set-vgre-env "VGRE_CLUSTER_UDP_WORKER_PORT" "7779"

# Master IP allowlist — comma-separated list of IP addresses workers will accept
# UDP broadcast from. Prevents rogue master injection on shared subnets.
# Leave unset to accept broadcasts from any master (development only).
# Example: set-vgre-env "VGRE_CLUSTER_MASTER_IP" "192.168.1.1,10.0.0.1"
# set-vgre-env "VGRE_CLUSTER_MASTER_IP" ""

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
