# VGRE Start — launch a master or worker node with one command (Windows).
#
# Prerequisites: run .\scripts\Setup-VGRECluster.ps1 once on each machine first.
#
# Usage:
#   .\scripts\Start-VGRE.ps1 --master                         Start master (runs the dashboard)
#   .\scripts\Start-VGRE.ps1 --worker                         Start worker (auto-discovers master)
#   .\scripts\Start-VGRE.ps1 --worker --master-ip 10.0.0.5   Start worker, connect to specific master
#   .\scripts\Start-VGRE.ps1 --worker --port 7778             Start worker on a non-default port
#   .\scripts\Start-VGRE.ps1 --test                           Quick local test: master + worker together

param(
    [switch]$master,
    [switch]$worker,
    [switch]$test,
    [string]$masterIp  = "",
    [string]$port      = "7777",
    [string]$threads   = ""
)

$ErrorActionPreference = "Stop"

$InstallDir = Join-Path $env:LOCALAPPDATA "VGRE"
$TokenFile  = if ($env:VGRE_TCP_AUTH_TOKEN_FILE) { $env:VGRE_TCP_AUTH_TOKEN_FILE }
              else { Join-Path $env:USERPROFILE ".vgre\token" }

# ── Determine mode ────────────────────────────────────────────────────────────
$mode = ""
if ($master) { $mode = "master" }
elseif ($worker) { $mode = "worker" }
elseif ($test)   { $mode = "test" }

if (-not $mode) {
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\scripts\Start-VGRE.ps1 --master"
    Write-Host "  .\scripts\Start-VGRE.ps1 --worker"
    Write-Host "  .\scripts\Start-VGRE.ps1 --worker --master-ip 192.168.1.50"
    Write-Host "  .\scripts\Start-VGRE.ps1 --test"
    exit 1
}

# ── Load Auth Token ───────────────────────────────────────────────────────────
if (Test-Path $TokenFile) {
    $env:VGRE_TCP_AUTH_TOKEN_FILE = $TokenFile
} elseif (-not $env:VGRE_TCP_AUTH_TOKEN) {
    Write-Host ""
    Write-Host "❌ No auth token found." -ForegroundColor Red
    Write-Host "   Run:  .\scripts\Setup-VGRECluster.ps1" -ForegroundColor Yellow
    Write-Host "   Or:   `$env:VGRE_TCP_AUTH_TOKEN = 'your-token'" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# ── Locate binaries ───────────────────────────────────────────────────────────
$workerExe    = Join-Path $InstallDir "vgre-worker.cmd"
if (-not (Test-Path $workerExe)) { $workerExe = Join-Path $InstallDir "vgre-worker.exe" }
if (-not (Test-Path $workerExe)) {
    Write-Host "❌ vgre-worker not found in $InstallDir" -ForegroundColor Red
    Write-Host "   Run .\scripts\vgre_sync.bat to build and install first." -ForegroundColor Yellow
    exit 1
}

$dashboardExe = Join-Path $InstallDir "Launch-VGRE-Dashboard.cmd"
if (-not (Test-Path $dashboardExe)) { $dashboardExe = Join-Path $InstallDir "vgre_dashboard.exe" }

# Add install lib dir to DLL search path for this session
$env:PATH = "$InstallDir\lib;$InstallDir;$env:PATH"

# ── Worker args ───────────────────────────────────────────────────────────────
$workerArgs = @("--port", $port)
if ($threads) { $workerArgs += @("--threads", $threads) }

# ── Start ─────────────────────────────────────────────────────────────────────
switch ($mode) {

    "master" {
        Write-Host "Starting VGRE Master Node..."
        Write-Host "  Port:  $port"
        Write-Host "  Token: $TokenFile"
        Write-Host ""
        $env:VGRE_PORT = $port
        Start-Process -FilePath $dashboardExe -NoNewWindow
        Write-Host "Dashboard launched." -ForegroundColor Green
    }

    "worker" {
        if ($masterIp) {
            $env:VGRE_CLUSTER_NODES = "${masterIp}:${port}"
            Write-Host "Starting VGRE Worker → master at ${masterIp}:${port}"
        } else {
            Write-Host "Starting VGRE Worker (auto-discovering master on local subnet)..."
        }
        Write-Host "  Port:  $port"
        Write-Host "  Token: $TokenFile"
        Write-Host ""
        & $workerExe @workerArgs
    }

    "test" {
        Write-Host "Starting local self-test (master + worker on same machine)..."
        Write-Host "  Token: $TokenFile"
        Write-Host "  Press Ctrl+C to stop."
        Write-Host ""
        # Start worker in a new window so both are visible
        $workerProc = Start-Process -FilePath $workerExe -ArgumentList $workerArgs -PassThru
        Start-Sleep -Seconds 1
        Write-Host "Worker started (PID $($workerProc.Id)). Launching master dashboard..."
        $env:VGRE_PORT = $port
        $dashProc = Start-Process -FilePath $dashboardExe -PassThru
        Write-Host "Dashboard started (PID $($dashProc.Id))."
        Write-Host "Press Enter to stop the test..." -ForegroundColor Yellow
        Read-Host | Out-Null
        Stop-Process -Id $workerProc.Id -ErrorAction SilentlyContinue
        Stop-Process -Id $dashProc.Id   -ErrorAction SilentlyContinue
        Write-Host "Test stopped." -ForegroundColor Green
    }
}
