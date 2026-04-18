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
#   .\scripts\Start-VGRE.ps1 --help                           Show this help

$ErrorActionPreference = "Stop"

$InstallDir = Join-Path $env:LOCALAPPDATA "VGRE"
$TokenFile  = if ($env:VGRE_TCP_AUTH_TOKEN_FILE) { $env:VGRE_TCP_AUTH_TOKEN_FILE }
              else { Join-Path $env:USERPROFILE ".vgre\token" }

# -- Parse arguments ----------------------------------------------------------
$mode     = ""
$masterIp = ""
$port     = "7777"
$threads  = ""

for ($i = 0; $i -lt $args.Length; $i++) {
    switch ($args[$i]) {
        "--master"    { $mode = "master" }
        "--worker"    { $mode = "worker" }
        "--test"      { $mode = "test"   }
        "--help"      { $mode = "help"   }
        "-h"          { $mode = "help"   }
        "--master-ip" { if (($i + 1) -lt $args.Length) { $masterIp = $args[++$i] } }
        "--port"      { if (($i + 1) -lt $args.Length) { $port     = $args[++$i] } }
        "--threads"   { if (($i + 1) -lt $args.Length) { $threads  = $args[++$i] } }
        default {
            Write-Host "Unknown option: $($args[$i])  (run .\scripts\Start-VGRE.ps1 --help)" -ForegroundColor Red
            exit 1
        }
    }
}

if ($mode -eq "help") {
    Get-Content $MyInvocation.MyCommand.Path | Where-Object { $_ -match "^#" } | ForEach-Object { $_ -replace "^# ?", "" }
    exit 0
}

if (-not $mode) {
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\scripts\Start-VGRE.ps1 --master"
    Write-Host "  .\scripts\Start-VGRE.ps1 --worker"
    Write-Host "  .\scripts\Start-VGRE.ps1 --worker --master-ip 192.168.1.50"
    Write-Host "  .\scripts\Start-VGRE.ps1 --test"
    exit 1
}

# -- Load Auth Token ----------------------------------------------------------
if (Test-Path $TokenFile) {
    $env:VGRE_TCP_AUTH_TOKEN_FILE = $TokenFile
} elseif (-not $env:VGRE_TCP_AUTH_TOKEN) {
    Write-Host ""
    Write-Host "[ERROR] No auth token found." -ForegroundColor Red
    Write-Host "   Run:  .\scripts\Setup-VGRECluster.ps1" -ForegroundColor Yellow
    Write-Host "   Or:   `$env:VGRE_TCP_AUTH_TOKEN = 'your-token'" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# -- Show SHA256 fingerprint --------------------------------------------------
# Hash the token text (no trailing newline) — identical to C++ SHA256(auth_token_str_).
# Do NOT use Get-FileHash: it hashes raw file bytes and will differ if the file has
# a BOM or trailing newline added by a text editor.
$_tokenText = ""
if (Test-Path $TokenFile) {
    $_tokenText = [System.IO.File]::ReadAllText($TokenFile).TrimEnd("`r", "`n")
} elseif ($env:VGRE_TCP_AUTH_TOKEN) {
    $_tokenText = $env:VGRE_TCP_AUTH_TOKEN
}
if ($_tokenText) {
    try {
        $_bytes = [System.Text.Encoding]::UTF8.GetBytes($_tokenText)
        $_sha   = [System.Security.Cryptography.SHA256]::Create()
        $_hash  = $_sha.ComputeHash($_bytes)
        $_fp    = ($_hash | ForEach-Object { $_.ToString("x2") }) -join ""
        Write-Host "Token fingerprint (SHA256): $($_fp.Substring(0,16))..." -ForegroundColor Cyan
        Write-Host "  (master and worker MUST show the same fingerprint)" -ForegroundColor Cyan
        Write-Host ""
    } catch {
        Write-Host "[WARN] Could not compute token fingerprint: $_" -ForegroundColor Yellow
    }
}

# -- Locate binaries ----------------------------------------------------------
$workerExe = Join-Path $InstallDir "vgre-worker.cmd"
if (-not (Test-Path $workerExe)) { $workerExe = Join-Path $InstallDir "vgre-worker.exe" }
if (-not (Test-Path $workerExe)) {
    Write-Host "[ERROR] vgre-worker not found in $InstallDir" -ForegroundColor Red
    Write-Host "   Run .\scripts\vgre_sync.bat to build and install first." -ForegroundColor Yellow
    exit 1
}

$dashboardExe = Join-Path $InstallDir "Launch-VGRE-Dashboard.cmd"
if (-not (Test-Path $dashboardExe)) { $dashboardExe = Join-Path $InstallDir "vgre_dashboard.exe" }

# Add install lib dir to DLL search path for this session
$env:PATH = "$InstallDir\lib;$InstallDir;$env:PATH"

# -- Worker args --------------------------------------------------------------
$workerArgs = @("--port", $port)
if ($threads) { $workerArgs += @("--threads", $threads) }

# -- Start --------------------------------------------------------------------
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
            Write-Host "Starting VGRE Worker -> master at ${masterIp}:${port}"
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
        Write-Host "  Press Enter to stop."
        Write-Host ""
        # Start worker in a separate window so both processes are visible
        $workerProc = Start-Process -FilePath $workerExe -ArgumentList $workerArgs -PassThru
        Start-Sleep -Seconds 1
        Write-Host "Worker started (PID $($workerProc.Id)). Launching master dashboard..."
        $env:VGRE_PORT = $port
        $dashProc = Start-Process -FilePath $dashboardExe -PassThru
        Write-Host "Dashboard started (PID $($dashProc.Id))." -ForegroundColor Green
        Write-Host "Press Enter to stop the test..." -ForegroundColor Yellow
        Read-Host | Out-Null
        Stop-Process -Id $workerProc.Id -ErrorAction SilentlyContinue
        Stop-Process -Id $dashProc.Id   -ErrorAction SilentlyContinue
        Write-Host "Test stopped." -ForegroundColor Green
    }
}
