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

# ---------------------------------------------------------------------------
# Helper: sanitize token file
#   Strips BOM, CRLF, trailing/leading whitespace and REWRITES the file clean.
#   This is critical: C++ std::getline strips only \r\n at the end.
#   If the file has extra spaces (from Notepad, paste, etc.) C++ and the
#   fingerprint computation must see the exact same bytes — so we clean the
#   file on disk, not just in memory.
# ---------------------------------------------------------------------------
function Repair-TokenFile {
    param([string]$File)
    if (-not (Test-Path $File)) { return $false }
    try {
        # ReadAllText handles UTF-8 BOM automatically
        $raw   = [System.IO.File]::ReadAllText($File, [System.Text.Encoding]::UTF8)
        $clean = $raw.Trim()   # remove ALL surrounding whitespace incl. CRLF, spaces
        if ($clean.Length -eq 0) {
            Write-Host "[ERROR] Token file is empty after sanitization: $File" -ForegroundColor Red
            return $false
        }
        if ($clean -ne $raw) {
            [System.IO.File]::WriteAllText($File, $clean)
            Write-Host "[INFO] Token file sanitized (removed extra whitespace/newlines)." -ForegroundColor DarkGray
        }
        return $true
    } catch {
        Write-Host "[ERROR] Could not read token file: $_" -ForegroundColor Red
        return $false
    }
}

# ---------------------------------------------------------------------------
# Helper: SHA256 fingerprint of the token text in the file.
#   MUST be called after Repair-TokenFile so the on-disk bytes match what
#   the C++ engine reads via std::getline.
# ---------------------------------------------------------------------------
function Get-TokenFingerprint {
    param([string]$File)
    if (-not (Test-Path $File)) { return "" }
    try {
        $text  = [System.IO.File]::ReadAllText($File, [System.Text.Encoding]::UTF8).Trim()
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($text)
        $sha   = [System.Security.Cryptography.SHA256]::Create()
        $hash  = $sha.ComputeHash($bytes)
        return ($hash | ForEach-Object { $_.ToString("x2") }) -join ""
    } catch { return "" }
}

# ---------------------------------------------------------------------------
# Helper: DLL pre-flight check
#   On Windows, a missing DLL causes the process to exit with no output at all
#   (Windows shows a dialog that PowerShell doesn't capture). We check the
#   most likely dependency before launching so we can give a useful message.
# ---------------------------------------------------------------------------
function Test-WorkerDependencies {
    param([string]$WorkerExe)
    $dir = Split-Path $WorkerExe -Parent
    $libDir = Join-Path $InstallDir "lib"

    $required = @("libvgre.dll", "libvgre_cudart.dll")
    $missing  = @()

    foreach ($dll in $required) {
        $found = (Test-Path (Join-Path $libDir $dll)) -or
                 (Test-Path (Join-Path $dir  $dll)) -or
                 (Test-Path (Join-Path $InstallDir $dll))
        if (-not $found) { $missing += $dll }
    }

    if ($missing.Count -gt 0) {
        Write-Host ""
        Write-Host "[ERROR] Required DLLs not found:" -ForegroundColor Red
        foreach ($dll in $missing) {
            Write-Host "        $dll (expected in $libDir)" -ForegroundColor Red
        }
        Write-Host ""
        Write-Host "  Run .\scripts\vgre_sync.bat to build and install VGRE first." -ForegroundColor Yellow
        Write-Host ""
        return $false
    }
    return $true
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
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
            Write-Host "Unknown option: $($args[$i])" -ForegroundColor Red
            Write-Host "Run: .\scripts\Start-VGRE.ps1 --help"
            exit 1
        }
    }
}

if ($mode -eq "help") {
    Get-Content $MyInvocation.MyCommand.Path |
        Where-Object { $_ -match "^#" } |
        ForEach-Object { $_ -replace "^# ?", "" }
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

# ---------------------------------------------------------------------------
# Load and validate auth token
# ---------------------------------------------------------------------------
if (Test-Path $TokenFile) {
    $env:VGRE_TCP_AUTH_TOKEN_FILE = $TokenFile

    # Sanitize the token file — fixes pasted tokens with CRLF / trailing spaces
    if (-not (Repair-TokenFile -File $TokenFile)) {
        Write-Host "[ERROR] Token file is invalid. Re-run .\scripts\Setup-VGRECluster.ps1" -ForegroundColor Red
        exit 1
    }

} elseif ($env:VGRE_TCP_AUTH_TOKEN) {
    # Token is in env var — fine, C++ will read it directly
} else {
    Write-Host ""
    Write-Host "[ERROR] No auth token found." -ForegroundColor Red
    Write-Host "   Run:  .\scripts\Setup-VGRECluster.ps1" -ForegroundColor Yellow
    Write-Host "   Or:   `$env:VGRE_TCP_AUTH_TOKEN = 'your-token'" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# ---------------------------------------------------------------------------
# Show SHA256 fingerprint
#   Hash is computed from the SANITIZED file bytes — identical to what the
#   C++ engine computes for auth_token_str_ after std::getline + strip loop.
# ---------------------------------------------------------------------------
$_fp = ""
if (Test-Path $TokenFile) {
    $_fp = Get-TokenFingerprint -File $TokenFile
} elseif ($env:VGRE_TCP_AUTH_TOKEN) {
    try {
        $_bytes = [System.Text.Encoding]::UTF8.GetBytes($env:VGRE_TCP_AUTH_TOKEN.Trim())
        $_sha   = [System.Security.Cryptography.SHA256]::Create()
        $_hash  = $_sha.ComputeHash($_bytes)
        $_fp    = ($_hash | ForEach-Object { $_.ToString("x2") }) -join ""
    } catch {}
}

if ($_fp) {
    Write-Host ""
    Write-Host "Token fingerprint (SHA256): $($_fp.Substring(0,16))..." -ForegroundColor Cyan
    Write-Host "  Master and worker MUST show the same fingerprint." -ForegroundColor Cyan
    Write-Host "  Verify on any node: .\scripts\Setup-VGRECluster.ps1 -ShowFingerprint" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "[WARN] Could not compute token fingerprint." -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# Locate binaries
# ---------------------------------------------------------------------------
$env:PATH = "$InstallDir\lib;$InstallDir;$env:PATH"

$workerExe = Join-Path $InstallDir "vgre-worker.cmd"
if (-not (Test-Path $workerExe)) { $workerExe = Join-Path $InstallDir "vgre-worker.exe" }

# Fallback: look in the repo's build directory (for development / first-time users)
if (-not (Test-Path $workerExe)) {
    $repoDir   = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
    $buildWorker = Join-Path $repoDir "build\src\advanced\vgre-worker.exe"
    if (Test-Path $buildWorker) {
        $workerExe = $buildWorker
        Write-Host "[INFO] Using worker from build directory: $workerExe" -ForegroundColor DarkGray
        # Add build lib path so DLLs resolve
        $buildLib = Join-Path $repoDir "build"
        $env:PATH = "$buildLib;$env:PATH"
    }
}

if (-not (Test-Path $workerExe)) {
    Write-Host "[ERROR] vgre-worker not found." -ForegroundColor Red
    Write-Host "   Expected: $InstallDir\vgre-worker.exe" -ForegroundColor Yellow
    Write-Host "   Run .\scripts\vgre_sync.bat to build and install first." -ForegroundColor Yellow
    exit 1
}

$dashboardExe = Join-Path $InstallDir "Launch-VGRE-Dashboard.cmd"
if (-not (Test-Path $dashboardExe)) { $dashboardExe = Join-Path $InstallDir "vgre_dashboard.exe" }

# ---------------------------------------------------------------------------
# Worker args
# ---------------------------------------------------------------------------
$workerArgs = @("--port", $port)
if ($threads) { $workerArgs += @("--threads", $threads) }

# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------
switch ($mode) {

    "master" {
        Write-Host "Starting VGRE Master Node..." -ForegroundColor Green
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
            Write-Host "Starting VGRE Worker..." -ForegroundColor Green
            Write-Host "  Connecting to master at: $masterIp`:$port"
        } else {
            Write-Host "Starting VGRE Worker..." -ForegroundColor Green
            Write-Host "  Mode: auto-discovering master on local subnet"
        }
        Write-Host "  Port:  $port"
        Write-Host "  Token: $TokenFile"
        Write-Host ""

        # DLL pre-flight — missing DLLs cause silent exit on Windows
        if (-not (Test-WorkerDependencies -WorkerExe $workerExe)) { exit 1 }

        Write-Host "Worker is running. Press Ctrl+C to stop." -ForegroundColor Yellow
        Write-Host "Scanning for master and waiting for connection..." -ForegroundColor Cyan
        Write-Host ""

        & $workerExe @workerArgs
        $ec = $LASTEXITCODE

        Write-Host ""
        if ($ec -eq 0) {
            Write-Host "Worker stopped cleanly." -ForegroundColor Green
        } else {
            Write-Host "[ERROR] Worker stopped with exit code $ec" -ForegroundColor Red
            Write-Host ""
            Write-Host "Common causes:" -ForegroundColor Yellow
            Write-Host "  1. Missing DLLs — run .\scripts\vgre_sync.bat to reinstall" -ForegroundColor Yellow
            Write-Host "  2. Port $port already in use — try --port 7778" -ForegroundColor Yellow
            Write-Host "  3. Token mismatch — re-run Setup-VGRECluster.ps1 -ShowFingerprint" -ForegroundColor Yellow
            Write-Host "     and compare fingerprint with master" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "For full diagnostics run:" -ForegroundColor Yellow
            Write-Host "  $workerExe --port $port 2>&1" -ForegroundColor Yellow
        }
    }

    "test" {
        Write-Host "Starting local self-test (master + worker on same machine)..." -ForegroundColor Green
        Write-Host "  Token: $TokenFile"
        Write-Host "  Press Enter to stop."
        Write-Host ""

        if (-not (Test-WorkerDependencies -WorkerExe $workerExe)) { exit 1 }

        # Start worker in a separate window so output is visible
        $workerProc = Start-Process -FilePath $workerExe -ArgumentList $workerArgs `
                        -PassThru -NoNewWindow
        Start-Sleep -Seconds 2

        if ($workerProc.HasExited) {
            Write-Host "[ERROR] Worker exited immediately (code: $($workerProc.ExitCode))" -ForegroundColor Red
            Write-Host "  Run .\scripts\vgre_sync.bat to rebuild and reinstall." -ForegroundColor Yellow
            exit 1
        }

        Write-Host "Worker running (PID $($workerProc.Id)). Launching master dashboard..." -ForegroundColor Green
        $env:VGRE_PORT = $port

        $dashProc = Start-Process -FilePath $dashboardExe -PassThru
        Write-Host "Dashboard running (PID $($dashProc.Id))." -ForegroundColor Green
        Write-Host ""
        Write-Host "Press Enter to stop the test..." -ForegroundColor Yellow
        Read-Host | Out-Null

        Stop-Process -Id $workerProc.Id -ErrorAction SilentlyContinue
        Stop-Process -Id $dashProc.Id   -ErrorAction SilentlyContinue
        Write-Host "Test stopped." -ForegroundColor Green
    }
}
