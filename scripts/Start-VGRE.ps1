# VGRE Start — launch a master or worker node with one command (Windows)

$ErrorActionPreference = "Stop"

$InstallDir = Join-Path $env:LOCALAPPDATA "VGRE"
$TokenFile = if ($env:VGRE_TCP_AUTH_TOKEN_FILE) {
    $env:VGRE_TCP_AUTH_TOKEN_FILE
} else {
    Join-Path $env:USERPROFILE ".vgre\token"
}

# -------------------------------
# Validate port
# -------------------------------
function Test-Port {
    param([string]$Port)
    return ($Port -match '^\d+$' -and [int]$Port -ge 1 -and [int]$Port -le 65535)
}

# -------------------------------
# Clean token file
# -------------------------------
function Repair-TokenFile {
    param([string]$File)

    if (-not (Test-Path $File)) { return $false }

    try {
        $raw = [System.IO.File]::ReadAllText($File)

        if ([string]::IsNullOrWhiteSpace($raw)) {
            Write-Host "[ERROR] Token file is empty." -ForegroundColor Red
            return $false
        }

        $clean = $raw.Trim()

        if ($clean -ne $raw) {
            [System.IO.File]::WriteAllText($File, $clean)
            Write-Host "[INFO] Token file sanitized." -ForegroundColor DarkGray
        }

        return $true
    }
    catch {
        Write-Host "[ERROR] Cannot read token file." -ForegroundColor Red
        return $false
    }
}

# -------------------------------
# Fingerprint
# -------------------------------
function Get-TokenFingerprint {
    param([string]$File)

    if (-not (Test-Path $File)) { return "" }

    try {
        $text = [System.IO.File]::ReadAllText($File).Trim()
        if ([string]::IsNullOrWhiteSpace($text)) { return "" }

        $bytes = [System.Text.Encoding]::UTF8.GetBytes($text)
        $sha = [System.Security.Cryptography.SHA256]::Create()
        $hash = $sha.ComputeHash($bytes)

        return ($hash | ForEach-Object { $_.ToString("x2") }) -join ""
    }
    catch {
        return ""
    }
}

# -------------------------------
# Check DLLs
# -------------------------------
function Test-WorkerDependencies {
    param([string]$WorkerExe)

    $libDir = Join-Path $InstallDir "lib"

    $required = @("vgre.dll", "vgre_cudart.dll")
    $missing = @()

    foreach ($dll in $required) {

        $paths = @(
            (Join-Path $libDir $dll),
            (Join-Path (Split-Path $WorkerExe -Parent) $dll),
            (Join-Path $InstallDir $dll)
        )

        $found = $false

        foreach ($p in $paths) {
            if (Test-Path $p) {
                $found = $true
                break
            }
        }

        if (-not $found) {
            $missing += $dll
        }
    }

    if ($missing.Count -gt 0) {
        Write-Host "[ERROR] Missing DLLs:" -ForegroundColor Red
        foreach ($dll in $missing) {
            Write-Host "  $dll" -ForegroundColor Red
        }
        return $false
    }

    return $true
}

# -------------------------------
# Parse arguments
# -------------------------------
$mode = ""
$masterIp = ""
$port = "7777"
$threads = ""

for ($i = 0; $i -lt $args.Length; $i++) {
    switch ($args[$i]) {
        "--master" { $mode = "master" }
        "--worker" { $mode = "worker" }
        "--test"   { $mode = "test" }

        "--master-ip" {
            if (($i + 1) -lt $args.Length) {
                $masterIp = $args[++$i]
            }
        }

        "--port" {
            if (($i + 1) -lt $args.Length) {
                $port = $args[++$i]
            }
        }

        "--threads" {
            if (($i + 1) -lt $args.Length) {
                $threads = $args[++$i]
            }
        }

        default {
            Write-Host "[ERROR] Unknown option: $($args[$i])" -ForegroundColor Red
            exit 1
        }
    }
}

if (-not (Test-Port $port)) {
    Write-Host "[ERROR] Invalid port number." -ForegroundColor Red
    exit 1
}

if (-not $mode) {
    Write-Host "Usage:"
    Write-Host "  --master | --worker | --test"
    exit 1
}

# -------------------------------
# Token validation
# -------------------------------
if (Test-Path $TokenFile) {
    $env:VGRE_TCP_AUTH_TOKEN_FILE = $TokenFile

    if (-not (Repair-TokenFile -File $TokenFile)) {
        exit 1
    }
}
elseif (-not $env:VGRE_TCP_AUTH_TOKEN) {
    Write-Host "[ERROR] No auth token found." -ForegroundColor Red
    exit 1
}

# -------------------------------
# Fingerprint display
# -------------------------------
$fp = Get-TokenFingerprint -File $TokenFile
if ($fp) {
    Write-Host "Token fingerprint: $($fp.Substring(0,16))..." -ForegroundColor Cyan
}

# -------------------------------
# Locate executables
# -------------------------------
$env:PATH = "$InstallDir\lib;$InstallDir;$env:PATH"

$workerExe = Join-Path $InstallDir "vgre-worker.exe"
$dashboardExe = Join-Path $InstallDir "vgre_dashboard.exe"

if (-not (Test-Path $workerExe)) {
    Write-Host "[ERROR] Worker not found." -ForegroundColor Red
    exit 1
}

# -------------------------------
# Worker args
# -------------------------------
$workerArgs = @("--port", $port)
if ($threads) {
    $workerArgs += @("--threads", $threads)
}

# -------------------------------
# Start modes
# -------------------------------
switch ($mode) {

    "master" {
        Write-Host "Starting master..." -ForegroundColor Green
        $env:VGRE_PORT = $port
        Start-Process -FilePath $dashboardExe
    }

    "worker" {
        if (-not (Test-WorkerDependencies -WorkerExe $workerExe)) {
            exit 1
        }

        if ($masterIp) {
            $env:VGRE_CLUSTER_NODES = "${masterIp}:${port}"
            Write-Host "Connecting to $masterIp`:$port"
        } else {
            Write-Host "Auto-discovering master..."
        }

        & $workerExe @workerArgs

        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Worker exited with code $LASTEXITCODE" -ForegroundColor Red
        }
    }

    "test" {
        Write-Host "Starting local test..." -ForegroundColor Green
        Write-Host "Press Enter to stop..."

        if (-not (Test-WorkerDependencies -WorkerExe $workerExe)) {
            exit 1
        }

        $workerProc = Start-Process -FilePath $workerExe `
            -ArgumentList $workerArgs `
            -PassThru

        Start-Sleep -Seconds 2

        if ($workerProc.HasExited) {
            Write-Host "[ERROR] Worker failed to start." -ForegroundColor Red
            exit 1
        }

        $env:VGRE_PORT = $port
        $dashProc = Start-Process -FilePath $dashboardExe -PassThru

        Read-Host | Out-Null

        Stop-Process -Id $workerProc.Id -ErrorAction SilentlyContinue
        Stop-Process -Id $dashProc.Id -ErrorAction SilentlyContinue

        Write-Host "Test stopped." -ForegroundColor Green
    }
}