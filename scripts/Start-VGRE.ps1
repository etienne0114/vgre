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
$mode          = ""
$masterIp      = ""    # --master-ip  (LAN: sets VGRE_CLUSTER_NODES)
$masterAddress = ""    # --master-address (WAN: sets VGRE_CLUSTER_MASTER_ADDRESS)
$port          = "7777"  # Must match kDefaultClusterPort in tcp_cluster_defaults.h
$threads       = ""

for ($i = 0; $i -lt $args.Length; $i++) {
    switch ($args[$i]) {
        "--master"  { $mode = "master" }
        "--worker"  { $mode = "worker" }
        "--test"    { $mode = "test"   }

        "--master-ip" {
            if (($i + 1) -lt $args.Length) {
                $masterIp = $args[++$i]
            }
        }

        # WAN / hostname / IPv6 direct connect.
        # Value format: IP:port, hostname:port, or [::1]:port
        "--master-address" {
            if (($i + 1) -lt $args.Length) {
                $masterAddress = $args[++$i]
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

        "--help" {
            Write-Host ""
            Write-Host "vgre-start -- VGRE Cluster Launcher"
            Write-Host ""
            Write-Host "Usage:"
            Write-Host "  vgre-start --master                          Start master + dashboard"
            Write-Host "  vgre-start --worker                          Start worker (LAN auto-discover)"
            Write-Host "  vgre-start --worker --master-ip <IP>         LAN: connect to specific master IP"
            Write-Host "  vgre-start --worker --master-address <H:P>   WAN: hostname/IPv4/IPv6 + port"
            Write-Host "  vgre-start --test                            Local self-test (master+worker)"
            Write-Host ""
            Write-Host "Options:"
            Write-Host "  --port <N>         TCP port (default 7777)"
            Write-Host "  --threads <N>      Worker thread count (default: auto)"
            Write-Host ""
            exit 0
        }

        default {
            Write-Host "[ERROR] Unknown option: $($args[$i])" -ForegroundColor Red
            Write-Host "        Run  vgre-start --help  for usage." -ForegroundColor Red
            exit 1
        }
    }
}

if (-not (Test-Port $port)) {
    Write-Host "[ERROR] Invalid port number: $port" -ForegroundColor Red
    exit 1
}

if (-not $mode) {
    Write-Host "Usage: vgre-start --master | --worker [--master-ip IP | --master-address HOST:PORT] | --test"
    Write-Host "       vgre-start --help"
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
# Prepend VGRE lib dir AND the LLVM runtime bin so vgre.dll and libomp.dll
# are both resolvable by the Windows loader before any DLL is mapped.
# -------------------------------
$LLVMBin = Join-Path $env:LOCALAPPDATA "VGRE\BuildTools\llvm\bin"
$env:PATH = "$InstallDir\lib;$InstallDir;$LLVMBin;$env:PATH"

$workerExe    = Join-Path $InstallDir "vgre-worker.exe"
$dashboardExe = Join-Path $InstallDir "vgre_dashboard.exe"

if (-not (Test-Path $workerExe)) {
    Write-Host "[ERROR] vgre-worker.exe not found at $workerExe" -ForegroundColor Red
    Write-Host "        Run  .\scripts\vgre_sync.bat  from the VGRE repository to build and install." -ForegroundColor Red
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
        if (-not (Test-Path $dashboardExe)) {
            Write-Host "[ERROR] vgre_dashboard.exe not found at $dashboardExe" -ForegroundColor Red
            Write-Host "        Run  .\scripts\vgre_sync.bat  from the VGRE repository to build and deploy the dashboard." -ForegroundColor Red
            exit 1
        }

        # Auto-detect real public IP so the master broadcasts with it.
        # Workers on different LANs then receive the correct address in the UDP ping.
        $discoverPs1 = Join-Path $InstallDir "scripts\vgre-discover.ps1"
        if (-not (Test-Path $discoverPs1)) {
            $discoverPs1 = Join-Path (Split-Path $PSCommandPath -Parent) "vgre-discover.ps1"
        }
        if (Test-Path $discoverPs1) {
            Write-Host "[...] Detecting public IP for WAN broadcast..." -ForegroundColor Cyan
            try {
                & powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
                    -File $discoverPs1 --set-master 2>$null
            } catch {}
        }

        Write-Host "Starting master..." -ForegroundColor Green
        $env:VGRE_PORT = $port
        Start-Process -FilePath $dashboardExe
    }

    "worker" {
        if (-not (Test-WorkerDependencies -WorkerExe $workerExe)) {
            exit 1
        }

        if ($masterAddress) {
            # WAN / explicit hostname:port — resolves via getaddrinfo in the C++ engine
            $env:VGRE_CLUSTER_MASTER_ADDRESS = $masterAddress
            Write-Host "Connecting to master at $masterAddress (WAN)" -ForegroundColor Cyan
        } elseif ($masterIp) {
            # Legacy LAN shorthand: only an IP was provided, append the port
            $env:VGRE_CLUSTER_NODES          = "${masterIp}:${port}"
            $env:VGRE_CLUSTER_MASTER_ADDRESS = "${masterIp}:${port}"
            Write-Host "Connecting to master at $masterIp`:$port" -ForegroundColor Cyan
        } else {
            Write-Host "Auto-discovering master on LAN (UDP broadcast)..." -ForegroundColor Cyan
        }

        & $workerExe @workerArgs

        if ($LASTEXITCODE -ne 0) {
            $hexCode = "0x{0:X8}" -f [uint32]$LASTEXITCODE
            Write-Host "[ERROR] Worker exited with code $LASTEXITCODE ($hexCode)" -ForegroundColor Red
            if ($LASTEXITCODE -eq -1073741819 -or $LASTEXITCODE -eq 0xC0000005 -or
                $LASTEXITCODE -eq [int]0xC0000005) {
                Write-Host "        STATUS_ACCESS_VIOLATION — possible causes:" -ForegroundColor Yellow
                Write-Host "          1. Missing DLL: check $InstallDir\lib for libomp.dll" -ForegroundColor Yellow
                Write-Host "          2. Run: .\scripts\vgre_sync.bat  to rebuild and redeploy" -ForegroundColor Yellow
                Write-Host "          3. See docs/TROUBLESHOOTING_WINDOWS.md for full diagnostics" -ForegroundColor Yellow
            }
        }
    }

    "test" {
        Write-Host "Starting local test..." -ForegroundColor Green
        Write-Host "Press Enter to stop..."

        if (-not (Test-WorkerDependencies -WorkerExe $workerExe)) {
            exit 1
        }

        if (-not (Test-Path $dashboardExe)) {
            Write-Host "[ERROR] vgre_dashboard.exe not found at $dashboardExe" -ForegroundColor Red
            Write-Host "        Run  .\scripts\vgre_sync.bat  from the VGRE repository to build and deploy the dashboard." -ForegroundColor Red
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