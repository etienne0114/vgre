# VGRE Start - launch a master or worker node with one command (Windows)

$ErrorActionPreference = "Stop"

$InstallDir = if ($env:VGRE_INSTALL_DIR) { $env:VGRE_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA "VGRE" }
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
# Locate LLVM bin folder dynamically
# -------------------------------
function Get-LLVMBinPath {
    # 1. Environment Variable
    if ($env:LLVM_DIR -and (Test-Path $env:LLVM_DIR)) {
        $dir = $env:LLVM_DIR
        for ($i = 0; $i -lt 4; $i++) {
            $bin = Join-Path $dir "bin"
            if (Test-Path (Join-Path $bin "clang.exe")) { return $bin }
            $dir = Split-Path $dir -Parent
        }
    }

    # 2. Clang on PATH
    $clang = Get-Command clang.exe -ErrorAction SilentlyContinue
    if ($clang) {
        return Split-Path $clang.Source -Parent
    }

    # 3. Registry
    $regPaths = @(
        "HKLM:\SOFTWARE\LLVM\LLVM",
        "HKLM:\SOFTWARE\WOW6432Node\LLVM\LLVM",
        "HKCU:\SOFTWARE\LLVM\LLVM",
        "HKCU:\SOFTWARE\WOW6432Node\LLVM\LLVM"
    )
    foreach ($regPath in $regPaths) {
        if (Test-Path $regPath) {
            $val = Get-ItemPropertyValue -Path $regPath -Name "" -ErrorAction SilentlyContinue
            if ($val -and (Test-Path $val)) {
                $bin = Join-Path $val "bin"
                if (Test-Path (Join-Path $bin "clang.exe")) { return $bin }
            }
        }
    }

    # 4. Visual Studio LLVM component (via vswhere if available)
    $pf86 = $env:ProgramFiles(x86)
    $vswhere = if ($pf86) { Join-Path $pf86 "Microsoft Visual Studio\Installer\vswhere.exe" } else { "" }
    if ($vswhere -and (Test-Path $vswhere)) {
        $vsLlvm = & $vswhere -latest -requires Microsoft.VisualStudio.Component.VC.Llvm.Clang -find VC\Tools\Llvm 2>$null
        if ($vsLlvm -and (Test-Path $vsLlvm)) {
            $bin = Join-Path $vsLlvm "bin"
            if (Test-Path (Join-Path $bin "clang.exe")) { return $bin }
        }
        $cmakeConfigs = & $vswhere -latest -find **\LLVMConfig.cmake 2>$null
        foreach ($cfg in $cmakeConfigs) {
            if ($cfg -and (Test-Path $cfg)) {
                $dir = Split-Path $cfg -Parent
                for ($i = 0; $i -lt 4; $i++) {
                    $bin = Join-Path $dir "bin"
                    if (Test-Path (Join-Path $bin "clang.exe")) { return $bin }
                    $dir = Split-Path $dir -Parent
                }
            }
        }
    }

    # 5. Fallback default paths
    $toolsRoot = if ($env:VGRE_TOOLS_ROOT) { $env:VGRE_TOOLS_ROOT } else { Join-Path $InstallDir "BuildTools" }
    $common = @(
        "$env:ProgramFiles\LLVM\bin",
        "$env:ProgramFiles(x86)\LLVM\bin",
        (Join-Path $toolsRoot "llvm\bin"),
        "$env:USERPROFILE\scoop\apps\llvm\current\bin",
        "$env:LOCALAPPDATA\Programs\LLVM\bin"
    )
    foreach ($path in $common) {
        if ($path -and (Test-Path (Join-Path $path "clang.exe"))) { return $path }
    }

    return $null
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
$LLVMBin = Get-LLVMBinPath
if ($LLVMBin) {
    $env:PATH = "$InstallDir\lib;$InstallDir;$LLVMBin;$env:PATH"
} else {
    $env:PATH = "$InstallDir\lib;$InstallDir;$env:PATH"
}

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
        # Auto-detect real public IP so the master broadcasts with it.
        # Workers on different LANs then receive the correct address in the UDP ping.
        $discoverPs1 = Join-Path $InstallDir "scripts\vgre-discover.ps1"
        if (-not (Test-Path $discoverPs1)) {
            $discoverPs1 = Join-Path (Split-Path $PSCommandPath -Parent) "vgre-discover.ps1"
        }
        if (Test-Path $discoverPs1) {
            Write-Host "[...] Detecting public IP for WAN broadcast..." -ForegroundColor Cyan
            try {
                # Pass --set-master as a positional string argument, not a named param.
                # Using -Command instead of -File avoids PowerShell treating '--set-master'
                # as a named parameter binding.
                $dp = Start-Process -FilePath "powershell.exe" `
                    -ArgumentList "-NoLogo","-NoProfile","-ExecutionPolicy","Bypass",`
                                  "-Command","& '$discoverPs1' '--set-master'" `
                    -NoNewWindow -PassThru
                if (-not $dp.WaitForExit(10000)) { $dp.Kill() }
            } catch {}
        }

        if (-not (Test-WorkerDependencies -WorkerExe $workerExe)) {
            exit 1
        }

        Write-Host "Starting master worker..." -ForegroundColor Green
        $env:VGRE_PORT = $port
        $masterWorkerArgs = $workerArgs + @("--master")
        $workerProc = Start-Process -FilePath $workerExe -ArgumentList $masterWorkerArgs -PassThru

        if (Test-Path $dashboardExe) {
            Write-Host "Starting dashboard..." -ForegroundColor Green
            # WorkingDirectory must be $InstallDir so Windows finds flutter_windows.dll
            # alongside the exe (DLL search order: exe dir, then system, then PATH).
            Start-Process -FilePath $dashboardExe -WorkingDirectory $InstallDir
        } else {
            Write-Host "[WARN] vgre_dashboard.exe not found - running headless (no UI)." -ForegroundColor Yellow
            Write-Host "       Re-run .\scripts\vgre_sync.bat to build and deploy the dashboard." -ForegroundColor Yellow
            Write-Host ""
            Write-Host "Master is running (PID $($workerProc.Id)). Press Ctrl+C to stop." -ForegroundColor Green
            # Keep the terminal alive without blocking on Read-Host
            try { $workerProc.WaitForExit() } catch { }
        }
    }

    "worker" {
        if (-not (Test-WorkerDependencies -WorkerExe $workerExe)) {
            exit 1
        }

        if ($masterAddress) {
            # WAN / explicit hostname:port - resolves via getaddrinfo in the C++ engine
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
            Write-Host "[ERROR] Worker exited with code $LASTEXITCODE" -ForegroundColor Red
            if ($LASTEXITCODE -eq -1073741819) {
                Write-Host "        STATUS_ACCESS_VIOLATION - possible missing DLL in $InstallDir\lib" -ForegroundColor Yellow
                Write-Host "        Run: .\scripts\vgre_sync.bat to rebuild and redeploy" -ForegroundColor Yellow
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
            Write-Host "[WARN] vgre_dashboard.exe not found - running test without dashboard UI." -ForegroundColor Yellow
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
        $dashProc = $null
        if (Test-Path $dashboardExe) {
            $dashProc = Start-Process -FilePath $dashboardExe -WorkingDirectory $InstallDir -PassThru
        }

        Write-Host "Press Ctrl+C to stop." -ForegroundColor Cyan
        try { $workerProc.WaitForExit() } catch { }
        if ($dashProc -and -not $dashProc.HasExited) {
            Stop-Process -Id $dashProc.Id -ErrorAction SilentlyContinue
        }
        Write-Host "Test stopped." -ForegroundColor Green
    }
}