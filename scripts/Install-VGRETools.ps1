# Install-VGRETools.ps1
# Installs VGRE CLI tools (vgre-token, vgre-start) to %LOCALAPPDATA%\VGRE\scripts\
# and adds that directory to User PATH — no build or admin rights required.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\Install-VGRETools.ps1
#
# After running, the following commands are available in THIS terminal immediately:
#   vgre-token generate
#   vgre-token fingerprint
#   vgre-token set <TOKEN>
#   Setup-VGRECluster   (alias for scripts\Setup-VGRECluster.ps1)

$ErrorActionPreference = "Stop"

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$InstallDir = Join-Path $env:LOCALAPPDATA "VGRE\scripts"
$VgreDir    = Join-Path $env:LOCALAPPDATA "VGRE"

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  VGRE CLI Tools Installation" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# ── Create install directories ────────────────────────────────────────────────
foreach ($dir in @($VgreDir, $InstallDir)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "[OK] Created $dir" -ForegroundColor Green
    }
}

# ── Copy CLI scripts ──────────────────────────────────────────────────────────
$tools = @(
    @{ Src = "vgre-token.ps1";          Dst = "vgre-token.ps1" },
    @{ Src = "vgre-token.bat";          Dst = "vgre-token.bat" },
    @{ Src = "Setup-VGRECluster.ps1";   Dst = "Setup-VGRECluster.ps1" },
    @{ Src = "Start-VGRE.ps1";          Dst = "Start-VGRE.ps1" },
    @{ Src = "vgre_env.ps1";            Dst = "vgre_env.ps1" }
)

foreach ($tool in $tools) {
    $src = Join-Path $ScriptDir $tool.Src
    $dst = Join-Path $InstallDir $tool.Dst
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $dst -Force
        Write-Host "[OK] $($tool.Src)  →  $InstallDir" -ForegroundColor Green
    }
}

# ── Create vgre-start.bat launcher (mirrors vgre-token.bat pattern) ───────────
$startBat = Join-Path $InstallDir "vgre-start.bat"
@'
@echo off
:: vgre-start.bat — Windows launcher for Start-VGRE.ps1
setlocal
set "SCRIPT_DIR=%~dp0"
set "PS1=%SCRIPT_DIR%Start-VGRE.ps1"
if not exist "%PS1%" set "PS1=%LOCALAPPDATA%\VGRE\scripts\Start-VGRE.ps1"
if not exist "%PS1%" (
    echo [ERROR] Start-VGRE.ps1 not found. Re-run Install-VGRETools.ps1.
    exit /b 1
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
exit /b %ERRORLEVEL%
'@ | Set-Content -Path $startBat -Encoding ASCII
Write-Host "[OK] vgre-start.bat   →  $InstallDir" -ForegroundColor Green

# ── Create Setup-VGRECluster.bat launcher ────────────────────────────────────
$setupBat = Join-Path $InstallDir "Setup-VGRECluster.bat"
@'
@echo off
:: Setup-VGRECluster.bat — Windows launcher for Setup-VGRECluster.ps1
setlocal
set "SCRIPT_DIR=%~dp0"
set "PS1=%SCRIPT_DIR%Setup-VGRECluster.ps1"
if not exist "%PS1%" set "PS1=%LOCALAPPDATA%\VGRE\scripts\Setup-VGRECluster.ps1"
if not exist "%PS1%" (
    echo [ERROR] Setup-VGRECluster.ps1 not found. Re-run Install-VGRETools.ps1.
    exit /b 1
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
exit /b %ERRORLEVEL%
'@ | Set-Content -Path $setupBat -Encoding ASCII
Write-Host "[OK] Setup-VGRECluster.bat  →  $InstallDir" -ForegroundColor Green

# ── Update User PATH (persistent) ────────────────────────────────────────────
$userPath = [Environment]::GetEnvironmentVariable("Path", "User") ?? ""
$dirsToAdd = @($VgreDir, $InstallDir)
$changed   = $false
foreach ($dir in $dirsToAdd) {
    if ($userPath -notlike "*$dir*") {
        $userPath += ";$dir"
        $changed   = $true
    }
}
if ($changed) {
    [Environment]::SetEnvironmentVariable("Path", $userPath.TrimStart(";"), "User")
    Write-Host "[OK] User PATH updated (persists across reboots)." -ForegroundColor Green
} else {
    Write-Host "[OK] Directories already in User PATH." -ForegroundColor Green
}

# ── Update current session PATH immediately — no restart needed ───────────────
foreach ($dir in $dirsToAdd) {
    if ($env:PATH -notlike "*$dir*") {
        $env:PATH = "$dir;$env:PATH"
    }
}
Write-Host "[OK] Current session PATH updated — tools available NOW." -ForegroundColor Cyan

# ── Set VGRE_TCP_AUTH_TOKEN_FILE if token file already exists ─────────────────
$tokenFile = Join-Path $env:USERPROFILE ".vgre\token"
if (Test-Path $tokenFile) {
    [Environment]::SetEnvironmentVariable("VGRE_TCP_AUTH_TOKEN_FILE", $tokenFile, "User")
    $env:VGRE_TCP_AUTH_TOKEN_FILE = $tokenFile
    Write-Host "[OK] VGRE_TCP_AUTH_TOKEN_FILE set to $tokenFile" -ForegroundColor Green
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Installation complete!" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Run these commands right now in this terminal:" -ForegroundColor White
Write-Host ""
Write-Host "  vgre-token generate        # create a cluster auth token" -ForegroundColor Yellow
Write-Host "  vgre-token fingerprint     # verify SHA-256 matches on all nodes" -ForegroundColor Yellow
Write-Host "  Setup-VGRECluster          # interactive cluster setup wizard" -ForegroundColor Yellow
Write-Host ""
Write-Host "Token file:  $tokenFile" -ForegroundColor DarkGray
Write-Host ""
