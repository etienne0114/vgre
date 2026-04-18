# VGRE Cluster Setup for Windows - run once on every machine that joins the cluster.
#
# What this does:
#   1. Creates or reuses an auth token shared by all nodes (master + workers).
#   2. Saves the token to %USERPROFILE%\.vgre\token
#   3. Persists VGRE_TCP_AUTH_TOKEN_FILE to your User environment so it loads automatically.
#   4. Prints the exact commands to start master and worker.
#
# Usage (run in PowerShell):
#   .\scripts\Setup-VGRECluster.ps1
#
# After this you only need:
#   .\scripts\Start-VGRE.ps1 --master          (machine running your CUDA app)
#   .\scripts\Start-VGRE.ps1 --worker          (each compute node, same subnet)
#   .\scripts\Start-VGRE.ps1 --worker --master-ip 192.168.1.50   (different subnet)

$ErrorActionPreference = "Stop"

$VgreDir   = Join-Path $env:USERPROFILE ".vgre"
$TokenFile = Join-Path $VgreDir "token"
$InstallDir = Join-Path $env:LOCALAPPDATA "VGRE"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "|       VGRE Cluster Setup                 |" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $VgreDir)) {
    New-Item -ItemType Directory -Path $VgreDir -Force | Out-Null
}

# -- Step 1: Auth Token --------------------------------------------------------
$keepExisting = $false
if (Test-Path $TokenFile) {
    $preview = (Get-Content $TokenFile -Raw).Substring(0, [Math]::Min(8, (Get-Content $TokenFile -Raw).Length))
    Write-Host "Existing token found: $preview... (truncated)" -ForegroundColor Yellow
    $ans = Read-Host "Keep existing token? [Y/n]"
    $keepExisting = ($ans -ne "n" -and $ans -ne "N")
}

if (-not $keepExisting) {
    Write-Host ""
    Write-Host "Auth Token Options:"
    Write-Host "  1) Generate a random secure token  (recommended)"
    Write-Host "  2) Enter your own token            (paste from another node)"
    $choice = Read-Host "Choice [1]"
    if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

    if ($choice -eq "2") {
        $secureToken = Read-Host "Enter token" -AsSecureString
        $token = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
            [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken))
        if ($token.Length -lt 8) {
            Write-Host "[ERROR] Token must be at least 8 characters. Aborting." -ForegroundColor Red
            exit 1
        }
    } else {
        # Generate 32 random bytes → 64 hex chars
        $bytes = New-Object byte[] 32
        [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $token = ($bytes | ForEach-Object { $_.ToString("x2") }) -join ""
        Write-Host "Generated token: $token" -ForegroundColor Green
    }

    [System.IO.File]::WriteAllText($TokenFile, $token)
    Write-Host "[OK] Token saved to $TokenFile" -ForegroundColor Green
}

# -- Step 2: Persist to User Environment --------------------------------------
$current = [System.Environment]::GetEnvironmentVariable("VGRE_TCP_AUTH_TOKEN_FILE", "User")
if ($current -ne $TokenFile) {
    [System.Environment]::SetEnvironmentVariable("VGRE_TCP_AUTH_TOKEN_FILE", $TokenFile, "User")
    Write-Host "[OK] VGRE_TCP_AUTH_TOKEN_FILE set in User environment" -ForegroundColor Green
} else {
    Write-Host "[OK] VGRE_TCP_AUTH_TOKEN_FILE already configured" -ForegroundColor Green
}
$env:VGRE_TCP_AUTH_TOKEN_FILE = $TokenFile

# -- Step 3: Ensure Install Dir in PATH ---------------------------------------
if (Test-Path $InstallDir) {
    $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$InstallDir*") {
        [System.Environment]::SetEnvironmentVariable("Path", "$InstallDir;$userPath", "User")
        Write-Host "[OK] Added $InstallDir to User PATH" -ForegroundColor Green
    }
}

# -- Done ----------------------------------------------------------------------
Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "|   Setup complete! Here is how to use your cluster:         |" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "|                                                            |"
Write-Host "|  MASTER node (machine running your CUDA app):              |"
Write-Host "|    .\scripts\Start-VGRE.ps1 --master                       |"
Write-Host "|                                                            |"
Write-Host "|  WORKER node (same subnet - auto-discovered):              |"
Write-Host "|    .\scripts\Start-VGRE.ps1 --worker                       |"
Write-Host "|                                                            |"
Write-Host "|  WORKER node (different subnet):                           |"
Write-Host "|    .\scripts\Start-VGRE.ps1 --worker --master-ip <IP>      |"
Write-Host "|                                                            |"
Write-Host "|  Copy the token file to each worker at the same path:      |"
Write-Host "|    scp $TokenFile user@worker:$TokenFile" -ForegroundColor Yellow
Write-Host "|    (or copy manually via USB / shared drive)               |"
Write-Host "|                                                            |"
Write-Host "|  NOTE: Open a NEW terminal for env changes to take effect. |"
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""
