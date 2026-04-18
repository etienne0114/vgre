# VGRE Cluster Setup for Windows — run once on every machine that joins the cluster.
#
# What this does:
#   1. Creates or reuses an auth token shared by all nodes (master + workers).
#   2. Saves the token to %USERPROFILE%\.vgre\token (no trailing newline, UTF-8).
#   3. Persists VGRE_TCP_AUTH_TOKEN_FILE to your User environment so it loads automatically.
#   4. Prints the exact commands to start master and worker.
#
# Usage (run in PowerShell):
#   .\scripts\Setup-VGRECluster.ps1
#   .\scripts\Setup-VGRECluster.ps1 -ShowFingerprint   (show current token fingerprint only)
#
# After this you only need:
#   .\scripts\Start-VGRE.ps1 --master          (machine running your CUDA app)
#   .\scripts\Start-VGRE.ps1 --worker          (each compute node, same subnet)
#   .\scripts\Start-VGRE.ps1 --worker --master-ip 192.168.1.50   (different subnet)

param(
    [switch]$ShowFingerprint   # Print the current token SHA256 fingerprint and exit
)

$ErrorActionPreference = "Stop"

$VgreDir    = Join-Path $env:USERPROFILE ".vgre"
$TokenFile  = Join-Path $VgreDir "token"
$InstallDir = Join-Path $env:LOCALAPPDATA "VGRE"

# -- Helper: SHA256 of token text (no trailing newline) -----------------------
# This matches exactly what the C++ engine computes for auth_token_str_.
# Do NOT use Get-FileHash — it hashes raw file bytes which may differ if a
# text editor adds a BOM or trailing newline.
function Get-TokenFingerprint {
    param([string]$File)
    if (-not (Test-Path $File)) { return "" }
    try {
        $text  = [System.IO.File]::ReadAllText($File).TrimEnd("`r", "`n")
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($text)
        $sha   = [System.Security.Cryptography.SHA256]::Create()
        $hash  = $sha.ComputeHash($bytes)
        return ($hash | ForEach-Object { $_.ToString("x2") }) -join ""
    } catch { return "" }
}

# -- -ShowFingerprint mode ----------------------------------------------------
if ($ShowFingerprint) {
    if (-not (Test-Path $TokenFile)) {
        Write-Host "[ERROR] No token found at $TokenFile — run Setup-VGRECluster.ps1 first." -ForegroundColor Red
        exit 1
    }
    $fp = Get-TokenFingerprint -File $TokenFile
    if (-not $fp) {
        Write-Host "[ERROR] Could not compute SHA256 fingerprint." -ForegroundColor Red
        exit 1
    }
    Write-Host "Token SHA256: $fp" -ForegroundColor Cyan
    Write-Host "Run this command on master AND worker — both must show identical output." -ForegroundColor Cyan
    exit 0
}

# -- Header -------------------------------------------------------------------
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "|       VGRE Cluster Setup                 |" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $VgreDir)) {
    New-Item -ItemType Directory -Path $VgreDir -Force | Out-Null
}

# -- Step 1: Auth Token -------------------------------------------------------
$keepExisting  = $false
$oldFingerprint = ""

if (Test-Path $TokenFile) {
    $oldFingerprint = Get-TokenFingerprint -File $TokenFile
    $preview = (Get-Content $TokenFile -Raw).Substring(0, [Math]::Min(8, (Get-Content $TokenFile -Raw).Length))
    Write-Host "Existing token found." -ForegroundColor Yellow
    if ($oldFingerprint) {
        Write-Host "  Current fingerprint (SHA256): $($oldFingerprint.Substring(0,16))..." -ForegroundColor Yellow
    } else {
        Write-Host "  Preview: $preview..." -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "WARNING: Replacing this token will DISCONNECT all workers using it." -ForegroundColor Red
    Write-Host "         They will need the new token file before they can reconnect." -ForegroundColor Red
    Write-Host ""
    $ans = Read-Host "Keep existing token? [Y/n]"
    $keepExisting = ($ans -ne "n" -and $ans -ne "N")
    if ($keepExisting) {
        Write-Host "[OK] Keeping existing token." -ForegroundColor Green
    }
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
        # Generate 32 random bytes -> 64 hex chars
        $bytes = New-Object byte[] 32
        [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $token = ($bytes | ForEach-Object { $_.ToString("x2") }) -join ""
        Write-Host "Generated token: $token" -ForegroundColor Green
    }

    # Write without trailing newline — matches how C++ reads it via std::getline + strip
    [System.IO.File]::WriteAllText($TokenFile, $token)

    # Compute fingerprint from the file we just wrote (single source of truth)
    $newFingerprint = Get-TokenFingerprint -File $TokenFile
    $fp16 = if ($newFingerprint) { $newFingerprint.Substring(0,16) } else { "(unavailable)" }
    $fp32 = if ($newFingerprint) { $newFingerprint.Substring(0,32) } else { "(unavailable)" }

    Write-Host "[OK] Token saved to $TokenFile" -ForegroundColor Green
    Write-Host "     SHA256 fingerprint: $fp16..." -ForegroundColor Green
    Write-Host ""
    Write-Host "+------------------------------------------------------------------+" -ForegroundColor Yellow
    Write-Host "|  WARNING: COPY THIS TOKEN FILE TO EVERY WORKER MACHINE          |" -ForegroundColor Yellow
    Write-Host "+------------------------------------------------------------------+" -ForegroundColor Yellow
    Write-Host "|  Token:  $token" -ForegroundColor Yellow
    Write-Host "|  SHA256: $fp32..." -ForegroundColor Yellow
    Write-Host "+------------------------------------------------------------------+" -ForegroundColor Yellow
    Write-Host "|  Copy command (run on THIS machine, replace WORKER_IP):          |" -ForegroundColor Yellow
    Write-Host "|    scp $TokenFile user@WORKER_IP:$TokenFile" -ForegroundColor Yellow
    Write-Host "|  (or copy manually via USB / shared drive)                       |" -ForegroundColor Yellow
    Write-Host "|                                                                  |" -ForegroundColor Yellow
    Write-Host "|  Verify both machines match (run on each):                       |" -ForegroundColor Yellow
    Write-Host "|    .\scripts\Setup-VGRECluster.ps1 -ShowFingerprint              |" -ForegroundColor Yellow
    Write-Host "|    Both machines MUST show: $fp16...           |" -ForegroundColor Yellow
    Write-Host "+------------------------------------------------------------------+" -ForegroundColor Yellow
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

# -- Always show the active fingerprint at the end ----------------------------
$activeFp = Get-TokenFingerprint -File $TokenFile
if ($activeFp) {
    Write-Host ""
    Write-Host "Active token fingerprint (SHA256): $($activeFp.Substring(0,16))..." -ForegroundColor Cyan
    Write-Host "  -> All cluster nodes must match this fingerprint." -ForegroundColor Cyan
    Write-Host "  -> Check any node at any time:" -ForegroundColor Cyan
    Write-Host "       .\scripts\Setup-VGRECluster.ps1 -ShowFingerprint" -ForegroundColor Cyan
}

# -- Done ---------------------------------------------------------------------
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
Write-Host "|    scp $TokenFile user@WORKER_IP:$TokenFile" -ForegroundColor Yellow
Write-Host "|    (or copy manually via USB / shared drive)               |"
Write-Host "|                                                            |"
Write-Host "|  NOTE: Open a NEW terminal for env changes to take effect. |"
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""
