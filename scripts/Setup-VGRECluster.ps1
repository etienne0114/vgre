param(
    [switch]$ShowFingerprint
)

$ErrorActionPreference = "Stop"

$VgreDir    = Join-Path $env:USERPROFILE ".vgre"
$TokenFile  = Join-Path $VgreDir "token"
$InstallDir = Join-Path $env:LOCALAPPDATA "VGRE"

# -- Helper: SHA256 of token text --------------------------------------------
function Get-TokenFingerprint {
    param([string]$File)

    if (-not (Test-Path $File)) { return "" }

    try {
        $text = [System.IO.File]::ReadAllText($File)

        if ([string]::IsNullOrWhiteSpace($text)) { return "" }

        $text  = $text.TrimEnd("`r", "`n")
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($text)

        $sha = [System.Security.Cryptography.SHA256]::Create()
        $hash = $sha.ComputeHash($bytes)

        return ($hash | ForEach-Object { $_.ToString("x2") }) -join ""
    }
    catch {
        return ""
    }
}

# -- Show fingerprint mode ----------------------------------------------------
if ($ShowFingerprint) {
    if (-not (Test-Path $TokenFile)) {
        Write-Host "[ERROR] No token found at $TokenFile" -ForegroundColor Red
        exit 1
    }

    $fp = Get-TokenFingerprint -File $TokenFile

    if (-not $fp) {
        Write-Host "[ERROR] Invalid or empty token file." -ForegroundColor Red
        exit 1
    }

    Write-Host "Token SHA256: $fp" -ForegroundColor Cyan
    Write-Host "Both master and worker must match." -ForegroundColor Cyan
    exit 0
}

# -- Header -------------------------------------------------------------------
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "|       VGRE Cluster Setup                 |" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Ensure directory exists
if (-not (Test-Path $VgreDir)) {
    New-Item -ItemType Directory -Path $VgreDir -Force | Out-Null
}

# -- Step 1: Token handling ---------------------------------------------------
$keepExisting = $false

if (Test-Path $TokenFile) {
    $existingContent = [System.IO.File]::ReadAllText($TokenFile)
    $existingContent = $existingContent.Trim()

    $oldFingerprint = Get-TokenFingerprint -File $TokenFile
    $preview = if ($existingContent.Length -gt 0) {
        $existingContent.Substring(0, [Math]::Min(8, $existingContent.Length))
    } else {
        "(empty)"
    }

    Write-Host "Existing token found." -ForegroundColor Yellow

    if ($oldFingerprint) {
        Write-Host "  Fingerprint: $($oldFingerprint.Substring(0,16))..." -ForegroundColor Yellow
    } else {
        Write-Host "  Preview: $preview..." -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "WARNING: Replacing token disconnects all workers!" -ForegroundColor Red
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
    Write-Host "  1) Generate secure token (recommended)"
    Write-Host "  2) Enter your own token"

    $choice = Read-Host "Choice [1]"
    if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

    if ($choice -eq "2") {

        $secureToken = Read-Host "Enter token" -AsSecureString

        $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
        try {
            $token = [Runtime.InteropServices.Marshal]::PtrToStringAuto($ptr)
        }
        finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
        }

        if ($token.Length -lt 8) {
            Write-Host "[ERROR] Token must be at least 8 characters." -ForegroundColor Red
            exit 1
        }

    } else {
        $bytes = New-Object byte[] 32
        [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)

        $token = ($bytes | ForEach-Object { $_.ToString("x2") }) -join ""
        Write-Host "Generated token: $token" -ForegroundColor Green
    }

    # Save token (no newline)
    [System.IO.File]::WriteAllText($TokenFile, $token)

    $newFingerprint = Get-TokenFingerprint -File $TokenFile

    if (-not $newFingerprint) {
        Write-Host "[ERROR] Failed to verify written token." -ForegroundColor Red
        exit 1
    }

    Write-Host "[OK] Token saved: $TokenFile" -ForegroundColor Green
    Write-Host "     SHA256: $($newFingerprint.Substring(0,16))..." -ForegroundColor Green

    Write-Host ""
    Write-Host "Copy this token file to ALL worker nodes." -ForegroundColor Yellow
    Write-Host "scp $TokenFile user@WORKER_IP:$TokenFile" -ForegroundColor Yellow
}

# -- Step 2: Environment variable --------------------------------------------
$current = [Environment]::GetEnvironmentVariable("VGRE_TCP_AUTH_TOKEN_FILE", "User")

if ($current -ne $TokenFile) {
    [Environment]::SetEnvironmentVariable("VGRE_TCP_AUTH_TOKEN_FILE", $TokenFile, "User")
    Write-Host "[OK] Environment variable set." -ForegroundColor Green
} else {
    Write-Host "[OK] Environment variable already set." -ForegroundColor Green
}

$env:VGRE_TCP_AUTH_TOKEN_FILE = $TokenFile

# -- Step 3: PATH handling ----------------------------------------------------
if (Test-Path $InstallDir) {

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")

    if (-not ($userPath -split ";" | Where-Object { $_ -eq $InstallDir })) {
        $newPath = "$InstallDir;$userPath"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")

        Write-Host "[OK] Added VGRE to PATH." -ForegroundColor Green
    }
}

# -- Final fingerprint --------------------------------------------------------
$activeFp = Get-TokenFingerprint -File $TokenFile

if ($activeFp) {
    Write-Host ""
    Write-Host "Active fingerprint: $($activeFp.Substring(0,16))..." -ForegroundColor Cyan
    Write-Host "All nodes MUST match this." -ForegroundColor Cyan
}

# -- Done ---------------------------------------------------------------------
Write-Host ""
Write-Host "==================== DONE ====================" -ForegroundColor Cyan
Write-Host "MASTER:"
Write-Host "  .\scripts\Start-VGRE.ps1 --master"
Write-Host ""
Write-Host "WORKER:"
Write-Host "  .\scripts\Start-VGRE.ps1 --worker"
Write-Host ""
Write-Host "DIFFERENT SUBNET:"
Write-Host "  .\scripts\Start-VGRE.ps1 --worker --master-ip <IP>"
Write-Host ""
Write-Host "IMPORTANT: Restart terminal after setup."
Write-Host "==============================================" -ForegroundColor Cyan