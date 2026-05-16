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

        # Sanitize the existing file in case it was edited manually and has
        # trailing CRLF / spaces that would cause a fingerprint mismatch.
        $raw   = [System.IO.File]::ReadAllText($TokenFile, [System.Text.Encoding]::UTF8)
        $clean = $raw.Trim()
        if ($clean -ne $raw) {
            [System.IO.File]::WriteAllText($TokenFile, $clean)
            Write-Host "[OK] Token file sanitized (removed extra whitespace)." -ForegroundColor Green
        }
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
        Write-Host "(Paste the token from another node — any surrounding whitespace is stripped automatically)" -ForegroundColor DarkGray
        $secureToken = Read-Host "Enter token" -AsSecureString

        $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
        try {
            $token = [Runtime.InteropServices.Marshal]::PtrToStringAuto($ptr)
        }
        finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
        }

        # Strip ALL surrounding whitespace (CRLF, spaces, tabs) that may be
        # included when pasting from terminal output or a text editor.
        $token = $token.Trim()

        if ($token.Length -lt 8) {
            Write-Host "[ERROR] Token must be at least 8 characters." -ForegroundColor Red
            exit 1
        }

        Write-Host "[OK] Token accepted ($($token.Length) chars)." -ForegroundColor Green

    } else {
        $bytes = New-Object byte[] 32
        [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)

        $token = ($bytes | ForEach-Object { $_.ToString("x2") }) -join ""
        Write-Host "Generated token: $token" -ForegroundColor Green
    }

    # Write WITHOUT trailing newline. C++ std::getline reads up to \n/EOF and
    # strips trailing \r\n — so writing a bare string ensures both sides hash
    # the identical bytes.
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

# -- Step 3: Install CLI tools into PATH -------------------------------------
$ScriptDir      = Split-Path -Parent $MyInvocation.MyCommand.Path
$TokenScriptDir = Join-Path $InstallDir "scripts"
if (-not (Test-Path $TokenScriptDir)) {
    New-Item -ItemType Directory -Path $TokenScriptDir -Force | Out-Null
}
# Copy all CLI tools that exist in the repo scripts/ directory
$cliFiles = @(
    "vgre-token.ps1", "vgre-token.bat",
    "Setup-VGRECluster.ps1",
    "Start-VGRE.ps1",
    "vgre_env.ps1",
    "Install-VGRETools.ps1"
)
foreach ($f in $cliFiles) {
    $src = Join-Path $ScriptDir $f
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination (Join-Path $TokenScriptDir $f) -Force
    }
}
Write-Host "[OK] CLI tools installed to $TokenScriptDir" -ForegroundColor Green

# -- Step 4: PATH handling ----------------------------------------------------
$userPath = [Environment]::GetEnvironmentVariable("Path", "User") ?? ""
$dirs = $userPath -split ";" | Where-Object { $_ -ne "" }
$changed = $false
foreach ($dir in @($InstallDir, $TokenScriptDir)) {
    if (-not ($dirs -contains $dir)) {
        $dirs += $dir
        $changed = $true
    }
}
if ($changed) {
    [Environment]::SetEnvironmentVariable("Path", ($dirs -join ";"), "User")
    Write-Host "[OK] Added VGRE to User PATH (persists across reboots)." -ForegroundColor Green
} else {
    Write-Host "[OK] VGRE already in User PATH." -ForegroundColor Green
}

# Also update the current PowerShell session — no restart needed.
foreach ($dir in @($InstallDir, $TokenScriptDir)) {
    if ($env:PATH -notlike "*$dir*") {
        $env:PATH = "$dir;$env:PATH"
    }
}
Write-Host "[OK] Current session PATH updated — vgre-token is available NOW." -ForegroundColor Cyan

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
Write-Host "TOKEN MANAGEMENT (run from any terminal):" -ForegroundColor White
Write-Host "  vgre-token generate          create / rotate token"
Write-Host "  vgre-token fingerprint       show SHA-256 for comparison"
Write-Host "  vgre-token set <TOKEN>       paste token from master"
Write-Host "  vgre-token copy              show copy command for workers"
Write-Host "  vgre-token verify            check master/worker match"
Write-Host ""
Write-Host "MASTER:"
Write-Host "  vgre-start --master   (or  .\scripts\Start-VGRE.ps1 --master)"
Write-Host ""
Write-Host "WORKER:"
Write-Host "  vgre-start --worker   (or  .\scripts\Start-VGRE.ps1 --worker)"
Write-Host ""
Write-Host "DIFFERENT SUBNET:"
Write-Host "  vgre-start --worker --master-ip <IP>"
Write-Host ""
Write-Host "IMPORTANT: Open a NEW terminal for PATH changes to take effect."
Write-Host "==============================================" -ForegroundColor Cyan