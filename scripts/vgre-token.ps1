# vgre-token.ps1 — VGRE Auth Token Manager (Windows / PowerShell)
#
# Run from ANYWHERE after install:
#   vgre-token generate              Generate a new secure 64-hex token
#   vgre-token show                  Print the stored token value
#   vgre-token fingerprint           Print the SHA-256 fingerprint
#   vgre-token set <TOKEN>           Store a specific token (from another node)
#   vgre-token verify                Check env-var matches the stored file
#   vgre-token copy                  Print the robocopy/scp command
#   vgre-token revoke                Delete the stored token
#
# Token file:  %USERPROFILE%\.vgre\token
# The env-var VGRE_TCP_AUTH_TOKEN_FILE is written to User scope (persists
# across sessions without needing to restart or source anything).
#
# Works on: Windows PowerShell 5.1+, PowerShell 7+
# No admin rights required.

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Command = "help",

    [Parameter(Position = 1)]
    [string]$TokenValue = ""
)

$ErrorActionPreference = "Stop"

# ── Paths ─────────────────────────────────────────────────────────────────────
$VgreDir   = Join-Path $env:USERPROFILE ".vgre"
$TokenFile = Join-Path $VgreDir "token"

# ── Helpers ───────────────────────────────────────────────────────────────────
function Ensure-VgreDir {
    if (-not (Test-Path $VgreDir)) {
        New-Item -ItemType Directory -Path $VgreDir -Force | Out-Null
    }
    # Restrict permissions to current user only (icacls on Windows)
    try {
        $acl = Get-Acl $VgreDir
        $acl.SetAccessRuleProtection($true, $false)
        $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
            $env:USERNAME, "FullControl", "ContainerInherit,ObjectInherit",
            "None", "Allow")
        $acl.AddAccessRule($rule)
        Set-Acl -Path $VgreDir -AclObject $acl -ErrorAction SilentlyContinue
    } catch {}
}

function Get-TokenSHA256 {
    param([string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) { return "" }
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
    $sha   = [System.Security.Cryptography.SHA256]::Create()
    $hash  = $sha.ComputeHash($bytes)
    return ($hash | ForEach-Object { $_.ToString("x2") }) -join ""
}

function Read-StoredToken {
    if (-not (Test-Path $TokenFile)) {
        Write-Host "[ERROR] No token found at $TokenFile. Run: vgre-token generate" `
            -ForegroundColor Red
        exit 1
    }
    return [System.IO.File]::ReadAllText($TokenFile, [System.Text.Encoding]::UTF8).Trim()
}

function Write-StoredToken {
    param([string]$Token)
    Ensure-VgreDir
    # Write WITHOUT trailing newline so hash bytes match POSIX systems
    [System.IO.File]::WriteAllText($TokenFile, $Token, [System.Text.Encoding]::UTF8)
    # Restrict file to current user
    try {
        $acl = Get-Acl $TokenFile
        $acl.SetAccessRuleProtection($true, $false)
        $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
            $env:USERNAME, "FullControl", "None", "None", "Allow")
        $acl.AddAccessRule($rule)
        Set-Acl -Path $TokenFile -AclObject $acl -ErrorAction SilentlyContinue
    } catch {}
}

function Set-PersistentEnvVar {
    param([string]$Name, [string]$Value)
    # User scope — no admin needed; survives reboots; visible to new terminals
    [Environment]::SetEnvironmentVariable($Name, $Value, "User")
    # Also set for the current process immediately
    [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
}

function Print-FingerprintBlock {
    param([string]$Token)
    $fp = Get-TokenSHA256 -Text $Token
    Write-Host ""
    Write-Host "Token fingerprint (SHA-256):" -ForegroundColor White
    Write-Host "  $fp" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Run  vgre-token fingerprint  on every node." -ForegroundColor DarkGray
    Write-Host "Both master and worker MUST show identical output." -ForegroundColor DarkGray
    Write-Host ""
}

function Add-ToWindowsPath {
    # Ensure VGRE install dir is in the user PATH so vgre-token.bat is callable
    $installDir = Join-Path $env:LOCALAPPDATA "VGRE"
    $scriptsDir = Join-Path $installDir "scripts"
    $userPath   = [Environment]::GetEnvironmentVariable("Path", "User") ?? ""
    $dirs = $userPath -split ";" | Where-Object { $_ -ne "" }
    $changed = $false
    foreach ($dir in @($installDir, $scriptsDir)) {
        if (-not ($dirs -contains $dir)) {
            $dirs += $dir
            $changed = $true
        }
    }
    if ($changed) {
        [Environment]::SetEnvironmentVariable("Path", ($dirs -join ";"), "User")
        Write-Host "[OK] Added VGRE to user PATH." -ForegroundColor Green
    }
}

# ── Commands ──────────────────────────────────────────────────────────────────

function Cmd-Generate {
    if (Test-Path $TokenFile) {
        $existing = Read-StoredToken
        $fp       = Get-TokenSHA256 -Text $existing
        Write-Host ""
        Write-Host "[WARN] A token already exists." -ForegroundColor Yellow
        Write-Host "  Fingerprint: $($fp.Substring(0,16))..." -ForegroundColor Yellow
        Write-Host "  Generating a new token will disconnect all workers." -ForegroundColor Yellow
        Write-Host ""
        $ans = Read-Host "Continue? [y/N]"
        if ($ans -ne "y" -and $ans -ne "Y") {
            Write-Host "[INFO] Aborted. Existing token kept." -ForegroundColor Cyan
            return
        }
    }

    Write-Host ""
    Write-Host "[...] Generating 256-bit random token..." -ForegroundColor Cyan

    # Cryptographically-secure random bytes (no OpenSSL dependency on Windows)
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $token = ($bytes | ForEach-Object { $_.ToString("x2") }) -join ""

    Write-StoredToken -Token $token
    Set-PersistentEnvVar -Name "VGRE_TCP_AUTH_TOKEN_FILE" -Value $TokenFile
    Add-ToWindowsPath

    Write-Host "[OK] Token saved to $TokenFile" -ForegroundColor Green
    Print-FingerprintBlock -Token $token
    Write-Host "[OK] VGRE_TCP_AUTH_TOKEN_FILE set in user environment." -ForegroundColor Green
    Write-Host ""
    Write-Host "Share the token file with workers:" -ForegroundColor Yellow
    Write-Host "  Copy-Item -Path $TokenFile -Destination \\WORKER\Users\$($env:USERNAME)\.vgre\token" `
        -ForegroundColor Yellow
    Write-Host ""
    Write-Host "New terminals will pick up the token automatically." -ForegroundColor DarkGray
}

function Cmd-Show {
    $tok = Read-StoredToken
    Write-Host ""
    Write-Host "Stored token:" -ForegroundColor White
    Write-Host "  $tok" -ForegroundColor Green
    Write-Host ""
    Write-Host "Location: $TokenFile" -ForegroundColor DarkGray
    Write-Host ""
}

function Cmd-Fingerprint {
    $tok = Read-StoredToken
    Print-FingerprintBlock -Token $tok
}

function Cmd-Set {
    param([string]$Token)
    if ([string]::IsNullOrWhiteSpace($Token)) {
        Write-Host "[ERROR] Usage: vgre-token set <TOKEN>" -ForegroundColor Red
        exit 1
    }
    $Token = $Token.Trim()
    if ($Token.Length -lt 8) {
        Write-Host "[ERROR] Token must be at least 8 characters." -ForegroundColor Red
        exit 1
    }
    Write-StoredToken -Token $Token
    Set-PersistentEnvVar -Name "VGRE_TCP_AUTH_TOKEN_FILE" -Value $TokenFile
    Add-ToWindowsPath
    Write-Host "[OK] Token saved to $TokenFile" -ForegroundColor Green
    Print-FingerprintBlock -Token $Token
    Write-Host "[OK] VGRE_TCP_AUTH_TOKEN_FILE set in user environment." -ForegroundColor Green
}

function Cmd-Verify {
    $tok    = Read-StoredToken
    $fileFP = Get-TokenSHA256 -Text $tok

    Write-Host ""
    Write-Host "File token fingerprint:" -ForegroundColor White
    Write-Host "  $fileFP" -ForegroundColor Cyan
    Write-Host "  ($TokenFile)" -ForegroundColor DarkGray
    Write-Host ""

    $envFilePath = [Environment]::GetEnvironmentVariable("VGRE_TCP_AUTH_TOKEN_FILE", "Process")
    $envTokRaw   = [Environment]::GetEnvironmentVariable("VGRE_TCP_AUTH_TOKEN",      "Process")

    if (![string]::IsNullOrWhiteSpace($envFilePath) -and (Test-Path $envFilePath)) {
        $envTok = [System.IO.File]::ReadAllText($envFilePath).Trim()
        $envFP  = Get-TokenSHA256 -Text $envTok
        if ($envFP -eq $fileFP) {
            Write-Host "[OK] VGRE_TCP_AUTH_TOKEN_FILE matches stored token." -ForegroundColor Green
        } else {
            Write-Host "[WARN] VGRE_TCP_AUTH_TOKEN_FILE points to a DIFFERENT token!" `
                -ForegroundColor Yellow
            Write-Host "  Env file fingerprint: $envFP" -ForegroundColor Yellow
        }
    } elseif (![string]::IsNullOrWhiteSpace($envTokRaw)) {
        $envFP = Get-TokenSHA256 -Text $envTokRaw.Trim()
        if ($envFP -eq $fileFP) {
            Write-Host "[OK] VGRE_TCP_AUTH_TOKEN env var matches stored token." -ForegroundColor Green
        } else {
            Write-Host "[WARN] VGRE_TCP_AUTH_TOKEN env var does NOT match stored token!" `
                -ForegroundColor Yellow
            Write-Host "  Env var fingerprint: $envFP" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[WARN] No VGRE_TCP_AUTH_TOKEN or VGRE_TCP_AUTH_TOKEN_FILE in process environment." `
            -ForegroundColor Yellow
        Write-Host "  Open a NEW terminal — the env var is set in user scope and needs a fresh shell." `
            -ForegroundColor DarkGray
    }
    Write-Host ""
}

function Cmd-Copy {
    if (-not (Test-Path $TokenFile)) {
        Write-Host "[ERROR] No token found. Run: vgre-token generate" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
    Write-Host "Copy token to a worker machine:" -ForegroundColor White
    Write-Host ""
    Write-Host "  PowerShell / Windows to Windows:" -ForegroundColor DarkGray
    Write-Host "    Copy-Item -Path `"$TokenFile`" -Destination `"\\WORKER_PC\Users\$($env:USERNAME)\.vgre\token`"" `
        -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  scp (if OpenSSH is installed):" -ForegroundColor DarkGray
    Write-Host "    scp `"$TokenFile`" user@WORKER_IP:~/.vgre/token" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Then on the worker run:  vgre-token fingerprint" -ForegroundColor DarkGray
    Write-Host "Both machines MUST show the same fingerprint." -ForegroundColor DarkGray
    Write-Host ""
}

function Cmd-Revoke {
    if (-not (Test-Path $TokenFile)) {
        Write-Host "[ERROR] No token found at $TokenFile" -ForegroundColor Red
        exit 1
    }
    $tok = Read-StoredToken
    $fp  = Get-TokenSHA256 -Text $tok
    Write-Host ""
    Write-Host "[WARN] About to DELETE the stored token." -ForegroundColor Yellow
    Write-Host "  Fingerprint: $($fp.Substring(0,16))..." -ForegroundColor Yellow
    Write-Host "  Location:    $TokenFile" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "This will disconnect ALL workers using this token." -ForegroundColor Red
    $ans = Read-Host "Continue? [y/N]"
    if ($ans -eq "y" -or $ans -eq "Y") {
        Remove-Item -Path $TokenFile -Force
        [Environment]::SetEnvironmentVariable("VGRE_TCP_AUTH_TOKEN_FILE", $null, "User")
        Write-Host "[OK] Token revoked." -ForegroundColor Green
    } else {
        Write-Host "[INFO] Aborted. Token kept." -ForegroundColor Cyan
    }
}

function Cmd-Help {
    Write-Host @"

vgre-token — VGRE Auth Token Manager

Usage:
  vgre-token generate              Generate a new secure 64-hex token
  vgre-token show                  Print the stored token value
  vgre-token fingerprint           Print the SHA-256 fingerprint
  vgre-token set <TOKEN>           Store a specific token (paste from another node)
  vgre-token verify                Check env-var and file token match
  vgre-token copy                  Print the copy command to share the token
  vgre-token revoke                Delete the stored token

Token file:  $TokenFile

Quick start:
  # On master — generate once:
  vgre-token generate

  # On every worker — paste the token from the master:
  vgre-token set <paste-token-here>
  # OR copy the file directly (see: vgre-token copy)

  # Verify both match:
  vgre-token fingerprint   # must be identical on master and worker
"@
}

# ── Entry point ───────────────────────────────────────────────────────────────
switch ($Command.ToLower()) {
    "generate"    { Cmd-Generate }
    "show"        { Cmd-Show }
    "fingerprint" { Cmd-Fingerprint }
    "set"         { Cmd-Set -Token $TokenValue }
    "verify"      { Cmd-Verify }
    "copy"        { Cmd-Copy }
    "revoke"      { Cmd-Revoke }
    { $_ -in "help","--help","-h","" } { Cmd-Help }
    default {
        Write-Host "[ERROR] Unknown command: $Command" -ForegroundColor Red
        Write-Host ""
        Cmd-Help
        exit 1
    }
}
