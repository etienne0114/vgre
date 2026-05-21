# vgre-discover.ps1 -- Detect real public IP and enable cross-LAN token-keyed discovery
#
# Usage:
#   vgre-discover                  Show public IP + worker connection command
#   vgre-discover --set-master     Set VGRE_CLUSTER_ADVERTISED_ADDRESS (for this session)
#   vgre-discover --register       Push master's IP to token-keyed KV store
#   vgre-discover --find [BUCKET]  Find master via token (worker side, prints connect cmd)
#   vgre-discover --unregister     Remove this master from the KV store
#
# Cross-LAN auto-discovery uses kvdb.io (free, no account after first-time setup).
# Each cluster gets a private bucket; only nodes with the same auth token can discover each other.

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Command = "",

    [Parameter(Position = 1)]
    [string]$Arg1 = ""
)

$ErrorActionPreference = "Stop"

$VgreDir        = Join-Path $env:USERPROFILE ".vgre"
$TokenFile      = Join-Path $VgreDir "token"
$PublicIpFile   = Join-Path $VgreDir "public_ip"
$BucketIdFile   = Join-Path $VgreDir "discovery_bucket"
$KvBase         = "https://kvdb.io"
$Port           = if ($env:VGRE_PORT) { $env:VGRE_PORT } else { "7777" }

if (-not (Test-Path $VgreDir)) {
    New-Item -ItemType Directory -Path $VgreDir -Force | Out-Null
}

# ── Helpers ───────────────────────────────────────────────────────────────────

function Get-TokenFingerprint {
    if (-not (Test-Path $TokenFile)) { return "" }
    $text = [System.IO.File]::ReadAllText($TokenFile).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) { return "" }
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($text)
    $sha   = [System.Security.Cryptography.SHA256]::Create()
    $hash  = $sha.ComputeHash($bytes)
    return ($hash | ForEach-Object { $_.ToString("x2") }) -join ""
}

function Get-PublicIp {
    $apis = @(
        "https://api.ipify.org",
        "https://ipecho.net/plain",
        "https://icanhazip.com",
        "https://checkip.amazonaws.com",
        "https://api4.my-ip.io/ip"
    )
    foreach ($api in $apis) {
        try {
            $ip = (Invoke-RestMethod -Uri $api -TimeoutSec 5 -ErrorAction Stop).Trim()
            if ($ip -match '^\d+\.\d+\.\d+\.\d+$') {
                return $ip
            }
        } catch {}
    }
    return $null
}

function Get-BucketId {
    if ($env:VGRE_DISCOVERY_BUCKET_ID) { return $env:VGRE_DISCOVERY_BUCKET_ID }
    if (Test-Path $BucketIdFile) { return [System.IO.File]::ReadAllText($BucketIdFile).Trim() }

    Write-Host "[INFO] No discovery bucket configured -- creating one at kvdb.io (free)..." -ForegroundColor Cyan
    try {
        $resp = Invoke-RestMethod -Method Post -Uri "${KvBase}/" `
            -Body "email=vgre-$(${env:COMPUTERNAME}.ToLower())@cluster.local" `
            -ContentType "application/x-www-form-urlencoded" `
            -TimeoutSec 10 -ErrorAction Stop
        $bid = $resp.bucket_id
        if ($bid) {
            [System.IO.File]::WriteAllText($BucketIdFile, $bid)
            return $bid
        }
    } catch {}

    Write-Host ""
    Write-Host "[ERROR] Could not create discovery bucket automatically." -ForegroundColor Red
    Write-Host "        1. Visit https://kvdb.io and create a free bucket." -ForegroundColor Red
    Write-Host "        2. set-vgre-env VGRE_DISCOVERY_BUCKET_ID '<your-bucket-id>'" -ForegroundColor Red
    Write-Host "        3. Re-run: vgre-discover --register" -ForegroundColor Red
    return $null
}

function Get-DiscoveryKey {
    $fp = Get-TokenFingerprint
    if ([string]::IsNullOrWhiteSpace($fp)) {
        Write-Host "[ERROR] No auth token found at $TokenFile" -ForegroundColor Red
        Write-Host "        Run: vgre-token generate" -ForegroundColor Red
        return $null
    }
    return "vgre-$($fp.Substring(0, 32))"
}

# ── Commands ──────────────────────────────────────────────────────────────────

function Cmd-Show {
    Write-Host ""
    Write-Host "=== VGRE Public IP Discovery ===" -ForegroundColor White
    Write-Host ""
    Write-Host "[...] Detecting public IP..." -ForegroundColor Cyan
    $ip = Get-PublicIp
    if (-not $ip) {
        Write-Host "[ERROR] Could not detect public IP. Check internet connectivity." -ForegroundColor Red
        exit 1
    }
    [System.IO.File]::WriteAllText($PublicIpFile, $ip)

    $fp = Get-TokenFingerprint

    Write-Host "Public IP  : $ip" -ForegroundColor Green
    Write-Host "TCP port   : $Port  (firewall must allow inbound TCP $Port)" -ForegroundColor White
    if ($fp) {
        Write-Host "Token FP   : $($fp.Substring(0,16))..." -ForegroundColor Cyan
    }
    Write-Host ""
    Write-Host "Share with workers on other networks:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  vgre-start --worker --master-address ${ip}:${Port}" -ForegroundColor Cyan
    Write-Host ""
    if ($fp) {
        Write-Host "For auto-discovery (workers use same token):" -ForegroundColor Yellow
        Write-Host "  vgre-discover --register           (run on master once)" -ForegroundColor Cyan
        Write-Host "  vgre-discover --find <BUCKET_ID>   (run on each worker)" -ForegroundColor Cyan
    }
    Write-Host ""
    Write-Host "NAT / firewall note:" -ForegroundColor DarkGray
    Write-Host "  Your router must port-forward TCP $Port to this machine." -ForegroundColor DarkGray
    Write-Host "  Or use a VPN (Tailscale / ZeroTier) -- no port-forward needed." -ForegroundColor DarkGray
    Write-Host ""
}

function Cmd-SetMaster {
    $ip = Get-PublicIp
    if (-not $ip) {
        Write-Host "[WARN] Could not detect public IP -- VGRE_CLUSTER_ADVERTISED_ADDRESS not set" -ForegroundColor Yellow
        return
    }
    [System.IO.File]::WriteAllText($PublicIpFile, $ip)
    $env:VGRE_CLUSTER_ADVERTISED_ADDRESS = "${ip}:${Port}"
    Write-Host "[OK] VGRE_CLUSTER_ADVERTISED_ADDRESS=${ip}:${Port}" -ForegroundColor Green
    Write-Host "     Master will broadcast this address so workers across NAT connect correctly." -ForegroundColor DarkGray
}

function Cmd-Register {
    Write-Host "[...] Detecting public IP..." -ForegroundColor Cyan
    $ip = Get-PublicIp
    if (-not $ip) {
        Write-Host "[ERROR] Could not detect public IP." -ForegroundColor Red
        exit 1
    }

    $key = Get-DiscoveryKey
    if (-not $key) { exit 1 }

    $bucket = Get-BucketId
    if (-not $bucket) { exit 1 }

    $fp      = Get-TokenFingerprint
    $fpShort = if ($fp) { $fp.Substring(0, 16) } else { "unknown" }
    $ts      = [System.DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $payload = "${ip}:${Port}|fp=${fpShort}|ts=${ts}"

    Write-Host "[...] Registering ${ip}:${Port} at kvdb.io bucket $bucket..." -ForegroundColor Cyan
    try {
        $resp = Invoke-WebRequest -Uri "${KvBase}/${bucket}/${key}" `
            -Method Post -Body $payload -ContentType "text/plain" `
            -TimeoutSec 10 -ErrorAction Stop -UseBasicParsing
        if ($resp.StatusCode -notin @(200, 201)) { throw "HTTP $($resp.StatusCode)" }
    } catch {
        Write-Host "[ERROR] Registration failed: $_" -ForegroundColor Red
        Write-Host "        Bucket: $bucket | Key: $key" -ForegroundColor Red
        exit 1
    }

    [System.IO.File]::WriteAllText($PublicIpFile, $ip)

    Write-Host ""
    Write-Host "[OK] Master registered successfully." -ForegroundColor Green
    Write-Host ""
    Write-Host "Bucket ID : $bucket" -ForegroundColor White
    Write-Host "Master IP : ${ip}:${Port}" -ForegroundColor Green
    Write-Host "Token FP  : ${fpShort}..." -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Share the Bucket ID with workers:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  vgre-discover --find $bucket" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Workers with the same token will connect to ${ip}:${Port} automatically." -ForegroundColor DarkGray
    Write-Host "Re-run --register if your public IP changes." -ForegroundColor DarkGray
    Write-Host ""
}

function Cmd-Find {
    param([string]$BucketArg = "")

    $bucket = $BucketArg
    if (-not $bucket) {
        if ($env:VGRE_DISCOVERY_BUCKET_ID) { $bucket = $env:VGRE_DISCOVERY_BUCKET_ID }
        elseif (Test-Path $BucketIdFile) { $bucket = [System.IO.File]::ReadAllText($BucketIdFile).Trim() }
    }
    if (-not $bucket) {
        Write-Host "[ERROR] Usage: vgre-discover --find <BUCKET_ID>" -ForegroundColor Red
        Write-Host "        Get the BUCKET_ID from the master: vgre-discover --register" -ForegroundColor Red
        exit 1
    }

    $key = Get-DiscoveryKey
    if (-not $key) { exit 1 }

    Write-Host "[...] Looking up master in bucket $bucket..." -ForegroundColor Cyan
    $payload = $null
    try {
        $payload = (Invoke-RestMethod -Uri "${KvBase}/${bucket}/${key}" `
            -TimeoutSec 10 -ErrorAction Stop).Trim()
    } catch {
        Write-Host "[ERROR] No master registered in bucket $bucket." -ForegroundColor Red
        Write-Host "        Ask the master to run: vgre-discover --register" -ForegroundColor Red
        exit 1
    }

    if (-not $payload) {
        Write-Host "[ERROR] Empty response from discovery bucket." -ForegroundColor Red
        exit 1
    }

    # payload: "IP:PORT|fp=FINGERPRINT|ts=TIMESTAMP"
    $addr       = $payload.Split('|')[0]
    $storedFp   = ($payload | Select-String 'fp=([^|]*)').Matches.Groups[1].Value
    $localFp    = Get-TokenFingerprint

    if ($storedFp -and $localFp -and $localFp.Substring(0,16) -ne $storedFp) {
        Write-Host "[ERROR] Token fingerprint mismatch!" -ForegroundColor Red
        Write-Host "  Registered: $storedFp" -ForegroundColor Red
        Write-Host "  Local:      $($localFp.Substring(0,16))" -ForegroundColor Red
        Write-Host "  Make sure you and the master have the same token." -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "[OK] Master found: $addr" -ForegroundColor Green
    Write-Host ""
    Write-Host "Connect with:" -ForegroundColor Yellow
    Write-Host "  vgre-start --worker --master-address $addr" -ForegroundColor Cyan
    Write-Host ""

    $ip = $addr.Split(':')[0]
    [System.IO.File]::WriteAllText($PublicIpFile, $ip)
    $env:VGRE_CLUSTER_MASTER_ADDRESS = $addr
    Write-Host "VGRE_CLUSTER_MASTER_ADDRESS=$addr" -ForegroundColor DarkGray
}

function Cmd-Unregister {
    $key = Get-DiscoveryKey
    if (-not $key) { exit 1 }
    $bucket = Get-BucketId
    if (-not $bucket) { exit 1 }
    try {
        Invoke-RestMethod -Uri "${KvBase}/${bucket}/${key}" `
            -Method Delete -TimeoutSec 10 -ErrorAction SilentlyContinue | Out-Null
    } catch {}
    Write-Host "[OK] Discovery registration removed." -ForegroundColor Green
}

# ── Entry point ───────────────────────────────────────────────────────────────
switch ($Command.ToLower()) {
    { $_ -in "", "--show", "show" }          { Cmd-Show }
    { $_ -in "--set-master", "set-master" }  { Cmd-SetMaster }
    { $_ -in "--register", "register" }      { Cmd-Register }
    { $_ -in "--find", "find" }              { Cmd-Find -BucketArg $Arg1 }
    { $_ -in "--unregister", "unregister" }  { Cmd-Unregister }
    { $_ -in "--help", "-h", "help" }        {
        Write-Host ""
        Write-Host "vgre-discover -- VGRE Public IP and Cross-LAN Discovery" -ForegroundColor White
        Write-Host ""
        Write-Host "Usage:" -ForegroundColor Yellow
        Write-Host "  vgre-discover                  Show public IP + connection command"
        Write-Host "  vgre-discover --set-master     Set advertised address for this session"
        Write-Host "  vgre-discover --register       Register master's IP (cross-LAN discovery)"
        Write-Host "  vgre-discover --find [BUCKET]  Find master via token"
        Write-Host "  vgre-discover --unregister     Remove registration"
        Write-Host ""
    }
    default {
        Write-Host "[ERROR] Unknown command: $Command  (run vgre-discover --help)" -ForegroundColor Red
        exit 1
    }
}
