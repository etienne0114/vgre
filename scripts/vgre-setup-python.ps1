<#
    vgre-setup-python.ps1 — install the vgre Python model CLI into a venv.

    The model subcommands (vgre generate / train / tokenize / info) run through
    the vgre Python package (a numpy-based LM). This installs it into a
    dedicated virtualenv at %USERPROFILE%\.vgre\venv — which the vgre.ps1 /
    vgre.bat dispatcher then finds automatically — so the model CLI works
    without touching the system Python.

    Called by vgre_sync.bat and Install-VGRETools.ps1 during a source install.
    Best-effort: prints guidance and exits 0 (never fails the install) unless
    -Strict is given.

    Usage:
      vgre-setup-python.ps1 [-PackageDir <path-to\bindings\python>] [-Strict]
#>
[CmdletBinding()]
param(
    [string] $PackageDir = "",
    [switch] $Strict
)

$ErrorActionPreference = "Continue"

# Default the package dir to <repo>\bindings\python relative to this script.
if (-not $PackageDir) {
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $PackageDir = Join-Path (Split-Path -Parent $ScriptDir) "bindings\python"
}

function Fail([string] $msg) {
    Write-Host "  [WARN] $msg" -ForegroundColor Yellow
    if ($Strict) { exit 1 } else { exit 0 }
}

if (-not (Test-Path $PackageDir)) {
    Fail "Python package not found at $PackageDir — skipping model CLI setup."
}

# Find a usable Python: prefer the launcher's 3.x, then python/python3.
$PyExe = $null
$PyArgs = @()
$launcher = Get-Command "py" -ErrorAction SilentlyContinue
if ($launcher) {
    & py -3 -c "import sys" 2>$null
    if ($LASTEXITCODE -eq 0) { $PyExe = "py"; $PyArgs = @("-3") }
}
if (-not $PyExe) {
    foreach ($p in @("python", "python3")) {
        if (Get-Command $p -ErrorAction SilentlyContinue) { $PyExe = $p; break }
    }
}
if (-not $PyExe) {
    Fail "python not found on PATH — install Python 3, then re-run. Model subcommands (vgre generate/train) will be unavailable."
}

$VenvDir = Join-Path $env:USERPROFILE ".vgre\venv"
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    Write-Host "  [..] Creating Python venv at $VenvDir" -ForegroundColor Cyan
    & $PyExe @PyArgs -m venv "$VenvDir" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPy)) {
        Fail "Could not create a venv. Model subcommands will be unavailable until you run: $PyExe -m venv $VenvDir"
    }
}

Write-Host "  [..] Installing the vgre package into the venv" -ForegroundColor Cyan
& $VenvPy -m pip install --quiet --upgrade pip 2>$null | Out-Null
& $VenvPy -m pip install --quiet "$PackageDir" 2>$null
if ($LASTEXITCODE -ne 0) {
    Fail "pip install failed (offline?). Retry later:  $VenvPy -m pip install `"$PackageDir`""
}

& $VenvPy -c "import vgre" 2>$null
if ($LASTEXITCODE -ne 0) {
    Fail "Installed, but 'import vgre' failed — model subcommands may not work."
}

Write-Host "  [OK] vgre model CLI ready — try:  vgre generate --prompt `"the `"" -ForegroundColor Green
exit 0
