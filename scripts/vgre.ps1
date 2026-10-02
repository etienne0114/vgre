<#
    vgre.ps1 — the unified VGRE command on Windows (PowerShell).

    Mirrors scripts/vgre.sh: dispatches model/tokenizer subcommands to the
    Python package CLI (python -m vgre) and cluster subcommands to the sibling
    vgre-<cmd> scripts. Installed into %LOCALAPPDATA%\VGRE\scripts by
    vgre_sync.bat, with vgre.bat as the cmd.exe launcher.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)] [string] $Command = "help",
    [Parameter(ValueFromRemainingArguments = $true)] [string[]] $Rest
)

$ErrorActionPreference = "Stop"
$VgreVersion = "0.1.4"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# The managed venv that the installers can populate with the vgre package.
$VgreVenvPy = Join-Path $env:USERPROFILE ".vgre\venv\Scripts\python.exe"

function Find-Python {
    # Prefer a Python that can actually import vgre: an override, then the
    # managed venv, then any interpreter on PATH.
    if ($env:VGRE_PYTHON -and (Test-Path $env:VGRE_PYTHON)) {
        & $env:VGRE_PYTHON -c "import vgre" 2>$null
        if ($LASTEXITCODE -eq 0) { return $env:VGRE_PYTHON }
    }
    if (Test-Path $VgreVenvPy) {
        & $VgreVenvPy -c "import vgre" 2>$null
        if ($LASTEXITCODE -eq 0) { return $VgreVenvPy }
    }
    foreach ($p in @("python", "py", "python3")) {
        $cmd = Get-Command $p -ErrorAction SilentlyContinue
        if ($cmd) { return $p }
    }
    if (Test-Path $VgreVenvPy) { return $VgreVenvPy }
    return $null
}

function Show-SetupHelp([string] $sub) {
    Write-Error @"
vgre: the Python model package isn't set up, so '$sub' can't run yet.

Set it up once (a venv keeps it isolated):
    python -m venv %USERPROFILE%\.vgre\venv
    %USERPROFILE%\.vgre\venv\Scripts\pip install vgre
    # …or from a source checkout:  ...\Scripts\pip install <vgre-source>\bindings\python

Then re-run:  vgre $sub ...
"@
}

function Show-Usage {
    @"
vgre - run CUDA and a local language model on the CPU (no GPU required).

Usage: vgre <command> [options]

Model & runtime (Python package - set up by the VGRE installer):
  info                 show version, native backend, library path, platform
  generate             generate text (trains a tiny demo model if no --model)
  train                train a small language model on a text corpus
  tokenize             byte / BPE tokenization helpers

Cluster & node (source install):
  start                start a master or worker node
  worker               run the worker binary directly
  token                manage the shared cluster auth token
  discover             find / publish the master's public IP
  connect-check        verify connectivity to a master before starting a worker
  dashboard            launch the Flutter monitoring dashboard

Other:
  version, --version   print the VGRE version and native backend status
  help,    --help      show this help

Run 'vgre <command> --help' for a command's own options.
"@ | Write-Output
}

function Show-Version {
    Write-Output "vgre $VgreVersion"
    $py = Find-Python
    if ($py) {
        & $py -c "import vgre" 2>$null
        if ($LASTEXITCODE -eq 0) {
            & $py -m vgre version 2>$null | Select-String -Pattern "native" | ForEach-Object { $_.Line }
            return
        }
    }
    Write-Output "native backend: model package not set up (run the VGRE installer, or see 'vgre generate')"
}

function Invoke-PythonCli([string] $sub, [string[]] $rest) {
    $py = Find-Python
    if (-not $py) { Write-Error "vgre: python is required for '$sub'."; exit 127 }
    & $py -c "import vgre" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Show-SetupHelp $sub
        exit 127
    }
    & $py -m vgre $sub @rest
    exit $LASTEXITCODE
}

function Invoke-Sibling([string] $name, [string[]] $rest) {
    $candidates = @(
        (Join-Path $ScriptDir "$name.ps1"),
        (Join-Path $ScriptDir "$name.bat"),
        (Join-Path $env:LOCALAPPDATA "VGRE\scripts\$name.ps1"),
        (Join-Path $env:LOCALAPPDATA "VGRE\scripts\$name.bat")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            if ($c.EndsWith(".ps1")) { & $c @rest } else { & $c @rest }
            exit $LASTEXITCODE
        }
    }
    Write-Error "vgre: '$name' is not available in this install (cluster/dashboard tools come from the source build)."
    exit 127
}

switch -Regex ($Command) {
    '^(-h|--help|help)$'       { Show-Usage; break }
    '^(-V|--version|version)$' { Show-Version; break }
    '^(info|generate|train|tokenize)$' { Invoke-PythonCli $Command $Rest; break }
    '^start$'     { Invoke-Sibling "vgre-start" $Rest; break }
    '^worker$'    { Invoke-Sibling "vgre-worker" $Rest; break }
    '^token$'     { Invoke-Sibling "vgre-token" $Rest; break }
    '^discover$'  { Invoke-Sibling "vgre-discover" $Rest; break }
    '^connect-check$' { Invoke-Sibling "vgre-connect-check" $Rest; break }
    '^dashboard$' { Invoke-Sibling "vgre-dashboard" $Rest; break }
    default {
        Write-Error "vgre: unknown command '$Command'"
        Show-Usage
        exit 2
    }
}
