$installDir = Join-Path $env:LOCALAPPDATA 'VGRE'
# The build itself (source tree + build/ output) is the directory Avast's
# real-time scanner actually contends with during a parallel ninja build —
# hundreds of .obj files created per second, each briefly opened for a scan
# right as cl.exe/ninja also wants it. That race is what causes intermittent
# "Cannot open include file" C1083 errors that vary by file/run and disappear
# on retry: not a code bug, a scanner-vs-compiler file lock race. Excluding
# only %LOCALAPPDATA%\VGRE (the installed app) doesn't cover this at all.
$sourceDir = 'C:\Users\Dell\Documents\vgre'
$exclusionTargets = @($installDir, $sourceDir)

# Method 1: Avast registry exclusions (file path exclusions)
$avastPaths = @(
    'HKLM:\SOFTWARE\AVAST Software\Avast\exclusions',
    'HKCU:\SOFTWARE\AVAST Software\Avast\exclusions'
)
$added = $false
foreach ($key in $avastPaths) {
    if (Test-Path $key) {
        $props = Get-ItemProperty $key -ErrorAction SilentlyContinue
        foreach ($target in $exclusionTargets) {
            $existing = ($props.PSObject.Properties | Where-Object { $_.Value -eq $target }).Count
            if ($existing -eq 0) {
                $idx = 0
                while (Get-ItemProperty $key -Name "$idx" -ErrorAction SilentlyContinue) { $idx++ }
                Set-ItemProperty -Path $key -Name "$idx" -Value $target -Type String
                Write-Host "[OK] Added $target to Avast exclusions at $key"
            } else {
                Write-Host "[OK] $target already in Avast exclusions at $key"
            }
            $added = $true
        }
        $props = $null
    }
}

# Method 2: Avast uses a different exclusion format - try the scan exclusions
$avastScanKey = 'HKLM:\SOFTWARE\AVAST Software\Avast\Scan\Exclusions'
if (-not $added -and (Test-Path 'HKLM:\SOFTWARE\AVAST Software\Avast')) {
    Write-Host "Avast installed. Trying to add exclusion via New-Item..."
    try {
        if (-not (Test-Path $avastScanKey)) {
            New-Item -Path $avastScanKey -Force | Out-Null
        }
        foreach ($target in $exclusionTargets) {
            $props = Get-ItemProperty $avastScanKey -ErrorAction SilentlyContinue
            $idx = 0
            while ($props -and (Get-Member -InputObject $props -Name "$idx" -ErrorAction SilentlyContinue)) { $idx++ }
            Set-ItemProperty -Path $avastScanKey -Name "$idx" -Value $target -Type String
            Write-Host "[OK] Added $target to Avast scan exclusions"
        }
        $added = $true
    } catch {
        Write-Host "[WARN] Could not add via registry: $_"
    }
}

if (-not $added) {
    Write-Host "[INFO] Could not auto-add Avast exclusion."
    Write-Host "       Please add manually in Avast > Menu > Settings > General > Exceptions:"
    foreach ($target in $exclusionTargets) { Write-Host "       $target" }
}

# Restore quarantined vgre_dashboard.exe from the build output
$bundleExe = 'C:\Users\Dell\Documents\vgre\vgre_dashboard\build\windows\x64\runner\Release\vgre_dashboard.exe'
$installExe = Join-Path $installDir 'vgre_dashboard.exe'
if ((Test-Path $bundleExe) -and (-not (Test-Path $installExe))) {
    Write-Host "Restoring vgre_dashboard.exe from build output..."
    Copy-Item $bundleExe $installExe -Force
    if (Test-Path $installExe) {
        Write-Host "[OK] vgre_dashboard.exe restored to $installDir"
    }
}
