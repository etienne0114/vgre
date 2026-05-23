$installDir = Join-Path $env:LOCALAPPDATA 'VGRE'

# Method 1: Avast registry exclusions (file path exclusions)
$avastPaths = @(
    'HKLM:\SOFTWARE\AVAST Software\Avast\exclusions',
    'HKCU:\SOFTWARE\AVAST Software\Avast\exclusions'
)
$added = $false
foreach ($key in $avastPaths) {
    if (Test-Path $key) {
        $props = Get-ItemProperty $key -ErrorAction SilentlyContinue
        $existing = ($props.PSObject.Properties | Where-Object { $_.Value -eq $installDir }).Count
        if ($existing -eq 0) {
            $idx = ($props.PSObject.Properties | Where-Object { $_.Name -match '^\d+$' } | Measure-Object).Count
            Set-ItemProperty -Path $key -Name "$idx" -Value $installDir -Type String
            Write-Host "[OK] Added $installDir to Avast exclusions at $key"
            $added = $true
        } else {
            Write-Host "[OK] $installDir already in Avast exclusions at $key"
            $added = $true
        }
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
        $props = Get-ItemProperty $avastScanKey -ErrorAction SilentlyContinue
        $idx = if ($props) { ($props.PSObject.Properties | Where-Object { $_.Name -match '^\d+$' } | Measure-Object).Count } else { 0 }
        Set-ItemProperty -Path $avastScanKey -Name "$idx" -Value $installDir -Type String
        Write-Host "[OK] Added $installDir to Avast scan exclusions"
        $added = $true
    } catch {
        Write-Host "[WARN] Could not add via registry: $_"
    }
}

if (-not $added) {
    Write-Host "[INFO] Could not auto-add Avast exclusion."
    Write-Host "       Please add manually in Avast > Menu > Settings > General > Exceptions:"
    Write-Host "       $installDir"
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
