# VGRE Environment Setup Script for PowerShell
# Usage: . ./scripts/vgre_env.ps1

Write-Host "Configuring VGRE environment..." -ForegroundColor Cyan

# Define a helper function to mimic 'export' behavior for users coming from bash
function Global:set-vgre-env($Name, $Value) {
    if ($null -ne $Name -and $Name -ne "") {
        Set-Item -Path "Env:$Name" -Value $Value
        Write-Host "Set $Name to '$Value'" -ForegroundColor Gray
    }
}

# Set Authentication Token only if explicitly provided by user.
# No hardcoded default secret is applied.
if ($null -eq $env:VGRE_TCP_AUTH_TOKEN -or $env:VGRE_TCP_AUTH_TOKEN -eq "") {
    Write-Host "VGRE_TCP_AUTH_TOKEN is not set (recommended for secure cluster mode)." -ForegroundColor Yellow
}

# Set Cluster Nodes (Default: localhost:7777)
if ($null -eq $env:VGRE_CLUSTER_NODES) {
    set-vgre-env "VGRE_CLUSTER_NODES" "127.0.0.1:7777"
}

# Add VGRE bin directory to PATH for the current session
$vgrePath = "$HOME\AppData\Local\VGRE"
if ($env:PATH -notlike "*$vgrePath*") {
    $env:PATH = "$vgrePath;$env:PATH"
    Write-Host "Added $vgrePath to PATH"
}

Write-Host "`nVGRE environment configured successfully." -ForegroundColor Green
Write-Host "You can now run 'vgre-worker' or 'vgre-dashboard' directly."
Write-Host "To set cluster variables, use: set-vgre-env VGRE_CLUSTER_NODES '192.168.1.50:7777'`n"
