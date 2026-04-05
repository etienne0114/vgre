$ErrorActionPreference = "Stop"

Write-Host "=== VGRE Build Tools Installation ===" -ForegroundColor Cyan
Write-Host "This script installs build tools without requiring Chocolatey." -ForegroundColor Yellow

$installDir = Join-Path $env:LOCALAPPDATA "VGRE\BuildTools"
if (-not (Test-Path $installDir)) {
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
    Write-Host "Created directory: $installDir" -ForegroundColor Green
}

function Get-CommandVersionLine {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

        [string[]]$Args = @("--version")
    )

    try {
        if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
            return $null
        }

        return (& $Command @Args 2>&1 | Select-Object -First 1)
    } catch {
        return $null
    }
}

function Remove-ExistingPathEntry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathToRemove
    )

    $userPath = [System.Environment]::GetEnvironmentVariable("PATH", [System.EnvironmentVariableTarget]::User)
    if ([string]::IsNullOrWhiteSpace($userPath)) {
        return
    }

    $parts = $userPath -split ";" | Where-Object { $_ -and $_.TrimEnd("\") -ne $PathToRemove.TrimEnd("\") }
    [System.Environment]::SetEnvironmentVariable("PATH", ($parts -join ";"), [System.EnvironmentVariableTarget]::User)
}

function Ensure-UserPathEntry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathToAdd
    )

    if (-not (Test-Path $PathToAdd)) {
        return
    }

    $currentPath = [System.Environment]::GetEnvironmentVariable("PATH", [System.EnvironmentVariableTarget]::User)
    if ([string]::IsNullOrWhiteSpace($currentPath)) {
        $currentPath = ""
    }

    $normalizedExisting = $currentPath -split ";" | ForEach-Object { $_.TrimEnd("\") }
    if ($normalizedExisting -contains $PathToAdd.TrimEnd("\")) {
        return
    }

    if ([string]::IsNullOrWhiteSpace($currentPath)) {
        $newPath = $PathToAdd
    } else {
        $newPath = $currentPath + ";" + $PathToAdd
    }

    [System.Environment]::SetEnvironmentVariable("PATH", $newPath, [System.EnvironmentVariableTarget]::User)
    Write-Host "Added to PATH: $PathToAdd" -ForegroundColor Green
}

function Download-FileVerified {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,

        [Parameter(Mandatory = $true)]
        [string]$OutputPath,

        [string]$Sha256
    )

    $needsDownload = $true
    $downloadTarget = $OutputPath
    if (Test-Path $OutputPath) {
        if ($Sha256) {
            try {
                $existingHash = (Get-FileHash -Path $OutputPath -Algorithm SHA256).Hash
                if ($existingHash -eq $Sha256) {
                    Write-Host "Verified existing download: $OutputPath" -ForegroundColor Green
                    $needsDownload = $false
                } else {
                    Write-Host "Existing file hash mismatch, removing corrupted download: $OutputPath" -ForegroundColor Yellow
                    Remove-Item -Path $OutputPath -Force
                }
            } catch {
                $downloadTarget = "$OutputPath.download"
                if (Test-Path $downloadTarget) {
                    Remove-Item -Path $downloadTarget -Force -ErrorAction SilentlyContinue
                }
                Write-Host "Existing archive is locked by another process. Downloading to a fresh file instead: $downloadTarget" -ForegroundColor Yellow
            }
        } else {
            Write-Host "File already exists: $OutputPath" -ForegroundColor Yellow
            $needsDownload = $false
        }
    }

    if ($needsDownload) {
        Write-Host "Downloading: $Url" -ForegroundColor Cyan
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $Url -OutFile $downloadTarget -UseBasicParsing
    }

    if ($Sha256) {
        $downloadedHash = (Get-FileHash -Path $downloadTarget -Algorithm SHA256).Hash
        if ($downloadedHash -ne $Sha256) {
            throw "Checksum verification failed for $downloadTarget. Expected $Sha256 but got $downloadedHash."
        }
        Write-Host "Checksum verified: $downloadTarget" -ForegroundColor Green
    }

    if ($downloadTarget -ne $OutputPath) {
        if (Test-Path $OutputPath) {
            Write-Host "Keeping locked partial archive in place and using verified download file: $downloadTarget" -ForegroundColor Yellow
        }
        return $downloadTarget
    }

    return $OutputPath
}

Write-Host "`n=== Installing CMake ===" -ForegroundColor Cyan

$cmakeVersion = "3.28.3"
$cmakeFolder = "cmake-$cmakeVersion-windows-x86_64"
$cmakePath = Join-Path $installDir "cmake"
$cmakeZip = Join-Path $installDir "cmake.zip"
$cmakeUrl = "https://github.com/Kitware/CMake/releases/download/v$cmakeVersion/$cmakeFolder.zip"

if (-not (Test-Path (Join-Path $cmakePath "bin\cmake.exe"))) {
    $cmakeDownloadedPath = Download-FileVerified -Url $cmakeUrl -OutputPath $cmakeZip

    Write-Host "Extracting CMake..." -ForegroundColor Cyan
    Expand-Archive -Path $cmakeDownloadedPath -DestinationPath $installDir -Force

    $expandedPath = Join-Path $installDir $cmakeFolder
    if (Test-Path $cmakePath) {
        Remove-Item -Path $cmakePath -Recurse -Force
    }

    Rename-Item -Path $expandedPath -NewName "cmake" -Force
    Remove-Item -Path $cmakeDownloadedPath -Force
    Write-Host "CMake extracted to: $cmakePath" -ForegroundColor Green
} else {
    Write-Host "CMake already exists at: $cmakePath" -ForegroundColor Yellow
}

Write-Host "`n=== Installing LLVM ===" -ForegroundColor Cyan

$llvmVersion = "18.1.0"
$llvmArchiveName = "clang+llvm-$llvmVersion-x86_64-pc-windows-msvc.tar.xz"
$llvmArchivePath = Join-Path $installDir $llvmArchiveName
$llvmUrl = "https://github.com/llvm/llvm-project/releases/download/llvmorg-$llvmVersion/$llvmArchiveName"
$llvmSha256 = "D128C0F5F7831C77D549296A910FC9972407FF028B720FB628FFA837ED7FF04E"
$llvmExtractRoot = Join-Path $installDir "llvm-extract"
$llvmExpandedPath = Join-Path $llvmExtractRoot "clang+llvm-$llvmVersion-x86_64-pc-windows-msvc"
$llvmPath = Join-Path $installDir "llvm"
$legacyInstallerPath = Join-Path $installDir "llvm-installer.exe"

if (Test-Path $legacyInstallerPath) {
    Write-Host "Removing cached NSIS installer because it is unreliable for this setup." -ForegroundColor Yellow
    try {
        Remove-Item -Path $legacyInstallerPath -Force
    } catch {
        Write-Host "Could not remove cached installer right now. Continuing with archive install." -ForegroundColor Yellow
    }
}

if (-not (Test-Path (Join-Path $llvmPath "bin\clang.exe"))) {
    $llvmDownloadedPath = Download-FileVerified -Url $llvmUrl -OutputPath $llvmArchivePath -Sha256 $llvmSha256

    if (Test-Path $llvmExtractRoot) {
        Remove-Item -Path $llvmExtractRoot -Recurse -Force
    }

    New-Item -ItemType Directory -Path $llvmExtractRoot -Force | Out-Null
    Write-Host "Extracting LLVM..." -ForegroundColor Cyan
    tar -xf $llvmDownloadedPath -C $llvmExtractRoot

    if (-not (Test-Path $llvmExpandedPath)) {
        throw "LLVM archive extracted, but expected folder was not found: $llvmExpandedPath"
    }

    if (Test-Path $llvmPath) {
        Remove-Item -Path $llvmPath -Recurse -Force
    }

    Move-Item -Path $llvmExpandedPath -Destination $llvmPath
    Remove-Item -Path $llvmExtractRoot -Recurse -Force
    if ($llvmDownloadedPath -ne $llvmArchivePath) {
        Remove-Item -Path $llvmDownloadedPath -Force -ErrorAction SilentlyContinue
    }
    Write-Host "LLVM extracted to: $llvmPath" -ForegroundColor Green
} else {
    Write-Host "LLVM already exists at: $llvmPath" -ForegroundColor Yellow
}

Write-Host "`n=== Updating Environment ===" -ForegroundColor Cyan

$cmakeBin = Join-Path $cmakePath "bin"
$llvmBin = Join-Path $llvmPath "bin"

Remove-ExistingPathEntry -PathToRemove "C:\Program Files\LLVM\bin"
Ensure-UserPathEntry -PathToAdd $cmakeBin
Ensure-UserPathEntry -PathToAdd $llvmBin

$llvmDir = Join-Path $llvmPath "lib\cmake\llvm"
if (Test-Path $llvmDir) {
    [System.Environment]::SetEnvironmentVariable("LLVM_DIR", $llvmDir, [System.EnvironmentVariableTarget]::User)
    Write-Host "Set LLVM_DIR to: $llvmDir" -ForegroundColor Green
}

Write-Host "`n=== Refreshing PATH In Current Session ===" -ForegroundColor Cyan
$machinePath = [System.Environment]::GetEnvironmentVariable("PATH", [System.EnvironmentVariableTarget]::Machine)
$userPath = [System.Environment]::GetEnvironmentVariable("PATH", [System.EnvironmentVariableTarget]::User)
$env:Path = "$machinePath;$userPath"
if (Test-Path $llvmDir) {
    $env:LLVM_DIR = $llvmDir
}

Write-Host "`n=== Verifying Installations ===" -ForegroundColor Cyan

$tools = @(
    @{ Name = "cmake"; Command = "cmake"; Args = @("--version") },
    @{ Name = "clang"; Command = "clang"; Args = @("--version") }
)

foreach ($tool in $tools) {
    $result = Get-CommandVersionLine -Command $tool.Command -Args $tool.Args
    if ($result) {
        Write-Host "$($tool.Name): $result" -ForegroundColor Green
    } else {
        Write-Host "$($tool.Name): Not found" -ForegroundColor Red
    }
}

Write-Host "`n=== Installation Complete ===" -ForegroundColor Green
Write-Host "Tools directory: $installDir" -ForegroundColor Cyan
Write-Host "LLVM_DIR: $llvmDir" -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Restart VS Code or open a new terminal"
Write-Host "2. Run: cd c:\Users\user\Documents\vgre"
Write-Host "3. Run: .\scripts\vgre_sync.bat"

Read-Host "`nPress Enter to exit"
