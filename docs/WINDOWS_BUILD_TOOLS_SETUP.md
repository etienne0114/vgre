# VGRE Build Tools Installation Guide - Windows

**Date**: April 4, 2026  
**Status**: Administrative Privileges Required

---

## 🚨 Current Issue

Chocolatey requires **Administrator privileges** to install packages. The current PowerShell session does not have elevated permissions.

## ✅ Solution: Manual Installation

Since we have **Visual Studio 2022 Build Tools** already installed, we can proceed with alternative methods:

### Option 1: Use VSCode Terminal (Elevated)
1. Right-click VS Code → "Run as Administrator"
2. Open integrated terminal
3. Run:
```powershell
choco install cmake llvm flutter -y
```

### Option 2: Direct Download & Installation

#### **Step 1: Install CMake**
1. Go to https://cmake.org/download/
2. Download: `cmake-3.29.x-windows-x86_64.msi`
3. Run installer, add to PATH
4. Verify: Open CMD and run `cmake --version`

#### **Step 2: Install LLVM**
1. Go to https://github.com/llvm/llvm-project/releases
2. Download: `LLVM-18.x.x-win64.exe`
3. Run installer, **CHECK**: Add LLVM to PATH
4. Verify: Open CMD and run `clang --version`

#### **Step 3: Install Flutter**
1. Go to https://flutter.dev/docs/get-started/install/windows
2. Download: `flutter_windows_x.x.x.zip` (stable channel)
3. Extract to `C:\src\flutter` or similar
4. Add to PATH: `C:\src\flutter\bin`
5. Verify: Open CMD and run `flutter --version`

#### **Step 4: Add Tools to PATH**

After installing, add to Windows PATH:
1. Open: `System Properties` → `Environment Variables`
2. Click `Path` under User Variables → `Edit`
3. Add new entries:
   - `C:\Program Files\CMake\bin`
   - `C:\Program Files\LLVM\bin`
   - `C:\src\flutter\bin` (or wherever you extracted Flutter)
4. Click OK, close all windows
5. **Restart terminal or VS Code**

### Option 3: Leverage Existing VS2022 Build Tools

We can use the C++ compiler that comes with VS2022 Build Tools without waiting for additional installs:

```powershell
# Set up VS environment
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

# Verify MSVC is available
cl.exe
```

Then:
1. Install just **CMake** (required by VGRE's CMakeLists.txt)
2. For LLVM, can use MSVC's built-in compiler initially
3. Install **Flutter** for dashboard

---

## 📋 Verification Checklist

After installation, verify all tools:

```powershell
# Run each and confirm versions appear:
cmake --version          # Should show: cmake version 3.16+
clang --version          # Should show: clang version 14+
flutter --version        # Should show: Flutter SDK version
```

---

## 🔧 Next Steps After Installation

Once tools are installed:

1. **Open fresh PowerShell** (NOT elevated, unless Chocolatey needed)
2. **Navigate to VGRE project**:
   ```powershell
   cd C:\Users\user\Documents\vgre
   ```

3. **Run build script**:
   ```powershell
   .\scripts\vgre_sync.bat
   ```

4. **Monitor build** (takes 8-13 minutes):
   - Native engine build
   - Flutter dashboard build
   - Deployment to `%LOCALAPPDATA%\VGRE`

5. **Launch dashboard**:
   ```powershell
   $env:Path += ";$env:LOCALAPPDATA\VGRE\lib"
   & "$env:LOCALAPPDATA\VGRE\vgre_dashboard.exe"
   ```

---

## 🆘 Troubleshooting

### "cmake: command not found"
- **Cause**: CMake not in PATH or not installed
- **Fix**: Check `System Properties` → `Environment Variables` → `Path`
- **Verify**: Close and reopen terminal after adding to PATH

### "cl.exe: command not found"
- **Cause**: MSVC not discoverable (VS Build Tools doesn't auto-add to PATH)
- **Fix**: Run vcvars batch file:
  ```powershell
  & "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  ```

### "flutter: command not found"
- **Cause**: Flutter SDK not in PATH
- **Fix**: Add `C:\src\flutter\bin` to PATH

### Build script fails with "flutter not found"
- **Cause**: Flutter not installed or not in PATH
- **Solution**: Install Flutter or run with specific installer permissions

---

## Recommended Installation Order

1. ✅ **CMake** (3.16+) - Essential for VGRE build system
2. ✅ **LLVM** (18+) - Used by VGRE compiler backend
3. ✅ **Flutter SDK** - For dashboard UI
4. ✅ **Visual Studio 2022 C++** - Already have Build Tools; C++ tools may need installation

Note: We already have VS2022 Build Tools, which provides the C++ compiler needed.

---

## Accelerated Path (Minimum Installation)

If time-limited, you can:

1. **Install CMake only** (manually or via website)
2. **Install LLVM18 only** (needed for kernel compilation)
3. **Skip Flutter** initially - Build native engine first:
   ```powershell
   mkdir build; cd build
   cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release --target vgre vgre_cudart vgre-worker
   ```
4. **Install Flutter later** for dashboard

---

## Status Updates

**Current**: Waiting for administrative access or manual tool installation  
**Next**: Will execute `vgre_sync.bat` once tools are in PATH  
**Time Estimate**: 10-15 minutes for tool installation, then 8-13 minutes for VGRE build

---

## 🚀 Global Command Access (vgre-worker)

By default, the VGRE build script (`vgre_sync.bat`) attempts to add the installation folder (`%LOCALAPPDATA%\VGRE`) to your User **PATH** permanently. This allows you to run `vgre-worker` from any terminal session started *after* the sync completes.

### Manual Verification
1.  Open a **NEW** Command Prompt or PowerShell window.
2.  Type: `vgre-worker --help`
3.  If you see "command not recognized", you may need to add it to your PATH manually.

### Manual PATH Update (Standard Method)

1.  Press `Win + R`, type `sysdm.cpl`, and press Enter.
2.  Navigate to the **Advanced** tab.
3.  Click **Environment Variables...**
4.  In the **User variables** section, find the variable named **Path** and select it.
5.  Click **Edit...**, then click **New**.
6.  Paste: `%LOCALAPPDATA%\VGRE`
7.  Click **OK** on all three open windows.
8.  **Important**: You must close and reopen any existing terminal windows (CMD, PowerShell, or VS Code terminals) for the change to take effect.

### Running with Dependencies
`vgre-worker.exe` depends on `vgre.dll` and `vgre_cudart.dll`. Both are copied into the same folder (`%LOCALAPPDATA%\VGRE`) during the sync process to ensure the command "just works".

---

