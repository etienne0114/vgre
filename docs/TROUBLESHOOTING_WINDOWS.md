# VGRE Windows Troubleshooting

This guide targets the startup failures seen in April 2026 builds:

- `Failed to load dynamic library ... vgre.dll ... (error code: 1114)`
- `vgre-worker.exe` exits with `-1073741795` (`0xC000001D`)

## 1) Rebuild with portable CPU flags

From project root:

```bat
.\scripts\vgre_sync.bat
```

By default this now builds with:

- `-DVGRE_ENABLE_NATIVE_SIMD=OFF`

This prevents binaries that only run on newer AVX-capable CPUs.

If you explicitly want host-tuned SIMD builds:

```bat
set VGRE_ENABLE_NATIVE_SIMD=1
.\scripts\vgre_sync.bat
```

## 2) Verify worker startup before launching dashboard

```bat
"%LOCALAPPDATA%\VGRE\vgre-worker.exe" --help
```

Expected: usage text and exit code `0`.

If it fails, close stale processes and retry:

```bat
taskkill /F /IM vgre_dashboard.exe /IM vgre-worker.exe /T
```

Then run sync again.

## 3) Verify DLL search paths and runtime dependencies

Ensure these exist:

- `%LOCALAPPDATA%\VGRE\vgre.dll`
- `%LOCALAPPDATA%\VGRE\lib\vgre.dll`
- `%LOCALAPPDATA%\VGRE\vgre_cudart.dll`
- `%LOCALAPPDATA%\VGRE\BuildTools\llvm\bin\libomp.dll`

Ensure PATH includes:

- `%LOCALAPPDATA%\VGRE`
- `%LOCALAPPDATA%\VGRE\lib`
- `%LOCALAPPDATA%\VGRE\BuildTools\llvm\bin`

## 4) Quick diagnosis of common errors

- `1114` while loading `vgre.dll`:
  - Most often unsupported CPU instruction set in built binary.
  - Can also be missing runtime DLL dependency.

- `0xC000001D` from `vgre-worker.exe`:
  - Illegal instruction at runtime; rebuild with `VGRE_ENABLE_NATIVE_SIMD=OFF`.

## 5) Launch

Use:

```bat
"%LOCALAPPDATA%\VGRE\Launch-VGRE-Dashboard.cmd"
```

This launcher sets PATH for VGRE and LLVM runtime dependencies before starting the dashboard.
