import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'dart:io';
import 'application/telemetry_bloc.dart';
import 'core/theme/vgre_theme.dart';
import 'infrastructure/bridge/vgre_ffi.dart';
import 'infrastructure/services/sqlite_service.dart';
import 'presentation/pages/dashboard_page.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const VgreBootstrapApp());
}

class VgreBootstrapApp extends StatefulWidget {
  const VgreBootstrapApp({super.key});

  @override
  State<VgreBootstrapApp> createState() => _VgreBootstrapAppState();
}

class _VgreBootstrapAppState extends State<VgreBootstrapApp> {
  VgreBridge? _bridge;
  SqliteService? _sqlite;
  String? _libPath;
  String? _error;

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    try {
      // Load ~/.vgre/env first so all runtime vars are available before the
      // bridge opens the shared library (LD_LIBRARY_PATH must be set first).
      _loadVgreEnvFile();

      final String libPath = _resolveLibPath();
      final VgreBridge bridge = VgreBridge(libPath);
      final SqliteService sqlite = SqliteService();

      _syncRuntimeEnvVars(bridge);

      // ── Critical: initialize and configure VGRE on the MAIN THREAD ─────────
      // BEFORE setState() triggers the first GTK+/Pango frame render.
      //
      // All five calls below MUST complete before setState() returns:
      //   1. vgre_init() starts BlockWorkerPool (96 threads), AdaptiveExecution
      //      calibration, and pre-warms lazy singletons.
      //   2. setServiceMode(true) starts the TCP cluster master (server listener,
      //      UDP announcer, worker-discovery threads).
      //   3-5. setProfilerEnabled / setBlockThreads / setBackgroundCompute
      //      configure global runtime flags.
      //
      // The telemetry isolate DOES NOT call bridge.init() or any configure
      // method. Doing so from the isolate's thread while GTK+ is rendering and
      // TCP cluster threads are live causes malloc(): invalid size (unsorted)
      // because concurrent heap activity (Pango text layout + TCP cluster
      // std::string allocs + setenv() realloc) corrupts glibc's unsorted bin.
      // Initialize and configure VGRE on the MAIN THREAD before setState().
      // Heap corruption on glibc malloc when Flutter (libc++) and VGRE
      // (libstdc++) threads contend on per-thread arenas is resolved by
      // preloading jemalloc in the launcher script.
      bridge.init();
      bridge.setProfilerEnabled(true);
      bridge.setBlockThreads(false);
      bridge.setBackgroundCompute(false);
      bridge.setServiceMode(true);

      if (!mounted) return;
      setState(() {
        _bridge = bridge;
        _sqlite = sqlite;
        _libPath = libPath;
      });
    } catch (e, stackTrace) {
      debugPrint('VGRE bootstrap failed: $e');
      debugPrint('$stackTrace');
      if (!mounted) return;
      setState(() {
        _error = e.toString();
      });
    }
  }

  /// Parse ~/.vgre/env (shell-style KEY="VALUE" lines) and push every variable
  /// into the Dart process environment via bridge.setEnvironmentVariable so
  /// that the native library picks them up through getenv().
  void _loadVgreEnvFile() {
    try {
      final home = Platform.environment['HOME'] ??
          Platform.environment['USERPROFILE'] ?? '';
      if (home.isEmpty) return;
      final envFile = File('$home/.vgre/env');
      if (!envFile.existsSync()) return;

      for (final raw in envFile.readAsLinesSync()) {
        final line = raw.trim();
        // Skip comments and blank lines
        if (line.isEmpty || line.startsWith('#')) continue;
        // Strip leading "export "
        final stripped = line.startsWith('export ') ? line.substring(7) : line;
        final eq = stripped.indexOf('=');
        if (eq < 1) continue;
        final key = stripped.substring(0, eq).trim();
        var val = stripped.substring(eq + 1).trim();
        // Strip surrounding quotes
        if ((val.startsWith('"') && val.endsWith('"')) ||
            (val.startsWith("'") && val.endsWith("'"))) {
          val = val.substring(1, val.length - 1);
        }
        // Expand simple ${VAR:-default} or $VAR references
        val = val.replaceAllMapped(
          RegExp(r'\$\{([^}:]+)(?::-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)'),
          (m) {
            final varName = m[1] ?? m[3] ?? '';
            final fallback = m[2] ?? '';
            return Platform.environment[varName] ?? fallback;
          },
        );
        if (key.isNotEmpty) {
          // Store in a local map; we'll push to native after the bridge is open.
          _pendingEnv[key] = val;
        }
      }
    } catch (e) {
      debugPrint('VGRE env file load failed: $e');
    }
  }

  final Map<String, String> _pendingEnv = {};

  String _resolveLibPath() {
    final String libName = Platform.isWindows
        ? 'vgre.dll'
        : (Platform.isMacOS ? 'libvgre.dylib' : 'libvgre.so');

    // 1. Explicit override from env file or environment
    final envPath = _pendingEnv['VGRE_LIB_PATH'] ??
        Platform.environment['VGRE_LIB_PATH'];
    if (envPath != null && File(envPath).existsSync()) return envPath;

    final String executablePath = Platform.resolvedExecutable;
    final String executableDir = executablePath.substring(
      0, executablePath.lastIndexOf(Platform.pathSeparator));

    // 2. Next to the binary's lib/ subdirectory (installed bundle)
    final bundlePath = '$executableDir${Platform.pathSeparator}lib'
        '${Platform.pathSeparator}$libName';
    if (File(bundlePath).existsSync()) return bundlePath;

    // 2b. macOS .app bundle: Contents/Frameworks/
    if (Platform.isMacOS) {
      final frameworksPath =
          '$executableDir${Platform.pathSeparator}..${Platform.pathSeparator}Frameworks'
          '${Platform.pathSeparator}$libName';
      if (File(frameworksPath).existsSync()) return frameworksPath;
    }

    // 3. Next to the binary itself
    final localPath = '$executableDir${Platform.pathSeparator}$libName';
    if (File(localPath).existsSync()) return localPath;

    // 4. Installed under ~/.local/share/VGRE/lib
    final home = Platform.environment['HOME'] ?? '';
    if (home.isNotEmpty) {
      final installedPath = '$home/.local/share/VGRE/lib/$libName';
      if (File(installedPath).existsSync()) return installedPath;
    }

    // 5. Repo build directory (developer mode)
    for (final rel in ['../build/$libName', 'build/$libName']) {
      if (File(rel).existsSync()) return rel;
    }

    // 6. Windows %LOCALAPPDATA%\VGRE
    if (Platform.isWindows) {
      final String? vgreInstallDir = Platform.environment['VGRE_INSTALL_DIR'];
      if (vgreInstallDir != null && vgreInstallDir.isNotEmpty) {
        for (final p in [
          '$vgreInstallDir\\lib\\$libName',
          '$vgreInstallDir\\$libName',
        ]) {
          if (File(p).existsSync()) return p;
        }
      }
      final appData = Platform.environment['LOCALAPPDATA'] ?? '';
      if (appData.isNotEmpty) {
        for (final p in [
          '$appData\\VGRE\\lib\\$libName',
          '$appData\\VGRE\\$libName',
        ]) {
          if (File(p).existsSync()) return p;
        }
      }
    }

    return libName; // last resort: dynamic linker search path
  }

  /// Push all env vars from ~/.vgre/env and the auth token into the native
  /// process via setenv() so the C++ runtime reads them through getenv().
  void _syncRuntimeEnvVars(VgreBridge bridge) {
    // Push everything parsed from ~/.vgre/env
    _pendingEnv.forEach((key, value) {
      bridge.setEnvironmentVariable(key, value);
    });

    // Auth token: file takes precedence over env so cluster workers always
    // use the correct token even if a stale VGRE_TCP_AUTH_TOKEN is set.
    if (!Platform.environment.containsKey('VGRE_TCP_AUTH_TOKEN') &&
        !_pendingEnv.containsKey('VGRE_TCP_AUTH_TOKEN')) {
      try {
        final home = Platform.environment['HOME'] ??
            Platform.environment['USERPROFILE'] ?? '';
        final tokenFile = File('$home/.vgre/token');
        if (tokenFile.existsSync()) {
          final token = tokenFile.readAsStringSync().trim();
          if (token.isNotEmpty) {
            bridge.setEnvironmentVariable('VGRE_TCP_AUTH_TOKEN', token);
          }
        }
      } catch (e) {
        debugPrint('Token sync failed: $e');
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return MaterialApp(
        title: 'VGRE Dashboard',
        debugShowCheckedModeBanner: false,
        theme: VgreTheme.darkTheme,
        home: _BootstrapErrorScreen(message: _error!),
      );
    }

    if (_bridge == null || _sqlite == null || _libPath == null) {
      return MaterialApp(
        title: 'VGRE Dashboard',
        debugShowCheckedModeBanner: false,
        theme: VgreTheme.darkTheme,
        home: const _BootstrapLoadingScreen(),
      );
    }

    return VgreDashboardApp(
      bridge: _bridge!,
      sqlite: _sqlite!,
      libPath: _libPath!,
    );
  }
}

class VgreDashboardApp extends StatelessWidget {
  final VgreBridge bridge;
  final SqliteService sqlite;
  final String libPath;

  const VgreDashboardApp({super.key, required this.bridge, required this.sqlite, required this.libPath});

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (context) =>
          TelemetryBloc(bridge: bridge, sqlite: sqlite)..add(const StartPolling()),
      child: MaterialApp(
        title: 'VGRE Dashboard',
        debugShowCheckedModeBanner: false,
        theme: VgreTheme.darkTheme,
        home: const DashboardPage(),
      ),
    );
  }
}

class _BootstrapLoadingScreen extends StatelessWidget {
  const _BootstrapLoadingScreen();

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      backgroundColor: VgreTheme.background,
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text(
              'Starting VGRE Dashboard...',
              style: TextStyle(color: Colors.white70),
            ),
          ],
        ),
      ),
    );
  }
}

class _BootstrapErrorScreen extends StatelessWidget {
  final String message;

  const _BootstrapErrorScreen({required this.message});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: VgreTheme.background,
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 720),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'VGRE Dashboard failed to start',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 24,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 16),
                const Text(
                  'Startup error:',
                  style: TextStyle(color: Colors.white70),
                ),
                const SizedBox(height: 8),
                SelectableText(
                  message,
                  style: const TextStyle(
                    color: Colors.redAccent,
                    fontFamily: 'Consolas',
                  ),
                ),
                const SizedBox(height: 24),
                const Text(
                  'Troubleshooting Steps:',
                  style: TextStyle(
                    color: Colors.white70,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 12),
                ...buildTroubleshootingSteps(message),
              ],
            ),
          ),
        ),
      ),
    );
  }

  List<Widget> buildTroubleshootingSteps(String errorMsg) {
    List<Widget> steps = [];
    
    if (errorMsg.contains('Failed to load') || errorMsg.contains('DLL')) {
      steps.addAll([
        const Text('1. Rebuilt VGRE with: .\\scripts\\vgre_sync.bat', style: TextStyle(color: Colors.white54)),
        const Text('2. Close all running instances and retry', style: TextStyle(color: Colors.white54)),
        const Text('3. Verify PATH includes: %LOCALAPPDATA%\\VGRE and %LOCALAPPDATA%\\VGRE\\lib', style: TextStyle(color: Colors.white54)),
        const Text('4. See docs/TROUBLESHOOTING_WINDOWS.md (error 1114 / 0xC000001D)', style: TextStyle(color: Colors.white54)),
      ]);
    } else if (errorMsg.contains('Cluster')) {
      steps.addAll([
        const Text('1. Ensure no other instance is running on port 7777', style: TextStyle(color: Colors.white54)),
        const Text('2. Check network connectivity', style: TextStyle(color: Colors.white54)),
      ]);
    }
    
    return steps.map((step) => Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: step,
    )).toList();
  }
}
