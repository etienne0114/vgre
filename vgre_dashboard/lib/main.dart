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
      final String libPath = _resolveLibPath();
      final VgreBridge bridge = VgreBridge(libPath);
      final SqliteService sqlite = SqliteService();

      _syncPersistentToken(bridge);

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

  String _resolveLibPath() {
    final String libName = Platform.isWindows
        ? 'vgre.dll'
        : (Platform.isMacOS ? 'libvgre.dylib' : 'libvgre.so');

    final String executablePath = Platform.resolvedExecutable;
    final String executableDir = executablePath.substring(
      0,
      executablePath.lastIndexOf(Platform.pathSeparator),
    );
    final String bundlePath =
        '$executableDir${Platform.pathSeparator}lib${Platform.pathSeparator}$libName';

    if (File(bundlePath).existsSync()) {
      return bundlePath;
    }

    final String localPath =
        '$executableDir${Platform.pathSeparator}$libName';
    if (File(localPath).existsSync()) {
      return localPath;
    }

    // Windows: Check %LOCALAPPDATA%\VGRE for installed version
    if (Platform.isWindows) {
      final String localAppData = Platform.environment['LOCALAPPDATA'] ?? '';
      if (localAppData.isNotEmpty) {
        final String vgreInstallPath = '$localAppData\\VGRE\\lib\\$libName';
        if (File(vgreInstallPath).existsSync()) {
          return vgreInstallPath;
        }
        final String vgreInstallRootPath = '$localAppData\\VGRE\\$libName';
        if (File(vgreInstallRootPath).existsSync()) {
          return vgreInstallRootPath;
        }
      }
    }

    return Platform.environment['VGRE_LIB_PATH'] ??
        (File('../build/$libName').existsSync() ? '../build/$libName' : libName);
  }

  void _syncPersistentToken(VgreBridge bridge) {
    if (Platform.environment.containsKey('VGRE_TCP_AUTH_TOKEN')) {
      return;
    }

    try {
      final String home =
          Platform.environment['HOME'] ??
          Platform.environment['USERPROFILE'] ??
          '';
      if (home.isEmpty) {
        return;
      }

      final tokenFile = File('$home/.vgre/token');
      if (!tokenFile.existsSync()) {
        return;
      }

      final token = tokenFile.readAsStringSync().trim();
      if (token.isNotEmpty) {
        bridge.setEnvironmentVariable('VGRE_TCP_AUTH_TOKEN', token);
      }
    } catch (e) {
      debugPrint('Token sync failed: $e');
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
          TelemetryBloc(bridge: bridge, sqlite: sqlite, libPath: libPath)..add(const StartPolling()),
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
