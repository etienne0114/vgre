import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'dart:io';
import 'application/telemetry_bloc.dart';
import 'core/theme/vgre_theme.dart';
import 'infrastructure/bridge/vgre_ffi.dart';
import 'presentation/pages/dashboard_page.dart';

void main() {
  // Initialize the bridge to the native VGRE engine
  // Initialize the bridge to the native VGRE engine using path-agnostic discovery
  String libPath;
  final String libName = Platform.isWindows
      ? 'vgre.dll'
      : (Platform.isMacOS ? 'libvgre.dylib' : 'libvgre.so');

  // 1. Check relative 'lib' directory (standard production bundle layout)
  final String executablePath = Platform.resolvedExecutable;
  final String executableDir =
      executablePath.substring(0, executablePath.lastIndexOf(Platform.pathSeparator));
  final String bundlePath =
      '$executableDir${Platform.pathSeparator}lib${Platform.pathSeparator}$libName';

  if (File(bundlePath).existsSync()) {
    libPath = bundlePath;
  } else {
    // 2. Check current directory (fallback for side-by-side deployment)
    final String localPath = '$executableDir${Platform.pathSeparator}$libName';
    if (File(localPath).existsSync()) {
      libPath = localPath;
    } else {
      // 3. Fallback to development/environment paths
      libPath =
          Platform.environment['VGRE_LIB_PATH'] ??
          (File('../build/$libName').existsSync()
              ? '../build/$libName'
              : libName);
    }
  }

  final VgreBridge bridge = VgreBridge(libPath);

  // Initialize the VGRE backend (equivalent to vgre_init)
  bridge.init();

  runApp(VgreDashboardApp(bridge: bridge));
}

class VgreDashboardApp extends StatelessWidget {
  final VgreBridge bridge;

  const VgreDashboardApp({super.key, required this.bridge});

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (context) => TelemetryBloc(bridge)..add(StartPolling()),
      child: MaterialApp(
        title: 'VGRE Dashboard',
        debugShowCheckedModeBanner: false,
        theme: VgreTheme.darkTheme,
        home: const DashboardPage(),
      ),
    );
  }
}
