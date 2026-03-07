import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'dart:io';
import 'application/telemetry_bloc.dart';
import 'core/theme/vgre_theme.dart';
import 'infrastructure/bridge/vgre_ffi.dart';
import 'presentation/pages/dashboard_page.dart';

import 'dart:io' show Platform;

void main() {
  // Initialize the bridge to the native VGRE engine
  String libPath;
  if (Platform.isWindows) {
    libPath = 'libvgre.dll'; // Must be in PATH or executable dir
  } else if (Platform.isMacOS) {
    libPath = 'libvgre.dylib';
  } else {
    // Portable loading: check local lib/ directory (relative to executable)
    // then fall back to development path.
    final String executablePath = Platform.resolvedExecutable;
    final String localPath = '${executablePath.substring(0, executablePath.lastIndexOf('/'))}/lib/libvgre.so';
    
    if (File(localPath).existsSync()) {
      libPath = localPath;
    } else {
      libPath = Platform.environment['VGRE_LIB_PATH'] ?? 
          (File('../build/libvgre.so').existsSync() 
              ? '../build/libvgre.so' 
              : 'libvgre.so');
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
