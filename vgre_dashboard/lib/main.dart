import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
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
    // Default Linux path for local development
    libPath = Platform.environment['VGRE_LIB_PATH'] ?? 
        '/home/umwami/Desktop/GPU emulator/virtual-gpu-runtime/build/libvgre.so';
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
