import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:equatable/equatable.dart';
import 'package:csv/csv.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ffi/ffi.dart';
import '../domain/models/telemetry.dart';
import '../infrastructure/bridge/vgre_ffi.dart';
import '../infrastructure/services/sqlite_service.dart';
import 'dart:isolate';

// ── Events ─────────────────────────────────────────────────────────────────
abstract class TelemetryEvent extends Equatable {
  const TelemetryEvent();
  @override
  List<Object?> get props => [];
}

class StartPolling extends TelemetryEvent {
  const StartPolling();
}

class StopPolling extends TelemetryEvent {
  const StopPolling();
}

class ToggleBackgroundCompute extends TelemetryEvent {
  final bool enabled;
  const ToggleBackgroundCompute(this.enabled);
  @override
  List<Object> get props => [enabled];
}

class ToggleServiceMode extends TelemetryEvent {
  final bool isMaster;
  const ToggleServiceMode(this.isMaster);
  @override
  List<Object> get props => [isMaster];
}

class ToggleBlockThreads extends TelemetryEvent {
  final bool enabled;
  const ToggleBlockThreads(this.enabled);
  @override
  List<Object> get props => [enabled];
}

class ToggleProfiler extends TelemetryEvent {
  final bool enabled;
  const ToggleProfiler(this.enabled);
  @override
  List<Object> get props => [enabled];
}

class ToggleClusterSecurity extends TelemetryEvent {
  final bool enabled;
  const ToggleClusterSecurity(this.enabled);
  @override
  List<Object> get props => [enabled];
}

class ResetCredits extends TelemetryEvent {
  const ResetCredits();
}

class SwitchDevice extends TelemetryEvent {
  final int deviceId;
  const SwitchDevice(this.deviceId);
  @override
  List<Object> get props => [deviceId];
}

class ExportKernelHistory extends TelemetryEvent {
  final String kernelName;
  const ExportKernelHistory(this.kernelName);
  @override
  List<Object> get props => [kernelName];
}

class UpdateTelemetry extends TelemetryEvent {
  final Telemetry telemetry;
  const UpdateTelemetry(this.telemetry);
  @override
  List<Object?> get props => [telemetry];
}

class SelectKernel extends TelemetryEvent {
  final String? kernelName;
  const SelectKernel(this.kernelName);
  @override
  List<Object?> get props => [kernelName];
}

// ── State ──────────────────────────────────────────────────────────────────
abstract class TelemetryState extends Equatable {
  const TelemetryState();
  @override
  List<Object?> get props => [];
}

class TelemetryInitial extends TelemetryState {}

class TelemetryActive extends TelemetryState {
  final Telemetry telemetry;
  final List<Telemetry> history;
  final String? selectedKernelName;
  final String deviceName;
  final String backendVersion;
  final int deviceCount;

  const TelemetryActive({
    required this.telemetry,
    required this.history,
    required this.deviceName,
    required this.backendVersion,
    required this.deviceCount,
    this.selectedKernelName,
  });
  @override
  List<Object?> get props => [
    telemetry,
    history,
    selectedKernelName,
    deviceName,
    backendVersion,
    deviceCount,
  ];
}

// ── BLoC ───────────────────────────────────────────────────────────────────
class TelemetryBloc extends Bloc<TelemetryEvent, TelemetryState> {
  final VgreBridge bridge;
  final SqliteService sqlite;
  final String libPath;
  Timer? _timer;

  SendPort? _pollSendPort;
  ReceivePort? _pollReceivePort;
  Isolate? _pollIsolate;

  int _currentDeviceId = 0;
  int get currentDeviceId => _currentDeviceId;
  int _deviceCount = 1;
  String _deviceName = "VGRE_VIRTUAL_GPU";
  bool _backgroundComputeActive = false;
  bool _serviceModeActive = true;
  bool _blockThreadsActive = false;
  Telemetry? _lastSmoothed;
  String? _selectedKernelName;
  KernelStat? _lastSelectedKernelStats;
  bool _profilerEnabled = true;
  final bool _clusterSecurityActive = false;
  String _backendVersion = '0.0.0';

  bool _clusterSecuritySupported = false;

  TelemetryBloc({
    required this.bridge,
    required this.sqlite,
    required this.libPath,
  }) : super(TelemetryInitial()) {
    on<StartPolling>((event, emit) async {
      await _startIsolate();
      _timer?.cancel();
      _timer = Timer.periodic(const Duration(milliseconds: 500), (_) {
        _requestPoll();
      });
      _requestPoll();
    });

    on<ToggleBackgroundCompute>((event, emit) {
      _pollSendPort?.send({
        'type': 'setBackgroundCompute',
        'enabled': event.enabled,
      });
      _backgroundComputeActive = event.enabled;
      if (state is TelemetryActive) {
        final s = state as TelemetryActive;
        emit(
          TelemetryActive(
            telemetry: s.telemetry.copyWith(
              backgroundComputeActive: event.enabled,
            ),
            history: s.history,
            deviceName: s.deviceName,
            backendVersion: s.backendVersion,
            deviceCount: s.deviceCount,
            selectedKernelName: s.selectedKernelName,
          ),
        );
      }
    });

    on<ToggleProfiler>((event, emit) {
      _pollSendPort?.send({
        'type': 'setProfilerEnabled',
        'enabled': event.enabled,
      });
      _profilerEnabled = event.enabled;
      if (state is TelemetryActive) {
        final s = state as TelemetryActive;
        emit(
          TelemetryActive(
            telemetry: s.telemetry.copyWith(profilerEnabled: event.enabled),
            history: s.history,
            deviceName: s.deviceName,
            backendVersion: s.backendVersion,
            deviceCount: s.deviceCount,
            selectedKernelName: s.selectedKernelName,
          ),
        );
      }
    });

    on<ToggleClusterSecurity>((event, emit) {
      if (!_clusterSecuritySupported) return;
      _pollSendPort?.send({
        'type': 'clusterSetSecurity',
        'enabled': event.enabled,
      });
    });

    on<ResetCredits>((event, emit) {
      _pollSendPort?.send({'type': 'creditsReset'});
    });

    on<ToggleServiceMode>((event, emit) {
      _pollSendPort?.send({
        'type': 'setServiceMode',
        'enabled': event.isMaster,
      });
      _serviceModeActive = event.isMaster;
      if (state is TelemetryActive) {
        final s = state as TelemetryActive;
        emit(
          TelemetryActive(
            telemetry: s.telemetry.copyWith(serviceModeActive: event.isMaster),
            history: s.history,
            deviceName: s.deviceName,
            backendVersion: s.backendVersion,
            deviceCount: s.deviceCount,
            selectedKernelName: s.selectedKernelName,
          ),
        );
      }
    });

    on<ToggleBlockThreads>((event, emit) {
      _pollSendPort?.send({
        'type': 'setBlockThreads',
        'enabled': event.enabled,
      });
      _blockThreadsActive = event.enabled;
      if (state is TelemetryActive) {
        final s = state as TelemetryActive;
        emit(
          TelemetryActive(
            telemetry: s.telemetry.copyWith(blockThreadsActive: event.enabled),
            history: s.history,
            deviceName: s.deviceName,
            backendVersion: s.backendVersion,
            deviceCount: s.deviceCount,
            selectedKernelName: s.selectedKernelName,
          ),
        );
      }
    });

    on<SwitchDevice>((event, emit) {
      _currentDeviceId = event.deviceId;
      _lastSmoothed = null;
      _pollSendPort?.send({'type': 'switchDevice', 'deviceId': event.deviceId});
      add(const StartPolling());
    });

    on<ExportKernelHistory>((event, emit) async {
      try {
        final history = await sqlite.getKernelHistory(
          event.kernelName,
          _currentDeviceId,
        );
        if (history.isEmpty) return;

        final List<List<dynamic>> rows = [
          [
            "Timestamp",
            "Duration (ms)",
            "GFLOPS",
            "Throughput (GB/s)",
            "Threads",
          ],
          ...history.map(
            (e) => [
              e.timestamp.toIso8601String(),
              e.durationMs,
              e.gflops,
              e.throughputGbps,
              e.threadsUsed,
            ],
          ),
        ];

        final csvData = const ListToCsvConverter().convert(rows);
        final directory =
            await getDownloadsDirectory() ??
            await getApplicationDocumentsDirectory();
        final path =
            "${directory.path}/vgre_${event.kernelName}_history_${DateTime.now().millisecondsSinceEpoch}.csv";
        final file = File(path);
        await file.writeAsString(csvData);
      } catch (e) {
        debugPrint("Export failed: $e");
      }
    });

    on<StopPolling>((event, emit) {
      _timer?.cancel();
      _stopIsolate();
    });

    on<SelectKernel>((event, emit) {
      _selectedKernelName = event.kernelName;
      _pollSendPort?.send({
        'type': 'selectKernel',
        'kernelName': event.kernelName,
      });
      if (state is TelemetryActive) {
        final s = state as TelemetryActive;
        emit(
          TelemetryActive(
            telemetry: s.telemetry,
            history: s.history,
            deviceName: _deviceName,
            backendVersion: _backendVersion,
            deviceCount: s.deviceCount,
            selectedKernelName: _selectedKernelName,
          ),
        );
      }
    });

    on<UpdateTelemetry>((event, emit) {
      final List<Telemetry> newHistory = List.from(
        (state is TelemetryActive) ? (state as TelemetryActive).history : [],
      );
      final smoothed = _smoothTelemetry(_lastSmoothed, event.telemetry);
      _lastSmoothed = smoothed;
      newHistory.add(smoothed);
      if (newHistory.length > 50) newHistory.removeAt(0);

      sqlite.saveTelemetry(event.telemetry, _currentDeviceId);

      emit(
        TelemetryActive(
          telemetry: event.telemetry.copyWith(
            backgroundComputeActive: _backgroundComputeActive,
            profilerEnabled: _profilerEnabled,
            blockThreadsActive: _blockThreadsActive,
            serviceModeActive: _serviceModeActive,
          ),
          history: newHistory,
          deviceName: _deviceName,
          backendVersion: _backendVersion,
          deviceCount: _deviceCount,
          selectedKernelName: _selectedKernelName,
        ),
      );
    });
  }

  Future<void> _startIsolate() async {
    if (_pollIsolate != null) return;

    _pollReceivePort = ReceivePort();
    _pollIsolate = await Isolate.spawn(_vgreIsolateEntryPoint, {
      'sendPort': _pollReceivePort!.sendPort,
      'libPath': libPath,
      'deviceId': _currentDeviceId,
    });

    _pollReceivePort!.listen((message) {
      if (message is SendPort) {
        // Do NOT send a 'configure' message here. _bootstrap() already called
        // bridge.init() + all configure methods on the main thread before
        // setState(). Sending another configure round from the isolate ~490ms
        // later hits the heap after GTK+ has been rendering and causes
        // malloc(): invalid size (unsorted) corruption.
        _pollSendPort = message;
      } else if (message is Map<String, dynamic>) {
        if (message['type'] == 'telemetry') {
          final t = message['data'] as Telemetry;
          _deviceCount = t.deviceCount;
          _deviceName = t.deviceName;
          _backendVersion = t.backendVersion;
          _clusterSecuritySupported = t.clusterSecuritySupported;
          _lastSelectedKernelStats = t.lastSelectedKernelStats;
          add(UpdateTelemetry(t));
        } else if (message['type'] == 'log') {
          debugPrint("[VGRE Isolate] ${message['message']}");
        }
      }
    });
  }

  void _stopIsolate() {
    _pollIsolate?.kill(priority: Isolate.immediate);
    _pollIsolate = null;
    _pollReceivePort?.close();
    _pollSendPort = null;
  }

  void _requestPoll() {
    _pollSendPort?.send({'type': 'poll'});
  }

  @override
  Future<void> close() {
    _timer?.cancel();
    _stopIsolate();
    return super.close();
  }

  Telemetry _smoothTelemetry(Telemetry? prev, Telemetry current) {
    if (prev == null) return current;
    const double alpha = 0.35;
    final smoothedCompute =
        prev.computeUtilization * (1.0 - alpha) +
        current.computeUtilization * alpha;
    final smoothedMem =
        prev.memoryBusUtilization * (1.0 - alpha) +
        current.memoryBusUtilization * alpha;
    final smoothedTemp =
        prev.temperature * (1.0 - alpha) + current.temperature * alpha;

    return current.copyWith(
      computeUtilization: smoothedCompute,
      memoryBusUtilization: smoothedMem,
      temperature: smoothedTemp,
    );
  }
}

// ── Background Polling Isolate ─────────────────────────────────────────────

void _vgreIsolateEntryPoint(Map<String, dynamic> args) {
  final SendPort mainSendPort = args['sendPort'];
  final String libPath = args['libPath'];
  final ReceivePort isolateReceivePort = ReceivePort();
  mainSendPort.send(isolateReceivePort.sendPort);

  late final VgreBridge bridge;
  try {
    bridge = VgreBridge(libPath);
    // DO NOT call bridge.init() here. _bootstrap() in main.dart already
    // called vgre_init() on the main thread before setState(). Calling it
    // again from this isolate thread while GTK+ is rendering and TCP cluster
    // threads are running causes a malloc(): invalid size (unsorted) crash:
    // vgre_init() acquires RuntimeEngine::recursive_mutex while the concurrent
    // heap activity from background threads corrupts glibc's unsorted bin.
    // The C++ runtime is already fully initialized — skip redundant init.
    mainSendPort.send({
      'type': 'log',
      'message': 'VGRE bridge ready (runtime pre-initialized)',
    });
  } catch (e) {
    mainSendPort.send({
      'type': 'log',
      'message': 'Failed to initialize VGRE bridge: $e',
    });
    return;
  }
  int currentDeviceId = args['deviceId'];

  bool securitySupported = false;

  // NOTE: Auth token propagation is intentionally omitted here.
  // _bootstrap() in main.dart calls _syncRuntimeEnvVars() → setenv() BEFORE
  // bridge.init() and setServiceMode(), so the token is already in the
  // process environment before any TCP cluster threads start.
  //
  // Calling setenv() here races with concurrent getenv() calls from live
  // TCP cluster threads (glibc setenv may realloc environ, getenv reads it
  // without a lock) → heap metadata corruption → malloc(): invalid size.

  // Fetch real device name and version string once at start
  String deviceName = "Unknown Device";
  String versionString = "unknown";
  try {
    final props = bridge.getDeviceProperties(currentDeviceId);
    deviceName = props['name'] as String;
    versionString = bridge.getVersion();
  } catch (_) {}
  try {
    bridge.getSecurityInfo();
    securitySupported = true;
  } catch (_) {
    securitySupported = false;
  }

  String? selectedKernelName;
  KernelStat? lastSelectedKernelStats;

  isolateReceivePort.listen((message) {
    if (message is! Map<String, dynamic>) return;

    try {
      switch (message['type']) {
        case 'configure':
          bridge.setServiceMode(message['serviceMode']);
          bridge.setProfilerEnabled(message['profilerEnabled']);
          bridge.setBackgroundCompute(message['backgroundCompute']);
          bridge.setBlockThreads(message['blockThreads']);
          break;
        case 'poll':
          final t = _doPollSync(
            bridge,
            currentDeviceId,
            deviceName,
            versionString,
            securitySupported,
            selectedKernelName,
            lastSelectedKernelStats,
          );
          lastSelectedKernelStats = t.lastSelectedKernelStats;
          mainSendPort.send({'type': 'telemetry', 'data': t});
          break;
        case 'setBackgroundCompute':
          bridge.setBackgroundCompute(message['enabled']);
          break;
        case 'setProfilerEnabled':
          bridge.setProfilerEnabled(message['enabled']);
          break;
        case 'clusterSetSecurity':
          bridge.clusterSetSecurity(message['enabled']);
          break;
        case 'creditsReset':
          bridge.creditsReset();
          break;
        case 'setServiceMode':
          bridge.setServiceMode(message['enabled']);
          break;
        case 'setBlockThreads':
          bridge.setBlockThreads(message['enabled']);
          break;
        case 'switchDevice':
          currentDeviceId = message['deviceId'];
          try {
            final props = bridge.getDeviceProperties(currentDeviceId);
            deviceName = props['name'] as String;
          } catch (_) {}
          break;
        case 'selectKernel':
          selectedKernelName = message['kernelName'];
          break;
      }
    } catch (e) {
      mainSendPort.send({'type': 'log', 'message': 'Poll Isolate Error: $e'});
    }
  });
}

Telemetry _doPollSync(
  VgreBridge bridge,
  int currentDeviceId,
  String deviceName,
  String versionString,
  bool securitySupported,
  String? selectedKernelName,
  KernelStat? lastSelectedKernelStats,
) {
  final ptr = calloc<VgreTelemetry>();
  try {
    final raw = bridge.getTelemetryWith(ptr);
    final logs = bridge.getLogs();
    final clusterData = bridge.getClusterNodes();

    SecurityInfo? securityInfo;
    try {
      final s = bridge.getSecurityInfo();
      securityInfo = SecurityInfo(
        cipherName: s['cipherName'] as String,
        keyFingerprint: s['keyFingerprint'] as String,
        sessionSeconds: s['sessionSeconds'] as double,
        isEncrypted: s['isEncrypted'] as bool,
        packetsSent: s['packetsSent'] as int,
        packetsReceived: s['packetsReceived'] as int,
        bytesSent: s['bytesSent'] as int,
        bytesReceived: s['bytesReceived'] as int,
      );
    } catch (_) {}

    final creditData = bridge.getCreditsAll();
    final List<ClusterNode> clusterNodes = clusterData.map((m) {
      final addr = m['address'] as String;
      final cred = creditData.firstWhere(
        (c) => c['address'] == addr,
        orElse: () => {},
      );
      return ClusterNode(
        address: addr,
        port: m['port'] as int,
        cpuCores: m['cpuCores'] as int,
        memoryBytes: m['memoryBytes'] as int,
        latencyMs: m['latencyMs'] as double,
        available: m['available'] as bool,
        igpuName: m['igpuName'] as String,
        totalCredits: (cred['totalCredits'] ?? 0.0) as double,
        totalDebits: (cred['totalDebits'] ?? 0.0) as double,
        balance: (cred['balance'] ?? 0.0) as double,
        transactionCount: (cred['transactionCount'] ?? 0) as int,
      );
    }).toList();

    List<KernelStat> topKernels = const [];
    try {
      final jsonStr = bridge.getProfilerJson(topN: 20);
      if (jsonStr != null) {
        final decoded = jsonDecode(jsonStr) as Map<String, dynamic>;
        final items = decoded['top_kernels'] as List<dynamic>? ?? [];

        List<KernelExecution> selectedHistory = const [];
        if (selectedKernelName != null) {
          final historyStr = bridge.getKernelHistoryJson(selectedKernelName);
          if (historyStr != null) {
            final historyItems = jsonDecode(historyStr) as List<dynamic>;
            selectedHistory = historyItems.map((h) {
              final hm = h as Map<String, dynamic>;
              return KernelExecution(
                timestamp: DateTime.fromMillisecondsSinceEpoch(
                  (hm['timestamp_ms'] ?? 0) as int,
                ),
                durationMs: (hm['duration_ms'] ?? 0).toDouble(),
                throughputGbps: (hm['throughput_gbps'] ?? 0).toDouble(),
                gflops: (hm['gflops'] ?? 0).toDouble(),
                threadsUsed: (hm['threads_used'] ?? 0) as int,
              );
            }).toList();
          }
        }

        topKernels = items.map((item) {
          final m = item as Map<String, dynamic>;
          final name = (m['name'] ?? 'kernel').toString();
          final k = KernelStat(
            name: name,
            invocations: (m['invocations'] ?? 0) as int,
            totalTimeMs: (m['total_time_ms'] ?? 0).toDouble(),
            avgTimeMs: (m['avg_time_ms'] ?? 0).toDouble(),
            minTimeMs: (m['min_time_ms'] ?? 0).toDouble(),
            maxTimeMs: (m['max_time_ms'] ?? 0).toDouble(),
            avgThroughputGbps: (m['avg_throughput_gbps'] ?? 0).toDouble(),
            avgGflops: (m['avg_gflops'] ?? 0).toDouble(),
            sourceCode: (m['source_code'] ?? '').toString(),
            irCode: (m['ir_code'] ?? '').toString(),
            history: name == selectedKernelName ? selectedHistory : const [],
          );
          if (name == selectedKernelName) lastSelectedKernelStats = k;
          return k;
        }).toList();
      }
    } catch (_) {}

    int deviceCount = 1;
    try {
      final count = bridge.getDeviceCount();
      if (count > 0) deviceCount = count;
    } catch (_) {}

    CacheStats cacheStats = const CacheStats();
    try {
      final cs = bridge.getCacheStats();
      if (cs != null) {
        cacheStats = CacheStats(
          l2Hits:      cs['l2Hits']      as int,
          l2Misses:    cs['l2Misses']    as int,
          l2Evictions: cs['l2Evictions'] as int,
          l2HitRate:   cs['l2HitRate']   as double,
          l1ConfigKb:  cs['l1ConfigKb']  as int,
          l2ConfigMb:  cs['l2ConfigMb']  as int,
        );
      }
    } catch (_) {}

    return Telemetry(
      timestamp: DateTime.fromMillisecondsSinceEpoch(raw.timestamp),
      gflops: raw.gflops,
      maxGflops: raw.maxGflops,
      computeUtilization: raw.computeUtilization,
      memoryBandwidth: raw.memoryBandwidthGbps,
      maxMemoryBandwidth: raw.maxMemoryBandwidthGbps,
      memoryBusUtilization: raw.memoryBusUtilization,
      memoryUsed: raw.memoryUsedBytes,
      memoryTotal: raw.memoryTotalBytes,
      totalPages: raw.totalPages,
      residentPages: raw.residentPages,
      evictedPages: raw.evictedPages,
      pageFaultRate: raw.pageFaultRate,
      uvmMap: List.generate(1024, (i) => raw.uvmMap[i]),
      activeKernels: raw.activeKernels,
      activeThreads: raw.activeThreads,
      clockSpeed: raw.deviceClockMhz.toInt(),
      avgLatency: raw.avgKernelLatencyMs,
      temperature: raw.deviceTemperature,
      eccEnabled: raw.eccEnabled != 0,
      backgroundComputeActive: raw.backgroundComputeActive != 0,
      serviceModeActive: true, // Master by definition in dashboard
      blockThreadsActive: false, // Default
      deviceName: deviceName,
      versionMajor: raw.versionMajor,
      versionMinor: raw.versionMinor,
      versionPatch: raw.versionPatch,
      logs: logs,
      topKernels: topKernels,
      clusterNodes: clusterNodes,
      securityInfo: securityInfo,
      profilerEnabled: true,
      backendVersion: versionString,
      clusterSecurityActive:
          (securityInfo?.isEncrypted ?? false) ||
          (securityInfo?.isHandshakePending ?? false) ||
          (securityInfo?.isSecurityEnabled ?? false),
      clusterSecuritySupported: securitySupported,
      deviceCount: deviceCount,
      lastSelectedKernelStats: lastSelectedKernelStats,
      cacheStats: cacheStats,
    );
  } finally {
    calloc.free(ptr);
  }
}
