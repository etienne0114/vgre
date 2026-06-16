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
  final List<String> disconnectNotices;
  const UpdateTelemetry(this.telemetry, {this.disconnectNotices = const []});
  @override
  List<Object?> get props => [telemetry, disconnectNotices];
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
  // Worker addresses that transitioned to unavailable since last poll tick.
  final List<String> disconnectNotices;
  // Wall-clock of the last successful backend telemetry poll. The UI compares it
  // to now to show LIVE vs STALE — so stale data is never presented as live.
  final DateTime? lastUpdated;

  const TelemetryActive({
    required this.telemetry,
    required this.history,
    required this.deviceName,
    required this.backendVersion,
    required this.deviceCount,
    this.selectedKernelName,
    this.disconnectNotices = const [],
    this.lastUpdated,
  });
  @override
  List<Object?> get props => [
    telemetry,
    history,
    selectedKernelName,
    deviceName,
    backendVersion,
    deviceCount,
    disconnectNotices,
    lastUpdated,
  ];
}

// ── BLoC ───────────────────────────────────────────────────────────────────
class TelemetryBloc extends Bloc<TelemetryEvent, TelemetryState> {
  final VgreBridge bridge;
  final SqliteService sqlite;
  Timer? _timer;

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
  String _backendVersion = '0.0.0';

  bool _clusterSecuritySupported = false;

  TelemetryBloc({
    required this.bridge,
    required this.sqlite,
  }) : super(TelemetryInitial()) {
    on<StartPolling>((event, emit) async {
      _timer?.cancel();
      _timer = Timer.periodic(const Duration(milliseconds: 500), (_) {
        _poll();
      });
      _poll();
    });

    on<ToggleBackgroundCompute>((event, emit) {
      bridge.setBackgroundCompute(event.enabled);
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
            lastUpdated: s.lastUpdated,
          ),
        );
      }
    });

    on<ToggleProfiler>((event, emit) {
      bridge.setProfilerEnabled(event.enabled);
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
            lastUpdated: s.lastUpdated,
          ),
        );
      }
    });

    on<ToggleClusterSecurity>((event, emit) {
      if (!_clusterSecuritySupported) return;
    });

    on<ResetCredits>((event, emit) {
    });

    on<ToggleServiceMode>((event, emit) {
      bridge.setServiceMode(event.isMaster);
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
            lastUpdated: s.lastUpdated,
          ),
        );
      }
    });

    on<ToggleBlockThreads>((event, emit) {
      bridge.setBlockThreads(event.enabled);
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
            lastUpdated: s.lastUpdated,
          ),
        );
      }
    });

    on<SwitchDevice>((event, emit) {
      _currentDeviceId = event.deviceId;
      _lastSmoothed = null;
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
    });

    on<SelectKernel>((event, emit) {
      _selectedKernelName = event.kernelName;
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
            disconnectNotices: s.disconnectNotices,
            lastUpdated: s.lastUpdated,
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
          disconnectNotices: event.disconnectNotices,
          lastUpdated: DateTime.now(),
        ),
      );
    });
  }

  // How long an unavailable node remains visible before being removed from the UI.
  // Exposed as a public constant so UI widgets can display the same countdown.
  static const kGracePeriodSeconds = 30;

  void _poll() {
    try {
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

        // Previous node list — used to preserve disconnect timestamps and
        // detect newly-disconnected workers.
        final prevNodes = (state is TelemetryActive)
            ? (state as TelemetryActive).telemetry.clusterNodes
            : const <ClusterNode>[];

        final now = DateTime.now();
        final disconnectNotices = <String>[];

        final List<ClusterNode> clusterNodes = clusterData.map((m) {
          final addr = m['address'] as String;
          final isAvailable = m['available'] as bool;
          final cred = creditData.firstWhere(
            (c) => c['address'] == addr,
            orElse: () => {},
          );

          // Preserve the moment a node first became unavailable.
          DateTime? disconnectedAt;
          if (!isAvailable) {
            DateTime? prevTime;
            for (final prev in prevNodes) {
              if (prev.address == addr) {
                prevTime = prev.disconnectedAt;
                break;
              }
            }
            disconnectedAt = prevTime ?? now;
            // Only emit a notice the first time (transition available→unavailable).
            if (prevTime == null && state is TelemetryActive) {
              disconnectNotices.add(addr);
            }
          }

          return ClusterNode(
            address: addr,
            port: m['port'] as int,
            cpuCores: m['cpuCores'] as int,
            memoryBytes: m['memoryBytes'] as int,
            latencyMs: m['latencyMs'] as double,
            available: isAvailable,
            igpuName: m['igpuName'] as String,
            totalCredits: (cred['totalCredits'] ?? 0.0) as double,
            totalDebits: (cred['totalDebits'] ?? 0.0) as double,
            balance: (cred['balance'] ?? 0.0) as double,
            transactionCount: (cred['transactionCount'] ?? 0) as int,
            disconnectedAt: disconnectedAt,
          );
        })
            // Remove nodes that have been unavailable past the grace period.
            .where((node) {
          if (node.available) return true;
          if (node.disconnectedAt == null) return false;
          return now.difference(node.disconnectedAt!).inSeconds <=
              kGracePeriodSeconds;
        }).toList();

        List<KernelStat> topKernels = const [];
        try {
          final jsonStr = bridge.getProfilerJson(topN: 20);
          if (jsonStr != null) {
            final decoded = jsonDecode(jsonStr) as Map<String, dynamic>;
            final items = decoded['top_kernels'] as List<dynamic>? ?? [];

            List<KernelExecution> selectedHistory = const [];
            if (_selectedKernelName != null) {
              final historyStr = bridge.getKernelHistoryJson(_selectedKernelName!);
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
                history: name == _selectedKernelName ? selectedHistory : const [],
              );
              if (name == _selectedKernelName) _lastSelectedKernelStats = k;
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

        String deviceName = "Unknown Device";
        String versionString = "unknown";
        try {
          final props = bridge.getDeviceProperties(_currentDeviceId);
          deviceName = props['name'] as String;
          versionString = bridge.getVersion();
        } catch (_) {}

        bool securitySupported = false;
        try {
          bridge.getSecurityInfo();
          securitySupported = true;
        } catch (_) {
          securitySupported = false;
        }

        final t = Telemetry(
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
          serviceModeActive: true,
          blockThreadsActive: false,
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
          lastSelectedKernelStats: _lastSelectedKernelStats,
          cacheStats: cacheStats,
        );

        _deviceCount = t.deviceCount;
        _deviceName = t.deviceName;
        _backendVersion = t.backendVersion;
        _clusterSecuritySupported = t.clusterSecuritySupported;
        _lastSelectedKernelStats = t.lastSelectedKernelStats;
        add(UpdateTelemetry(t, disconnectNotices: disconnectNotices));
      } finally {
        calloc.free(ptr);
      }
    } catch (e) {
      debugPrint("Poll error: $e");
    }
  }

  @override
  Future<void> close() {
    _timer?.cancel();
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
