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
  List<Object?> get props =>
      [telemetry, history, selectedKernelName, deviceName, backendVersion, deviceCount];
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
  bool _clusterSecurityActive = false;
  String _backendVersion = '0.0.0';
  static bool _checkClusterSecuritySupport() {
    if (Platform.environment.containsKey('VGRE_TCP_AUTH_TOKEN')) return true;
    try {
      final home = Platform.environment['HOME'] ?? '';
      if (home.isNotEmpty) {
        final tokenFile = File('$home/.vgre/token');
        if (tokenFile.existsSync()) {
          final token = tokenFile.readAsStringSync().trim();
          return token.isNotEmpty;
        }
      }
    } catch (_) {}
    return false;
  }

  late final bool _clusterSecuritySupported = _checkClusterSecuritySupport();

  TelemetryBloc({required this.bridge, required this.sqlite}) : super(TelemetryInitial()) {
    on<StartPolling>((event, emit) async {
      // Initialize as master service (Dashboard)
      try {
        bridge.setServiceMode(true);
      } catch (e) {
        debugPrint("Failed to start VGRE IPC Service: $e");
      }
      try {
        bridge.setProfilerEnabled(true);
      } catch (e) {
        debugPrint("Failed to enable profiler: $e");
      }

      // Load history from SQLite
      try {
        final history = await sqlite.getTelemetryHistory(_currentDeviceId);
        if (history.isNotEmpty) {
          _lastSmoothed = history.last;
          emit(TelemetryActive(
            telemetry: history.last,
            history: history,
            deviceName: _deviceName,
            backendVersion: _backendVersion,
            deviceCount: _deviceCount,
            selectedKernelName: _selectedKernelName,
          ));
        }
      } catch (e) {
        debugPrint("Failed to load telemetry history: $e");
      }

      try {
        _deviceCount = bridge.getDeviceCount();
      } catch (e) {
        debugPrint("Failed to fetch device count: $e");
      }

      // Fetch device info once
      try {
        final props = bridge.getDeviceProperties(_currentDeviceId);
        _deviceName = props['name'] as String;
      } catch (e) {
        debugPrint("Failed to fetch device info: $e");
      }

      try {
        _backendVersion = bridge.getVersion();
      } catch (e) {
        debugPrint("Failed to read backend version: $e");
      }

      _timer?.cancel();
      _timer = Timer.periodic(const Duration(milliseconds: 500), (_) {
        _pollOnce();
      });
      _pollOnce();
    });

    on<ToggleBackgroundCompute>((event, emit) {
      try {
        final res = bridge.setBackgroundCompute(event.enabled);
        if (res == 0) {
          _backgroundComputeActive = event.enabled;
          
          // REFINEMENT: Emit immediate state update so UI feels responsive
          if (state is TelemetryActive) {
            final s = state as TelemetryActive;
            emit(TelemetryActive(
              telemetry: s.telemetry.copyWith(backgroundComputeActive: event.enabled),
              history: s.history,
              deviceName: s.deviceName,
              backendVersion: s.backendVersion,
              deviceCount: s.deviceCount,
              selectedKernelName: s.selectedKernelName,
            ));
          }
        } else {
          debugPrint("Failed to toggle background compute: $res");
        }
      } catch (e) {
        debugPrint("Failed to toggle background compute: $e");
      }
    });

    on<ToggleProfiler>((event, emit) {
      try {
        final res = bridge.setProfilerEnabled(event.enabled);
        if (res == 0) {
          _profilerEnabled = event.enabled;
        } else {
          debugPrint("Failed to toggle profiler: $res");
        }
      } catch (e) {
        debugPrint("Failed to toggle profiler: $e");
      }
    });

    on<ToggleClusterSecurity>((event, emit) {
      if (!_clusterSecuritySupported) {
        debugPrint("Cluster security toggle blocked: token not set");
        return;
      }
      try {
        final res = bridge.clusterSetSecurity(event.enabled);
        if (res == 0) {
          _clusterSecurityActive = event.enabled;
        } else {
          debugPrint("Failed to toggle cluster security: $res");
        }
      } catch (e) {
        debugPrint("Failed to toggle cluster security: $e");
      }
    });

    on<ResetCredits>((event, emit) {
      try {
        final res = bridge.creditsReset();
        if (res != 0) {
          debugPrint("Failed to reset credits: $res");
          return;
        }
        _pollOnce();
      } catch (e) {
        debugPrint("Failed to reset credits: $e");
      }
    });

    on<ToggleServiceMode>((event, emit) {
      final res = bridge.setServiceMode(event.isMaster);
      if (res == 0) {
        _serviceModeActive = event.isMaster;
      } else {
        debugPrint("Failed to switch service mode: $res");
      }
    });

    on<ToggleBlockThreads>((event, emit) {
      final res = bridge.setBlockThreads(event.enabled);
      if (res == 0) {
        _blockThreadsActive = event.enabled;
      } else {
        debugPrint("Failed to toggle block threads: $res");
      }
    });

    on<SwitchDevice>((event, emit) {
      _currentDeviceId = event.deviceId;
      _lastSmoothed = null; // Reset smoothing for new device
      // Refresh device info
      try {
        final props = bridge.getDeviceProperties(_currentDeviceId);
        _deviceName = props['name'] as String;
      } catch (e) {
        debugPrint("Failed to fetch device info: $e");
      }
      add(const StartPolling());
    });

    on<ExportKernelHistory>((event, emit) async {
      try {
        final history = await sqlite.getKernelHistory(event.kernelName, _currentDeviceId);
        if (history.isEmpty) {
          debugPrint("No history found for ${event.kernelName} to export");
          return;
        }

        final List<List<dynamic>> rows = [
          ["Timestamp", "Duration (ms)", "GFLOPS", "Throughput (GB/s)", "Threads"],
          ...history.map((e) => [
                e.timestamp.toIso8601String(),
                e.durationMs,
                e.gflops,
                e.throughputGbps,
                e.threadsUsed,
              ])
        ];

        final csvData = const ListToCsvConverter().convert(rows);
        final directory = await getDownloadsDirectory() ?? await getApplicationDocumentsDirectory();
        final path = "${directory.path}/vgre_${event.kernelName}_history_${DateTime.now().millisecondsSinceEpoch}.csv";
        final file = File(path);
        await file.writeAsString(csvData);
        debugPrint("Hardware-authoritative export completed: $path");
      } catch (e) {
        debugPrint("Failed to export kernel history: $e");
      }
    });

    on<StopPolling>((event, emit) {
      _timer?.cancel();
    });

    on<SelectKernel>((event, emit) {
      _selectedKernelName = event.kernelName;
      // We don't reset _lastSelectedKernelStats here because the next poll will update it
      // if it exists in the new topKernels list.
      if (state is TelemetryActive) {
        final s = state as TelemetryActive;
        emit(TelemetryActive(
          telemetry: s.telemetry,
          history: s.history,
          deviceName: _deviceName,
          backendVersion: _backendVersion,
          deviceCount: s.deviceCount,
          selectedKernelName: _selectedKernelName,
        ));
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

      // Persist to SQLite
      sqlite.saveTelemetry(event.telemetry, _currentDeviceId);

      emit(TelemetryActive(
        telemetry: event.telemetry,
        history: newHistory,
        deviceName: _deviceName,
        backendVersion: _backendVersion,
        deviceCount: _deviceCount,
        selectedKernelName: _selectedKernelName,
      ));
    });
  }

  void _pollOnce() {
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
      } catch (e) {
        debugPrint("Failed to fetch security info: $e");
      }

      final creditData = bridge.getCreditsAll();
      final creditLedger = creditData.map((c) {
        return CreditEntry(
          address: c['address'] as String,
          totalCredits: (c['totalCredits'] ?? 0.0) as double,
          totalDebits: (c['totalDebits'] ?? 0.0) as double,
          balance: (c['balance'] ?? 0.0) as double,
          transactionCount: (c['transactionCount'] ?? 0) as int,
          lastActivity: (c['lastActivity'] ?? 0) as int,
        );
      }).toList(growable: false);

      final Map<String, Map<String, dynamic>> creditsByAddr = {
        for (var c in creditData) c['address'] as String: c
      };

      final List<ClusterNode> clusterNodes = clusterData.map((m) {
        final addr = m['address'] as String;
        final cred = creditsByAddr[addr];
        return ClusterNode(
          address: addr,
          port: m['port'] as int,
          cpuCores: m['cpuCores'] as int,
          memoryBytes: m['memoryBytes'] as int,
          latencyMs: m['latencyMs'] as double,
          available: m['available'] as bool,
          igpuName: m['igpuName'] as String,
          totalCredits: (cred?['totalCredits'] ?? 0.0) as double,
          totalDebits: (cred?['totalDebits'] ?? 0.0) as double,
          balance: (cred?['balance'] ?? 0.0) as double,
          transactionCount: (cred?['transactionCount'] ?? 0) as int,
        );
      }).toList(growable: false);

      if (securityInfo != null) {
        _clusterSecurityActive = securityInfo.isEncrypted;
      }

      List<KernelStat> topKernels = const [];
      try {
        final jsonStr = bridge.getProfilerJson(topN: 20);
        if (jsonStr != null) {
          final decoded = jsonDecode(jsonStr) as Map<String, dynamic>;
          final items = decoded['top_kernels'] as List<dynamic>? ?? [];

          // Fetch history for selected kernel if any
          List<KernelExecution> selectedHistory = const [];
          if (_selectedKernelName != null) {
            try {
              final historyStr = bridge.getKernelHistoryJson(_selectedKernelName!);
              if (historyStr != null) {
                final historyItems = jsonDecode(historyStr) as List<dynamic>;
                selectedHistory = historyItems.map((h) {
                  final hm = h as Map<String, dynamic>;
                  return KernelExecution(
                    timestamp: DateTime.fromMillisecondsSinceEpoch(
                        (hm['timestamp_ms'] ?? 0) as int),
                    durationMs: (hm['duration_ms'] ?? 0).toDouble(),
                    throughputGbps: (hm['throughput_gbps'] ?? 0).toDouble(),
                    gflops: (hm['gflops'] ?? 0).toDouble(),
                    threadsUsed: (hm['threads_used'] ?? 0) as int,
                  );
                }).toList();
              }
            } catch (e) {
              debugPrint("History fetch failed for $_selectedKernelName: $e");
            }
          }

          topKernels = items.map((item) {
            final m = item as Map<String, dynamic>;
            final name = (m['name'] ?? 'kernel').toString();
            
            // Check for new history items to persist (authoritative timestamp comparison)
            if (name == _selectedKernelName && selectedHistory.isNotEmpty) {
                final lastSavedTs = _lastSelectedKernelStats?.history.isNotEmpty == true 
                    ? _lastSelectedKernelStats!.history.last.timestamp 
                    : DateTime.fromMillisecondsSinceEpoch(0);
                
                if (selectedHistory.last.timestamp.isAfter(lastSavedTs)) {
                    sqlite.saveKernelExecution(name, selectedHistory.last, _currentDeviceId);
                }
            }

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
            // Cache if this is the currently selected kernel
            if (k.name == _selectedKernelName) {
              _lastSelectedKernelStats = k;
            }
            return k;
          }).toList(growable: false);
        }
      } catch (e) {
        debugPrint('Profiler JSON parse failed: $e');
        topKernels = const [];
      }

      List<MemoryAllocation> allocations = const [];
      List<MemoryPool> memoryPools = const [];
      try {
        final memJson = bridge.getMemoryInfoJson();
        if (memJson != null) {
          final decoded = jsonDecode(memJson) as Map<String, dynamic>;
          final allocs = decoded['allocations'] as List<dynamic>? ?? [];
          allocations = allocs.map<MemoryAllocation>((a) {
            final m = a as Map<String, dynamic>;
            return MemoryAllocation(
              ptr: (m['ptr'] ?? '0x0').toString(),
              size: (m['size'] ?? 0) as int,
              isManaged: (m['managed'] ?? false) as bool,
              isResident: (m['resident'] ?? false) as bool,
              deviceId: (m['device'] ?? 0) as int,
            );
          }).toList(growable: false);

          final pools = decoded['pools'] as List<dynamic>? ?? [];
          memoryPools = pools.map<MemoryPool>((p) {
            final m = p as Map<String, dynamic>;
            return MemoryPool(
              id: (m['id'] ?? 0) as int,
              blockSize: (m['blockSize'] ?? 0) as int,
              totalAllocated: (m['total'] ?? 0) as int,
              peakAllocated: (m['peak'] ?? 0) as int,
              activeCount: (m['active'] ?? 0) as int,
              freeCount: (m['free'] ?? 0) as int,
            );
          }).toList(growable: false);
        }
      } catch (e) {
        debugPrint('Memory JSON parse failed: $e');
      }

      // Dynamic Smoothing: React faster to spikes, smoother for stable regions
      double alpha = 0.25;
      if (_lastSmoothed != null) {
        final gflopsDiff = (raw.gflops - _lastSmoothed!.gflops).abs();
        if (gflopsDiff > 10.0) alpha = 0.55; // Fast reaction to heavy workloads
      }

      final smoothed = Telemetry(
        timestamp: DateTime.fromMillisecondsSinceEpoch(raw.timestamp),
        gflops: _lastSmoothed == null
            ? raw.gflops
            : _lastSmoothed!.gflops * (1 - alpha) + raw.gflops * alpha,
        maxGflops: raw.maxGflops,
        computeUtilization: _lastSmoothed == null
            ? raw.computeUtilization
            : _lastSmoothed!.computeUtilization * (1 - alpha) +
                raw.computeUtilization * alpha,
        memoryBandwidth: _lastSmoothed == null
            ? raw.memoryBandwidthGbps
            : _lastSmoothed!.memoryBandwidth * (1 - alpha) +
                raw.memoryBandwidthGbps * alpha,
        maxMemoryBandwidth: raw.maxMemoryBandwidthGbps,
        memoryBusUtilization: _lastSmoothed == null
            ? raw.memoryBusUtilization
            : _lastSmoothed!.memoryBusUtilization * (1 - alpha) +
                raw.memoryBusUtilization * alpha,
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
        backgroundComputeActive: _backgroundComputeActive,
        serviceModeActive: _serviceModeActive,
        blockThreadsActive: _blockThreadsActive,
        deviceName: _deviceName,
        versionMajor: raw.versionMajor,
        versionMinor: raw.versionMinor,
        logs: logs,
        topKernels: topKernels,
        computeQuality: MetricQuality.measured,
        memoryQuality: MetricQuality.measured,
        uvmQuality: MetricQuality.measured,
        temperatureQuality: MetricQuality.measured,
        clusterNodes: clusterNodes,
        securityInfo: securityInfo,
        creditLedger: creditLedger,
        profilerEnabled: _profilerEnabled,
        backendVersion: _backendVersion,
        clusterSecurityActive: _clusterSecurityActive,
        clusterSecuritySupported: _clusterSecuritySupported,
        lastSelectedKernelStats: _lastSelectedKernelStats,
        allocations: allocations,
        memoryPools: memoryPools,
        deviceCount: _deviceCount,
      );
      add(UpdateTelemetry(smoothed));
    } catch (e) {
      debugPrint('FFI Error: $e');
    } finally {
      calloc.free(ptr);
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
        prev.computeUtilization * (1.0 - alpha) + current.computeUtilization * alpha;
    final smoothedMem =
        prev.memoryBusUtilization * (1.0 - alpha) + current.memoryBusUtilization * alpha;
    final smoothedTemp =
        prev.temperature * (1.0 - alpha) + current.temperature * alpha;

    return Telemetry(
      timestamp: current.timestamp,
      gflops: current.gflops,
      maxGflops: current.maxGflops,
      computeUtilization: smoothedCompute,
      memoryBandwidth: current.memoryBandwidth,
      maxMemoryBandwidth: current.maxMemoryBandwidth,
      memoryBusUtilization: smoothedMem,
      memoryUsed: current.memoryUsed,
      memoryTotal: current.memoryTotal,
      totalPages: current.totalPages,
      residentPages: current.residentPages,
      evictedPages: current.evictedPages,
      pageFaultRate: current.pageFaultRate,
      uvmMap: current.uvmMap,
      activeKernels: current.activeKernels,
      activeThreads: current.activeThreads,
      clockSpeed: current.clockSpeed,
      avgLatency: current.avgLatency,
      temperature: smoothedTemp,
      eccEnabled: current.eccEnabled,
      backgroundComputeActive: current.backgroundComputeActive,
      serviceModeActive: current.serviceModeActive,
      blockThreadsActive: current.blockThreadsActive,
      deviceName: current.deviceName,
      versionMajor: current.versionMajor,
      versionMinor: current.versionMinor,
      logs: current.logs,
      topKernels: current.topKernels,
      clusterNodes: current.clusterNodes,
      securityInfo: current.securityInfo,
      creditLedger: current.creditLedger,
      profilerEnabled: current.profilerEnabled,
      backendVersion: current.backendVersion,
      clusterSecurityActive: current.clusterSecurityActive,
      computeQuality: current.computeQuality,
      memoryQuality: current.memoryQuality,
      uvmQuality: current.uvmQuality,
      temperatureQuality: current.temperatureQuality,
      lastSelectedKernelStats: current.lastSelectedKernelStats,
      allocations: current.allocations,
      memoryPools: current.memoryPools,
    );
  }
}
