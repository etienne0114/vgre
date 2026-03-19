import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'package:flutter/foundation.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:equatable/equatable.dart';
import '../domain/models/telemetry.dart';
import '../infrastructure/bridge/vgre_ffi.dart';

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

class UpdateTelemetry extends TelemetryEvent {
  final Telemetry telemetry;
  const UpdateTelemetry(this.telemetry);
  @override
  List<Object?> get props => [telemetry];
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
  const TelemetryActive({required this.telemetry, required this.history});
  @override
  List<Object?> get props => [telemetry, history];
}

// ── BLoC ───────────────────────────────────────────────────────────────────
class TelemetryBloc extends Bloc<TelemetryEvent, TelemetryState> {
  final VgreBridge bridge;
  Timer? _timer;

  String _deviceName = "VGRE_VIRTUAL_GPU";
  bool _backgroundComputeActive = false;
  bool _serviceModeActive = true;
  bool _blockThreadsActive = false;
  Telemetry? _lastSmoothed;

  TelemetryBloc(this.bridge) : super(TelemetryInitial()) {
    on<StartPolling>((event, emit) {
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

      // Fetch device info once
      try {
        final props = bridge.getDeviceProperties(0);
        _deviceName = props['name'] as String;
      } catch (e) {
        debugPrint("Failed to fetch device info: $e");
      }

      _timer?.cancel();
      _timer = Timer.periodic(const Duration(milliseconds: 500), (timer) {
          try {
            final ptr = calloc<VgreTelemetry>();
          try {
            final raw = bridge.getTelemetryWith(ptr);
            final logs = bridge.getLogs();
            List<KernelStat> topKernels = const [];
            try {
              final jsonStr = bridge.getProfilerJson(topN: 5);
              if (jsonStr != null) {
                final decoded = jsonDecode(jsonStr) as Map<String, dynamic>;
                final items = decoded['top_kernels'] as List<dynamic>? ?? [];
                topKernels = items.map((item) {
                  final m = item as Map<String, dynamic>;
                  return KernelStat(
                    name: (m['name'] ?? 'kernel').toString(),
                    invocations: (m['invocations'] ?? 0) as int,
                    totalTimeMs: (m['total_time_ms'] ?? 0).toDouble(),
                    avgTimeMs: (m['avg_time_ms'] ?? 0).toDouble(),
                    avgThroughputGbps:
                        (m['avg_throughput_gbps'] ?? 0).toDouble(),
                    avgGflops: (m['avg_gflops'] ?? 0).toDouble(),
                  );
                }).toList(growable: false);
              }
            } catch (e) {
              debugPrint('Profiler JSON parse failed: $e');
              topKernels = const [];
            }

          final bool hasProfilerStats = topKernels.isNotEmpty;
          final data = Telemetry(
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
              backgroundComputeActive:
                  _backgroundComputeActive, // Use local state
              serviceModeActive: _serviceModeActive,
              blockThreadsActive: _blockThreadsActive,
              deviceName: _deviceName,
              versionMajor: raw.versionMajor,
              versionMinor: raw.versionMinor,
              logs: logs,
              topKernels: topKernels,
              computeQuality: hasProfilerStats
                  ? MetricQuality.measured
                  : MetricQuality.estimated,
              memoryQuality: hasProfilerStats
                  ? MetricQuality.measured
                  : MetricQuality.estimated,
              uvmQuality: MetricQuality.simulated,
              temperatureQuality: MetricQuality.estimated,
            );
            add(UpdateTelemetry(data));
          } finally {
            calloc.free(ptr);
          }
        } catch (e) {
          debugPrint('FFI Error: $e');
        }
      });
    });

    on<ToggleBackgroundCompute>((event, emit) {
      try {
        final res = bridge.setBackgroundCompute(event.enabled);
        if (res == 0) {
          _backgroundComputeActive = event.enabled;
        } else {
          debugPrint("Failed to toggle background compute: $res");
        }
      } catch (e) {
        debugPrint("Failed to toggle background compute: $e");
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

    on<StopPolling>((event, emit) {
      _timer?.cancel();
    });

    on<UpdateTelemetry>((event, emit) {
      final List<Telemetry> newHistory = List.from(
        (state is TelemetryActive) ? (state as TelemetryActive).history : [],
      );
      final smoothed = _smoothTelemetry(_lastSmoothed, event.telemetry);
      _lastSmoothed = smoothed;
      newHistory.add(smoothed);
      if (newHistory.length > 50) newHistory.removeAt(0);

      emit(TelemetryActive(telemetry: event.telemetry, history: newHistory));
    });
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
    );
  }
}
