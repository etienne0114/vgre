import 'dart:async';
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

  TelemetryBloc(this.bridge) : super(TelemetryInitial()) {
    on<StartPolling>((event, emit) {
      // Initialize as master service (Dashboard)
      try {
        bridge.setServiceMode(true);
      } catch (e) {
        debugPrint("Failed to start VGRE IPC Service: $e");
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
              deviceName: _deviceName,
              versionMajor: raw.versionMajor,
              versionMinor: raw.versionMinor,
              logs: logs,
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
        _backgroundComputeActive = event.enabled;
        bridge.setBackgroundCompute(event.enabled);
      } catch (e) {
        debugPrint("Failed to toggle background compute: $e");
        _backgroundComputeActive = !event.enabled; // Revert on failure
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

    on<StopPolling>((event, emit) {
      _timer?.cancel();
    });

    on<UpdateTelemetry>((event, emit) {
      final List<Telemetry> newHistory = List.from(
        (state is TelemetryActive) ? (state as TelemetryActive).history : [],
      );
      newHistory.add(event.telemetry);
      if (newHistory.length > 50) newHistory.removeAt(0);

      emit(TelemetryActive(telemetry: event.telemetry, history: newHistory));
    });
  }

  @override
  Future<void> close() {
    _timer?.cancel();
    return super.close();
  }
}
