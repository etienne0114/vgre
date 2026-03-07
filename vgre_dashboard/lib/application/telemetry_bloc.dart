import 'dart:async';
import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:equatable/equatable.dart';
import 'dart:convert';
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

  TelemetryBloc(this.bridge) : super(TelemetryInitial()) {
    on<StartPolling>((event, emit) {
      // Initialize as master service (Dashboard)
      try {
        bridge.setServiceMode(true);
      } catch (e) {
        print("Failed to start VGRE IPC Service: $e");
      }

      // Fetch device info once
      try {
        final props = bridge.getDeviceProperties(0);
        _deviceName = props['name'] as String;
      } catch (e) {
        print("Failed to fetch device info: $e");
      }

      _timer?.cancel();
      _timer = Timer.periodic(const Duration(milliseconds: 500), (timer) {
        try {
          final ptr = calloc<VgreTelemetry>();
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
            backgroundComputeActive: _backgroundComputeActive, // Use local state
            deviceName: _deviceName,
            versionMajor: raw.versionMajor,
            versionMinor: raw.versionMinor,
            logs: logs,
          );
          calloc.free(ptr);
          add(UpdateTelemetry(data));
        } catch (e) {
          print('FFI Error: $e');
        }
      });
    });

    on<ToggleBackgroundCompute>((event, emit) {
      _backgroundComputeActive = event.enabled;
      bridge.setBackgroundCompute(event.enabled);
    });

    on<StopPolling>((event, emit) {
      _timer?.cancel();
    });

    on<UpdateTelemetry>((event, emit) {
      final List<Telemetry> newHistory = List.from(
          (state is TelemetryActive) ? (state as TelemetryActive).history : []);
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
