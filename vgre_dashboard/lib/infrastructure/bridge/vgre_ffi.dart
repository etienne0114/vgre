import 'dart:ffi';
import 'dart:io';
import 'dart:convert';
import 'package:ffi/ffi.dart';

// ── VGRE C API Types ───────────────────────────────────────────────────────
final class VgreTelemetry extends Struct {
  @Int64()
  external int timestamp;

  @Float()
  external double gflops;
  @Float()
  external double maxGflops;
  @Float()
  external double computeUtilization;

  @Float()
  external double memoryBandwidthGbps;
  @Float()
  external double maxMemoryBandwidthGbps;
  @Float()
  external double memoryBusUtilization;

  @Uint64()
  external int memoryUsedBytes;
  @Uint64()
  external int memoryTotalBytes;

  @Uint64()
  external int totalPages;
  @Uint64()
  external int residentPages;
  @Uint64()
  external int evictedPages;
  @Float()
  external double pageFaultRate;

  @Array(1024)
  external Array<Uint8> uvmMap;

  @Int32()
  external int activeKernels;
  @Int32()
  external int activeThreads;

  @Float()
  external double deviceClockMhz;
  @Float()
  external double avgKernelLatencyMs;
  @Float()
  external double deviceTemperature;
  @Int32()
  external int eccEnabled;
  @Int32()
  external int simulationEnabled;
}

final class VgreDeviceProperties extends Struct {
  @Array(256)
  external Array<Uint8> name;
  @Uint64()
  external int totalGlobalMem;
  @Uint64()
  external int sharedMemPerBlock;
  @Int32()
  external int maxThreadsPerBlock;
  @Array(3)
  external Array<Int32> maxThreadsDim;
  @Array(3)
  external Array<Int32> maxGridSize;
  @Int32()
  external int warpSize;
  @Int32()
  external int multiProcessorCount;
  @Int32()
  external int major;
  @Int32()
  external int minor;
  @Int32()
  external int clockRate;
  @Uint64()
  external int totalConstMem;
}

typedef InitFunc = Int32 Function();
typedef Init = int Function();

typedef ShutdownFunc = Int32 Function();
typedef Shutdown = int Function();

typedef GetVersionFunc = Pointer<Utf8> Function();
typedef GetVersion = Pointer<Utf8> Function();

typedef SetSimulationModeFunc = Int32 Function(Int32);
typedef SetSimulationMode = int Function(int);

typedef SetServiceModeFunc = Int32 Function(Int32);
typedef SetServiceMode = int Function(int);

typedef GetTelemetryFunc = Int32 Function(Pointer<VgreTelemetry>);
typedef GetTelemetry = int Function(Pointer<VgreTelemetry>);

typedef GetDevicePropertiesFunc = Int32 Function(
    Int32, Pointer<VgreDeviceProperties>);
typedef GetDeviceProperties = int Function(int, Pointer<VgreDeviceProperties>);

typedef GetLogsFunc = Int32 Function(
    Pointer<Pointer<Pointer<Utf8>>>, Pointer<Int32>);
typedef GetLogs = int Function(Pointer<Pointer<Pointer<Utf8>>>, Pointer<Int32>);

typedef FreeLogsFunc = Void Function(Pointer<Pointer<Utf8>>, Int32);
typedef FreeLogs = void Function(Pointer<Pointer<Utf8>>, int);

// ── VGRE FFI Bridge ────────────────────────────────────────────────────────
class VgreBridge {
  late final DynamicLibrary _lib;
  late final Init _init;
  late final Shutdown _shutdown;
  late final GetTelemetry _getTelemetry;
  late final GetDeviceProperties _getDeviceProperties;
  late final GetLogs _getLogs;
  late final FreeLogs _freeLogs;
  late final GetVersion _getVersion;
  late final SetSimulationMode _setSimulationMode;
  late final SetServiceMode _setServiceMode;

  VgreBridge(String libPath) {
    _lib = DynamicLibrary.open(libPath);
    _init = _lib.lookupFunction<InitFunc, Init>('vgre_init');
    _shutdown = _lib.lookupFunction<ShutdownFunc, Shutdown>('vgre_shutdown');
    _getTelemetry =
        _lib.lookupFunction<GetTelemetryFunc, GetTelemetry>('vgre_get_telemetry');
    _getDeviceProperties = _lib.lookupFunction<GetDevicePropertiesFunc,
        GetDeviceProperties>('vgre_get_device_properties');
    _getLogs = _lib.lookupFunction<GetLogsFunc, GetLogs>('vgre_get_logs');
    _freeLogs = _lib.lookupFunction<FreeLogsFunc, FreeLogs>('vgre_free_logs');
    _getVersion =
        _lib.lookupFunction<GetVersionFunc, GetVersion>('vgre_get_version');
    _setSimulationMode = _lib.lookupFunction<SetSimulationModeFunc,
        SetSimulationMode>('vgre_set_simulation_mode');
    _setServiceMode = _lib.lookupFunction<SetServiceModeFunc,
        SetServiceMode>('vgre_set_service_mode');
  }

  int init() => _init();
  int shutdown() => _shutdown();
  String getVersion() => _getVersion().toDartString();
  int setSimulationMode(bool enabled) => _setSimulationMode(enabled ? 1 : 0);
  int setServiceMode(bool isMaster) => _setServiceMode(isMaster ? 1 : 0);

  Map<String, dynamic> getDeviceProperties(int deviceId) {
    final ptr = calloc<VgreDeviceProperties>();
    try {
      final res = _getDeviceProperties(deviceId, ptr);
      if (res != 0) throw Exception('Failed to get device properties: $res');
      
      final props = ptr.ref;
      final List<int> nameBytes = [];
      for (int i = 0; i < 256; i++) {
        if (props.name[i] == 0) break;
        nameBytes.add(props.name[i]);
      }
      
      return {
        'name': utf8.decode(nameBytes),
        'clockRate': props.clockRate,
        'totalGlobalMem': props.totalGlobalMem,
      };
    } finally {
      calloc.free(ptr);
    }
  }

  VgreTelemetry getTelemetryWith(Pointer<VgreTelemetry> ptr) {
    final res = _getTelemetry(ptr);
    if (res != 0) throw Exception('Failed to get telemetry: $res');
    return ptr.ref;
  }

  List<String> getLogs() {
    final countPtr = calloc<Int32>();
    final bufferPtr = calloc<Pointer<Pointer<Utf8>>>();

    try {
      final res = _getLogs(bufferPtr, countPtr);
      if (res != 0) throw Exception('Failed to get logs: $res');

      final count = countPtr.value;
      if (count == 0 || bufferPtr.value == nullptr) return [];

      final List<String> logs = [];
      final lines = bufferPtr.value;
      for (int i = 0; i < count; i++) {
        logs.add(lines[i].toDartString());
      }

      _freeLogs(lines, count);
      return logs;
    } finally {
      calloc.free(countPtr);
      calloc.free(bufferPtr);
    }
  }
}
