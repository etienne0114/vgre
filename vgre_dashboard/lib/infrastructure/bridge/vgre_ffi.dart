import 'dart:ffi';
import 'dart:convert';
import 'package:ffi/ffi.dart';

// ── VGRE C API Types ───────────────────────────────────────────────────────
final class VgreTelemetry extends Struct {
  @Uint64()
  external int timestamp;
  @Uint64()
  external int versionMajor;
  @Uint64()
  external int versionMinor;

  @Double()
  external double gflops;
  @Double()
  external double maxGflops;
  @Double()
  external double computeUtilization;

  @Double()
  external double memoryBandwidthGbps;
  @Double()
  external double maxMemoryBandwidthGbps;
  @Double()
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
  @Double()
  external double pageFaultRate;

  @Int64()
  external int activeKernels;
  @Int64()
  external int activeThreads;

  @Double()
  external double deviceClockMhz;
  @Double()
  external double avgKernelLatencyMs;
  @Double()
  external double deviceTemperature;
  @Int64()
  external int eccEnabled;
  @Int64()
  external int backgroundComputeActive;

  @Array(1024)
  external Array<Uint8> uvmMap;
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

final class VgreClusterNode extends Struct {
  @Array(64)
  external Array<Uint8> address;
  @Int32()
  external int port;
  @Int32()
  external int cpuCores;
  @Uint64()
  external int memoryBytes;
  @Double()
  external double latencyMs;
  @Int32()
  external int available;
  @Array(64)
  external Array<Uint8> igpuName;
}

final class VgreSecurityInfo extends Struct {
  @Array(64)
  external Array<Uint8> cipherName;
  @Array(65)
  external Array<Uint8> keyFingerprint;
  @Double()
  external double sessionSeconds;
  @Int32()
  external int isEncrypted;
  @Uint64()
  external int packetsSent;
  @Uint64()
  external int packetsReceived;
  @Uint64()
  external int bytesSent;
  @Uint64()
  external int bytesReceived;
}

final class VgreCreditInfo extends Struct {
  @Array(64)
  external Array<Uint8> address;
  @Double()
  external double totalCredits;
  @Double()
  external double totalDebits;
  @Double()
  external double balance;
  @Uint64()
  external int lastActivity;
  @Int32()
  external int transactionCount;
}


typedef InitFunc = Int32 Function();
typedef Init = int Function();

typedef ShutdownFunc = Int32 Function();
typedef Shutdown = int Function();

typedef GetVersionFunc = Pointer<Utf8> Function();
typedef GetVersion = Pointer<Utf8> Function();

typedef SetBackgroundComputeFunc = Int32 Function(Int32);
typedef SetBackgroundCompute = int Function(int);

typedef SetServiceModeFunc = Int32 Function(Int32);
typedef SetServiceMode = int Function(int);

typedef SetBlockThreadsFunc = Int32 Function(Int32);
typedef SetBlockThreads = int Function(int);

typedef GetTelemetryFunc = Int32 Function(Pointer<VgreTelemetry>);
typedef GetTelemetry = int Function(Pointer<VgreTelemetry>);
typedef GetMemoryInfoJsonFunc = Int32 Function(Pointer<Pointer<Utf8>>);
typedef GetMemoryInfoJson = int Function(Pointer<Pointer<Utf8>>);

typedef GetDevicePropertiesFunc =
    Int32 Function(Int32, Pointer<VgreDeviceProperties>);
typedef GetDeviceProperties = int Function(int, Pointer<VgreDeviceProperties>);

typedef GetLogsFunc =
    Int32 Function(Pointer<Pointer<Pointer<Utf8>>>, Pointer<Int32>);
typedef GetLogs = int Function(Pointer<Pointer<Pointer<Utf8>>>, Pointer<Int32>);

typedef FreeLogsFunc = Void Function(Pointer<Pointer<Utf8>>, Int32);
typedef FreeLogs = void Function(Pointer<Pointer<Utf8>>, int);

typedef GetProfilerJsonFunc = Int32 Function(Pointer<Pointer<Utf8>>, Int32);
typedef GetProfilerJson = int Function(Pointer<Pointer<Utf8>>, int);

typedef GetKernelHistoryJsonFunc = Int32 Function(
    Pointer<Utf8>, Pointer<Pointer<Utf8>>);
typedef GetKernelHistoryJson = int Function(
    Pointer<Utf8>, Pointer<Pointer<Utf8>>);

typedef FreeStringFunc = Void Function(Pointer<Utf8>);
typedef FreeString = void Function(Pointer<Utf8>);

typedef SetProfilerEnabledFunc = Int32 Function(Int32);
typedef SetProfilerEnabled = int Function(int);

typedef GetClusterNodesFunc = Int32 Function(Pointer<VgreClusterNode>, Pointer<Int32>);
typedef GetClusterNodes = int Function(Pointer<VgreClusterNode>, Pointer<Int32>);


typedef ClusterSetSecurityFunc = Int32 Function(Int32);
typedef ClusterSetSecurity = int Function(int);

typedef GetSecurityInfoFunc = Int32 Function(Pointer<VgreSecurityInfo>);
typedef GetSecurityInfo = int Function(Pointer<VgreSecurityInfo>);

typedef GetCreditsAllFunc = Int32 Function(Pointer<VgreCreditInfo>, Pointer<Int32>);
typedef GetCreditsAll = int Function(Pointer<VgreCreditInfo>, Pointer<Int32>);

typedef CreditsResetFunc = Int32 Function();
typedef CreditsReset = int Function();


// ── VGRE FFI Bridge ────────────────────────────────────────────────────────
class VgreBridge {
  late final DynamicLibrary _lib;
  late final Init _init;
  late final Shutdown _shutdown;
  late final GetTelemetry _getTelemetry;
  late final GetDeviceProperties _getDeviceProperties;
  late final GetLogs _getLogs;
  late final FreeLogs _freeLogs;
  late final GetProfilerJson _getProfilerJson;
  late final GetKernelHistoryJson _getKernelHistoryJson;
  late final FreeString _freeString;
  late final SetProfilerEnabled _setProfilerEnabled;
  late final GetMemoryInfoJson _getMemoryInfoJson;
  late final GetVersion _getVersion;
  late final SetBackgroundCompute _setBackgroundCompute;
  late final SetServiceMode _setServiceMode;
  late final SetBlockThreads _setBlockThreads;
  late final GetClusterNodes _getClusterNodes;
  late final ClusterSetSecurity _clusterSetSecurity;
  late final GetSecurityInfo _getSecurityInfo;
  late final GetCreditsAll _getCreditsAll;
  late final CreditsReset _creditsReset;


  VgreBridge(String libPath) {
    _lib = DynamicLibrary.open(libPath);
    _init = _lib.lookupFunction<InitFunc, Init>('vgre_init');
    _shutdown = _lib.lookupFunction<ShutdownFunc, Shutdown>('vgre_shutdown');
    _getTelemetry = _lib.lookupFunction<GetTelemetryFunc, GetTelemetry>(
      'vgre_get_telemetry',
    );
    _getDeviceProperties = _lib
        .lookupFunction<GetDevicePropertiesFunc, GetDeviceProperties>(
          'vgre_get_device_properties',
        );
    _getLogs = _lib.lookupFunction<GetLogsFunc, GetLogs>('vgre_get_logs');
    _freeLogs = _lib.lookupFunction<FreeLogsFunc, FreeLogs>('vgre_free_logs');
    _getProfilerJson =
        _lib.lookupFunction<GetProfilerJsonFunc, GetProfilerJson>(
          'vgre_get_profiler_json',
        );
    _getKernelHistoryJson = _lib.lookupFunction<GetKernelHistoryJsonFunc,
        GetKernelHistoryJson>('vgre_get_kernel_history_json');
    _freeString =
        _lib.lookupFunction<FreeStringFunc, FreeString>('vgre_free_string');
    _setProfilerEnabled =
        _lib.lookupFunction<SetProfilerEnabledFunc, SetProfilerEnabled>(
          'vgre_set_profiler_enabled',
        );
    _getMemoryInfoJson =
        _lib.lookupFunction<GetMemoryInfoJsonFunc, GetMemoryInfoJson>(
          'vgre_get_memory_info_json',
        );
    _getVersion = _lib.lookupFunction<GetVersionFunc, GetVersion>(
      'vgre_get_version',
    );
    _setBackgroundCompute = _lib
        .lookupFunction<SetBackgroundComputeFunc, SetBackgroundCompute>(
          'vgre_set_background_compute',
        );
    _setServiceMode = _lib.lookupFunction<SetServiceModeFunc, SetServiceMode>(
      'vgre_set_service_mode',
    );
    _setBlockThreads =
        _lib.lookupFunction<SetBlockThreadsFunc, SetBlockThreads>(
          'vgre_set_block_threads',
        );
    _getClusterNodes = _lib.lookupFunction<GetClusterNodesFunc, GetClusterNodes>(
        'vgre_get_cluster_nodes');
    _clusterSetSecurity = _lib.lookupFunction<ClusterSetSecurityFunc, ClusterSetSecurity>(
        'vgre_cluster_set_security');
    _getSecurityInfo = _lib.lookupFunction<GetSecurityInfoFunc, GetSecurityInfo>(
        'vgre_cluster_get_security_info');
    _getCreditsAll = _lib.lookupFunction<GetCreditsAllFunc, GetCreditsAll>(
        'vgre_credits_get_all');
    _creditsReset = _lib.lookupFunction<CreditsResetFunc, CreditsReset>(
        'vgre_credits_reset');
  }


  int init() => _init();
  int shutdown() => _shutdown();
  String getVersion() => _getVersion().toDartString();
  int setBackgroundCompute(bool enabled) =>
      _setBackgroundCompute(enabled ? 1 : 0);
  int setServiceMode(bool isMaster) => _setServiceMode(isMaster ? 1 : 0);
  int setBlockThreads(bool enabled) => _setBlockThreads(enabled ? 1 : 0);
  int setProfilerEnabled(bool enabled) => _setProfilerEnabled(enabled ? 1 : 0);

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

  String? getProfilerJson({int topN = 5}) {
    final outPtr = calloc<Pointer<Utf8>>();
    try {
      final res = _getProfilerJson(outPtr, topN);
      if (res != 0) throw Exception('Failed to get profiler json: $res');
      if (outPtr.value == nullptr) return null;
      final jsonStr = outPtr.value.toDartString();
      _freeString(outPtr.value);
      return jsonStr;
    } finally {
      calloc.free(outPtr);
    }
  }

  String? getMemoryInfoJson() {
    final outPtr = calloc<Pointer<Utf8>>();
    try {
      final res = _getMemoryInfoJson(outPtr);
      if (res != 0) throw Exception('Failed to get memory info json: $res');
      if (outPtr.value == nullptr) return null;
      final jsonStr = outPtr.value.toDartString();
      _freeString(outPtr.value);
      return jsonStr;
    } finally {
      calloc.free(outPtr);
    }
  }

  List<Map<String, dynamic>> getClusterNodes() {
    final countPtr = calloc<Int32>();
    countPtr.value = 32; // Max 32 nodes for now
    final nodesPtr = calloc<VgreClusterNode>(32);

    try {
      final res = _getClusterNodes(nodesPtr, countPtr);
      if (res != 0) throw Exception('Failed to get cluster nodes: $res');

      final count = countPtr.value;
      final List<Map<String, dynamic>> nodes = [];

      for (int i = 0; i < count; i++) {
        final node = nodesPtr[i];
        final List<int> addrBytes = [];
        for (int j = 0; j < 64; j++) {
          if (node.address[j] == 0) break;
          addrBytes.add(node.address[j]);
        }

        final List<int> igpuBytes = [];
        for (int j = 0; j < 64; j++) {
          if (node.igpuName[j] == 0) break;
          igpuBytes.add(node.igpuName[j]);
        }

        nodes.add({
          'address': utf8.decode(addrBytes),
          'port': node.port,
          'cpuCores': node.cpuCores,
          'memoryBytes': node.memoryBytes,
          'latencyMs': node.latencyMs,
          'available': node.available != 0,
          'igpuName': utf8.decode(igpuBytes),
        });
      }
      return nodes;
    } finally {
      calloc.free(countPtr);
      calloc.free(nodesPtr);
    }
  }

  int clusterSetSecurity(bool enabled) => _clusterSetSecurity(enabled ? 1 : 0);

  Map<String, dynamic> getSecurityInfo() {
    final ptr = calloc<VgreSecurityInfo>();
    try {
      final res = _getSecurityInfo(ptr);
      if (res != 0) throw Exception('Failed to get security info: $res');

      final info = ptr.ref;
      final cipherBytes = <int>[];
      for (int i = 0; i < 64; i++) {
        if (info.cipherName[i] == 0) break;
        cipherBytes.add(info.cipherName[i]);
      }

      final fpBytes = <int>[];
      for (int i = 0; i < 65; i++) {
        if (info.keyFingerprint[i] == 0) break;
        fpBytes.add(info.keyFingerprint[i]);
      }

      return {
        'cipherName': utf8.decode(cipherBytes),
        'keyFingerprint': utf8.decode(fpBytes),
        'sessionSeconds': info.sessionSeconds,
        'isEncrypted': info.isEncrypted != 0,
        'packetsSent': info.packetsSent,
        'packetsReceived': info.packetsReceived,
        'bytesSent': info.bytesSent,
        'bytesReceived': info.bytesReceived,
      };
    } finally {
      calloc.free(ptr);
    }
  }

  List<Map<String, dynamic>> getCreditsAll() {
    final countPtr = calloc<Int32>();
    countPtr.value = 32;
    final nodesPtr = calloc<VgreCreditInfo>(32);

    try {
      final res = _getCreditsAll(nodesPtr, countPtr);
      if (res != 0) throw Exception('Failed to get credits: $res');

      final count = countPtr.value;
      final List<Map<String, dynamic>> credits = [];

      for (int i = 0; i < count; i++) {
        final info = nodesPtr[i];
        final addrBytes = <int>[];
        for (int j = 0; j < 64; j++) {
          if (info.address[j] == 0) break;
          addrBytes.add(info.address[j]);
        }

        credits.add({
          'address': utf8.decode(addrBytes),
          'totalCredits': info.totalCredits,
          'totalDebits': info.totalDebits,
          'balance': info.balance,
          'lastActivity': info.lastActivity,
          'transactionCount': info.transactionCount,
        });
      }
      return credits;
    } finally {
      calloc.free(countPtr);
      calloc.free(nodesPtr);
    }
  }

  int creditsReset() => _creditsReset();

  String? getKernelHistoryJson(String kernelName) {
    final namePtr = kernelName.toNativeUtf8();
    final outPtr = calloc<Pointer<Utf8>>();
    try {
      final res = _getKernelHistoryJson(namePtr, outPtr);
      if (res != 0) throw Exception('Failed to get kernel history: $res');
      if (outPtr.value == nullptr) return null;
      final jsonStr = outPtr.value.toDartString();
      _freeString(outPtr.value);
      return jsonStr;
    } finally {
      calloc.free(namePtr);
      calloc.free(outPtr);
    }
  }
}
