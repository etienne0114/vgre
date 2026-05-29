// Web stub for VgreBridge — dart:ffi is not available on web.
// All methods return safe defaults so the app compiles for web targets.
// On native platforms, vgre_ffi.dart is used instead via the conditional
// export in vgre_bridge.dart.

// Stub Telemetry structure — matches VgreTelemetry fields used by the bloc.
class VgreTelemetry {
  int timestamp = 0;
  int versionMajor = 0;
  int versionMinor = 0;
  int versionPatch = 0;
  double gflops = 0;
  double maxGflops = 0;
  double computeUtilization = 0;
  double memoryBandwidthGbps = 0;
  double maxMemoryBandwidthGbps = 0;
  double memoryBusUtilization = 0;
  int memoryUsedBytes = 0;
  int memoryTotalBytes = 0;
  int totalPages = 0;
  int residentPages = 0;
  int evictedPages = 0;
  double pageFaultRate = 0;
  int activeKernels = 0;
  int activeThreads = 0;
  double deviceClockMhz = 0;
  double avgKernelLatencyMs = 0;
  double deviceTemperature = 0;
  int eccEnabled = 0;
  int backgroundComputeActive = 0;
  List<int> uvmMap = List.filled(1024, 0);
}

// Stub bridge — no native library is loaded on web.
class VgreBridge {
  VgreBridge(String libPath);

  void setEnvironmentVariable(String name, String value) {}

  int getDeviceCount() => 0;
  int init() => 0;
  int shutdown() => 0;
  String getVersion() => '0.0.0-web';
  int setBackgroundCompute(bool enabled) => 0;
  int setServiceMode(bool isMaster) => 0;
  int setBlockThreads(bool enabled) => 0;
  int setProfilerEnabled(bool enabled) => 0;

  Map<String, dynamic> getDeviceProperties(int deviceId) => {
        'name': 'Web (no native device)',
        'clockRate': 0,
        'totalGlobalMem': 0,
      };

  VgreTelemetry getTelemetryWith(dynamic ptr) => VgreTelemetry();

  List<String> getLogs() => [];

  String? getProfilerJson({int topN = 5}) => null;
  String? getMemoryInfoJson() => null;

  List<Map<String, dynamic>> getClusterNodes() => [];
  int clusterSetSecurity(bool enabled) => 0;
  Map<String, dynamic> getSecurityInfo() => {
        'cipherName': '',
        'keyFingerprint': '',
        'sessionSeconds': 0.0,
        'isEncrypted': false,
        'packetsSent': 0,
        'packetsReceived': 0,
        'bytesSent': 0,
        'bytesReceived': 0,
      };

  List<Map<String, dynamic>> getCreditsAll() => [];
  int creditsReset() => 0;

  Map<String, dynamic>? getCacheStats() => null;
  void resetCacheStats() {}

  String? getKernelHistoryJson(String kernelName) => null;
}
