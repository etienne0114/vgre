import 'package:equatable/equatable.dart';

enum MetricQuality {
  measured,
}

class CacheStats extends Equatable {
  final int l2Hits;
  final int l2Misses;
  final int l2Evictions;
  final double l2HitRate;
  final int l1ConfigKb;
  final int l2ConfigMb;

  const CacheStats({
    this.l2Hits = 0,
    this.l2Misses = 0,
    this.l2Evictions = 0,
    this.l2HitRate = 0.0,
    this.l1ConfigKb = 32,
    this.l2ConfigMb = 6,
  });

  int get l2Total => l2Hits + l2Misses;
  double get l2HitPercent => l2HitRate * 100.0;

  @override
  List<Object?> get props =>
      [l2Hits, l2Misses, l2Evictions, l2HitRate, l1ConfigKb, l2ConfigMb];
}

extension MetricQualityLabel on MetricQuality {
  String get label {
    switch (this) {
      case MetricQuality.measured:
        return "MEASURED";
    }
  }
}

class SecurityInfo extends Equatable {
  final String cipherName;
  final String keyFingerprint;
  final double sessionSeconds;
  final bool isEncrypted;
  final int packetsSent;
  final int packetsReceived;
  final int bytesSent;
  final int bytesReceived;

  const SecurityInfo({
    required this.cipherName,
    required this.keyFingerprint,
    required this.sessionSeconds,
    required this.isEncrypted,
    required this.packetsSent,
    required this.packetsReceived,
    required this.bytesSent,
    required this.bytesReceived,
  });

  bool get isHandshakePending => cipherName.contains("PENDING");

  /// True when security_enabled_ is set on the C++ side (token provided,
  /// enableSecurity() called). False only when cipher is "NONE (plaintext)".
  bool get isSecurityEnabled =>
      cipherName.isNotEmpty && cipherName != "NONE (plaintext)";

  @override
  List<Object?> get props => [
        cipherName,
        keyFingerprint,
        sessionSeconds,
        isEncrypted,
        packetsSent,
        packetsReceived,
        bytesSent,
        bytesReceived,
      ];
}

class ClusterNode extends Equatable {
  final String address;
  final int port;
  final int cpuCores;
  final int memoryBytes;
  final double latencyMs;
  final bool available;
  final String igpuName;
  final String platform;   // node OS: "Linux"/"macOS"/"Windows" ("" if unknown)
  final String arch;       // node CPU arch: "x86_64"/"arm64"
  final String hostname;   // node hostname
  final int inFlightKernels;  // kernels dispatched here, not yet returned
  final int kernelsCompleted; // cumulative kernels this node has executed

  // Phase 5: Credits
  final double totalCredits;
  final double totalDebits;
  final double balance;
  final int transactionCount;

  // Disconnect tracking: set to the moment the node first went unavailable;
  // null while the node is connected.
  final DateTime? disconnectedAt;

  const ClusterNode({
    required this.address,
    required this.port,
    required this.cpuCores,
    required this.memoryBytes,
    required this.latencyMs,
    required this.available,
    required this.igpuName,
    this.platform = '',
    this.arch = '',
    this.hostname = '',
    this.inFlightKernels = 0,
    this.kernelsCompleted = 0,
    this.totalCredits = 0.0,
    this.totalDebits = 0.0,
    this.balance = 0.0,
    this.transactionCount = 0,
    this.disconnectedAt,
  });

  @override
  List<Object?> get props => [
        address,
        port,
        cpuCores,
        memoryBytes,
        latencyMs,
        available,
        igpuName,
        platform,
        arch,
        hostname,
        inFlightKernels,
        kernelsCompleted,
        totalCredits,
        totalDebits,
        balance,
        transactionCount,
        disconnectedAt,
      ];
}


class CreditEntry extends Equatable {
  final String address;
  final double totalCredits;
  final double totalDebits;
  final double balance;
  final int transactionCount;
  final int lastActivity;

  const CreditEntry({
    required this.address,
    required this.totalCredits,
    required this.totalDebits,
    required this.balance,
    required this.transactionCount,
    required this.lastActivity,
  });

  @override
  List<Object?> get props => [
        address,
        totalCredits,
        totalDebits,
        balance,
        transactionCount,
        lastActivity,
      ];
}

class MemoryAllocation extends Equatable {
  final String ptr;
  final int size;
  final bool isManaged;
  final bool isResident;
  final int deviceId;

  const MemoryAllocation({
    required this.ptr,
    required this.size,
    required this.isManaged,
    required this.isResident,
    required this.deviceId,
  });

  @override
  List<Object?> get props => [ptr, size, isManaged, isResident, deviceId];
}

class MemoryPool extends Equatable {
  final int id;
  final int blockSize;
  final int totalAllocated;
  final int peakAllocated;
  final int activeCount;
  final int freeCount;

  const MemoryPool({
    required this.id,
    required this.blockSize,
    required this.totalAllocated,
    required this.peakAllocated,
    required this.activeCount,
    required this.freeCount,
  });

  @override
  List<Object?> get props => [
        id,
        blockSize,
        totalAllocated,
        peakAllocated,
        activeCount,
        freeCount,
      ];
}


class Telemetry extends Equatable {
  final DateTime timestamp;
  
  // Compute
  final double gflops;
  final double maxGflops;
  final double computeUtilization;
  
  // Memory
  final double memoryBandwidth;
  final double maxMemoryBandwidth;
  final double memoryBusUtilization;
  final int memoryUsed;
  final int memoryTotal;
  
  // UVM
  final int totalPages;
  final int residentPages;
  final int evictedPages;
  final double pageFaultRate;
  final List<int> uvmMap;

  // Device
  final int activeKernels;
  final int activeThreads;
  final int clockSpeed;
  final double avgLatency;
  final double temperature;
  final bool eccEnabled;
  final bool backgroundComputeActive;
  final bool serviceModeActive;
  final bool blockThreadsActive;
  final String deviceName;
  final int versionMajor;
  final int versionMinor;
  final int versionPatch;
  final List<String> logs;
  final List<KernelStat> topKernels;
  final MetricQuality computeQuality;
  final MetricQuality memoryQuality;
  final MetricQuality uvmQuality;
  final MetricQuality temperatureQuality;
  final List<ClusterNode> clusterNodes;
  final SecurityInfo? securityInfo;
  final List<CreditEntry> creditLedger;
  final bool profilerEnabled;
  final String backendVersion;
  final bool clusterSecurityActive;
  final bool clusterSecuritySupported;
  final KernelStat? lastSelectedKernelStats;
  final List<MemoryAllocation> allocations;
  final List<MemoryPool> memoryPools;
  final int deviceCount;
  final CacheStats cacheStats;

  const Telemetry({
    required this.timestamp,
    required this.gflops,
    required this.maxGflops,
    required this.computeUtilization,
    required this.memoryBandwidth,
    required this.maxMemoryBandwidth,
    required this.memoryBusUtilization,
    required this.memoryUsed,
    required this.memoryTotal,
    required this.totalPages,
    required this.residentPages,
    required this.evictedPages,
    required this.pageFaultRate,
    required this.uvmMap,
    required this.activeKernels,
    required this.activeThreads,
    required this.clockSpeed,
    required this.avgLatency,
    required this.temperature,
    required this.eccEnabled,
    this.backgroundComputeActive = false,
    this.serviceModeActive = true,
    this.blockThreadsActive = false,
    this.deviceName = "VGRE_DEVICE",
    this.versionMajor = 0,
    this.versionMinor = 1,
    this.versionPatch = 1,
    this.logs = const [],
    this.topKernels = const [],
    this.computeQuality = MetricQuality.measured,
    this.memoryQuality = MetricQuality.measured,
    this.uvmQuality = MetricQuality.measured,
    this.temperatureQuality = MetricQuality.measured,
    this.clusterNodes = const [],
    this.securityInfo,
    this.creditLedger = const [],
    this.profilerEnabled = true,
    this.backendVersion = '0.0.0',
    this.clusterSecurityActive = false,
    this.clusterSecuritySupported = false,
    this.lastSelectedKernelStats,
    this.allocations = const [],
    this.memoryPools = const [],
    this.deviceCount = 1,
    this.cacheStats = const CacheStats(),
  });

  double get memoryUsagePercent =>
      memoryTotal > 0 ? (memoryUsed / memoryTotal) * 100 : 0;

  Telemetry copyWith({
    DateTime? timestamp,
    double? gflops,
    double? maxGflops,
    double? computeUtilization,
    double? memoryBandwidth,
    double? maxMemoryBandwidth,
    double? memoryBusUtilization,
    int? memoryUsed,
    int? memoryTotal,
    int? totalPages,
    int? residentPages,
    int? evictedPages,
    double? pageFaultRate,
    List<int>? uvmMap,
    int? activeKernels,
    int? activeThreads,
    int? clockSpeed,
    double? avgLatency,
    double? temperature,
    bool? eccEnabled,
    bool? backgroundComputeActive,
    bool? serviceModeActive,
    bool? blockThreadsActive,
    String? deviceName,
    int? versionMajor,
    int? versionMinor,
    int? versionPatch,
    List<String>? logs,
    List<KernelStat>? topKernels,
    MetricQuality? computeQuality,
    MetricQuality? memoryQuality,
    MetricQuality? uvmQuality,
    MetricQuality? temperatureQuality,
    List<ClusterNode>? clusterNodes,
    SecurityInfo? securityInfo,
    List<CreditEntry>? creditLedger,
    bool? profilerEnabled,
    String? backendVersion,
    bool? clusterSecurityActive,
    bool? clusterSecuritySupported,
    KernelStat? lastSelectedKernelStats,
    List<MemoryAllocation>? allocations,
    List<MemoryPool>? memoryPools,
    int? deviceCount,
    CacheStats? cacheStats,
  }) {
    return Telemetry(
      timestamp: timestamp ?? this.timestamp,
      gflops: gflops ?? this.gflops,
      maxGflops: maxGflops ?? this.maxGflops,
      computeUtilization: computeUtilization ?? this.computeUtilization,
      memoryBandwidth: memoryBandwidth ?? this.memoryBandwidth,
      maxMemoryBandwidth: maxMemoryBandwidth ?? this.maxMemoryBandwidth,
      memoryBusUtilization: memoryBusUtilization ?? this.memoryBusUtilization,
      memoryUsed: memoryUsed ?? this.memoryUsed,
      memoryTotal: memoryTotal ?? this.memoryTotal,
      totalPages: totalPages ?? this.totalPages,
      residentPages: residentPages ?? this.residentPages,
      evictedPages: evictedPages ?? this.evictedPages,
      pageFaultRate: pageFaultRate ?? this.pageFaultRate,
      uvmMap: uvmMap ?? this.uvmMap,
      activeKernels: activeKernels ?? this.activeKernels,
      activeThreads: activeThreads ?? this.activeThreads,
      clockSpeed: clockSpeed ?? this.clockSpeed,
      avgLatency: avgLatency ?? this.avgLatency,
      temperature: temperature ?? this.temperature,
      eccEnabled: eccEnabled ?? this.eccEnabled,
      backgroundComputeActive: backgroundComputeActive ?? this.backgroundComputeActive,
      serviceModeActive: serviceModeActive ?? this.serviceModeActive,
      blockThreadsActive: blockThreadsActive ?? this.blockThreadsActive,
      deviceName: deviceName ?? this.deviceName,
      versionMajor: versionMajor ?? this.versionMajor,
      versionMinor: versionMinor ?? this.versionMinor,
      versionPatch: versionPatch ?? this.versionPatch,
      logs: logs ?? this.logs,
      topKernels: topKernels ?? this.topKernels,
      computeQuality: computeQuality ?? this.computeQuality,
      memoryQuality: memoryQuality ?? this.memoryQuality,
      uvmQuality: uvmQuality ?? this.uvmQuality,
      temperatureQuality: temperatureQuality ?? this.temperatureQuality,
      clusterNodes: clusterNodes ?? this.clusterNodes,
      securityInfo: securityInfo ?? this.securityInfo,
      creditLedger: creditLedger ?? this.creditLedger,
      profilerEnabled: profilerEnabled ?? this.profilerEnabled,
      backendVersion: backendVersion ?? this.backendVersion,
      clusterSecurityActive: clusterSecurityActive ?? this.clusterSecurityActive,
      clusterSecuritySupported: clusterSecuritySupported ?? this.clusterSecuritySupported,
      lastSelectedKernelStats: lastSelectedKernelStats ?? this.lastSelectedKernelStats,
      allocations: allocations ?? this.allocations,
      memoryPools: memoryPools ?? this.memoryPools,
      deviceCount: deviceCount ?? this.deviceCount,
      cacheStats: cacheStats ?? this.cacheStats,
    );
  }

  @override
  List<Object?> get props => [
        timestamp,
        gflops,
        maxGflops,
        computeUtilization,
        memoryBandwidth,
        maxMemoryBandwidth,
        memoryBusUtilization,
        memoryUsed,
        memoryTotal,
        totalPages,
        residentPages,
        evictedPages,
        pageFaultRate,
        uvmMap,
        activeKernels,
        activeThreads,
        clockSpeed,
        avgLatency,
        temperature,
        eccEnabled,
        backgroundComputeActive,
        serviceModeActive,
        blockThreadsActive,
        deviceName,
        versionMajor,
        versionMinor,
        versionPatch,
        logs,
        topKernels,
        computeQuality,
        memoryQuality,
        uvmQuality,
        temperatureQuality,
        clusterNodes,
        securityInfo,
        creditLedger,
        profilerEnabled,
        backendVersion,
        clusterSecurityActive,
        clusterSecuritySupported,
        lastSelectedKernelStats,
        allocations,
        memoryPools,
        deviceCount,
        cacheStats,
      ];
}


class KernelExecution extends Equatable {
  final DateTime timestamp;
  final double durationMs;
  final double throughputGbps;
  final double gflops;
  final int threadsUsed;

  const KernelExecution({
    required this.timestamp,
    required this.durationMs,
    required this.throughputGbps,
    required this.gflops,
    required this.threadsUsed,
  });

  @override
  List<Object?> get props => [
        timestamp,
        durationMs,
        throughputGbps,
        gflops,
        threadsUsed,
      ];
}

class KernelStat extends Equatable {
  final String name;
  final int invocations;
  final double totalTimeMs;
  final double avgTimeMs;
  final double minTimeMs;
  final double maxTimeMs;
  final double avgThroughputGbps;
  final double avgGflops;
  final String sourceCode;
  final String irCode;
  final List<KernelExecution> history;

  const KernelStat({
    required this.name,
    required this.invocations,
    required this.totalTimeMs,
    required this.avgTimeMs,
    required this.minTimeMs,
    required this.maxTimeMs,
    required this.avgThroughputGbps,
    required this.avgGflops,
    required this.sourceCode,
    required this.irCode,
    this.history = const [],
  });

  @override
  List<Object?> get props => [
        name,
        invocations,
        totalTimeMs,
        avgTimeMs,
        minTimeMs,
        maxTimeMs,
        avgThroughputGbps,
        avgGflops,
        sourceCode,
        irCode,
        history,
      ];
}
