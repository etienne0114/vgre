import 'package:equatable/equatable.dart';

enum MetricQuality {
  measured,
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
  
  // Phase 5: Credits
  final double totalCredits;
  final double totalDebits;
  final double balance;
  final int transactionCount;

  const ClusterNode({
    required this.address,
    required this.port,
    required this.cpuCores,
    required this.memoryBytes,
    required this.latencyMs,
    required this.available,
    required this.igpuName,
    this.totalCredits = 0.0,
    this.totalDebits = 0.0,
    this.balance = 0.0,
    this.transactionCount = 0,
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
        totalCredits,
        totalDebits,
        balance,
        transactionCount,
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
    this.versionMajor = 1,
    this.versionMinor = 0,
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
  });

  double get memoryUsagePercent =>
      memoryTotal > 0 ? (memoryUsed / memoryTotal) * 100 : 0;

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
      ];
}
