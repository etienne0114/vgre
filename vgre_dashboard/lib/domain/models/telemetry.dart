import 'package:equatable/equatable.dart';

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
  final bool simulationEnabled;
  final String deviceName;
  final List<String> logs;

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
    this.simulationEnabled = false,
    this.deviceName = "VGRE_DEVICE",
    this.logs = const [],
  });

  double get memoryUsagePercent => (memoryUsed / memoryTotal) * 100;

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
      simulationEnabled,
      deviceName,
      logs,
    ];
}
