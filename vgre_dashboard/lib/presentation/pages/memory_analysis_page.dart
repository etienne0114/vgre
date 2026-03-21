import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import '../../application/telemetry_bloc.dart';
import '../../domain/models/telemetry.dart';
import '../../core/theme/vgre_theme.dart';
import '../widgets/glass_card.dart';

class MemoryAnalysisPage extends StatelessWidget {
  const MemoryAnalysisPage({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<TelemetryBloc, TelemetryState>(
      builder: (context, state) {
        if (state is! TelemetryActive) {
          return const Center(child: CircularProgressIndicator());
        }

        final telemetry = state.telemetry;
        final history = state.history;

        return SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(24, 16, 24, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        "MEMORY ANALYTICS",
                        style: VgreTheme.headingStyle.copyWith(fontSize: 28),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        "Deep heap inspection, UVM page tracking, and pool analytics.",
                        style: VgreTheme.bodyStyle.copyWith(color: Colors.white70),
                      ),
                    ],
                  ),
                  const Spacer(),
                  _statChip("TOTAL", _formatBytes(telemetry.memoryTotal)),
                  const SizedBox(width: 12),
                  _statChip("USED", _formatBytes(telemetry.memoryUsed), color: VgreTheme.primaryNeon),
                ],
              ),
              const SizedBox(height: 24),
              
              // Allocation Table & Pool Inspector Row
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // High-fidelity Allocation Table
                  Expanded(
                    flex: 3,
                    child: GlassCard(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            "ACTIVE HEAP ALLOCATIONS",
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              letterSpacing: 1.5,
                              fontSize: 12,
                            ),
                          ),
                          const SizedBox(height: 16),
                          Container(
                            height: 400,
                            decoration: BoxDecoration(
                              color: Colors.black26,
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
                            ),
                            child: telemetry.allocations.isEmpty
                                ? const Center(
                                    child: Text(
                                      "NO ACTIVE ALLOCATIONS",
                                      style: TextStyle(color: Colors.white24, fontSize: 10),
                                    ),
                                  )
                                : ListView.separated(
                                    padding: const EdgeInsets.all(12),
                                    itemCount: telemetry.allocations.length,
                                    separatorBuilder: (_, __) => Divider(color: Colors.white.withValues(alpha: 0.05), height: 1),
                                    itemBuilder: (context, index) {
                                      final a = telemetry.allocations[index];
                                      return _allocationRow(a);
                                    },
                                  ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(width: 24),
                  // Memory Pool Inspector
                  Expanded(
                    flex: 2,
                    child: Column(
                      children: [
                        _uvmMapPanel(telemetry),
                        const SizedBox(height: 24),
                        _poolInspector(telemetry),
                      ],
                    ),
                  ),
                ],
              ),
              
              const SizedBox(height: 24),
              _trendPanel(history),
            ],
          ),
        );
      },
    );
  }

  Widget _statChip(String label, String value, {Color color = Colors.white38}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white.withValues(alpha: 0.1)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Text(label, style: const TextStyle(fontSize: 10, color: Colors.white38, fontWeight: FontWeight.bold)),
          Text(value, style: TextStyle(fontSize: 14, color: color, fontWeight: FontWeight.bold, letterSpacing: 1)),
        ],
      ),
    );
  }

  Widget _allocationRow(MemoryAllocation a) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: a.isManaged ? Colors.purple.withValues(alpha: 0.1) : Colors.blue.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Icon(
              a.isManaged ? Icons.layers : Icons.memory,
              size: 14,
              color: a.isManaged ? Colors.purpleAccent : Colors.blueAccent,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  a.ptr,
                  style: const TextStyle(
                    fontFamily: 'monospace',
                    fontSize: 12,
                    color: Colors.white70,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  a.isManaged ? "Unified Managed Memory" : "Device Physical Memory",
                  style: const TextStyle(fontSize: 10, color: Colors.white38),
                ),
              ],
            ),
          ),
          _kv("SIZE", _formatBytes(a.size)),
          const SizedBox(width: 24),
          _kv("DEVICE", "GPU:${a.deviceId}"),
          const SizedBox(width: 24),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: a.isResident ? Colors.green.withValues(alpha: 0.1) : Colors.orange.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(4),
              border: Border.all(color: a.isResident ? Colors.green.withValues(alpha: 0.2) : Colors.orange.withValues(alpha: 0.2)),
            ),
            child: Text(
              a.isResident ? "RESIDENT" : "EVICTED",
              style: TextStyle(
                fontSize: 8,
                fontWeight: FontWeight.bold,
                color: a.isResident ? Colors.greenAccent : Colors.orangeAccent,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _kv(String k, String v) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        Text(k, style: const TextStyle(fontSize: 9, color: Colors.white24, fontWeight: FontWeight.bold)),
        Text(v, style: const TextStyle(fontSize: 11, color: Colors.white70, fontWeight: FontWeight.bold)),
      ],
    );
  }

  Widget _uvmMapPanel(Telemetry telemetry) {
    return GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "UVM RESIDENCY RADAR",
            style: TextStyle(
              fontWeight: FontWeight.bold,
              letterSpacing: 1.5,
              fontSize: 11,
            ),
          ),
          const SizedBox(height: 16),
          SizedBox(
            height: 140,
            child: GridView.builder(
              physics: const NeverScrollableScrollPhysics(),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 32,
                mainAxisSpacing: 1,
                crossAxisSpacing: 1,
              ),
              itemCount: 1024,
              itemBuilder: (context, index) {
                final isResident = telemetry.uvmMap[index] == 1;
                return Container(
                  decoration: BoxDecoration(
                    color: isResident
                        ? VgreTheme.primaryNeon.withValues(alpha: 0.7)
                        : Colors.white.withValues(alpha: 0.03),
                    borderRadius: BorderRadius.circular(1),
                  ),
                );
              },
            ),
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _statMini("PAGES", "${telemetry.totalPages}"),
              _statMini("RESIDENT", "${telemetry.residentPages}"),
              _statMini("FAULT RATE", "${telemetry.pageFaultRate.toStringAsFixed(1)}/s"),
            ],
          ),
        ],
      ),
    );
  }

  Widget _statMini(String l, String v) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(l, style: const TextStyle(fontSize: 8, color: Colors.white24)),
        Text(v, style: const TextStyle(fontSize: 10, color: Colors.white70, fontWeight: FontWeight.bold)),
      ],
    );
  }

  Widget _poolInspector(Telemetry telemetry) {
    return GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "MEMORY POOL HIERARCHY",
            style: TextStyle(
              fontWeight: FontWeight.bold,
              letterSpacing: 1.5,
              fontSize: 11,
            ),
          ),
          const SizedBox(height: 16),
          if (telemetry.memoryPools.isEmpty)
            const Center(
              child: Padding(
                padding: EdgeInsets.all(20),
                child: Text("NO POOLS CONFIGURED", style: TextStyle(color: Colors.white12, fontSize: 10)),
              ),
            )
          else
            ...telemetry.memoryPools.map((p) => _poolItem(p)),
        ],
      ),
    );
  }

  Widget _poolItem(MemoryPool p) {
    final util = p.totalAllocated > 0 ? (p.totalAllocated / p.peakAllocated) : 0.0;
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Text("POOL #${p.id}", style: const TextStyle(fontWeight: FontWeight.bold, color: VgreTheme.secondaryNeon, fontSize: 12)),
              const Spacer(),
              Text("BLOCK: ${p.blockSize}B", style: const TextStyle(color: Colors.white38, fontSize: 10)),
            ],
          ),
          const SizedBox(height: 8),
          LinearProgressIndicator(
            value: util.toDouble(),
            backgroundColor: Colors.white.withValues(alpha: 0.05),
            color: VgreTheme.secondaryNeon,
            minHeight: 2,
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _statMini("ACTIVE", "${p.activeCount}"),
              _statMini("FREE", "${p.freeCount}"),
              _statMini("ALLOCATED", _formatBytes(p.totalAllocated)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _trendPanel(List<Telemetry> history) {
    if (history.isEmpty) return const SizedBox.shrink();
    
    final usageSpots = _buildSpots(history)
        .map((value) => FlSpot(value[0], value[1]))
        .toList(growable: false);
    final residentSpots = _buildSpots(history, resident: true)
        .map((value) => FlSpot(value[0], value[1]))
        .toList(growable: false);

    return GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Text(
                "HEAP UTILIZATION TREND",
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.5,
                  fontSize: 12,
                ),
              ),
              const Spacer(),
              _trendLegend("USED %", VgreTheme.primaryNeon),
              const SizedBox(width: 16),
              _trendLegend("RESIDENT %", VgreTheme.secondaryNeon),
            ],
          ),
          const SizedBox(height: 16),
          SizedBox(
            height: 180,
            child: LineChart(
              LineChartData(
                minY: 0,
                maxY: 100,
                gridData: FlGridData(
                  show: true,
                  drawHorizontalLine: true,
                  getDrawingHorizontalLine: (value) => FlLine(
                    color: Colors.white.withValues(alpha: 0.05),
                  ),
                  drawVerticalLine: false,
                ),
                titlesData: const FlTitlesData(show: false),
                borderData: FlBorderData(show: false),
                lineBarsData: [
                  if (usageSpots.isNotEmpty)
                    _lineData(VgreTheme.primaryNeon, usageSpots),
                  if (residentSpots.isNotEmpty)
                    _lineData(VgreTheme.secondaryNeon, residentSpots),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _trendLegend(String label, Color color) {
    return Row(
      children: [
        Container(width: 8, height: 8, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
        const SizedBox(width: 6),
        Text(
          label,
          style: const TextStyle(color: VgreTheme.textMuted, fontSize: 10),
        ),
      ],
    );
  }

  LineChartBarData _lineData(Color color, List<FlSpot> spots) {
    return LineChartBarData(
      spots: spots,
      isCurved: true,
      color: color,
      barWidth: 2,
      dotData: const FlDotData(show: false),
      belowBarData: BarAreaData(
        show: true,
        gradient: LinearGradient(
          colors: [color.withValues(alpha: 0.15), color.withValues(alpha: 0.0)],
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
        ),
      ),
    );
  }

  String _formatBytes(int bytes) {
    if (bytes <= 0) return '0 B';
    const suffixes = ['B', 'KB', 'MB', 'GB', 'TB'];
    double size = bytes.toDouble();
    int index = 0;
    while (size >= 1024 && index < suffixes.length - 1) {
      size /= 1024;
      index++;
    }
    return '${size.toStringAsFixed(size < 10 ? 2 : 0)} ${suffixes[index]}';
  }

  List<List<double>> _buildSpots(List<Telemetry> history, {bool resident = false}) {
    final List<List<double>> result = [];
    for (int i = 0; i < history.length; i++) {
      final t = history[i];
      final double value = resident
          ? (t.totalPages > 0 ? (t.residentPages / t.totalPages) * 100 : 0.0)
          : t.memoryUsagePercent;
      result.add([i.toDouble(), value]);
    }
    return result;
  }
}
