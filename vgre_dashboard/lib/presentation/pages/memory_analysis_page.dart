import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import '../../application/telemetry_bloc.dart';
import '../../domain/models/telemetry.dart';
import '../../core/theme/vgre_theme.dart';
import '../widgets/glass_card.dart';
import '../widgets/glow_gauge.dart';

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
        final residentPercent = telemetry.totalPages > 0
            ? (telemetry.residentPages / telemetry.totalPages) * 100
            : 0.0;
        final usageSpots = _buildSpots(history)
            .map((value) => FlSpot(value[0], value[1]))
            .toList(growable: false);
        final residentSpots = _buildSpots(history, resident: true)
            .map((value) => FlSpot(value[0], value[1]))
            .toList(growable: false);

        return SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                "MEMORY ANALYSIS",
                style: VgreTheme.headingStyle.copyWith(fontSize: 28),
              ),
              const SizedBox(height: 8),
              Text(
                "Live utilization, UVM residency, and bandwidth tracking.",
                style: VgreTheme.bodyStyle.copyWith(color: Colors.white70),
              ),
              const SizedBox(height: 24),
              Wrap(
                runSpacing: 24,
                spacing: 18,
                children: [
                  _memoryGaugeCard(telemetry),
                  _bandwidthGaugeCard(telemetry),
                  _pageFaultPanel(telemetry, residentPercent),
                ],
              ),
              const SizedBox(height: 24),
              GlassCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Text(
                          "MEMORY TREND",
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
                    const SizedBox(height: 12),
                    SizedBox(
                      height: 220,
                      child: (usageSpots.isEmpty && residentSpots.isEmpty)
                          ? const Center(
                              child: Text(
                                'Waiting for telemetry... ',
                                style: TextStyle(color: Colors.white38),
                              ),
                            )
                          : LineChart(
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
              ),
              const SizedBox(height: 24),
              GlassCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Expanded(
                          child: Text(
                            "UVM RESIDENCY",
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              letterSpacing: 1.5,
                              fontSize: 12,
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          telemetry.uvmQuality.label,
                          style: const TextStyle(
                            color: VgreTheme.textMuted,
                            fontSize: 10,
                            letterSpacing: 1,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    SizedBox(
                      height: 180,
                      child: GridView.builder(
                        physics: const NeverScrollableScrollPhysics(),
                        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                          crossAxisCount: 32,
                          mainAxisSpacing: 2,
                          crossAxisSpacing: 2,
                        ),
                        itemCount: 1024,
                        itemBuilder: (context, index) {
                          final isResident = telemetry.uvmMap[index] == 1;
                          return Container(
                            decoration: BoxDecoration(
                              color: isResident
                                  ? VgreTheme.primaryNeon.withValues(alpha: 0.8)
                                  : Colors.white.withValues(alpha: 0.04),
                              borderRadius: BorderRadius.circular(1),
                            ),
                          );
                        },
                      ),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        _memoryStatItem("RESIDENT", "${telemetry.residentPages}"),
                        const SizedBox(width: 16),
                        _memoryStatItem("EVICTED", "${telemetry.evictedPages}"),
                        const SizedBox(width: 16),
                          _memoryStatItem(
                            "PAGE FAULTS",
                            '${telemetry.pageFaultRate.toStringAsFixed(1)}/s',
                          ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _memoryGaugeCard(Telemetry telemetry) {
    return GlassCard(
      child: GlowGauge(
        value: telemetry.memoryUsagePercent,
        max: 100,
        label: "MEMORY USAGE",
        unit: "%",
        color: VgreTheme.primaryNeon,
        info: [
          {
            'key': 'TOTAL',
            'value': _formatBytes(telemetry.memoryTotal),
          },
          {
            'key': 'USED',
            'value': _formatBytes(telemetry.memoryUsed),
          },
          {
            'key': 'BANDWIDTH',
            'value': '${telemetry.memoryBusUtilization.toStringAsFixed(1)}%',
          },
        ],
      ),
    );
  }

  Widget _bandwidthGaugeCard(Telemetry telemetry) {
    return GlassCard(
      child: GlowGauge(
        value: telemetry.memoryBandwidth,
        max: telemetry.maxMemoryBandwidth > 0 ? telemetry.maxMemoryBandwidth : 1,
        label: "BANDWIDTH",
        unit: "GB/s",
        color: VgreTheme.secondaryNeon,
        info: [
          {
            'key': 'UTIL',
            'value': '${telemetry.memoryBusUtilization.toStringAsFixed(1)}%',
          },
          {
            'key': 'MAX',
            'value': telemetry.maxMemoryBandwidth.toStringAsFixed(1),
          },
          {
            'key': 'FAULTS',
            'value': '${telemetry.pageFaultRate.toStringAsFixed(1)}/s',
          },
        ],
      ),
    );
  }

  Widget _pageFaultPanel(Telemetry telemetry, double residentPercent) {
    return GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "PAGE PEAKS",
            style: TextStyle(
              fontWeight: FontWeight.bold,
              letterSpacing: 1.5,
              fontSize: 12,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _pageFaultBubble(
                "FAULT RATE",
                '${telemetry.pageFaultRate.toStringAsFixed(1)}/s',
                color: Colors.orangeAccent,
              ),
              _pageFaultBubble(
                "RESIDENT",
                '${residentPercent.toStringAsFixed(1)}%',
                color: VgreTheme.neonGreen,
              ),
              _pageFaultBubble(
                "TOTAL PAGES",
                telemetry.totalPages.toString(),
                color: Colors.white70,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _pageFaultBubble(String title, String value, {Color color = Colors.white}) {
    return Expanded(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Text(
            title,
            style: const TextStyle(fontSize: 10, color: Colors.white38),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: TextStyle(fontWeight: FontWeight.bold, color: color, fontSize: 16),
          ),
        ],
      ),
    );
  }

  Widget _memoryStatItem(String label, String value) {
    return Expanded(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: const TextStyle(color: VgreTheme.textMuted, fontSize: 10),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
          ),
        ],
      ),
    );
  }

  Widget _trendLegend(String label, Color color) {
    return Row(
      children: [
        Container(width: 12, height: 12, color: color),
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
      belowBarData: BarAreaData(show: true, color: color.withValues(alpha: 0.15)),
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
