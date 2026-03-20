import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:fl_chart/fl_chart.dart';
import '../../application/telemetry_bloc.dart';
import '../../domain/models/telemetry.dart';
import '../../core/theme/vgre_theme.dart';
import '../widgets/glass_card.dart';
import '../widgets/glow_gauge.dart';

import '../widgets/navigation_sidebar.dart';
import 'kernel_explorer_page.dart';
import 'cluster_topology_page.dart';
import 'hardware_tuning_page.dart';

class DashboardPage extends StatefulWidget {
  const DashboardPage({super.key});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  int _selectedIndex = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          color: VgreTheme.background,
        ),
        child: Row(
          children: [
            // Left Sidebar
            LayoutBuilder(
              builder: (context, constraints) {
                final bool isCompact =
                    MediaQuery.sizeOf(context).width < VgreTheme.desktopLimit;
                return VgreNavigationSidebar(
                  selectedIndex: _selectedIndex,
                  isCompact: isCompact,
                  onDestinationSelected: (index) {
                    setState(() {
                      _selectedIndex = index;
                    });
                  },
                );
              },
            ),

            // Main Content Area
            Expanded(
              child: AnimatedSwitcher(
                duration: const Duration(milliseconds: 300),
                transitionBuilder: (Widget child, Animation<double> animation) {
                  return FadeTransition(
                    opacity: animation,
                    child: SlideTransition(
                      position: Tween<Offset>(
                        begin: const Offset(0.02, 0),
                        end: Offset.zero,
                      ).animate(animation),
                      child: child,
                    ),
                  );
                },
                child: _buildCurrentScreen(),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCurrentScreen() {
    switch (_selectedIndex) {
      case 0:
        return const DashboardOverviewContent();
      case 1:
        return const KernelExplorerPage();
      case 2:
        return const ClusterTopologyPage();
      case 3:
        return const HardwareTuningPage();
      default:
        return Center(
          child: Text(
            "FEATURE ${_selectedIndex + 1} (WIP)",
            style: const TextStyle(color: VgreTheme.textMuted, letterSpacing: 2),
          ),
        );
    }
  }
}

class DashboardOverviewContent extends StatelessWidget {
  const DashboardOverviewContent({super.key});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: BlocBuilder<TelemetryBloc, TelemetryState>(
          builder: (context, state) {
            if (state is! TelemetryActive) {
              return const Center(child: CircularProgressIndicator());
            }
            final activeState = state;
            final data = activeState.telemetry;
            final history = activeState.history;

            return Column(
              children: [
                _buildHeader(context, data),
                const SizedBox(height: 24),
                Expanded(
                  child: Column(
                    children: [
                      // Top Section: Gauges and UVM Map
                      Expanded(
                        flex: 5,
                        child: Row(
                          children: [
                            Expanded(
                              child: GlassCard(
                                child: GlowGauge(
                                  value: data.gflops,
                                  max: data.maxGflops,
                                  label: "GFLOPS Performance",
                                  unit: "GFLOPS",
                                  color: VgreTheme.primaryNeon,
                                  info: [
                                    {
                                      'key': 'Quality',
                                      'value': data.computeQuality.label,
                                    },
                                    {
                                      'key': 'Util',
                                      'value':
                                          '${data.computeUtilization.toStringAsFixed(0)}%',
                                    },
                                    {
                                      'key': 'Cores',
                                      'value': '${data.activeThreads}',
                                    },
                                    {
                                      'key': 'Temp',
                                      'value':
                                          '${data.temperature.toStringAsFixed(1)}°C',
                                    },
                                  ],
                                ),
                              ),
                            ),
                            const SizedBox(width: 20),
                            Expanded(
                              child: GlassCard(
                                child: GlowGauge(
                                  value: data.memoryBandwidth,
                                  max: data.maxMemoryBandwidth,
                                  label: "Memory Bandwidth",
                                  unit: "GB/s",
                                  color: VgreTheme.secondaryNeon,
                                  info: [
                                    {
                                      'key': 'Quality',
                                      'value': data.memoryQuality.label,
                                    },
                                    {
                                      'key': 'HBM2',
                                      'value':
                                          '${data.memoryUsagePercent.toStringAsFixed(0)}%',
                                    },
                                    {
                                      'key': 'Clock',
                                      'value': '${data.clockSpeed} MHz',
                                    },
                                    {
                                      'key': 'ECC',
                                      'value': data.eccEnabled
                                          ? 'OK'
                                          : 'ERR',
                                    },
                                  ],
                                ),
                              ),
                            ),
                            const SizedBox(width: 20),
                            Expanded(child: _buildMemoryMapPanel(data)),
                          ],
                        ),
                      ),
                      const SizedBox(height: 20),
                      // Middle Section: Workload Chart
                      Expanded(
                        flex: 3,
                        child: _buildWorkloadChart(history),
                      ),
                      const SizedBox(height: 20),
                      // Bottom Section: Console
                      Expanded(
                        flex: 3,
                        child: Row(
                          children: [
                            Expanded(
                              flex: 2,
                              child: _buildKernelPanel(data),
                            ),
                            const SizedBox(width: 20),
                            Expanded(
                              flex: 3,
                              child: _TerminalConsole(logs: data.logs),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context, Telemetry data) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white10),
      ),
      child: LayoutBuilder(
        builder: (context, constraints) {
          final bool isNarrow = constraints.maxWidth < 1100;
          
          final infoItems = [
            _headerInfoItem("CLOCK", "${data.clockSpeed} MHz"),
            const SizedBox(width: 24),
            _headerInfoItem("ECC", data.eccEnabled ? "ENABLED" : "DISABLED"),
            const SizedBox(width: 24),
            _headerToggle("BACKGROUND COMPUTE", data.backgroundComputeActive, (val) {
              context.read<TelemetryBloc>().add(ToggleBackgroundCompute(val));
            }),
            const SizedBox(width: 24),
            _headerToggle("SERVICE", data.serviceModeActive, (val) {
              context.read<TelemetryBloc>().add(ToggleServiceMode(val));
            }, color: VgreTheme.primaryNeon),
            const SizedBox(width: 24),
            _headerToggle("BLOCK THREADS", data.blockThreadsActive, (val) {
              context.read<TelemetryBloc>().add(ToggleBlockThreads(val));
            }, color: VgreTheme.secondaryNeon),
          ];

          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(Icons.memory, color: VgreTheme.primaryNeon, size: 28),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Text(
                      data.deviceName.toUpperCase(),
                      style: GoogleFonts.orbitron(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 2,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  if (!isNarrow) ...[
                    const Spacer(),
                    ...infoItems,
                  ],
                ],
              ),
              if (isNarrow) ...[
                const SizedBox(height: 16),
                Wrap(
                  spacing: 24,
                  runSpacing: 12,
                  crossAxisAlignment: WrapCrossAlignment.center,
                  children: [
                    _headerInfoItem("CLOCK", "${data.clockSpeed} MHz"),
                    _headerInfoItem("ECC", data.eccEnabled ? "ENABLED" : "DISABLED"),
                    _headerToggle("BACKGROUND COMPUTE", data.backgroundComputeActive, (val) {
                      context.read<TelemetryBloc>().add(ToggleBackgroundCompute(val));
                    }),
                    _headerToggle("SERVICE", data.serviceModeActive, (val) {
                      context.read<TelemetryBloc>().add(ToggleServiceMode(val));
                    }, color: VgreTheme.primaryNeon),
                    _headerToggle("BLOCK THREADS", data.blockThreadsActive, (val) {
                      context.read<TelemetryBloc>().add(ToggleBlockThreads(val));
                    }, color: VgreTheme.secondaryNeon),
                  ],
                ),
              ],
            ],
          );
        },
      ),
    );
  }

  Widget _headerToggle(
    String label,
    bool value,
    ValueChanged<bool> onChanged, {
    Color? color,
  }) {
    return Row(
      children: [
        Text(
          "$label: ",
          style: const TextStyle(color: VgreTheme.textMuted, fontSize: 10),
        ),
        SizedBox(
          height: 24,
          child: Switch(
            value: value,
            onChanged: onChanged,
            activeThumbColor: Colors.white,
            activeTrackColor: color ?? VgreTheme.secondaryNeon,
            inactiveThumbColor: VgreTheme.textMuted,
            inactiveTrackColor: Colors.white.withValues(alpha: 0.05),
            materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
          ),
        ),
      ],
    );
  }

  Widget _headerInfoItem(String label, String value, {Color? color}) {
    return Row(
      children: [
        Text(
          "$label: ",
          style: const TextStyle(color: VgreTheme.textMuted, fontSize: 11),
        ),
        Text(
          value,
          style: TextStyle(
            color: color ?? Colors.white,
            fontWeight: FontWeight.bold,
            fontSize: 11,
            letterSpacing: 1,
          ),
        ),
      ],
    );
  }

  Widget _buildMemoryMapPanel(Telemetry data) {
    return GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Expanded(
                child: Text(
                  "UVM MEMORY MAP",
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    letterSpacing: 1.5,
                    fontSize: 12,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              const SizedBox(width: 8),
              Text(
                "32x32 GRID",
                style: TextStyle(color: VgreTheme.textMuted, fontSize: 10),
              ),
              const SizedBox(width: 12),
              Text(
                data.uvmQuality.label,
                style: const TextStyle(
                  color: VgreTheme.textMuted,
                  fontSize: 10,
                  letterSpacing: 1,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Expanded(
            child: GridView.builder(
              physics: const NeverScrollableScrollPhysics(),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 32,
                mainAxisSpacing: 2,
                crossAxisSpacing: 2,
              ),
              itemCount: 1024,
              itemBuilder: (context, index) {
                bool isResident = data.uvmMap[index] == 1;
                return Container(
                  decoration: BoxDecoration(
                    color: isResident
                        ? VgreTheme.primaryNeon.withValues(alpha: 0.8)
                        : Colors.white.withValues(alpha: 0.05),
                    borderRadius: BorderRadius.circular(1),
                    boxShadow: isResident
                        ? [
                            BoxShadow(
                              color: VgreTheme.primaryNeon.withValues(alpha: 0.3),
                              blurRadius: 2,
                            ),
                          ]
                        : null,
                  ),
                );
              },
            ),
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(child: _uvmLegendItem("TOTAL PAGES", "${data.totalPages}")),
              const SizedBox(width: 8),
              Expanded(
                child: _uvmLegendItem(
                  "RESIDENT (GREEN)",
                  "${data.residentPages}",
                  color: VgreTheme.primaryNeon,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(child: _uvmLegendItem("EVICTED (DIM)", "${data.evictedPages}")),
              const SizedBox(width: 8),
              Expanded(
                child: _uvmLegendItem(
                  "PAGE FAULTS",
                  "${data.pageFaultRate.toStringAsFixed(1)}/s",
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _uvmLegendItem(String label, String value, {Color? color}) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(
            color: color ?? Colors.white.withValues(alpha: 0.1),
            borderRadius: BorderRadius.circular(1),
          ),
        ),
        const SizedBox(width: 8),
        Flexible(
          child: Text(
            label,
            style: const TextStyle(color: VgreTheme.textMuted, fontSize: 10),
            overflow: TextOverflow.ellipsis,
          ),
        ),
        const SizedBox(width: 4),
        Text(
          value,
          style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 10),
        ),
      ],
    );
  }

  Widget _buildWorkloadChart(List<Telemetry> history) {
    return GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Text(
                "GPU WORKLOAD & HEAT",
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.5,
                  fontSize: 12,
                ),
              ),
              const Spacer(),
              _chartLabel("COMPUTE", VgreTheme.primaryNeon),
              const SizedBox(width: 16),
              _chartLabel("MEM I/O", VgreTheme.secondaryNeon),
              const SizedBox(width: 16),
              _chartLabel("HEAT", Colors.redAccent),
            ],
          ),
          const SizedBox(height: 24),
          Expanded(
            child: LineChart(
              LineChartData(
                minY: 0,
                maxY: 100,
                gridData: FlGridData(
                  show: true,
                  drawVerticalLine: false,
                  getDrawingHorizontalLine: (v) =>
                      FlLine(color: Colors.white.withValues(alpha: 0.05)),
                ),
                titlesData: const FlTitlesData(show: false),
                borderData: FlBorderData(show: false),
                lineBarsData: [
                  _lineData(
                    VgreTheme.primaryNeon,
                    history.map((t) => t.computeUtilization).toList(),
                  ),
                  _lineData(
                    VgreTheme.secondaryNeon,
                    history.map((t) => t.memoryBusUtilization).toList(),
                  ),
                  _lineData(
                    Colors.redAccent,
                    history.map((t) => t.temperature).toList(),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  LineChartBarData _lineData(Color color, List<double> values) {
    return LineChartBarData(
      spots: values
          .asMap()
          .entries
          .map((e) => FlSpot(e.key.toDouble(), e.value))
          .toList(),
      isCurved: true,
      color: color,
      barWidth: 2,
      isStrokeCapRound: true,
      dotData: const FlDotData(show: false),
      belowBarData: BarAreaData(show: true, color: color.withValues(alpha: 0.1)),
    );
  }

  Widget _chartLabel(String text, Color color) {
    return Row(
      children: [
        Container(width: 12, height: 12, color: color),
        const SizedBox(width: 8),
        Text(
          text,
          style: const TextStyle(color: VgreTheme.textMuted, fontSize: 10),
        ),
      ],
    );
  }

  Widget _buildKernelPanel(Telemetry data) {
    return GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "HOT KERNELS (TOP 5)",
            style: TextStyle(
              fontWeight: FontWeight.bold,
              letterSpacing: 1.5,
              fontSize: 12,
            ),
          ),
          const SizedBox(height: 16),
          Expanded(
            child: data.topKernels.isEmpty
                ? Center(
                    child: Text(
                      "Profiler disabled or no data yet",
                      style: TextStyle(
                        color: VgreTheme.textMuted,
                        fontSize: 12,
                      ),
                    ),
                  )
                : ListView.separated(
                    itemCount: data.topKernels.length,
                    separatorBuilder: (_, index) => const SizedBox(height: 8),
                    itemBuilder: (context, index) {
                      final k = data.topKernels[index];
                      return Row(
                        children: [
                          Expanded(
                            child: Text(
                              k.name,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 12,
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Text(
                            "${k.avgTimeMs.toStringAsFixed(2)} ms",
                            style: TextStyle(
                              color: VgreTheme.textMuted,
                              fontSize: 11,
                            ),
                          ),
                          const SizedBox(width: 12),
                          Text(
                            "${k.avgGflops.toStringAsFixed(1)} GF",
                            style: TextStyle(
                              color: VgreTheme.primaryNeon,
                              fontSize: 11,
                            ),
                          ),
                        ],
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}

class _TerminalConsole extends StatefulWidget {
  final List<String> logs;
  const _TerminalConsole({required this.logs});

  @override
  State<_TerminalConsole> createState() => _TerminalConsoleState();
}

class _TerminalConsoleState extends State<_TerminalConsole> {
  final List<String> _allLogs = [];

  @override
  void didUpdateWidget(_TerminalConsole oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.logs.isNotEmpty) {
      if (_allLogs.length != widget.logs.length ||
          (_allLogs.isNotEmpty && _allLogs.last != widget.logs.last)) {
        _allLogs.clear();
        _allLogs.addAll(widget.logs);
        setState(() {});
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return GlassCard(
      padding: const EdgeInsets.all(0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
            decoration: const BoxDecoration(
              border: Border(bottom: BorderSide(color: Colors.white10)),
            ),
            child: const Row(
              children: [
                Text(
                  "TERMINAL CONSOLE / KERNEL LOGS",
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    letterSpacing: 1.5,
                    fontSize: 11,
                  ),
                ),
                Spacer(),
                Icon(Icons.remove, size: 16, color: VgreTheme.textMuted),
                SizedBox(width: 12),
                Icon(
                  Icons.check_box_outline_blank,
                  size: 14,
                  color: VgreTheme.textMuted,
                ),
                SizedBox(width: 12),
                Icon(Icons.close, size: 16, color: VgreTheme.textMuted),
              ],
            ),
          ),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: SelectionArea(
                child: ListView.builder(
                  itemCount: _allLogs.length,
                  reverse: true, // Show latest at bottom
                  itemBuilder: (context, index) {
                    final log = _allLogs[_allLogs.length - 1 - index];
                    return _consoleLine(log);
                  },
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _consoleLine(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Text(
        text,
        style: GoogleFonts.firaCode(
          fontSize: 12,
          color: text.contains('[ERROR]')
              ? Colors.redAccent
              : VgreTheme.textBody,
        ),
      ),
    );
  }
}
