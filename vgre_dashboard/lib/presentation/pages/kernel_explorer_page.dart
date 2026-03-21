import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../application/telemetry_bloc.dart';
import '../../domain/models/telemetry.dart';
import '../../core/theme/vgre_theme.dart';
import '../widgets/glass_card.dart';

class KernelExplorerPage extends StatefulWidget {
  const KernelExplorerPage({super.key});

  @override
  State<KernelExplorerPage> createState() => _KernelExplorerPageState();
}

class _KernelExplorerPageState extends State<KernelExplorerPage> {
  // Selection is now managed by TelemetryBloc for persistence

  @override
  Widget build(BuildContext context) {
    final bool isDesktop = VgreTheme.isDesktop(context);

    return BlocBuilder<TelemetryBloc, TelemetryState>(
      builder: (context, state) {
        if (state is! TelemetryActive) {
          return const Center(child: CircularProgressIndicator());
        }
        final data = state.telemetry;
        final kernels = data.topKernels;

        // Reactive Selection Resolution (from BLoC state)
        final String? effectiveSelectionName = state.selectedKernelName;
        KernelStat? selectedKernel;

        if (effectiveSelectionName != null && kernels.isNotEmpty) {
          try {
            selectedKernel = kernels.firstWhere(
              (k) => k.name == effectiveSelectionName,
            );
          } catch (_) {}
        }
        
        // PERSISTENCE FALLBACK: If not in live top-list, use cached version from model
        selectedKernel ??= data.lastSelectedKernelStats;

        // Auto-select "background_compute" or first available IF no selection exists
        if (selectedKernel == null &&
            kernels.isNotEmpty &&
            effectiveSelectionName == null) {
          final bgKernel = kernels.any((k) => k.name == "background_compute")
              ? kernels.firstWhere((k) => k.name == "background_compute")
              : kernels.first;
          selectedKernel = bgKernel;
          // Synchronize back to BLoC
          WidgetsBinding.instance.addPostFrameCallback((_) {
            context.read<TelemetryBloc>().add(SelectKernel(bgKernel.name));
          });
        }

        return Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                "KERNEL EXPLORER",
                style: Theme.of(context).textTheme.displayLarge?.copyWith(
                  fontSize: 24,
                  letterSpacing: 4,
                ),
              ),
              const SizedBox(height: 8),
              const Text(
                "REAL-TIME KERNEL REGISTRY AND PERFORMANCE ANALYTICS",
                style: TextStyle(
                  color: VgreTheme.textMuted,
                  fontSize: 10,
                  letterSpacing: 2,
                ),
              ),
              const SizedBox(height: 32),
              Expanded(
                child: isDesktop
                    ? Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Expanded(
                            flex: 2,
                            child: _buildKernelList(kernels, selectedKernel),
                          ),
                          const SizedBox(width: 24),
                          Expanded(
                            flex: 3,
                            child: _buildDetailsColumn(selectedKernel),
                          ),
                        ],
                      )
                    : ListView(
                        children: [
                          _buildKernelList(
                            kernels,
                            selectedKernel,
                            height: 400,
                          ),
                          const SizedBox(height: 24),
                          _buildDetailsColumn(selectedKernel, height: 600),
                        ],
                      ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildDetailsColumn(KernelStat? selectedKernel, {double? height}) {
    return SizedBox(
      height: height,
      child: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        child: Column(
          children: [
            _KernelDetailsPanel(selectedKernel: selectedKernel),
            const SizedBox(height: 16),
            SizedBox(
              height: 400, // Explicit height for the IDE view to give it scrolling room
              child: _KernelSourcePanel(selectedKernel: selectedKernel),
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 300, // Explicit height for terminal logs
              child: _KernelLogsPanel(kernelName: selectedKernel?.name),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildKernelList(
    List<KernelStat> kernels,
    KernelStat? selectedKernel, {
    double? height,
  }) {
    return SizedBox(
      height: height,
      child: GlassCard(
        padding: EdgeInsets.zero,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.all(20.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    "REGISTERED KERNELS",
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                      letterSpacing: 1,
                    ),
                  ),
                  IconButton(
                    icon: const Icon(
                      Icons.refresh,
                      size: 16,
                      color: VgreTheme.primaryNeon,
                    ),
                    padding: EdgeInsets.zero,
                    constraints: const BoxConstraints(),
                    onPressed: () {
                      context.read<TelemetryBloc>().add(const StartPolling());
                    },
                  ),
                ],
              ),
            ),
            const Divider(color: Colors.white10, height: 1),
            Expanded(
              child: kernels.isEmpty
                  ? const Center(
                      child: Text(
                        "No kernels active",
                        style: TextStyle(color: VgreTheme.textMuted),
                      ),
                    )
                  : ListView.separated(
                      itemCount: kernels.length,
                      separatorBuilder: (_, index) =>
                          const Divider(color: Colors.white10, height: 1),
                      itemBuilder: (context, index) {
                        final kernel = kernels[index];
                        final bool isSelected =
                            selectedKernel?.name == kernel.name;
                        return Padding(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 4,
                          ),
                          child: Material(
                            color: isSelected
                                ? VgreTheme.primaryNeon.withValues(alpha: 0.15)
                                : Colors.transparent,
                            borderRadius: BorderRadius.circular(12),
                            clipBehavior: Clip.antiAlias,
                            child: InkWell(
                              onTap: () {
                                context.read<TelemetryBloc>().add(
                                  SelectKernel(kernel.name),
                                );
                              },
                              hoverColor: VgreTheme.primaryNeon.withValues(
                                alpha: 0.05,
                              ),
                              splashColor: VgreTheme.primaryNeon.withValues(
                                alpha: 0.2,
                              ),
                              child: Container(
                                padding: const EdgeInsets.all(12),
                                decoration: BoxDecoration(
                                  borderRadius: BorderRadius.circular(12),
                                  border: Border.all(
                                    color: isSelected
                                        ? VgreTheme.primaryNeon.withValues(
                                            alpha: 0.5,
                                          )
                                        : Colors.white.withValues(alpha: 0.05),
                                    width: 1,
                                  ),
                                  boxShadow: isSelected
                                      ? [
                                          BoxShadow(
                                            color: VgreTheme.primaryNeon
                                                .withValues(alpha: 0.1),
                                            blurRadius: 10,
                                            spreadRadius: 1,
                                          ),
                                        ]
                                      : [],
                                ),
                                child: Row(
                                  children: [
                                    _PulseIndicator(active: isSelected),
                                    const SizedBox(width: 16),
                                    Expanded(
                                      child: Column(
                                        crossAxisAlignment:
                                            CrossAxisAlignment.start,
                                        children: [
                                          Text(
                                            kernel.name,
                                            style: TextStyle(
                                              fontSize: 14,
                                              fontWeight: isSelected
                                                  ? FontWeight.bold
                                                  : FontWeight.w500,
                                              color: isSelected
                                                  ? Colors.white
                                                  : Colors.white70,
                                              letterSpacing: 0.5,
                                            ),
                                          ),
                                          const SizedBox(height: 4),
                                          Text(
                                            "${kernel.invocations} executions",
                                            style: TextStyle(
                                              fontSize: 11,
                                              color: isSelected
                                                  ? VgreTheme.primaryNeon
                                                        .withValues(alpha: 0.8)
                                                  : VgreTheme.textMuted,
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                    Icon(
                                      isSelected
                                          ? Icons.radio_button_checked
                                          : Icons.chevron_right,
                                      color: isSelected
                                          ? VgreTheme.primaryNeon
                                          : VgreTheme.textMuted,
                                      size: 18,
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

class _KernelDetailsPanel extends StatelessWidget {
  final KernelStat? selectedKernel;
  const _KernelDetailsPanel({this.selectedKernel});

  @override
  Widget build(BuildContext context) {
    return GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text(
            "EXECUTION METRICS",
            style: TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 12,
              letterSpacing: 1,
            ),
          ),
          const SizedBox(height: 16),
          if (selectedKernel == null)
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 32),
              child: Center(
                child: Text(
                  "No kernel selected",
                  style: TextStyle(color: VgreTheme.textMuted),
                ),
              ),
            )
          else
            Padding(
                padding: const EdgeInsets.symmetric(vertical: 4),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    Row(
                      children: [
                        Expanded(child: _buildStatCard("AVG LATENCY", "${selectedKernel!.avgTimeMs.toStringAsFixed(2)} ms", VgreTheme.primaryNeon, Icons.timer)),
                        const SizedBox(width: 16),
                        Expanded(child: _buildStatCard("RANGE", "${selectedKernel!.minTimeMs.toStringAsFixed(2)}-${selectedKernel!.maxTimeMs.toStringAsFixed(2)} ms", VgreTheme.textMuted, Icons.compare_arrows)),
                      ],
                    ),
                    const SizedBox(height: 16),
                    _buildLinearGauge("AVG GFLOPS", selectedKernel!.avgGflops, 200.0, VgreTheme.secondaryNeon),
                    const SizedBox(height: 16),
                    _buildLinearGauge("THROUGHPUT", selectedKernel!.avgThroughputGbps, 1500.0, Colors.orangeAccent, unit: "GB/s"),
                    const SizedBox(height: 16),
                     Row(
                      children: [
                        const Icon(Icons.data_usage, size: 14, color: Colors.purpleAccent),
                        const SizedBox(width: 8),
                         const Text(
                          "TOTAL INVOCATIONS:",
                          style: TextStyle(
                            color: VgreTheme.textMuted,
                            fontSize: 10,
                            letterSpacing: 1,
                          ),
                        ),
                        const Spacer(),
                        Text(
                          selectedKernel!.invocations.toString(),
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 14,
                            fontWeight: FontWeight.bold,
                            fontFamily: 'Orbitron',
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
        ],
      ),
    );
  }

  Widget _buildStatCard(String label, String value, Color color, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, size: 14, color: color),
              const SizedBox(width: 6),
              Text(
                label,
                style: const TextStyle(color: VgreTheme.textMuted, fontSize: 10, letterSpacing: 0.5),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            value,
            style: TextStyle(color: color, fontSize: 16, fontWeight: FontWeight.bold, fontFamily: 'Orbitron'),
          ),
        ],
      ),
    );
  }

  Widget _buildLinearGauge(String label, double value, double max, Color color, {String unit = "GFlops"}) {
     final fraction = (max > 0) ? (value / max).clamp(0.0, 1.0) : 0.0;
     return Column(
       crossAxisAlignment: CrossAxisAlignment.start,
       children: [
         Row(
           mainAxisAlignment: MainAxisAlignment.spaceBetween,
           children: [
             Text(
               label,
               style: const TextStyle(color: VgreTheme.textMuted, fontSize: 10, fontWeight: FontWeight.bold, letterSpacing: 1),
             ),
             Text(
               "${value.toStringAsFixed(1)} $unit",
               style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.bold, fontFamily: 'Orbitron'),
             ),
           ],
         ),
         const SizedBox(height: 8),
         Stack(
           children: [
             Container(
               height: 6,
               decoration: BoxDecoration(
                 color: Colors.white.withValues(alpha: 0.05),
                 borderRadius: BorderRadius.circular(3),
               ),
             ),
             FractionallySizedBox(
               widthFactor: fraction,
               child: Container(
                 height: 6,
                 decoration: BoxDecoration(
                   color: color,
                   borderRadius: BorderRadius.circular(3),
                   boxShadow: [BoxShadow(color: color.withValues(alpha: 0.5), blurRadius: 4)],
                 ),
               ),
             ),
           ],
         ),
       ],
     );
  }
}

class _KernelSourcePanel extends StatefulWidget {
  final KernelStat? selectedKernel;
  const _KernelSourcePanel({this.selectedKernel});

  @override
  State<_KernelSourcePanel> createState() => _KernelSourcePanelState();
}

class _KernelSourcePanelState extends State<_KernelSourcePanel> {
  bool _showIR = false;

  @override
  Widget build(BuildContext context) {
    return GlassCard(
      padding: EdgeInsets.zero,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
            decoration: const BoxDecoration(
              border: Border(bottom: BorderSide(color: Colors.white10)),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    const Icon(
                      Icons.code,
                      size: 16,
                      color: VgreTheme.primaryNeon,
                    ),
                    const SizedBox(width: 12),
                    Text(
                      widget.selectedKernel != null
                          ? widget.selectedKernel!.name.toUpperCase()
                          : "KERNEL SOURCE VIEW",
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        letterSpacing: 1.5,
                        fontSize: 10,
                      ),
                    ),
                  ],
                ),
                if (widget.selectedKernel != null)
                  Row(
                    children: [
                      _toggleButton("SOURCE", !_showIR),
                      const SizedBox(width: 8),
                      _toggleButton("LLVM-IR", _showIR),
                    ],
                  ),
              ],
            ),
          ),
          Expanded(
            child: widget.selectedKernel == null
                ? const Center(
                    child: Text(
                      "Select a kernel to view cross-compiled IR",
                      style: TextStyle(
                        color: VgreTheme.textMuted,
                        fontSize: 12,
                      ),
                    ),
                  )
                : _buildIdeViewer(),
          ),
        ],
      ),
    );
  }

  Widget _buildIdeViewer() {
    final code = _showIR
        ? (widget.selectedKernel!.irCode.isNotEmpty
            ? widget.selectedKernel!.irCode
            : "// No LLVM-IR available [Binary Module]")
        : (widget.selectedKernel!.sourceCode.isNotEmpty
            ? widget.selectedKernel!.sourceCode
            : "// No source available [Pre-registered]");
    
    final lines = code.split('\n');

    return Container(
      width: double.infinity,
      color: const Color(0xFF1E1E1E), // Classic IDE dark theme
      child: ListView.builder(
        padding: const EdgeInsets.symmetric(vertical: 16),
        itemCount: lines.length,
        itemBuilder: (context, index) {
          return Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 48,
                padding: const EdgeInsets.only(right: 12),
                alignment: Alignment.centerRight,
                decoration: const BoxDecoration(
                  border: Border(right: BorderSide(color: Color(0xFF333333))),
                ),
                child: Text(
                  "${index + 1}",
                  style: const TextStyle(color: Color(0xFF6E7681), fontSize: 12, fontFamily: 'FiraCode'),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: SelectableText(
                  lines[index],
                  style: GoogleFonts.firaCode(
                    color: _showIR
                        ? Colors.greenAccent.withValues(alpha: 0.9)
                        : const Color(0xFF9CDCFE), // VS Code Light Blue
                    fontSize: 12,
                    height: 1.5,
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _toggleButton(String label, bool active) {
    return InkWell(
      onTap: () => setState(() => _showIR = (label == "LLVM-IR")),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: active
              ? VgreTheme.primaryNeon.withValues(alpha: 0.1)
              : Colors.transparent,
          borderRadius: BorderRadius.circular(4),
          border: Border.all(
            color: active
                ? VgreTheme.primaryNeon.withValues(alpha: 0.5)
                : Colors.white10,
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: active ? VgreTheme.primaryNeon : VgreTheme.textMuted,
            fontSize: 9,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
    );
  }
}

class _PulseIndicator extends StatefulWidget {
  final bool active;
  const _PulseIndicator({required this.active});

  @override
  State<_PulseIndicator> createState() => _PulseIndicatorState();
}

class _PulseIndicatorState extends State<_PulseIndicator>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.active) {
      return Container(
        width: 10,
        height: 10,
        decoration: const BoxDecoration(
          color: VgreTheme.textMuted,
          shape: BoxShape.circle,
        ),
      );
    }

    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: VgreTheme.primaryNeon,
            boxShadow: [
              BoxShadow(
                color: VgreTheme.primaryNeon.withValues(
                  alpha: 0.8 * (1.0 - _controller.value),
                ),
                blurRadius: 10 * _controller.value,
                spreadRadius: 4 * _controller.value,
              ),
            ],
          ),
        );
      },
    );
  }
}

class _KernelLogsPanel extends StatelessWidget {
  final String? kernelName;
  const _KernelLogsPanel({this.kernelName});

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<TelemetryBloc, TelemetryState>(
      builder: (context, state) {
        final logs = (state is TelemetryActive)
            ? state.telemetry.logs
            : <String>[];
        final filteredLogs = kernelName == null
            ? logs
            : logs.where((log) => log.contains(kernelName!)).toList();

        return GlassCard(
          padding: EdgeInsets.zero,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 20,
                  vertical: 12,
                ),
                decoration: const BoxDecoration(
                  border: Border(bottom: BorderSide(color: Colors.white10)),
                  color: Color(0xFF18181A),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Row(
                      children: [
                        const Icon(
                          Icons.terminal,
                          size: 16,
                          color: Colors.white,
                        ),
                        const SizedBox(width: 12),
                        Text(
                          kernelName != null
                              ? "${kernelName!.toUpperCase()} STDOUT/STDERR"
                              : "SYSTEM KERNEL LOGS",
                          style: const TextStyle(
                            fontWeight: FontWeight.bold,
                            letterSpacing: 1.5,
                            fontSize: 10,
                          ),
                        ),
                      ],
                    ),
                    if (kernelName != null)
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 2,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.1),
                          borderRadius: BorderRadius.circular(4),
                          border: Border.all(
                            color: Colors.white.withValues(alpha: 0.3),
                          ),
                        ),
                        child: const Text(
                          "FILTERED",
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 8,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                  ],
                ),
              ),
              Expanded(
                child: filteredLogs.isEmpty
                    ? Center(
                        child: Text(
                          kernelName == null
                              ? "No system logs available"
                              : "No logs found for '$kernelName'",
                          style: const TextStyle(
                            color: VgreTheme.textMuted,
                            fontSize: 11,
                          ),
                        ),
                      )
                    : Container(
                        padding: EdgeInsets.zero,
                        width: double.infinity,
                        color: const Color(0xFF0D0D0E),
                        child: SelectionArea(
                          child: ListView.separated(
                            padding: const EdgeInsets.all(16),
                            physics: const BouncingScrollPhysics(),
                            reverse: true, // Auto-scroll behavior via reverse logic
                            itemCount: filteredLogs.length,
                            separatorBuilder: (context, index) => const Divider(color: Colors.white10, height: 8),
                            itemBuilder: (context, index) {
                              // We use reverse because we want newest at bottom (ListView reverse flips ordering)
                              final rawLog = filteredLogs[filteredLogs.length - 1 - index];
                              return _buildLogEntry(rawLog);
                            },
                          ),
                        ),
                      ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildLogEntry(String log) {
     final bool isError = log.contains("[ERROR]");
     final bool isWarning = log.contains("[WARN]");
     final bool isSuccess = log.contains("COMPLETED") || log.contains("SUCCESS");
     
     Color levelColor = Colors.white70;
     if (isError) {
       levelColor = Colors.redAccent;
     } else if (isWarning) {
       levelColor = Colors.orangeAccent;
     } else if (isSuccess) {
       levelColor = VgreTheme.neonGreen;
     } else if (log.contains("heart-beat")) {
       levelColor = VgreTheme.secondaryNeon;
     }

     // Attempt to extract timestamp: [2026-03-21 ...]
     String timestamp = "";
     String content = log;
     final tMatch = RegExp(r'^\[(.*?)\]\s*(.*)$').firstMatch(log);
     if (tMatch != null) {
       timestamp = "[${tMatch.group(1)}]";
       content = tMatch.group(2) ?? "";
     }

     return Row(
       crossAxisAlignment: CrossAxisAlignment.start,
       children: [
         if (timestamp.isNotEmpty)
           SizedBox(
             width: 155,
             child: Text(
               timestamp,
               style: GoogleFonts.firaCode(color: const Color(0xFF555555), fontSize: 10),
             ),
           ),
         Expanded(
           child: Text(
             content,
             style: GoogleFonts.firaCode(
               color: levelColor,
               fontSize: 11,
             ),
           ),
         ),
       ],
     );
  }
}
