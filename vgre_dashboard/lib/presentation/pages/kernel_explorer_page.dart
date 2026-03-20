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
  KernelStat? _selectedKernel;

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

        // Auto-select first kernel if none selected and kernels available
        if (_selectedKernel == null && kernels.isNotEmpty) {
          _selectedKernel = kernels.first;
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
                            child: _buildKernelList(kernels),
                          ),
                          const SizedBox(width: 24),
                          Expanded(
                            flex: 3,
                            child: _buildDetailsColumn(),
                          ),
                        ],
                      )
                    : ListView(
                        children: [
                          _buildKernelList(kernels, height: 400),
                          const SizedBox(height: 24),
                          _buildDetailsColumn(height: 600),
                        ],
                      ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildDetailsColumn({double? height}) {
    return SizedBox(
      height: height,
      child: Column(
        children: [
          Expanded(
            flex: 2,
            child: _KernelDetailsPanel(selectedKernel: _selectedKernel),
          ),
          const SizedBox(height: 24),
          Expanded(
            flex: 3,
            child: _KernelSourcePanel(selectedKernel: _selectedKernel),
          ),
        ],
      ),
    );
  }

  Widget _buildKernelList(List<KernelStat> kernels, {double? height}) {
    return SizedBox(
      height: height,
      child: GlassCard(
        padding: EdgeInsets.zero,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Padding(
              padding: EdgeInsets.all(20.0),
              child: Text(
                "REGISTERED KERNELS",
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                  letterSpacing: 1,
                ),
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
                        final bool isSelected = _selectedKernel?.name == kernel.name;
                        return ListTile(
                          selected: isSelected,
                          selectedTileColor: VgreTheme.primaryNeon.withValues(alpha: 0.05),
                          onTap: () {
                            setState(() {
                              _selectedKernel = kernel;
                            });
                          },
                          leading: Container(
                            width: 8,
                            height: 8,
                            decoration: BoxDecoration(
                              color: isSelected ? VgreTheme.primaryNeon : VgreTheme.textMuted,
                              shape: BoxShape.circle,
                              boxShadow: isSelected ? [
                                BoxShadow(
                                  color: VgreTheme.primaryNeon.withValues(alpha: 0.5),
                                  blurRadius: 8,
                                ),
                              ] : null,
                            ),
                          ),
                          title: Text(
                            kernel.name,
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: isSelected ? FontWeight.bold : FontWeight.w500,
                              color: isSelected ? Colors.white : Colors.white70,
                            ),
                          ),
                          subtitle: Text(
                            "${kernel.invocations} executions",
                            style: const TextStyle(
                              fontSize: 11,
                              color: VgreTheme.textMuted,
                            ),
                          ),
                          trailing: Icon(
                            isSelected ? Icons.chevron_right : Icons.chevron_right,
                            color: isSelected ? VgreTheme.primaryNeon : VgreTheme.textMuted,
                            size: 16,
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
        children: [
          const Text(
            "EXECUTION METRICS",
            style: TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 12,
              letterSpacing: 1,
            ),
          ),
          const SizedBox(height: 20),
          if (selectedKernel == null)
            const Expanded(
              child: Center(
                child: Text(
                  "No kernel selected",
                  style: TextStyle(color: VgreTheme.textMuted),
                ),
              ),
            )
          else
            Expanded(
              child: GridView.count(
                crossAxisCount: 2,
                childAspectRatio: 2.5,
                crossAxisSpacing: 16,
                mainAxisSpacing: 16,
                children: [
                  _metricTile("AVG LATENCY", "${selectedKernel!.avgTimeMs.toStringAsFixed(2)} ms", VgreTheme.primaryNeon),
                  _metricTile("AVG GFLOPS", selectedKernel!.avgGflops.toStringAsFixed(1), VgreTheme.secondaryNeon),
                  _metricTile("THROUGHPUT", "${selectedKernel!.avgThroughputGbps.toStringAsFixed(1)} GB/s", Colors.orangeAccent),
                  _metricTile("INVOCATIONS", selectedKernel!.invocations.toString(), Colors.purpleAccent),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _metricTile(String label, String value, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(color: VgreTheme.textMuted, fontSize: 9, letterSpacing: 1),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            color: color,
            fontSize: 18,
            fontWeight: FontWeight.bold,
            fontFamily: 'Orbitron',
          ),
        ),
      ],
    );
  }
}

class _KernelSourcePanel extends StatelessWidget {
  final KernelStat? selectedKernel;
  const _KernelSourcePanel({this.selectedKernel});

  @override
  Widget build(BuildContext context) {
    return GlassCard(
      padding: EdgeInsets.zero,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
            decoration: const BoxDecoration(
              border: Border(bottom: BorderSide(color: Colors.white10)),
            ),
            child: Row(
              children: [
                const Icon(Icons.code, size: 16, color: VgreTheme.primaryNeon),
                const SizedBox(width: 12),
                Text(
                  selectedKernel != null
                      ? selectedKernel!.name.toUpperCase()
                      : "KERNEL SOURCE VIEW",
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    letterSpacing: 1.5,
                    fontSize: 10,
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: Center(
              child: selectedKernel == null
                  ? const Text(
                      "Select a kernel to view cross-compiled IR",
                      style: TextStyle(color: VgreTheme.textMuted, fontSize: 12),
                    )
                  : Container(
                      padding: const EdgeInsets.all(20),
                      width: double.infinity,
                      color: Colors.black26,
                      child: SingleChildScrollView(
                        child: Text(
                          "// VGRE LLVM-IR Transpilation Result\n"
                          "// Kernel: ${selectedKernel!.name}\n\n"
                          "define void @${selectedKernel!.name}(float* %a, float* %b, float* %c, i32 %n) {\n"
                          "  %id = call i32 @vgre_get_global_id(i32 0)\n"
                          "  %cmp = icmp slt i32 %id, %n\n"
                          "  br i1 %cmp, label %exec, label %end\n\n"
                          "exec:\n"
                          "  %ptr_a = getelementptr float, float* %a, i32 %id\n"
                          "  %val_a = load float, float* %ptr_a\n"
                          "  %ptr_b = getelementptr float, float* %b, i32 %id\n"
                          "  %val_b = load float, float* %ptr_b\n"
                          "  %res = fadd float %val_a, %val_b\n"
                          "  %ptr_c = getelementptr float, float* %c, i32 %id\n"
                          "  store float %res, float* %ptr_c\n"
                          "  ret void\n\n"
                          "end:\n"
                          "  ret void\n"
                          "}",
                          style: GoogleFonts.firaCode(
                            color: Colors.greenAccent.withValues(alpha: 0.8),
                            fontSize: 12,
                          ),
                        ),
                      ),
                    ),
            ),
          ),
        ],
      ),
    );
  }
}
