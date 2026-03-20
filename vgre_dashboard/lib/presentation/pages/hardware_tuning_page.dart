import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import '../../application/telemetry_bloc.dart';
import '../widgets/glass_card.dart';
import '../../core/theme/vgre_theme.dart';

class HardwareTuningPage extends StatelessWidget {
  const HardwareTuningPage({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<TelemetryBloc, TelemetryState>(
      builder: (context, state) {
        if (state is! TelemetryActive) {
          return const Center(child: CircularProgressIndicator());
        }

        final telemetry = state.telemetry;

        return SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildHeader(),
              const SizedBox(height: 32),
              LayoutBuilder(
                builder: (context, constraints) {
                  return Wrap(
                    spacing: 24,
                    runSpacing: 24,
                    children: [
                      _buildTuningSection(
                        context,
                        "RUNTIME ENGINE",
                        "Configure core execution behavior and synchronization.",
                        [
                          _buildToggleTile(
                            context,
                            "Background Compute",
                            "Enables autonomous workload generation for stress testing.",
                            telemetry.backgroundComputeActive,
                            (val) => context.read<TelemetryBloc>().add(ToggleBackgroundCompute(val)),
                            icon: Icons.speed,
                          ),
                          _buildToggleTile(
                            context,
                            "Block-Level Threads",
                            "Toggles per-block OS thread execution for __syncthreads correctness.",
                            telemetry.blockThreadsActive,
                            (val) => context.read<TelemetryBloc>().add(ToggleBlockThreads(val)),
                            icon: Icons.layers,
                            color: VgreTheme.secondaryNeon,
                          ),
                        ],
                        constraints.maxWidth,
                      ),
                      _buildTuningSection(
                        context,
                        "SERVICE & CONNECTIVITY",
                        "Manage IPC and cluster communication modes.",
                        [
                          _buildToggleTile(
                            context,
                            "Cluster Service Mode",
                            "When enabled, this node acts as a Master for telemetry aggregation.",
                            telemetry.serviceModeActive,
                            (val) => context.read<TelemetryBloc>().add(ToggleServiceMode(val)),
                            icon: Icons.hub,
                            color: VgreTheme.primaryNeon,
                          ),
                        ],
                        constraints.maxWidth,
                      ),
                      _buildTuningSection(
                        context,
                        "HARDWARE EMULATION",
                        "Adjust virtual hardware properties and error correction.",
                        [
                          _buildInfoTile(
                            "Virtual Clock Speed",
                            "${telemetry.clockSpeed} MHz",
                            "Real-time host-linked frequency calibration.",
                            icon: Icons.timer,
                          ),
                          _buildInfoTile(
                            "ECC Capability",
                            telemetry.eccEnabled ? "ENABLED" : "DISABLED",
                            "Error Correction Code reporting for memory integrity.",
                            icon: Icons.security,
                            color: telemetry.eccEnabled ? VgreTheme.neonGreen : Colors.white38,
                          ),
                        ],
                        constraints.maxWidth,
                      ),
                    ],
                  );
                },
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildHeader() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          "HARDWARE TUNING",
          style: VgreTheme.headingStyle.copyWith(fontSize: 28),
        ),
        const SizedBox(height: 8),
        Text(
          "Authoritative control over virtual hardware and runtime heuristics.",
          style: VgreTheme.bodyStyle.copyWith(color: Colors.white70),
        ),
      ],
    );
  }

  Widget _buildTuningSection(BuildContext context, String title, String subtitle,
      List<Widget> children, double maxWidth) {
    final bool isDesktop = VgreTheme.isDesktop(context);
    double width = isDesktop ? (maxWidth - 48) / 2 : maxWidth;
    return SizedBox(
      width: width,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: VgreTheme.headingStyle.copyWith(fontSize: 16, color: VgreTheme.neonCyan),
          ),
          const SizedBox(height: 4),
          Text(
            subtitle,
            style: VgreTheme.bodyStyle.copyWith(fontSize: 12, color: Colors.white38),
          ),
          const SizedBox(height: 16),
          ...children,
        ],
      ),
    );
  }

  Widget _buildToggleTile(BuildContext context, String title, String description,
      bool value, ValueChanged<bool> onChanged,
      {IconData? icon, Color color = VgreTheme.primaryNeon}) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      child: GlassCard(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon ?? Icons.settings, color: color),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: VgreTheme.bodyStyle.copyWith(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    description,
                    style: VgreTheme.bodyStyle.copyWith(fontSize: 11, color: Colors.white60),
                  ),
                ],
              ),
            ),
            Switch(
              value: value,
              onChanged: onChanged,
              activeThumbColor: Colors.white,
              activeTrackColor: color,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoTile(String title, String value, String description,
      {IconData? icon, Color color = VgreTheme.neonCyan}) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      child: GlassCard(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon ?? Icons.info_outline, color: color),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: VgreTheme.bodyStyle.copyWith(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    description,
                    style: VgreTheme.bodyStyle.copyWith(fontSize: 11, color: Colors.white60),
                  ),
                ],
              ),
            ),
            Text(
              value,
              style: VgreTheme.headingStyle.copyWith(fontSize: 18, color: color),
            ),
          ],
        ),
      ),
    );
  }
}
