import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import '../../application/telemetry_bloc.dart';
import '../../core/theme/vgre_theme.dart';
import '../widgets/glass_card.dart';

class DashboardSettingsPage extends StatelessWidget {
  const DashboardSettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<TelemetryBloc, TelemetryState>(
      builder: (context, state) {
        if (state is! TelemetryActive) {
          return const Center(child: CircularProgressIndicator());
        }

        final telemetry = state.telemetry;
        final isDesktop = VgreTheme.isDesktop(context);
        final double panelWidth = isDesktop
            ? (MediaQuery.sizeOf(context).width - 96) / 2
            : double.infinity;

        return SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                "DASHBOARD SETTINGS",
                style: VgreTheme.headingStyle.copyWith(fontSize: 28),
              ),
              const SizedBox(height: 8),
              Text(
                "Control profiler, security, and credit ledger state for the live backend.",
                style: VgreTheme.bodyStyle.copyWith(color: Colors.white70),
              ),
              const SizedBox(height: 24),
              Wrap(
                spacing: 24,
                runSpacing: 24,
                children: [
                  SizedBox(
                    width: panelWidth,
                    child: GlassCard(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            "BACKEND INFO",
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              letterSpacing: 1.5,
                              fontSize: 12,
                            ),
                          ),
                          const SizedBox(height: 16),
                          _infoRow("Version", telemetry.backendVersion),
                          const SizedBox(height: 8),
                          _infoRow("Protocol", "v${telemetry.versionMajor}.${telemetry.versionMinor}.${telemetry.versionPatch}"),
                          const SizedBox(height: 8),
                          _infoRow(
                            "Last Update",
                            telemetry.timestamp.toIso8601String(),
                          ),
                        ],
                      ),
                    ),
                  ),
                  SizedBox(
                    width: panelWidth,
                    child: GlassCard(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            "PROFILER",
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              letterSpacing: 1.5,
                              fontSize: 12,
                            ),
                          ),
                          const SizedBox(height: 16),
                          _toggleRow(
                            context,
                            label: "Enable profiler",
                            description:
                                "Collect kernel metrics for every launch and chart visibility.",
                            enabled: telemetry.profilerEnabled,
                            onChanged: (value) => context
                                .read<TelemetryBloc>()
                                .add(ToggleProfiler(value)),
                            color: VgreTheme.primaryNeon,
                          ),
                          const Divider(color: Colors.white10),
                          _toggleRow(
                            context,
                            label: "Secure cluster channel",
                            description:
                                "Toggle Phase 5 encryption for remote workers (requires VPN).",
                            enabled: telemetry.clusterSecurityActive,
                            onChanged: telemetry.clusterSecuritySupported
                                ? (value) => context
                                    .read<TelemetryBloc>()
                                    .add(ToggleClusterSecurity(value))
                                : null,
                            color: VgreTheme.neonGreen,
                          ),
                          if (!telemetry.clusterSecuritySupported)
                            Padding(
                              padding: const EdgeInsets.only(top: 4),
                              child: Text(
                                "Set VGRE_TCP_AUTH_TOKEN before enabling Phase 5 security.",
                                style: VgreTheme.bodyStyle.copyWith(
                                  fontSize: 10,
                                  color: Colors.orangeAccent,
                                ),
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
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
                          "CLUSTER SECURITY",
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            letterSpacing: 1.5,
                            fontSize: 12,
                          ),
                        ),
                        const Spacer(),
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 10,
                            vertical: 4,
                          ),
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(4),
                            border: Border.all(
                              color: !telemetry.clusterSecurityActive
                                  ? Colors.white10
                                  : (telemetry.securityInfo?.isEncrypted == true
                                      ? VgreTheme.neonGreen
                                      : Colors.orange),
                            ),
                          ),
                          child: Text(
                            !telemetry.clusterSecurityActive
                                ? "DISABLED"
                                : (telemetry.securityInfo?.isEncrypted == true
                                    ? "ENCRYPTED"
                                    : "STANDBY"),
                            style: TextStyle(
                              color: !telemetry.clusterSecurityActive
                                  ? VgreTheme.textMuted
                                  : (telemetry.securityInfo?.isEncrypted == true
                                      ? VgreTheme.neonGreen
                                      : Colors.orange),
                              fontSize: 11,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        )
                      ],
                    ),
                    const SizedBox(height: 16),
                    Text(
                      telemetry.securityInfo?.cipherName ?? "No secure cipher configured",
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      telemetry.securityInfo?.keyFingerprint ?? "Fingerprint unavailable",
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: Colors.white60,
                            fontFamily: 'monospace',
                          ),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        _securityMetric("Packets", telemetry.securityInfo?.packetsSent ?? 0),
                        const SizedBox(width: 16),
                        _securityMetric("Bytes", telemetry.securityInfo?.bytesSent ?? 0),
                        const SizedBox(width: 16),
                        _securityMetric("Sessions", telemetry.securityInfo?.sessionSeconds.toInt() ?? 0),
                      ],
                    )
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
                        const Text(
                          "CREDIT LEDGER",
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            letterSpacing: 1.5,
                            fontSize: 12,
                          ),
                        ),
                        const Spacer(),
                        ElevatedButton.icon(
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.white10,
                            foregroundColor: Colors.white,
                            minimumSize: const Size(140, 36),
                          ),
                          onPressed: () {
                            context.read<TelemetryBloc>().add(const ResetCredits());
                          },
                          icon: const Icon(Icons.refresh, size: 16),
                          label: const Text("Reset Credits"),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    if (telemetry.creditLedger.isEmpty)
                      Padding(
                        padding: const EdgeInsets.symmetric(vertical: 24),
                        child: Text(
                          "No entries in the resource ledger yet.",
                          style: VgreTheme.bodyStyle.copyWith(color: Colors.white54),
                        ),
                      )
                    else
                      ListView.separated(
                        shrinkWrap: true,
                        physics: const NeverScrollableScrollPhysics(),
                        itemCount: telemetry.creditLedger.length,
                        separatorBuilder: (context, index) => const Divider(color: Colors.white10),
                        itemBuilder: (context, index) {
                          final entry = telemetry.creditLedger[index];
                          return ListTile(
                            contentPadding: EdgeInsets.zero,
                            title: Text(
                              entry.address,
                              style: const TextStyle(fontWeight: FontWeight.bold),
                            ),
                            subtitle: Text(
                              "Credits: ${entry.totalCredits.toStringAsFixed(2)} · Debits: ${entry.totalDebits.toStringAsFixed(2)}",
                              style: const TextStyle(color: VgreTheme.textMuted),
                            ),
                            trailing: Column(
                              crossAxisAlignment: CrossAxisAlignment.end,
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Text(
                                  entry.balance.toStringAsFixed(2),
                                  style: const TextStyle(
                                    color: VgreTheme.neonGreen,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                Text(
                                  "${entry.transactionCount} TX",
                                  style: const TextStyle(fontSize: 10, color: Colors.white54),
                                ),
                              ],
                            ),
                          );
                        },
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

  Widget _infoRow(String label, String value) {
    return Row(
      children: [
        Text(
          "$label:",
          style: const TextStyle(color: VgreTheme.textMuted, fontSize: 11),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            value,
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
        ),
      ],
    );
  }

  Widget _toggleRow(
    BuildContext context, {
    required String label,
    required String description,
    required bool enabled,
    ValueChanged<bool>? onChanged,
    Color color = VgreTheme.primaryNeon,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(Icons.toggle_on, color: color),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label, style: const TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 2),
                Text(
                  description,
                  style: VgreTheme.bodyStyle.copyWith(fontSize: 11, color: Colors.white60),
                ),
              ],
            ),
          ),
          Switch(
            value: enabled,
            onChanged: onChanged,
            activeThumbColor: Colors.white,
            activeTrackColor: color,
          ),
        ],
      ),
    );
  }

  Widget _securityMetric(String label, int value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(color: VgreTheme.textMuted, fontSize: 10),
        ),
        const SizedBox(height: 4),
        Text(
          value.toString(),
          style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
        ),
      ],
    );
  }
}
