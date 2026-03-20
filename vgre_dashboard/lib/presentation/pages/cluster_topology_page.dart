import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'dart:math' as math;
import '../../application/telemetry_bloc.dart';
import '../../domain/models/telemetry.dart';
import '../widgets/glass_card.dart';
import '../../core/theme/vgre_theme.dart';

class ClusterTopologyPage extends StatelessWidget {
  const ClusterTopologyPage({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<TelemetryBloc, TelemetryState>(
      builder: (context, state) {
        if (state is! TelemetryActive) {
          return const Center(child: CircularProgressIndicator());
        }

        final telemetry = state.telemetry;
        final nodes = telemetry.clusterNodes;

        return SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildHeader(),
              const SizedBox(height: 24),
              _buildSecuritySummary(telemetry.securityInfo),
              const SizedBox(height: 32),
              _buildTopologyGraph(context, nodes),
              const SizedBox(height: 32),
              _buildNodeList(nodes),
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
          "CLUSTER TOPOLOGY",
          style: VgreTheme.headingStyle.copyWith(fontSize: 28),
        ),
        const SizedBox(height: 8),
        Text(
          "Real-time node discovery and interconnect visualization.",
          style: VgreTheme.bodyStyle.copyWith(color: Colors.white70),
        ),
      ],
    );
  }

  Widget _buildTopologyGraph(BuildContext context, List<ClusterNode> nodes) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final double size = math.min(constraints.maxWidth, constraints.maxHeight);
        final double radius = (size / 2.8).clamp(100.0, 180.0);

        return GlassCard(
          height: 500,
          child: Stack(
            children: [
              Positioned.fill(
                child: CustomPaint(
                  painter: TopologyPainter(nodes.length, radius: radius),
                ),
              ),
              Center(
                child: _buildNodeIcon(
                  "MASTER",
                  "Localhost",
                  VgreTheme.neonBlue,
                  isMaster: true,
                ),
              ),
              ...List.generate(nodes.length, (index) {
                final angle = (2 * math.pi * index) / nodes.length;
                final x = radius * math.cos(angle);
                final y = radius * math.sin(angle);

                final node = nodes[index];
                return Center(
                  child: Transform.translate(
                    offset: Offset(x, y),
                    child: _buildNodeIcon(
                      "WORKER ${index + 1}",
                      node.address,
                      node.available ? VgreTheme.neonGreen : Colors.redAccent,
                      latency: "${node.latencyMs.toStringAsFixed(1)}ms",
                    ),
                  ),
                );
              }),
            ],
          ),
        );
      },
    );
  }

  Widget _buildNodeIcon(String label, String sublabel, Color color,
      {bool isMaster = false, String? latency}) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: isMaster ? 80 : 60,
          height: isMaster ? 80 : 60,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: color.withValues(alpha: 0.1),
            border: Border.all(color: color, width: 2),
            boxShadow: [
              BoxShadow(
                color: color.withValues(alpha: 0.3),
                blurRadius: 15,
                spreadRadius: 2,
              ),
            ],
          ),
          child: Icon(
            isMaster ? Icons.dns : Icons.settings_input_component,
            color: color,
            size: isMaster ? 40 : 30,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          label,
          style: VgreTheme.bodyStyle.copyWith(
            fontWeight: FontWeight.bold,
            color: color,
            fontSize: 12,
          ),
        ),
        Text(
          sublabel,
          style: VgreTheme.bodyStyle.copyWith(
            fontSize: 10,
            color: Colors.white60,
          ),
        ),
        if (latency != null)
          Text(
            latency,
            style: VgreTheme.bodyStyle.copyWith(
              fontSize: 10,
              color: VgreTheme.neonCyan,
            ),
          ),
      ],
    );
  }

    );
  }

  Widget _buildSecuritySummary(SecurityInfo? info) {
    final bool isSecured = info?.isEncrypted ?? false;
    final color = isSecured ? VgreTheme.neonGreen : Colors.orangeAccent;

    return GlassCard(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                isSecured ? Icons.security : Icons.security_update_warning,
                color: color,
                size: 24,
              ),
              const SizedBox(width: 12),
              Text(
                "CLUSTER SECURITY: ${isSecured ? 'ENCRYPTED' : 'LOW'}",
                style: VgreTheme.headingStyle.copyWith(
                  fontSize: 14,
                  color: color,
                  letterSpacing: 2,
                ),
              ),
              const Spacer(),
              if (isSecured)
                _securityStat("SESSION", "${info!.sessionSeconds.toInt()}s"),
              const SizedBox(width: 24),
              if (isSecured)
                _securityStat("CIPHER", info!.cipherName),
            ],
          ),
          if (isSecured) ...[
            const SizedBox(height: 20),
            Row(
              children: [
                Expanded(
                  child: _securityDetail(
                    "FINGERPRINT",
                    info!.keyFingerprint,
                    Icons.fingerprint,
                  ),
                ),
                const SizedBox(width: 32),
                _securityDetail(
                  "TRAFFIC",
                  "${(info.bytesSent / 1024).toStringAsFixed(1)} KB SENT",
                  Icons.swap_vert,
                ),
                const SizedBox(width: 32),
                _securityDetail(
                  "PACKETS",
                  "${info.packetsSent} PKTS",
                  Icons.inventory_2_outlined,
                ),
              ],
            ),
          ] else
            Padding(
              padding: const EdgeInsets.only(top: 12),
              child: Text(
                "Communication is currently unencrypted. Set VGRE_TCP_AUTH_TOKEN to enable Phase 5 Secure Channel.",
                style: VgreTheme.bodyStyle.copyWith(
                  fontSize: 12,
                  color: Colors.white38,
                  fontStyle: FontStyle.italic,
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _securityStat(String label, String value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        Text(
          label,
          style: const TextStyle(color: Colors.white38, fontSize: 10),
        ),
        Text(
          value,
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
            fontSize: 12,
          ),
        ),
      ],
    );
  }

  Widget _securityDetail(String label, String value, IconData icon) {
    return Row(
      children: [
        Icon(icon, size: 16, color: Colors.white30),
        const SizedBox(width: 8),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              label,
              style: const TextStyle(color: Colors.white30, fontSize: 9),
            ),
            Text(
              value,
              style: const TextStyle(
                color: Colors.white70,
                fontSize: 11,
                fontFamily: 'monospace',
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildNodeList(List<ClusterNode> nodes) {

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          "CONNECTED NODES",
          style: VgreTheme.headingStyle.copyWith(fontSize: 18),
        ),
        const SizedBox(height: 16),
        if (nodes.isEmpty)
          Text(
            "No remote workers detected. Running in standalone mode.",
            style: VgreTheme.bodyStyle.copyWith(color: Colors.white38),
          )
        else
          ...nodes.map((node) => _buildNodeEntry(node)),
      ],
    );
  }

  Widget _buildNodeEntry(ClusterNode node) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white10),
      ),
      child: Row(
        children: [
          _buildStatusIndicator(node.available),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  node.address,
                  style: VgreTheme.bodyStyle.copyWith(fontWeight: FontWeight.bold),
                ),
                Text(
                  "CPU: ${node.cpuCores} Cores | RAM: ${(node.memoryBytes / (1024 * 1024 * 1024)).toStringAsFixed(1)} GB",
                  style: VgreTheme.bodyStyle.copyWith(fontSize: 12, color: Colors.white60),
                ),
              ],
            ),
          ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  "${node.latencyMs.toStringAsFixed(2)} ms",
                  style: VgreTheme.bodyStyle.copyWith(
                    color: VgreTheme.neonCyan,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  "LATENCY",
                  style: VgreTheme.bodyStyle.copyWith(fontSize: 10, color: Colors.white38),
                ),
              ],
            ),
          ),
          const SizedBox(width: 24),
          _buildBillingCol(node),
        ],
      ),
    );
  }

  Widget _buildBillingCol(ClusterNode node) {
    return SizedBox(
      width: 120,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.account_balance_wallet_outlined, size: 14, color: VgreTheme.neonGreen),
              const SizedBox(width: 4),
              Text(
                node.balance.toStringAsFixed(2),
                style: GoogleFonts.firaCode(
                  color: VgreTheme.neonGreen,
                  fontWeight: FontWeight.bold,
                  fontSize: 14,
                ),
              ),
            ],
          ),
          Text(
            "CREDITS",
            style: VgreTheme.bodyStyle.copyWith(fontSize: 10, color: Colors.white38),
          ),
          const SizedBox(height: 4),
          Text(
            "${node.transactionCount} TX",
            style: VgreTheme.bodyStyle.copyWith(fontSize: 9, color: Colors.white30),
          ),
        ],
      ),
    );
  }


  Widget _buildStatusIndicator(bool active) {
    return Container(
      width: 12,
      height: 12,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: active ? VgreTheme.neonGreen : Colors.redAccent,
        boxShadow: [
          BoxShadow(
            color: (active ? VgreTheme.neonGreen : Colors.redAccent).withValues(alpha: 0.5),
            blurRadius: 8,
          ),
        ],
      ),
    );
  }
}

class TopologyPainter extends CustomPainter {
  final int nodeCount;
  final double radius;

  TopologyPainter(this.nodeCount, {required this.radius});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white12
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;

    final center = Offset(size.width / 2, size.height / 2);

    for (int i = 0; i < nodeCount; i++) {
      final angle = (2 * math.pi * i) / nodeCount;
      final x = center.dx + radius * math.cos(angle);
      final y = center.dy + radius * math.sin(angle);

      // Draw connection line
      canvas.drawLine(center, Offset(x, y), paint);

      // Draw pulse glow effect
      final glowPaint = Paint()
        ..shader = LinearGradient(
          colors: [VgreTheme.neonBlue.withValues(alpha: 0.5), VgreTheme.neonBlue.withValues(alpha: 0)],
        ).createShader(Rect.fromPoints(center, Offset(x, y)))
        ..strokeWidth = 3;
      
      canvas.drawLine(center, Offset(x, y), glowPaint);
    }

    // Draw circular orbits
    canvas.drawCircle(center, radius, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
