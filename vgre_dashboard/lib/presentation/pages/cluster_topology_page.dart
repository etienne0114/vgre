import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'dart:math' as math;
import 'package:google_fonts/google_fonts.dart';
import '../../application/telemetry_bloc.dart';

import '../../domain/models/telemetry.dart';
import '../widgets/glass_card.dart';
import '../../core/theme/vgre_theme.dart';
import '../widgets/cluster_heatmap_widget.dart';

class ClusterTopologyPage extends StatelessWidget {
  const ClusterTopologyPage({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocConsumer<TelemetryBloc, TelemetryState>(
      listenWhen: (prev, curr) =>
          curr is TelemetryActive && curr.disconnectNotices.isNotEmpty,
      listener: (context, state) {
        if (state is! TelemetryActive) return;
        for (final addr in state.disconnectNotices) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Row(
                children: [
                  const Icon(Icons.wifi_off, color: Colors.redAccent, size: 16),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      'Worker $addr disconnected',
                      style: const TextStyle(color: Colors.white),
                    ),
                  ),
                ],
              ),
              backgroundColor: const Color(0xFF2A0A0A),
              duration: const Duration(seconds: 6),
              behavior: SnackBarBehavior.floating,
            ),
          );
        }
      },
      buildWhen: (prev, curr) {
        if (prev is! TelemetryActive || curr is! TelemetryActive) return true;
        return prev.telemetry != curr.telemetry;
      },
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
              SizedBox(
                height: 200,
                child: ClusterHeatmapWidget(
                  nodeTemperatures: {
                    for (var node in nodes) node.address: 35.0 + (node.latencyMs.clamp(0.0, 50.0)),
                  },
                  nodePowerUsage: {
                    for (var node in nodes) node.address: 5.0 + (node.cpuCores * 15.0),
                  },
                ),
              ),
              const SizedBox(height: 32),
              _Topology3DView(nodes: nodes),
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
          "Real-time 3D node discovery and interconnect visualization.",
          style: VgreTheme.bodyStyle.copyWith(color: Colors.white70),
        ),
      ],
    );
  }

  Widget _buildSecuritySummary(SecurityInfo? info) {
    // isSecure: true when security is active (either fully encrypted or
    // enabled/standby with a configured cipher — not plaintext).
    final bool isSecure =
        (info?.isEncrypted ?? false) || (info?.isSecurityEnabled ?? false);
    final bool isPending =
        isSecure && (info?.cipherName.contains("PENDING") ?? false);

    return GlassCard(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                !isSecure
                    ? Icons.gpp_bad
                    : (info?.isEncrypted == true ? Icons.vpn_lock : Icons.vpn_key),
                color: !isSecure
                    ? Colors.redAccent
                    : (info?.isEncrypted == true
                        ? VgreTheme.neonGreen
                        : Colors.orangeAccent),
                size: 20,
              ),
              const SizedBox(width: 8),
              Text(
                !isSecure
                    ? 'VGRE BUILT-IN VPN: DISABLED'
                    : (info?.isEncrypted == true
                        ? 'VGRE BUILT-IN VPN: SECURED'
                        : (isPending
                            ? 'VGRE BUILT-IN VPN: CONNECTING...'
                            : 'VGRE BUILT-IN VPN: STANDBY')),
                style: TextStyle(
                  color: !isSecure
                      ? Colors.redAccent
                      : (info?.isEncrypted == true
                          ? VgreTheme.neonGreen
                          : Colors.orangeAccent),
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.2,
                ),
              ),
            ],
          ),
          if (info?.isEncrypted == true && info != null) ...[
            // Fully encrypted session — show cipher details and traffic stats
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                _securityStat("CIPHER", info.cipherName),
                _securityStat("FINGERPRINT", info.keyFingerprint),
              ],
            ),
            const SizedBox(height: 12),
            const Divider(color: Colors.white10),
            const SizedBox(height: 8),
            Row(
              children: [
                _securityDetail(
                  "TRAFFIC",
                  "${((info.bytesSent + info.bytesReceived) / 1024).toStringAsFixed(1)} KB",
                  Icons.swap_vert,
                ),
                const SizedBox(width: 32),
                _securityDetail(
                  "PACKETS",
                  "${info.packetsSent + info.packetsReceived} PKTS",
                  Icons.inventory_2_outlined,
                ),
              ],
            ),
          ] else if (isSecure && info != null) ...[
            // Security enabled but no fully-encrypted session yet (STANDBY)
            const SizedBox(height: 8),
            Text(
              info.cipherName,
              style: VgreTheme.bodyStyle.copyWith(
                fontSize: 12,
                color: Colors.orangeAccent,
                fontStyle: FontStyle.italic,
              ),
            ),
          ] else
            Padding(
              padding: const EdgeInsets.only(top: 12),
              child: Text(
                "Phase 5 VPN disabled. Enable 'Secure cluster channel' in Dashboard Settings.",
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
    // Compute countdown seconds remaining before node is removed from the list.
    final disconnectedAt = node.disconnectedAt;
    String? countdownText;
    if (!node.available && disconnectedAt != null) {
      const gracePeriod = 30;
      final elapsed = DateTime.now().difference(disconnectedAt).inSeconds;
      final remaining = gracePeriod - elapsed;
      if (remaining > 0) countdownText = 'Removing in ${remaining}s';
    }

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: node.available
            ? Colors.white.withValues(alpha: 0.05)
            : Colors.red.withValues(alpha: 0.04),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: node.available ? Colors.white10 : Colors.redAccent.withValues(alpha: 0.25),
        ),
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
                  style: VgreTheme.bodyStyle.copyWith(
                    fontWeight: FontWeight.bold,
                    color: node.available ? Colors.white : Colors.white60,
                  ),
                ),
                Text(
                  "CPU: ${node.cpuCores} Cores | RAM: ${(node.memoryBytes / (1024 * 1024 * 1024)).toStringAsFixed(1)} GB",
                  style: VgreTheme.bodyStyle.copyWith(fontSize: 12, color: Colors.white60),
                ),
                if (!node.available) ...[
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      const Icon(Icons.wifi_off, size: 11, color: Colors.redAccent),
                      const SizedBox(width: 4),
                      Text(
                        countdownText ?? 'DISCONNECTED',
                        style: const TextStyle(
                          color: Colors.redAccent,
                          fontSize: 10,
                          letterSpacing: 0.8,
                        ),
                      ),
                    ],
                  ),
                ],
              ],
            ),
          ),
          Column(
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

// ── Animated 3D Topology View ──────────────────────────────────────────────

class _Topology3DView extends StatefulWidget {
  final List<ClusterNode> nodes;

  const _Topology3DView({required this.nodes});

  @override
  State<_Topology3DView> createState() => _Topology3DViewState();
}

class _Topology3DViewState extends State<_Topology3DView>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 20),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GlassCard(
      height: 520,
      child: AnimatedBuilder(
        animation: _controller,
        builder: (context, _) {
          return CustomPaint(
            size: const Size(double.infinity, 520),
            painter: Topology3DPainter(
              nodeCount: widget.nodes.length,
              nodes: widget.nodes,
              rotationAngle: _controller.value * 2 * math.pi,
              pulsePhase: _controller.value,
            ),
          );
        },
      ),
    );
  }
}

// ── 3D Perspective Projected Topology Painter ──────────────────────────────

class _Node3D {
  final double x, y, z;
  _Node3D(this.x, this.y, this.z);
}

class Topology3DPainter extends CustomPainter {
  final int nodeCount;
  final List<ClusterNode> nodes;
  final double rotationAngle;
  final double pulsePhase;

  Topology3DPainter({
    required this.nodeCount,
    required this.nodes,
    required this.rotationAngle,
    required this.pulsePhase,
  });

  /// Perspective projection: maps 3D coordinates to 2D screen space.
  Offset _projectToScreen(_Node3D node, Size size, double focalLength) {
    // Camera at z = -focalLength looking towards +Z
    double perspective = focalLength / (focalLength + node.z);
    double sx = size.width / 2 + node.x * perspective;
    double sy = size.height / 2 + node.y * perspective;
    return Offset(sx, sy);
  }

  /// Scale factor for depth-based size attenuation
  double _depthScale(double z, double focalLength) {
    return focalLength / (focalLength + z);
  }

  @override
  void paint(Canvas canvas, Size size) {
    const double focalLength = 600.0;
    const double orbitRadius = 160.0;

    // Master node at center (slight Z offset for depth)
    final master = _Node3D(0, 0, 0);
    final masterScreen = _projectToScreen(master, size, focalLength);

    // Position worker nodes in a 3D orbital ring
    List<_Node3D> workerPositions = [];
    List<Offset> workerScreenPositions = [];
    List<double> workerDepths = [];

    for (int i = 0; i < nodeCount; i++) {
      double baseAngle = (2 * math.pi * i) / nodeCount;
      double angle = baseAngle + rotationAngle;

      // 3D ring with Y-axis tilt for perspective effect
      double x = orbitRadius * math.cos(angle);
      double z = orbitRadius * math.sin(angle) * 0.4; // Flatten the Z for a tilted ring
      double y = orbitRadius * math.sin(angle) * 0.6; // Some Y spread

      final node3d = _Node3D(x, y, z);
      workerPositions.add(node3d);
      workerScreenPositions.add(_projectToScreen(node3d, size, focalLength));
      workerDepths.add(z);
    }

    // Draw orbital path (elliptical ring)
    _drawOrbitalPath(canvas, size, focalLength, orbitRadius);

    // Sort by depth for correct rendering order (back to front)
    List<int> sortedIndices = List.generate(nodeCount, (i) => i);
    sortedIndices.sort((a, b) => workerDepths[b].compareTo(workerDepths[a]));

    // Draw connections from master to each worker
    for (int idx in sortedIndices) {
      final workerScreen = workerScreenPositions[idx];
      double ds = _depthScale(workerDepths[idx], focalLength);
      double alpha = (ds * 0.8).clamp(0.15, 0.9);

      // Animated pulse: packet traveling along the connection
      double pulse = (pulsePhase * 3 + idx * 0.3) % 1.0;
      Offset pulsePos = Offset.lerp(masterScreen, workerScreen, pulse)!;

      // Connection line with gradient
      final linePaint = Paint()
        ..shader = LinearGradient(
          colors: [
            VgreTheme.neonBlue.withValues(alpha: alpha * 0.6),
            VgreTheme.neonCyan.withValues(alpha: alpha * 0.3),
          ],
        ).createShader(Rect.fromPoints(masterScreen, workerScreen))
        ..strokeWidth = 1.5 * ds
        ..style = PaintingStyle.stroke;

      canvas.drawLine(masterScreen, workerScreen, linePaint);

      // Pulse dot
      final pulsePaint = Paint()
        ..color = VgreTheme.neonCyan.withValues(alpha: alpha * 0.8)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3);
      canvas.drawCircle(pulsePos, 3.0 * ds, pulsePaint);
    }

    // Draw master node
    _drawNodeGlow(canvas, masterScreen, VgreTheme.neonBlue, 22, 1.0, "MASTER");

    // Draw worker nodes (back to front)
    for (int idx in sortedIndices) {
      double ds = _depthScale(workerDepths[idx], focalLength);
      double nodeRadius = 14 * ds;

      Color nodeColor;
      if (idx < nodes.length) {
        nodeColor = nodes[idx].available ? VgreTheme.neonGreen : Colors.redAccent;
      } else {
        nodeColor = VgreTheme.neonGreen;
      }

      String label = "W${idx + 1}";
      _drawNodeGlow(canvas, workerScreenPositions[idx], nodeColor, nodeRadius, ds, label);
    }
  }

  void _drawOrbitalPath(Canvas canvas, Size size, double focalLength, double radius) {
    final path = Path();
    const int segments = 64;
    for (int i = 0; i <= segments; i++) {
      double angle = (2 * math.pi * i / segments) + rotationAngle;
      double x = radius * math.cos(angle);
      double z = radius * math.sin(angle) * 0.4;
      double y = radius * math.sin(angle) * 0.6;

      final pt = _projectToScreen(_Node3D(x, y, z), size, focalLength);
      if (i == 0) {
        path.moveTo(pt.dx, pt.dy);
      } else {
        path.lineTo(pt.dx, pt.dy);
      }
    }
    path.close();

    final orbitPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.06)
      ..strokeWidth = 1.0
      ..style = PaintingStyle.stroke;
    canvas.drawPath(path, orbitPaint);
  }

  void _drawNodeGlow(
      Canvas canvas, Offset pos, Color color, double radius, double scale, String label) {
    // Outer glow
    final glowPaint = Paint()
      ..color = color.withValues(alpha: 0.15 * scale)
      ..maskFilter = MaskFilter.blur(BlurStyle.normal, 12 * scale);
    canvas.drawCircle(pos, radius * 1.6, glowPaint);

    // Node circle background
    final bgPaint = Paint()
      ..color = color.withValues(alpha: 0.12 * scale);
    canvas.drawCircle(pos, radius, bgPaint);

    // Node circle border
    final borderPaint = Paint()
      ..color = color.withValues(alpha: 0.7 * scale)
      ..strokeWidth = 2.0 * scale
      ..style = PaintingStyle.stroke;
    canvas.drawCircle(pos, radius, borderPaint);

    // Label
    final textSpan = TextSpan(
      text: label,
      style: TextStyle(
        color: color.withValues(alpha: 0.9 * scale),
        fontSize: 9 * scale,
        fontWeight: FontWeight.bold,
        letterSpacing: 1.5,
      ),
    );
    final textPainter = TextPainter(
      text: textSpan,
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(
      canvas,
      Offset(pos.dx - textPainter.width / 2, pos.dy + radius + 6 * scale),
    );
  }

  @override
  bool shouldRepaint(covariant Topology3DPainter oldDelegate) =>
      rotationAngle != oldDelegate.rotationAngle ||
      pulsePhase != oldDelegate.pulsePhase ||
      nodeCount != oldDelegate.nodeCount;
}
