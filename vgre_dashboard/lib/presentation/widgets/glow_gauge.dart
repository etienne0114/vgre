import 'dart:math';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../core/theme/vgre_theme.dart';

class GlowGauge extends StatelessWidget {
  final double value;
  final double max;
  final String label;
  final String unit;
  final Color color;
  final List<Map<String, String>> info;

  const GlowGauge({
    super.key,
    required this.value,
    this.max = 100,
    required this.label,
    required this.unit,
    this.color = VgreTheme.primaryNeon,
    this.info = const [],
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label.toUpperCase(),
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                letterSpacing: 2,
                fontWeight: FontWeight.bold,
                color: Colors.white70,
              ),
        ),
        const SizedBox(height: 12),
        Center(
          child: SizedBox(
            width: 140,
            height: 140,
            child: Stack(
              alignment: Alignment.center,
              children: [
                CustomPaint(
                  size: const Size(140, 140),
                  painter: _GaugePainter(
                    percent: value / max,
                    color: color,
                  ),
                ),
                Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      value >= 1000 ? (value / 1000).toStringAsFixed(1) : value.toStringAsFixed(1),
                      style: Theme.of(context).textTheme.displayLarge?.copyWith(
                            fontSize: 32,
                            color: Colors.white,
                          ),
                    ),
                    Text(
                      value >= 1000 ? "T${unit.substring(1)}" : unit,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: VgreTheme.textMuted,
                            fontSize: 12,
                          ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      "Max: ${max.toStringAsFixed(1)} $unit",
                      style: const TextStyle(color: VgreTheme.textMuted, fontSize: 10),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 4),
        // Gauge Info Rows
        Flexible(
          child: ListView(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            children: info.map((item) => Padding(
                  padding: const EdgeInsets.only(bottom: 2),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(item['key']!.toUpperCase(),
                          style: const TextStyle(
                              color: VgreTheme.textMuted,
                              fontSize: 9,
                              fontWeight: FontWeight.bold)),
                      Text(item['value']!,
                          style: GoogleFonts.firaCode(
                              color: color,
                              fontSize: 9,
                              fontWeight: FontWeight.bold)),
                    ],
                  ),
                )).toList(),
          ),
        ),
      ],
    );
  }

  IconData _getIcon(String key) {
    switch (key.toLowerCase()) {
      case 'util':
        return Icons.speed;
      case 'cores':
        return Icons.memory;
      case 'temp':
        return Icons.thermostat;
      case 'clock':
        return Icons.timer;
      case 'hbm2':
        return Icons.storage;
      default:
        return Icons.info_outline;
    }
  }
}

class _GaugePainter extends CustomPainter {
  final double percent;
  final Color color;

  _GaugePainter({required this.percent, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = min(size.width / 2, size.height / 2);
    const strokeWidth = 8.0;

    // Background track
    final bgPaint = Paint()
      ..color = Colors.white.withOpacity(0.05)
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth;

    canvas.drawCircle(center, radius - strokeWidth / 2, bgPaint);

    // Progress arc
    final progressPaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    // Glow effect
    final glowPaint = Paint()
      ..color = color.withOpacity(0.3)
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth * 2.5
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 10);

    final rect = Rect.fromCircle(center: center, radius: radius - strokeWidth / 2);
    const startAngle = -pi / 2;
    final sweepAngle = 2 * pi * percent;

    canvas.drawArc(rect, startAngle, sweepAngle, false, glowPaint);
    canvas.drawArc(rect, startAngle, sweepAngle, false, progressPaint);
  }

  @override
  bool shouldRepaint(covariant _GaugePainter oldDelegate) =>
      oldDelegate.percent != percent;
}
