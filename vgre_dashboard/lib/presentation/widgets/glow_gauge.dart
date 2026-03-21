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
    return LayoutBuilder(
      builder: (context, constraints) {
        final double labelHeight = 22.0;
        final double infoItemHeight = 16.0;
        final double reservedHeight = labelHeight + 16 + 8 + (info.length * infoItemHeight);
        final double initialGaugeSize = (constraints.maxHeight - reservedHeight).clamp(70.0, 160.0);

        return Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 3,
                  height: 14,
                  decoration: BoxDecoration(
                    color: color,
                    borderRadius: BorderRadius.circular(2),
                    boxShadow: VgreTheme.neonShadow(color, blur: 8),
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  label.toUpperCase(),
                  style: GoogleFonts.orbitron(
                    letterSpacing: 2,
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Expanded(
              child: Center(
                child: TweenAnimationBuilder<double>(
                  tween: Tween<double>(begin: 0, end: value),
                  duration: const Duration(milliseconds: 800),
                  curve: Curves.easeOutCubic,
                  builder: (context, animValue, child) {
                    return SizedBox(
                      width: initialGaugeSize,
                      height: initialGaugeSize,
                      child: Stack(
                        alignment: Alignment.center,
                        children: [
                          CustomPaint(
                            size: Size(initialGaugeSize, initialGaugeSize),
                            painter: _GaugePainter(
                              percent: animValue / max,
                              color: color,
                            ),
                          ),
                          Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Text(
                                animValue >= 1000 
                                    ? (animValue / 1000).toStringAsFixed(1) 
                                    : animValue.toStringAsFixed(1),
                                style: GoogleFonts.orbitron(
                                  fontSize: initialGaugeSize < 100 ? 18 : 28,
                                  fontWeight: FontWeight.w900,
                                  color: Colors.white,
                                  shadows: [
                                    Shadow(
                                      color: color.withValues(alpha: 0.5),
                                      blurRadius: 12,
                                    ),
                                  ],
                                ),
                              ),
                              Text(
                                animValue >= 1000 ? "T${unit.substring(1)}" : unit,
                                style: GoogleFonts.inter(
                                  color: VgreTheme.textMuted,
                                  fontSize: initialGaugeSize < 100 ? 9 : 11,
                                  fontWeight: FontWeight.bold,
                                  letterSpacing: 1,
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ),
            ),
            const SizedBox(height: 12),
            if (constraints.maxHeight > 200) 
              ListView(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                children: info.map((item) => Padding(
                      padding: const EdgeInsets.only(bottom: 4),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(item['key']!.toUpperCase(),
                              style: const TextStyle(
                                  color: VgreTheme.textMuted,
                                  fontSize: 9,
                                  fontWeight: FontWeight.w600)),
                          Text(item['value']!,
                              style: GoogleFonts.firaCode(
                                  color: color,
                                  fontSize: 10,
                                  fontWeight: FontWeight.bold)),
                        ],
                      ),
                    )).toList(),
              ),
          ],
        );
      },
    );
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
    const strokeWidth = 10.0;

    // Background track (Glassy)
    final bgPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.03)
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth;

    canvas.drawCircle(center, radius - strokeWidth / 2, bgPaint);

    final rect = Rect.fromCircle(center: center, radius: radius - strokeWidth / 2);
    const startAngle = -pi / 2;
    final sweepAngle = 2 * pi * percent;

    // Glow layers for depth
    for (int i = 1; i <= 3; i++) {
      final glowPaint = Paint()
        ..color = color.withValues(alpha: 0.3 / i)
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth + (i * 8)
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, i * 6.0);
      canvas.drawArc(rect, startAngle, sweepAngle, false, glowPaint);
    }

    // Main Progress arc
    final progressPaint = Paint()
      ..shader = SweepGradient(
        colors: [color.withValues(alpha: 0.5), color],
        stops: const [0.0, 1.0],
        transform: const GradientRotation(-pi / 2),
      ).createShader(rect)
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    canvas.drawArc(rect, startAngle, sweepAngle, false, progressPaint);

    // Dynamic Dot at the end
    final dotPaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.fill;
    
    final dotGlow = Paint()
      ..color = color.withValues(alpha: 0.8)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8);

    final dotPos = Offset(
      center.dx + (radius - strokeWidth / 2) * cos(startAngle + sweepAngle),
      center.dy + (radius - strokeWidth / 2) * sin(startAngle + sweepAngle),
    );

    canvas.drawCircle(dotPos, 4, dotGlow);
    canvas.drawCircle(dotPos, 2.5, dotPaint);
  }

  @override
  bool shouldRepaint(covariant _GaugePainter oldDelegate) =>
      oldDelegate.percent != percent;
}
