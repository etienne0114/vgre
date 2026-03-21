import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class VgreTheme {
  // --- Core Colors ---
  static const Color primaryNeon = Color(0xFF00FFD1);
  static const Color primaryGlow = Color(0x3300FFD1);
  static const Color secondaryNeon = Color(0xFF00A2FF);
  static const Color secondaryGlow = Color(0x3300A2FF);
  
  static const Color neonCyan = Color(0xFF00E5FF);
  static const Color neonGreen = Color(0xFF39FF14);
  static const Color neonBlue = Color(0xFF007BFF);
  static const Color neonPink = Color(0xFFFF00E5);
  static const Color neonRed = Color(0xFFFF3131);

  static const Color background = Color(0xFF060608);
  static const Color surface = Color(0xFF0F1115);
  static const Color surfaceLighter = Color(0xFF1A1D23);
  
  static const Color textBody = Color(0xFFE8E8E8);
  static const Color textMuted = Color(0xFF71717A);

  // --- Gradients ---
  static const LinearGradient primaryGradient = LinearGradient(
    colors: [primaryNeon, neonCyan],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  static const LinearGradient secondaryGradient = LinearGradient(
    colors: [secondaryNeon, neonBlue],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  static const LinearGradient heatGradient = LinearGradient(
    colors: [neonRed, Colors.orangeAccent],
    begin: Alignment.topCenter,
    end: Alignment.bottomCenter,
  );

  // --- Glassmorphism Styles ---
  static BoxDecoration glassDecoration({
    double blur = 20,
    double opacity = 0.05,
    BorderRadius? borderRadius,
  }) =>
      BoxDecoration(
        color: Colors.white.withValues(alpha: opacity),
        borderRadius: borderRadius ?? BorderRadius.circular(24),
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.1),
          width: 1.5,
        ),
      );

  // --- Glow Styles ---
  static List<BoxShadow> neonShadow(Color color, {double blur = 15}) => [
        BoxShadow(
          color: color.withValues(alpha: 0.5),
          blurRadius: blur,
          spreadRadius: -2,
        ),
        BoxShadow(
          color: color.withValues(alpha: 0.2),
          blurRadius: blur * 2,
          spreadRadius: -5,
        ),
      ];

  static TextStyle get bodyStyle => GoogleFonts.inter(
        color: textBody,
        fontSize: 13,
      );

  static TextStyle get headingStyle => GoogleFonts.orbitron(
        color: primaryNeon,
        fontWeight: FontWeight.bold,
        letterSpacing: 2,
      );

  static const double mobileLimit = 600;
  static const double desktopLimit = 1024;

  static bool isMobile(BuildContext context) =>
      MediaQuery.sizeOf(context).width < mobileLimit;

  static bool isTablet(BuildContext context) =>
      MediaQuery.sizeOf(context).width >= mobileLimit &&
      MediaQuery.sizeOf(context).width < desktopLimit;

  static bool isDesktop(BuildContext context) =>
      MediaQuery.sizeOf(context).width >= desktopLimit;

  static ThemeData get darkTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: background,
      colorScheme: const ColorScheme.dark(
        primary: primaryNeon,
        secondary: secondaryNeon,
        surface: surface,
      ),
      dividerTheme: DividerThemeData(
        color: Colors.white.withValues(alpha: 0.05),
        thickness: 1,
      ),
      textSelectionTheme: const TextSelectionThemeData(
        selectionColor: Color(0x6600FFD1),
        selectionHandleColor: primaryNeon,
        cursorColor: primaryNeon,
      ),
      textTheme: GoogleFonts.orbitronTextTheme().copyWith(
        bodyLarge: GoogleFonts.inter(color: textBody),
        bodyMedium: GoogleFonts.inter(color: textBody),
        displayLarge: GoogleFonts.orbitron(
          color: primaryNeon,
          fontWeight: FontWeight.bold,
          letterSpacing: 3,
        ),
      ),
      cardTheme: CardThemeData(
        color: surface.withValues(alpha: 0.9),
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
          side: BorderSide(color: Colors.white.withValues(alpha: 0.05)),
        ),
      ),
    );
  }
}
