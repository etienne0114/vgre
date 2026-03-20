import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class VgreTheme {
  static const Color primaryNeon = Color(0xFF00FFD1);
  static const Color secondaryNeon = Color(0xFF00A2FF);
  static const Color background = Color(0xFF0A0A0B);
  static const Color surface = Color(0xFF161618);
  static const Color textBody = Color(0xFFE1E1E1);
  static const Color textMuted = Color(0xFF888888);
  static const Color neonCyan = Color(0xFF00E5FF);
  static const Color neonGreen = Color(0xFF39FF14);
  static const Color neonBlue = Color(0xFF007BFF);

  static TextStyle get bodyStyle => GoogleFonts.inter(
        color: textBody,
        fontSize: 13,
      );

  static TextStyle get headingStyle => GoogleFonts.orbitron(
        color: primaryNeon,
        fontWeight: FontWeight.bold,
        letterSpacing: 1.5,
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
      textSelectionTheme: const TextSelectionThemeData(
        selectionColor: Color(0x4D00FFD1), // primaryNeon with 0.3 alpha
        selectionHandleColor: primaryNeon,
        cursorColor: primaryNeon,
      ),
      textTheme: GoogleFonts.orbitronTextTheme().copyWith(
        bodyLarge: GoogleFonts.inter(color: textBody),
        bodyMedium: GoogleFonts.inter(color: textBody),
        displayLarge: GoogleFonts.orbitron(
          color: primaryNeon,
          fontWeight: FontWeight.bold,
          letterSpacing: 2,
        ),
      ),
      cardTheme: CardThemeData(
        color: surface.withValues(alpha: 0.8),
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: textMuted.withValues(alpha: 0.2)),
        ),
      ),
    );
  }
}
