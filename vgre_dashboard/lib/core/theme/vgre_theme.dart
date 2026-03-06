import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class VgreTheme {
  static const Color primaryNeon = Color(0xFF00FFD1);
  static const Color secondaryNeon = Color(0xFF00A2FF);
  static const Color background = Color(0xFF0A0A0B);
  static const Color surface = Color(0xFF161618);
  static const Color textBody = Color(0xFFE1E1E1);
  static const Color textMuted = Color(0xFF888888);

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
        color: surface.withOpacity(0.8),
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: textMuted.withOpacity(0.2)),
        ),
      ),
    );
  }
}
