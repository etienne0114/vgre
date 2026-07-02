import 'package:url_launcher/url_launcher.dart';

/// Canonical VGRE documentation destinations, opened from the dashboard's
/// "Help & Docs" navigation. Each entry deep-links to the matching page of the
/// docs website (docs/site/, served at the base URL below); clicking it opens
/// the user's default browser.
class DocsLinks {
  /// Public docs base. Override at build time with
  /// `--dart-define=VGRE_DOCS_URL=https://your-host/docs`.
  static const String base = String.fromEnvironment(
    'VGRE_DOCS_URL',
    defaultValue: 'https://vgre.dev/docs',
  );

  static String page(String file) => '$base/$file';

  static const String home = base;
  static String get quickStart => page('quickstart.html');
  static String get userGuide => page('running-cuda.html');
  static String get apiReference => page('api.html');
  static String get localAi => page('local-ai.html');
  static String get cluster => page('cluster.html');
  static String get debugging => page('debugging.html');
  static String get github => 'https://github.com/vgre-org/vgre-runtime';

  /// Opens [url] in an external browser. Returns false if no handler exists
  /// (surfaced to the caller so the UI can show a fallback).
  static Future<bool> open(String url) async {
    final uri = Uri.parse(url);
    if (!await canLaunchUrl(uri)) return false;
    return launchUrl(uri, mode: LaunchMode.externalApplication);
  }
}
