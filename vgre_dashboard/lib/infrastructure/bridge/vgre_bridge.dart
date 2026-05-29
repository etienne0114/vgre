// Platform-conditional VgreBridge export.
// On native platforms (Linux, macOS, Windows) the real FFI bridge is used.
// On web (dart.library.html) a compile-time stub is used instead so that
// the dashboard can be compiled for web without pulling in dart:ffi.
export 'vgre_ffi.dart' if (dart.library.html) 'vgre_bridge_web.dart';
