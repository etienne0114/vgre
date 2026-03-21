import 'dart:async';
import 'dart:io';
import 'package:path/path.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';
import 'package:path_provider/path_provider.dart';
import '../../domain/models/telemetry.dart';

class SqliteService {
  static Database? _database;
  static const String _dbName = 'vgre_metrics.db';

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    // Initialize FFI for desktop support if needed
    if (Platform.isWindows || Platform.isLinux) {
      sqfliteFfiInit();
      databaseFactory = databaseFactoryFfi;
    }

    final directory = await getApplicationSupportDirectory();
    final path = join(directory.path, _dbName);

    return await openDatabase(
      path,
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE telemetry_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            deviceId INTEGER NOT NULL,
            gflops REAL,
            maxGflops REAL,
            computeUtilization REAL,
            memoryBandwidth REAL,
            maxMemoryBandwidth REAL,
            memoryBusUtilization REAL,
            memoryUsed INTEGER,
            memoryTotal INTEGER,
            temperature REAL,
            clockSpeed INTEGER,
            activeKernels INTEGER,
            activeThreads INTEGER
          )
        ''');

        await db.execute('''
          CREATE TABLE kernel_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            deviceId INTEGER NOT NULL,
            durationMs REAL,
            throughputGbps REAL,
            gflops REAL,
            threadsUsed INTEGER
          )
        ''');

        await db.execute('CREATE INDEX idx_telemetry_ts ON telemetry_history(timestamp)');
        await db.execute('CREATE INDEX idx_kernel_name_ts ON kernel_history(name, timestamp)');
      },
    );
  }

  // ── Telemetry Operations ───────────────────────────────────────────────────

  Future<void> saveTelemetry(Telemetry telemetry, int deviceId) async {
    final db = await database;
    await db.insert(
      'telemetry_history',
      {
        'timestamp': telemetry.timestamp.millisecondsSinceEpoch,
        'deviceId': deviceId,
        'gflops': telemetry.gflops,
        'maxGflops': telemetry.maxGflops,
        'computeUtilization': telemetry.computeUtilization,
        'memoryBandwidth': telemetry.memoryBandwidth,
        'maxMemoryBandwidth': telemetry.maxMemoryBandwidth,
        'memoryBusUtilization': telemetry.memoryBusUtilization,
        'memoryUsed': telemetry.memoryUsed,
        'memoryTotal': telemetry.memoryTotal,
        'temperature': telemetry.temperature,
        'clockSpeed': telemetry.clockSpeed,
        'activeKernels': telemetry.activeKernels,
        'activeThreads': telemetry.activeThreads,
      },
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  Future<List<Telemetry>> getTelemetryHistory(int deviceId, {int limit = 100}) async {
    final db = await database;
    final List<Map<String, dynamic>> maps = await db.query(
      'telemetry_history',
      where: 'deviceId = ?',
      whereArgs: [deviceId],
      orderBy: 'timestamp DESC',
      limit: limit,
    );

    return List.generate(maps.length, (i) {
      final m = maps[i];
      return Telemetry(
        timestamp: DateTime.fromMillisecondsSinceEpoch(m['timestamp']),
        gflops: m['gflops'],
        maxGflops: m['maxGflops'],
        computeUtilization: m['computeUtilization'],
        memoryBandwidth: m['memoryBandwidth'],
        maxMemoryBandwidth: m['maxMemoryBandwidth'],
        memoryBusUtilization: m['memoryBusUtilization'],
        memoryUsed: m['memoryUsed'],
        memoryTotal: m['memoryTotal'],
        totalPages: 0, // Not persisted for history to keep DB light
        residentPages: 0,
        evictedPages: 0,
        pageFaultRate: 0,
        uvmMap: const [],
        activeKernels: m['activeKernels'],
        activeThreads: m['activeThreads'],
        clockSpeed: m['clockSpeed'],
        avgLatency: 0,
        temperature: m['temperature'],
        eccEnabled: false,
      );
    }).reversed.toList(); // Return in chronological order for charts
  }

  // ── Kernel History Operations ──────────────────────────────────────────────

  Future<void> saveKernelExecution(String name, KernelExecution exec, int deviceId) async {
    final db = await database;
    await db.insert(
      'kernel_history',
      {
        'name': name,
        'timestamp': exec.timestamp.millisecondsSinceEpoch,
        'deviceId': deviceId,
        'durationMs': exec.durationMs,
        'throughputGbps': exec.throughputGbps,
        'gflops': exec.gflops,
        'threadsUsed': exec.threadsUsed,
      },
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  Future<List<KernelExecution>> getKernelHistory(String name, int deviceId) async {
    final db = await database;
    final List<Map<String, dynamic>> maps = await db.query(
      'kernel_history',
      where: 'name = ? AND deviceId = ?',
      whereArgs: [name, deviceId],
      orderBy: 'timestamp ASC',
    );

    return List.generate(maps.length, (i) {
      final m = maps[i];
      return KernelExecution(
        timestamp: DateTime.fromMillisecondsSinceEpoch(m['timestamp']),
        durationMs: m['durationMs'],
        throughputGbps: m['throughputGbps'],
        gflops: m['gflops'],
        threadsUsed: m['threadsUsed'],
      );
    });
  }

  Future<void> clearHistory() async {
    final db = await database;
    await db.delete('telemetry_history');
    await db.delete('kernel_history');
  }
}
