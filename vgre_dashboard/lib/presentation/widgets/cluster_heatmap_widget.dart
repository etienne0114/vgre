import 'package:flutter/material.dart';
import 'dart:math' as math;

class ClusterHeatmapWidget extends StatelessWidget {
  final Map<String, double> nodeTemperatures; // IP -> Temp
  final Map<String, double> nodePowerUsage; // IP -> Watts

  const ClusterHeatmapWidget({
    super.key,
    required this.nodeTemperatures,
    required this.nodePowerUsage,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.3),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.blueAccent.withValues(alpha: 0.5)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.grid_view, color: Colors.orangeAccent),
              const SizedBox(width: 8),
              Text(
                "Authoritative Cluster Heatmap",
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Expanded(
            child: LayoutBuilder(
              builder: (context, constraints) {
                final nodes = nodeTemperatures.keys.toList();
                if (nodes.isEmpty) {
                  return const Center(
                    child: Text(
                      "No remote nodes connected",
                      style: TextStyle(color: Colors.white54, fontSize: 14),
                    ),
                  );
                }
                final columns = math.max(1, (math.sqrt(nodes.length)).ceil());

                return GridView.builder(
                  gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: columns,
                    mainAxisSpacing: 8,
                    crossAxisSpacing: 8,
                  ),
                  itemCount: nodes.length,
                  itemBuilder: (context, index) {
                    final ip = nodes[index];
                    final temp = nodeTemperatures[ip] ?? 0.0;
                    final power = nodePowerUsage[ip] ?? 0.0;

                    // Color gradient based on temperature
                    final color = Color.lerp(
                      Colors.blueAccent.withValues(alpha: 0.8),
                      Colors.redAccent.withValues(alpha: 0.8),
                      (temp - 30) / 50, // 30C to 80C range
                    )!;

                    return Tooltip(
                      message:
                          "Node: $ip\nTemp: ${temp.toStringAsFixed(1)}°C\nPower: ${power.toStringAsFixed(1)}W",
                      child: Container(
                        decoration: BoxDecoration(
                          color: color,
                          borderRadius: BorderRadius.circular(8),
                          boxShadow: [
                            BoxShadow(
                              color: color.withValues(alpha: 0.4),
                              blurRadius: 10,
                              spreadRadius: 2,
                            ),
                          ],
                        ),
                        child: Center(
                          child: Text(
                            "${temp.toStringAsFixed(0)}°",
                            style: const TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.bold,
                              fontSize: 12,
                            ),
                          ),
                        ),
                      ),
                    );
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
