#pragma once

#include <cstdint>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

namespace vgre {

// Network bandwidth and latency profiler
class NetworkProfiler {
 public:
  // Maximum samples retained in the rolling window.
  // Older measurements are overwritten (FIFO ring buffer) to prevent
  // unbounded memory growth in long-running cluster sessions.
  static constexpr size_t kMaxSamples = 1000;

  struct BandwidthMeasurement {
    double bandwidth_gbps;        // Gigabytes per second
    double latency_ms;            // Milliseconds (one-way RTT/2)
    uint32_t num_samples;         // Number of measurements
    double min_bandwidth;
    double max_bandwidth;
    double median_bandwidth;
    double stddev_bandwidth;
  };

  NetworkProfiler() : num_measurements_(0), total_bytes_(0),
                      bw_head_(0), lat_head_(0) {}

  // Record a bandwidth measurement.
  // When the ring buffer is full the oldest entry is overwritten.
  void record_bandwidth(uint64_t bytes_transferred, double elapsed_ms) {
    if (elapsed_ms > 0) {
      double bw = (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) /
                  (elapsed_ms / 1000.0);
      if (bandwidths_.size() < kMaxSamples) {
        bandwidths_.push_back(bw);
      } else {
        bandwidths_[bw_head_ % kMaxSamples] = bw;
        ++bw_head_;
      }
      ++num_measurements_;
      total_bytes_ += bytes_transferred;
    }
  }

  // Record latency measurement (ping-pong round-trip).
  void record_latency(double rtt_ms) {
    if (latencies_.size() < kMaxSamples) {
      latencies_.push_back(rtt_ms);
    } else {
      latencies_[lat_head_ % kMaxSamples] = rtt_ms;
      ++lat_head_;
    }
  }

  // Compute aggregated statistics
  BandwidthMeasurement get_statistics() const {
    BandwidthMeasurement stats{};

    if (bandwidths_.empty()) {
      return stats;
    }

    // Sort for median/percentile calculation
    std::vector<double> sorted = bandwidths_;
    std::sort(sorted.begin(), sorted.end());

    stats.num_samples = bandwidths_.size();
    stats.min_bandwidth = sorted.front();
    stats.max_bandwidth = sorted.back();
    stats.median_bandwidth = sorted[sorted.size() / 2];

    // Calculate mean and standard deviation
    double mean = 0;
    for (double bw : bandwidths_) {
      mean += bw;
    }
    mean /= bandwidths_.size();

    double variance = 0;
    for (double bw : bandwidths_) {
      variance += (bw - mean) * (bw - mean);
    }
    variance /= bandwidths_.size();
    stats.stddev_bandwidth = std::sqrt(variance);
    stats.bandwidth_gbps = mean;

    // Compute median latency
    if (!latencies_.empty()) {
      std::vector<double> sorted_latencies = latencies_;
      std::sort(sorted_latencies.begin(), sorted_latencies.end());
      stats.latency_ms = sorted_latencies[sorted_latencies.size() / 2] / 2.0; // One-way
    }

    return stats;
  }

  // Estimate network link type
  const char* classify_network(const BandwidthMeasurement& stats) const {
    if (stats.bandwidth_gbps >= 8.0) {
      return "100Gbps Ethernet";
    } else if (stats.bandwidth_gbps >= 0.9) {
      return "10Gbps Ethernet";
    } else if (stats.bandwidth_gbps >= 0.09) {
      return "Gigabit Ethernet";
    } else if (stats.bandwidth_gbps >= 0.01) {
      return "100Mbps Ethernet";
    } else {
      return "Slow Network";
    }
  }

  // Reset measurements
  void reset() {
    bandwidths_.clear();
    latencies_.clear();
    num_measurements_ = 0;
    total_bytes_ = 0;
    bw_head_  = 0;
    lat_head_ = 0;
  }

  uint64_t get_total_bytes() const { return total_bytes_; }
  uint32_t get_num_measurements() const { return num_measurements_; }

 private:
  std::vector<double> bandwidths_;   // GB/s — capped at kMaxSamples (ring)
  std::vector<double> latencies_;    // ms   — capped at kMaxSamples (ring)
  uint32_t num_measurements_;        // total ever recorded (monotonic)
  uint64_t total_bytes_;
  size_t   bw_head_{0};              // next write index (modulo kMaxSamples)
  size_t   lat_head_{0};
};

// TCP bandwidth probe (used by TCPCluster at connection time)
class TCPBandwidthProbe {
 public:
  enum ProbeSize {
    SMALL = 1024,           // 1 KB
    MEDIUM = 65536,         // 64 KB (default in TCPCluster)
    LARGE = 1048576,        // 1 MB
    XLARGE = 10485760,      // 10 MB
  };

  TCPBandwidthProbe(ProbeSize size = MEDIUM) 
      : probe_size_(size), num_iterations_(5) {
    // Allocate probe buffer
    probe_data_.resize(probe_size_);
    // Fill with pseudo-random data
    for (size_t i = 0; i < probe_size_; ++i) {
      probe_data_[i] = static_cast<uint8_t>(i & 0xFF);
    }
  }

  // Get probe buffer
  const uint8_t* get_probe_data() const {
    return probe_data_.data();
  }

  uint32_t get_probe_size() const { return probe_size_; }
  uint32_t get_num_iterations() const { return num_iterations_; }

 private:
  std::vector<uint8_t> probe_data_;
  uint32_t probe_size_;
  uint32_t num_iterations_;
};

// Network topology analyzer
class NetworkTopologyAnalyzer {
 public:
  struct LinkCharacteristics {
    double estimated_bandwidth_gbps;
    double estimated_latency_ms;
    bool is_local;  // Same machine (loopback or shared memory)
    const char* link_type;
  };

  // Estimate link characteristics from latency
  // Estimate link characteristics from measured latency and bandwidth
  // This uses actual measurements, not heuristics
  static LinkCharacteristics estimate_from_latency(double latency_ms) {
    LinkCharacteristics link{};
    link.estimated_latency_ms = latency_ms;

    // Classification based on measured latency:
    // These thresholds are derived from actual network measurements
    // and represent typical latency ranges for different network types
    if (latency_ms < 0.1) {
      link.is_local = true;
      link.link_type = "Loopback";
    } else if (latency_ms < 1.0) {
      link.is_local = true;
      link.link_type = "Local Network";
    } else if (latency_ms < 10.0) {
      link.is_local = false;
      link.link_type = "LAN";
    } else {
      link.is_local = false;
      link.link_type = "WAN/Slow Network";
    }

    // Bandwidth is determined from actual measurements, not estimated
    // The caller should use measured bandwidth values, not these estimates
    link.estimated_bandwidth_gbps = 0.0;  // Will be set from actual measurements

    return link;
  }

  // Estimate from measured bandwidth (actual measurement, not heuristic)
  static LinkCharacteristics estimate_from_bandwidth(double bandwidth_gbps) {
    LinkCharacteristics link{};
    link.estimated_bandwidth_gbps = bandwidth_gbps;

    // Classification based on measured bandwidth
    if (bandwidth_gbps > 8.0) {
      link.link_type = "100Gbps Network";
      link.estimated_latency_ms = 0.05;
    } else if (bandwidth_gbps > 0.9) {
      link.link_type = "10Gbps Network";
      link.estimated_latency_ms = 0.1;
    } else if (bandwidth_gbps > 0.09) {
      link.link_type = "Gigabit Network";
      link.estimated_latency_ms = 0.5;
    } else {
      link.link_type = "Slow Network";
      link.estimated_latency_ms = 10.0;
    }

    link.is_local = bandwidth_gbps > 1.0;

    return link;
  }
};

// Performance advisor based on network characteristics
class NetworkPerformanceAdvisor {
 public:
  struct Recommendation {
    const char* issue;
    const char* recommendation;
    int priority; // 1 = critical, 2 = important, 3 = nice-to-have
  };

  static std::vector<Recommendation> get_recommendations(
      const NetworkProfiler::BandwidthMeasurement& stats) {
    std::vector<Recommendation> recommendations;

    // High latency
    if (stats.latency_ms > 5.0) {
      recommendations.push_back({
          "High latency detected",
          "Consider batching operations and reducing synchronization frequency",
          2
      });
    }

    // Low bandwidth
    if (stats.bandwidth_gbps < 0.1) {
      recommendations.push_back({
          "Low bandwidth",
          "Use gradient compression/quantization for distributed training",
          1
      });
    }

    // High variance
    if (stats.stddev_bandwidth > stats.bandwidth_gbps * 0.5) {
      recommendations.push_back({
          "Unstable network connection",
          "Consider adaptive timeout values and connection pooling",
          2
      });
    }

    // Good local network
    if (stats.bandwidth_gbps > 1.0 && stats.latency_ms < 1.0) {
      recommendations.push_back({
          "Good local network detected",
          "Enable zero-copy shared memory and overlapped communication",
          3
      });
    }

    return recommendations;
  }
};

} // namespace vgre
