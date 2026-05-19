#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <cstring>
#include <string>
#include <map>

namespace vgre {

// Gradient checkpointing: save activations selectively during forward pass
// Recompute them during backward pass to reduce peak memory usage

class ActivationCheckpoint {
 public:
  using ForwardFn = std::function<void(const float*, float*, uint32_t)>;
  using BackwardFn = std::function<void(const float*, const float*, float*, uint32_t)>;

  ActivationCheckpoint() : size_bytes_(0) {}

  // Save activation to checkpoint
  void save(const float* activation, uint32_t num_elements) {
    size_bytes_ = num_elements * sizeof(float);
    data_.resize(num_elements);
    memcpy(data_.data(), activation, size_bytes_);
  }

  // Load activation from checkpoint
  void load(float* activation) const {
    memcpy(activation, data_.data(), size_bytes_);
  }

  bool is_valid() const { return !data_.empty(); }
  uint32_t num_elements() const { return data_.size(); }
  size_t size_bytes() const { return size_bytes_; }

 private:
  std::vector<float> data_;
  size_t size_bytes_;
};

// Gradient checkpointing strategy
enum class CheckpointStrategy {
  NONE,           // No checkpointing
  EVERY_K_LAYERS, // Checkpoint every K layers
  RANDOM,         // Randomly checkpoint with probability p
  BALANCED,       // Balanced memory/compute tradeoff (recommended)
};

class GradientCheckpointManager {
 public:
  GradientCheckpointManager(CheckpointStrategy strategy = CheckpointStrategy::BALANCED,
                           uint32_t checkpoint_every_k = 2,
                           float checkpoint_probability = 0.5f)
      : strategy_(strategy),
        checkpoint_every_k_(checkpoint_every_k),
        checkpoint_probability_(checkpoint_probability),
        total_forward_recomputes_(0),
        total_activation_checkpoints_(0) {}

  // Register a layer for checkpointing
  void register_layer(uint32_t layer_id,
                     const std::string& layer_name,
                     uint32_t activation_size_bytes) {
    LayerInfo info{layer_id, layer_name, activation_size_bytes};
    layers_[layer_id] = info;
  }

  // Should this layer be checkpointed?
  bool should_checkpoint(uint32_t layer_id) const {
    if (strategy_ == CheckpointStrategy::NONE) {
      return false;
    }

    auto it = layers_.find(layer_id);
    if (it == layers_.end()) {
      return false;
    }

    switch (strategy_) {
      case CheckpointStrategy::EVERY_K_LAYERS:
        return (layer_id % checkpoint_every_k_) == 0;
      
      case CheckpointStrategy::RANDOM:
        // Deterministic seeding based on layer ID
        return (((layer_id * 2654435761U) ^ 0x9E3779B9) % 100) < 
               (checkpoint_probability_ * 100);
      
      case CheckpointStrategy::BALANCED:
        // Checkpoint every other layer for typical transformer
        return (layer_id % 2) == 0;
      
      default:
        return false;
    }
  }

  // Save checkpoint
  void save_checkpoint(uint32_t layer_id, const float* activation) {
    auto it = layers_.find(layer_id);
    if (it == layers_.end()) {
      return;
    }

    uint32_t num_elements = it->second.activation_size_bytes / sizeof(float);
    checkpoints_[layer_id].save(activation, num_elements);
    total_activation_checkpoints_++;
  }

  // Load checkpoint
  const ActivationCheckpoint* load_checkpoint(uint32_t layer_id) const {
    auto it = checkpoints_.find(layer_id);
    if (it == checkpoints_.end()) {
      return nullptr;
    }
    return &it->second;
  }

  // Estimate memory savings
  float estimate_memory_savings() const {
    // Typical: save ~50% of activations (every other layer)
    // Results in ~50% memory savings for transformer blocks
    switch (strategy_) {
      case CheckpointStrategy::NONE:
        return 0.0f;
      case CheckpointStrategy::EVERY_K_LAYERS:
        return 1.0f - (1.0f / checkpoint_every_k_);
      case CheckpointStrategy::RANDOM:
        return checkpoint_probability_;
      case CheckpointStrategy::BALANCED:
        return 0.5f; // Every other layer
      default:
        return 0.0f;
    }
  }

  // Compute recompute cost (fraction of original forward time)
  float estimate_recompute_cost() const {
    // Recompute ~50% of layers costs ~50% of forward pass time
    switch (strategy_) {
      case CheckpointStrategy::NONE:
        return 0.0f;
      case CheckpointStrategy::EVERY_K_LAYERS:
        return 1.0f / checkpoint_every_k_;
      case CheckpointStrategy::RANDOM:
        return 1.0f - checkpoint_probability_;
      case CheckpointStrategy::BALANCED:
        return 0.5f;
      default:
        return 0.0f;
    }
  }

  // Reset statistics
  void reset_stats() {
    total_forward_recomputes_ = 0;
    total_activation_checkpoints_ = 0;
  }

  uint64_t get_total_recomputes() const { return total_forward_recomputes_; }
  uint64_t get_total_checkpoints() const { return total_activation_checkpoints_; }

  void record_recompute() { ++total_forward_recomputes_; }

 private:
  struct LayerInfo {
    uint32_t layer_id;
    std::string layer_name;
    uint32_t activation_size_bytes;
  };

  CheckpointStrategy strategy_;
  uint32_t checkpoint_every_k_;
  float checkpoint_probability_;
  std::map<uint32_t, LayerInfo> layers_;
  std::map<uint32_t, ActivationCheckpoint> checkpoints_;
  uint64_t total_forward_recomputes_;
  uint64_t total_activation_checkpoints_;

};

// Example usage pattern for a transformer block:
/*
GradientCheckpointManager gcp(CheckpointStrategy::BALANCED);

// Forward pass
for (uint32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
  auto layer = transformer_layers[layer_id];
  
  // Forward computation
  float* activation = layer.forward(x);
  
  // Checkpoint if needed
  if (gcp.should_checkpoint(layer_id)) {
    gcp.save_checkpoint(layer_id, activation);
  }
  
  x = activation;
}

// Backward pass
for (int32_t layer_id = num_layers - 1; layer_id >= 0; --layer_id) {
  // Recompute forward if checkpointed
  const auto* ckpt = gcp.load_checkpoint(layer_id);
  if (ckpt) {
    gcp.record_recompute();
    x = recompute_forward(layer_id, x);
  }
  
  // Backward computation
  float* grad_input = backward(layer_id, x, grad_output);
}
*/

} // namespace vgre
