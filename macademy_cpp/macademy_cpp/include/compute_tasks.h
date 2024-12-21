#pragma once
#include <vector>
#include <span>
#include <memory>
#include <string>
#include <variant>

namespace macademy {
class Network;
class TrainingSuite;
class IBuffer;
class IComputeDevice;

struct UniformDistribution
{
    float range;
};

/// <summary>
/// A class representing an opaque handle to a neural network compiled for a specific device
/// </summary>
struct NetworkResourceHandle
{
    NetworkResourceHandle(Network& network, IComputeDevice& compute_device);

    void SynchronizeNetworkData();

    void AllocateTrainingResources(uint32_t training_sample_count);
    void AllocateBatchEvalResources(uint32_t batch_count) const;
    void AllocateMutationBuffer();

    void FreeCachedResources();

    IComputeDevice* GetComputeDevice() { return m_compute_device; }

    IComputeDevice* m_compute_device = nullptr;
    Network* m_network = nullptr;

    std::unique_ptr<IBuffer> m_weights;
    std::unique_ptr<IBuffer> m_layer_config_buffer;
    mutable std::unique_ptr<IBuffer> m_layer_result_buffer_a;
    mutable std::unique_ptr<IBuffer> m_layer_result_buffer_b;

    std::unique_ptr<IBuffer> m_mutation_buffer;
    std::unique_ptr<IBuffer> m_input_buffer;
    std::unique_ptr<IBuffer> m_desired_output_buffer;
    std::unique_ptr<IBuffer> m_activations_zvalues_buffer;
    std::unique_ptr<IBuffer> m_delta_k_buffer;
    std::unique_ptr<IBuffer> m_gradient_buffer;
};

using MutationDistribution = std::variant<UniformDistribution>;

class ComputeTasks
{
  public:
    std::vector<float> Evaluate(const NetworkResourceHandle& network, std::span<const float> input) const;

    std::vector<float> EvaluateBatch(uint32_t batch_size, const NetworkResourceHandle& network, std::span<const float> input) const;
#if 0
    void ApplyRandomMutation(NetworkResourceHandle& network_handle, MutationDistribution weight_mutation_distribution, MutationDistribution bias_mutation_distribution);

    void Train(NetworkResourceHandle& network, const TrainingSuite& training_suite, uint32_t trainingDataBegin, uint32_t trainingDataEnd) const;
#endif
};
} // namespace macademy