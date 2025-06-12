#pragma once
#include <vector>
#include <span>
#include <memory>
#include <string>
#include <variant>

namespace macademy {
class Network;
struct TrainingSuite;
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
    void AllocateBatchEvalResources() const;
    void AllocateMutationBuffer();

    void FreeCachedResources();

    IComputeDevice* GetComputeDevice() { return m_compute_device; }

    IComputeDevice* const m_compute_device = nullptr;
    Network* const m_network = nullptr;

    std::vector<std::unique_ptr<IBuffer>> m_tensor_buffers;
    mutable std::unique_ptr<IBuffer> m_layer_result_buffer_a;
    mutable std::unique_ptr<IBuffer> m_layer_result_buffer_b;

    std::unique_ptr<IBuffer> m_input_buffer;
    std::unique_ptr<IBuffer> m_desired_output_buffer;
    std::unique_ptr<IBuffer> m_delta_k_buffer_a;
    std::unique_ptr<IBuffer> m_delta_k_buffer_b;
    std::vector<std::unique_ptr<IBuffer>> m_gradient_buffers;
    std::vector<std::unique_ptr<IBuffer>> m_activation_buffers;
    std::vector<std::unique_ptr<IBuffer>> m_zvalue_buffers;

    std::vector<std::unique_ptr<IBuffer>> m_mutation_buffers;
};

using MutationDistribution = std::variant<UniformDistribution>;

class ComputeTasks
{
  public:
    std::vector<float> Evaluate(const NetworkResourceHandle& network, std::span<const float> input) const;

    void TrainMinibatch(NetworkResourceHandle& network, const TrainingSuite& training_suite, uint64_t trainingDataBegin, uint64_t trainingDataEnd) const;

    void ApplyRandomMutation(NetworkResourceHandle& network_handle, MutationDistribution weight_mutation_distribution, MutationDistribution bias_mutation_distribution);
};

} // namespace macademy